"""
Racing line optimizer for EV optimal line racing.

Stage 1: Path Optimization — minimize total squared curvature
Stage 2: Velocity Profile — maximize speed subject to friction/power limits
Stage 3: Energy Analysis — compute EV energy consumption and regen

Two path solvers:
  1. Custom: Projected BFGS (Alg 6.6 with bound projection)
  2. Library: scipy.optimize.minimize with SLSQP

SCP solver (joint path+speed):
  - Linearizes nonlinear dynamics constraints around current iterate
  - Solves LP sub-problem with the Simplex algorithm (Ch 12)
  - Gradients computed with JAX automatic differentiation (Ch 2.4)

References:
  - Kochenderfer & Wheeler, Algorithms for Optimization, 2nd ed.
    - BFGS: Algorithm 6.6, eq (6.26)
    - Adam: Algorithm 5.8
    - Simplex: Algorithms 12.1–12.5
"""

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.ndimage import uniform_filter1d


# ── Racing line parameterization ─────────────────────────────────────────

def alpha_to_raceline(alpha, centerline, normals):
    """Convert lateral offsets to a 2D racing line."""
    return centerline + alpha[:, np.newaxis] * normals


def compute_curvature_from_path(path):
    """Compute curvature at each point of a closed 2D path (periodic).

    Uses periodic finite differences to avoid boundary artifacts
    that np.gradient introduces on non-periodic data.
    """
    n = len(path)
    # Pad with wraparound for periodic central differences
    pad = 3
    x = np.concatenate([path[-pad:, 0], path[:, 0], path[:pad, 0]])
    y = np.concatenate([path[-pad:, 1], path[:, 1], path[:pad, 1]])

    dx = np.gradient(x, edge_order=2)
    dy = np.gradient(y, edge_order=2)
    ddx = np.gradient(dx, edge_order=2)
    ddy = np.gradient(dy, edge_order=2)

    # Trim padding
    dx = dx[pad:pad+n]
    dy = dy[pad:pad+n]
    ddx = ddx[pad:pad+n]
    ddy = ddy[pad:pad+n]

    speed = np.sqrt(dx**2 + dy**2)
    speed = np.maximum(speed, 1e-10)
    kappa = (dx * ddy - dy * ddx) / speed**3
    return kappa


# ── Objective functions ──────────────────────────────────────────────────

def objective_curvature(alpha, centerline, normals, car_params=None):
    """f(alpha) = sum of squared curvature + tiny smoothness regularizer."""
    path = alpha_to_raceline(alpha, centerline, normals)
    kappa = compute_curvature_from_path(path)
    f_curv = np.sum(kappa**2)
    dalpha = np.diff(alpha, append=alpha[0])
    f_smooth = 1e-5 * np.sum(dalpha**2)
    return f_curv + f_smooth


def objective_laptime(alpha, centerline, normals, car_params):
    """f(alpha) = lap time = sum(ds / v), where v comes from velocity profile."""
    path = alpha_to_raceline(alpha, centerline, normals)
    kappa = compute_curvature_from_path(path)
    ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
    v = compute_velocity_profile(path, kappa, car_params, ds)
    lap_time = np.sum(ds / np.maximum(v, 0.5))
    # small smoothness regularizer to help convergence
    dalpha = np.diff(alpha, append=alpha[0])
    return lap_time + 1e-4 * np.sum(dalpha**2)


def objective_energy(alpha, centerline, normals, car_params):
    """f(alpha) = net energy consumption (drive - regen)."""
    path = alpha_to_raceline(alpha, centerline, normals)
    kappa = compute_curvature_from_path(path)
    ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
    v = compute_velocity_profile(path, kappa, car_params, ds)
    energy = compute_energy(path, v, car_params, ds)
    dalpha = np.diff(alpha, append=alpha[0])
    return energy['net_energy_kJ'] + 1e-4 * np.sum(dalpha**2)


# Map of available objectives
OBJECTIVES = {
    'curvature': objective_curvature,
    'laptime': objective_laptime,
    'energy': objective_energy,
}


def gradient(alpha, centerline, normals, car_params=None, obj_func=None, eps=1e-5):
    """Finite-difference gradient of the objective."""
    if obj_func is None:
        obj_func = objective_curvature
    n = len(alpha)
    f0 = obj_func(alpha, centerline, normals, car_params)
    grad = np.zeros(n)
    for i in range(n):
        alpha_p = alpha.copy()
        alpha_p[i] += eps
        grad[i] = (obj_func(alpha_p, centerline, normals, car_params) - f0) / eps
    return grad


# ── Projected Adam (Alg 5.8 with bound projection) ──────────────────────

def projected_adam(f, grad_f, x0, lb, ub, max_iter=3000, lr=0.5,
                   b1=0.9, b2=0.999, tol=1e-12):
    """
    Vanilla Adam optimizer (Alg 5.8) with projection onto box [lb, ub].
    Cosine annealing learning rate schedule.
    No budget constraints — runs until convergence or max_iter.
    """
    x = np.clip(x0.copy(), lb, ub)
    ea = 1e-8
    mv = np.zeros_like(x)
    ms = np.zeros_like(x)
    history = [(0, f(x))]

    for t in range(1, max_iter + 1):
        g = grad_f(x)

        # Adam moment updates (Alg 5.8)
        mv = b1 * mv + (1 - b1) * g
        ms = b2 * ms + (1 - b2) * g**2
        m_hat = mv / (1 - b1**t)
        v_hat = ms / (1 - b2**t)

        # cosine annealing learning rate
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / max_iter))
        lr_t = max(lr_t, lr * 0.01)  # floor

        # Adam step + projection
        x -= lr_t * m_hat / (np.sqrt(v_hat) + ea)
        x = np.clip(x, lb, ub)

        # log every 50 iters (cheap — just one f eval)
        if t % 50 == 0:
            fval = f(x)
            history.append((t, fval))

            # convergence check on recent window
            if len(history) > 10:
                recent = [h[1] for h in history[-10:]]
                if max(recent) - min(recent) < tol:
                    break

    return x, history


# ── Projected BFGS (Alg 6.6 with bound projection) ──────────────────────

def projected_bfgs(f, grad_f, x0, lb, ub, max_iter=500, tol=1e-8):
    """
    BFGS quasi-Newton method (Alg 6.6) with projection onto box [lb, ub].
    Used for fine-tuning after Adam warm-start.
    """
    n = len(x0)
    x = np.clip(x0, lb, ub)
    Q = np.eye(n)
    g = grad_f(x)
    f_best = f(x)
    x_best = x.copy()
    history = [(0, f_best)]

    for k in range(1, max_iter + 1):
        pg = x - np.clip(x - g, lb, ub)
        if np.linalg.norm(pg) < tol:
            break

        d = -Q @ g

        # projected backtracking line search
        alpha = 1.0
        fx = f(x)
        for _ in range(40):
            x_trial = np.clip(x + alpha * d, lb, ub)
            if f(x_trial) <= fx + 1e-4 * g @ (x_trial - x):
                break
            alpha *= 0.5
        x_new = np.clip(x + alpha * d, lb, ub)
        g_new = grad_f(x_new)

        f_new = f(x_new)
        if f_new < f_best:
            f_best = f_new
            x_best = x_new.copy()

        # BFGS inverse Hessian update (eq 6.26)
        s = x_new - x
        y = g_new - g
        sy = s @ y
        if sy > 1e-12:
            rho_k = 1.0 / sy
            I = np.eye(n)
            V = I - rho_k * np.outer(s, y)
            Q = V @ Q @ V.T + rho_k * np.outer(s, s)

        x = x_new
        g = g_new
        history.append((k, f_new))

    return x_best, history




# ── Solvers ──────────────────────────────────────────────────────────────

def solve_scipy(centerline, normals, widths, alpha0=None,
                obj='curvature', car_params=None):
    """Solve racing line optimization using scipy SLSQP."""
    n = len(centerline)
    if alpha0 is None:
        alpha0 = np.zeros(n)

    half_w = widths  # widths is already center-to-boundary (half-width)
    bounds = [(-hw, hw) for hw in half_w]
    obj_func = OBJECTIVES[obj]

    def f(alpha):
        return obj_func(alpha, centerline, normals, car_params)

    def g(alpha):
        return gradient(alpha, centerline, normals, car_params, obj_func)

    result = scipy_minimize(f, alpha0, jac=g, method='SLSQP',
                            bounds=bounds,
                            options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
    return result.x, result


def solve_custom(centerline, normals, widths, alpha0=None,
                 obj='curvature', car_params=None):
    """
    Two-phase custom solver:
      Phase 1: Projected Adam (robust global convergence)
      Phase 2: Projected BFGS (fast local refinement)

    Same iteration count for all objectives. For laptime/energy,
    warm-starts from the min-curvature solution (which is cheap to compute).
    """
    n = len(centerline)
    if alpha0 is None:
        alpha0 = np.zeros(n)

    half_w = widths  # widths is already center-to-boundary (half-width)
    lb = -half_w
    ub = half_w
    obj_func = OBJECTIVES[obj]

    def f(alpha):
        return obj_func(alpha, centerline, normals, car_params)

    def g(alpha):
        return gradient(alpha, centerline, normals, car_params, obj_func)

    # For expensive objectives, warm-start from curvature if starting from zero
    if obj in ('laptime', 'energy') and np.allclose(alpha0, 0):
        print(f"    [warm-start] solving curvature first...")
        alpha_warm, _ = solve_custom(centerline, normals, widths,
                                      alpha0, obj='curvature', car_params=car_params)
        alpha0 = alpha_warm

    # Same iterations for all objectives
    adam_iter, bfgs_iter = 1500, 300
    # Lower LR for non-smooth objectives (velocity profile has kinks)
    lr = 0.1 if obj in ('laptime', 'energy') else 0.3

    # Phase 1: Vanilla Adam
    alpha_adam, hist_adam = projected_adam(f, g, alpha0, lb, ub,
                                          max_iter=adam_iter, lr=lr)

    # Phase 2: BFGS refinement
    alpha_opt, hist_bfgs = projected_bfgs(f, g, alpha_adam, lb, ub,
                                           max_iter=bfgs_iter, tol=1e-10)

    # merge histories
    offset = hist_adam[-1][0] if hist_adam else 0
    history = hist_adam + [(h[0] + offset, h[1]) for h in hist_bfgs]

    return alpha_opt, history


# ── SCP: Joint path+speed optimization ───────────────────────────────────

def _jax_curvature_jacobian(alpha, centerline, normals):
    """
    Compute curvature and its Jacobian ∂κ/∂α using JAX autodiff (Ch 2.4).

    Falls back to finite differences if JAX is not available.
    """
    try:
        import jax
        import jax.numpy as jnp

        def _curvature_jax(a):
            path = centerline + a[:, None] * normals
            n = len(a)
            pad = 3
            x = jnp.concatenate([path[-pad:, 0], path[:, 0], path[:pad, 0]])
            y = jnp.concatenate([path[-pad:, 1], path[:, 1], path[:pad, 1]])
            dx = jnp.gradient(x); dy = jnp.gradient(y)
            ddx = jnp.gradient(dx); ddy = jnp.gradient(dy)
            dx = dx[pad:pad+n]; dy = dy[pad:pad+n]
            ddx = ddx[pad:pad+n]; ddy = ddy[pad:pad+n]
            speed = jnp.sqrt(dx**2 + dy**2)
            speed = jnp.maximum(speed, 1e-10)
            return (dx * ddy - dy * ddx) / speed**3

        a_jax = jnp.array(alpha)
        kappa = np.array(_curvature_jax(a_jax))
        dkappa_da = np.array(jax.jacobian(_curvature_jax)(a_jax))
        return kappa, dkappa_da

    except ImportError:
        # Fallback: finite differences
        path = alpha_to_raceline(alpha, centerline, normals)
        kappa = compute_curvature_from_path(path)
        n = len(alpha)
        eps_fd = 1e-4
        dkappa_da = np.zeros((n, n))
        for j in range(n):
            a_p = alpha.copy()
            a_p[j] += eps_fd
            dkappa_da[:, j] = (compute_curvature_from_path(
                alpha_to_raceline(a_p, centerline, normals)) - kappa) / eps_fd
        return kappa, dkappa_da


def solve_scp(centerline, normals, widths, car_params, alpha0=None,
              rho=3.0, eps=1e-2, max_iters=10):
    """
    Joint path+speed optimization via Sequential Convex Programming (SCP).

    Mirrors HW2 cart-pole SCP structure (cartpole_swingup_constrained.py):
      - affinize: linearize nonlinear constraints (cornering/accel/braking)
                  around current iterate (alpha, v)  [like affinize(f, s, u)]
      - scp_iteration: solve LP sub-problem with Simplex (Ch 12)
                       [like scp_iteration with cvxpy in HW2]
      - outer loop: iterate until |ΔJ| < eps or max_iters
                    [like solve_swingup_scp]

    Variables: x = [alpha_0..alpha_{n-1},  v_0..v_{n-1}]
    Objective: min Σ(ds_i / v_i)  (true lap time, linearized per iteration)
    Constraints (all >= 0, linearized):
      - Cornering:    a_lat - v_i² |κ_i(α)| >= 0
      - Acceleration: v_i² + 2*a_lon*ds_i(α) - v_{i+1}² >= 0
      - Braking:      v_{i+1}² + 2*a_brk*ds_i(α) - v_i² >= 0
    Trust region:  |Δα_i| ≤ rho,  |Δv_i| ≤ 3*rho

    LP sub-problem solved with the Simplex algorithm (Alg 12.1–12.5).
    Curvature Jacobian ∂κ/∂α computed with JAX autodiff (Ch 2.4).
    """
    from simplex import solve_lp
    import time

    n = len(centerline)
    p = car_params
    g = 9.81
    rho_init = rho
    a_lat = p.mu * g * 0.85
    a_lon = min(p.F_drive_max / p.mass, p.mu * g * 0.6)
    a_brk = min(p.F_brake_max / p.mass, p.mu * g * 0.9)
    half_w = widths

    # Warm-start: alpha from caller, v from forward-backward pass
    if alpha0 is None:
        alpha0 = np.zeros(n)
    path0 = alpha_to_raceline(alpha0, centerline, normals)
    kappa0 = compute_curvature_from_path(path0)
    ds0 = np.linalg.norm(np.diff(path0, axis=0, append=path0[0:1]), axis=1)
    alpha = alpha0.copy()
    v = compute_velocity_profile(path0, kappa0, car_params, ds0)

    J_prev = np.inf
    history = []

    for it in range(max_iters):
        t_iter = time.time()

        # ── Current quantities ──
        path = alpha_to_raceline(alpha, centerline, normals)
        dpath = np.diff(path, axis=0, append=path[0:1])
        ds = np.linalg.norm(dpath, axis=1)
        v_next = np.roll(v, -1)

        # Curvature + Jacobian via JAX (or FD fallback)
        kappa, dkappa_da = _jax_curvature_jacobian(alpha, centerline, normals)

        J = np.sum(ds / np.maximum(v, 1e-3))
        history.append(J)
        dJ = abs(J_prev - J)
        print(f"    [SCP] iter {it:2d}: J={J:.4f}  dJ={dJ:.5f}")
        if it > 0 and dJ < eps:
            print(f"    [SCP] converged after {it} iterations.")
            break
        J_prev = J

        # ── Affinize: compute constraint Jacobians ──
        # (like affinize(f, s, u) in HW2 cartpole_swingup_constrained.py)

        # Analytical ∂ds_i/∂α_j — banded: only j=i and j=(i+1)%n nonzero
        dds_da = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dds_da[i, i]   = -np.dot(dpath[i], normals[i])   / ds[i]
            dds_da[i, ip1] +=  np.dot(dpath[i], normals[ip1]) / ds[i]

        sgn_k = np.where(kappa >= 0, 1.0, -1.0)

        # ── Linearized objective gradient ──
        c_obj = np.concatenate([
            dds_da.T @ (1.0 / np.maximum(v, 1e-3)),
            -ds / np.maximum(v, 1e-3)**2,
        ])

        # ── Linearized constraints: A_ub @ Δx ≤ b_ub ──

        # Cornering: c_c_i = a_lat - v_i² |κ_i|
        dc_c_da = -(v**2)[:, None] * sgn_k[:, None] * dkappa_da
        dc_c_dv = np.diag(-2.0 * v * np.abs(kappa))
        c_c0 = a_lat - v**2 * np.abs(kappa)

        # Acceleration: c_a_i = v_i² + 2*a_lon*ds_i - v_{i+1}²
        dc_a_da = 2.0 * a_lon * dds_da
        dc_a_dv = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dc_a_dv[i, i]   =  2.0 * v[i]
            dc_a_dv[i, ip1] = -2.0 * v_next[i]
        c_a0 = v**2 + 2.0 * a_lon * ds - v_next**2

        # Braking: c_b_i = v_{i+1}² + 2*a_brk*ds_i - v_i²
        dc_b_da = 2.0 * a_brk * dds_da
        dc_b_dv = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dc_b_dv[i, i]   = -2.0 * v[i]
            dc_b_dv[i, ip1] =  2.0 * v_next[i]
        c_b0 = v_next**2 + 2.0 * a_brk * ds - v**2

        A_ub = np.vstack([
            np.hstack([-dc_c_da, -dc_c_dv]),
            np.hstack([-dc_a_da, -dc_a_dv]),
            np.hstack([-dc_b_da, -dc_b_dv]),
        ])
        b_ub = np.concatenate([c_c0, c_a0, c_b0])

        # ── Bounds: trust region ∩ absolute variable bounds ──
        lb_scp = np.concatenate([
            np.maximum(-rho,      -half_w - alpha),
            np.maximum(-rho*3.0,   1.0    - v),
        ])
        ub_scp = np.concatenate([
            np.minimum( rho,       half_w - alpha),
            np.minimum( rho*3.0,   p.v_max - v),
        ])

        # ── Solve LP sub-problem with Simplex (Alg 12.1–12.5) ──
        #
        #   min  c_obj^T Δx         (linearized lap time)
        #   s.t. A_ub Δx ≤ b_ub    (linearized physics constraints)
        #        lb ≤ Δx ≤ ub       (trust region + absolute bounds)
        #
        # Converted to equality form (eq 12.8–12.10) with slack variables,
        # then solved with the two-phase simplex (Alg 12.5).
        t_lp = time.time()
        delta = solve_lp(c_obj, A_ub, b_ub, lb_scp, ub_scp)
        t_lp = time.time() - t_lp

        max_viol = float(np.max(np.maximum(A_ub @ delta - b_ub, 0)))
        obj_decrease = float(c_obj @ delta)
        print(f"           LP: obj_Δ={obj_decrease:.4f}  max_viol={max_viol:.4f}  "
              f"rho={rho:.2f}  time={t_lp:.1f}s  total={time.time()-t_iter:.1f}s")

        # Adaptive trust region (like ρ in HW2 solve_swingup_scp)
        if max_viol > 0.5:
            damping = min(0.5 / max(max_viol, 1e-6), 1.0)
            delta *= damping
            rho = max(rho * 0.5, 0.1)
            print(f"    [SCP] damping step by {damping:.2f}, rho -> {rho:.2f}")

        # Accept step
        alpha_new = np.clip(alpha + delta[:n], -half_w, half_w)
        v_new     = np.clip(v     + delta[n:],  1.0, p.v_max)

        # Evaluate actual objective at new point
        path_new = alpha_to_raceline(alpha_new, centerline, normals)
        ds_new = np.linalg.norm(np.diff(path_new, axis=0, append=path_new[0:1]), axis=1)
        J_new = np.sum(ds_new / np.maximum(v_new, 1e-3))

        if J_new < J:
            # Good step — accept and maybe grow trust region
            alpha, v = alpha_new, v_new
            rho = min(rho * 1.2, rho_init)
        else:
            # Step increased objective — accept but shrink trust region
            alpha, v = alpha_new, v_new
            rho = max(rho * 0.5, 0.1)
            print(f"    [SCP] J increased ({J:.4f} -> {J_new:.4f}), rho -> {rho:.2f}")

    return alpha, v, history


# ── Stage 2: Velocity profile optimization ───────────────────────────────

def compute_velocity_profile(path, kappa, car_params, ds=None):
    """
    Compute optimal velocity at each point of a racing line.

    Uses forward-backward integration:
    1. Cornering limit: v_max = sqrt(a_lat_max / |kappa|)
    2. Forward pass: acceleration-limited
    3. Backward pass: braking-limited
    4. Power limit: v <= P_max / F_drive

    Args:
        path: (N, 2) racing line points
        kappa: (N,) curvature at each point
        car_params: CarParams with mu, P_max, mass, etc.
        ds: (N,) arc-length increments (computed if None)
    Returns:
        v_profile: (N,) optimal speed at each point (m/s)
    """
    n = len(path)
    p = car_params

    # arc-length increments
    if ds is None:
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)

    g = 9.81
    # friction limits
    a_lat_max = p.mu * g * 0.85  # 85% of max grip for safety margin
    a_lon_max = min(p.F_drive_max / p.mass, p.mu * g * 0.6)  # acceleration
    a_brake = min(p.F_brake_max / p.mass, p.mu * g * 0.9)     # braking

    # Step 1: cornering speed limit
    abs_kappa = np.abs(kappa)
    abs_kappa = np.maximum(abs_kappa, 1e-6)
    v_corner = np.sqrt(a_lat_max / abs_kappa)
    v_corner = np.minimum(v_corner, p.v_max)

    # Step 2: power limit (vectorized)
    # P_max >= v * (0.5*rho*Cd*A*v^2 + Crr*m*g) => solve for max v
    F_roll_const = p.C_roll * p.mass * g
    v_candidates = np.linspace(p.v_max, 5.0, 50)
    P_needed = v_candidates * (0.5 * p.rho * p.C_d * p.A_front * v_candidates**2 + F_roll_const)
    # find highest v where P_needed <= P_max
    valid = P_needed <= p.P_max
    v_power_limit = v_candidates[np.argmax(valid)] if np.any(valid) else 5.0

    v_profile = np.minimum(v_corner, v_power_limit)

    # Step 3: forward pass (acceleration limited)
    # v_{i+1}^2 <= v_i^2 + 2 * a_max * ds_i
    for i in range(1, 2 * n):  # two laps to handle wraparound
        idx = i % n
        prev = (i - 1) % n
        v_max_accel = np.sqrt(max(v_profile[prev]**2 + 2 * a_lon_max * ds[prev], 0))
        v_profile[idx] = min(v_profile[idx], v_max_accel)

    # Step 4: backward pass (braking limited)
    for i in range(2 * n - 2, -1, -1):
        idx = i % n
        nxt = (i + 1) % n
        v_max_brake = np.sqrt(max(v_profile[nxt]**2 + 2 * a_brake * ds[idx], 0))
        v_profile[idx] = min(v_profile[idx], v_max_brake)

    return v_profile


# ── Stage 3: EV energy analysis ──────────────────────────────────────────

def compute_energy(path, v_profile, car_params, ds=None):
    """
    Compute energy consumption along a racing line with given velocity profile.

    Returns dict with:
        drive_energy_kJ: total energy from battery
        regen_energy_kJ: energy recovered through regenerative braking
        net_energy_kJ: total - regen
        energy_per_m: J/m average
        lap_time_s: total time for one lap
        segment_power: (N,) power at each segment (W)
        segment_time: (N,) time for each segment (s)
    """
    n = len(path)
    p = car_params
    g = 9.81

    if ds is None:
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)

    dt_seg = ds / np.maximum(v_profile, 0.5)  # time per segment

    # acceleration at each point
    dv = np.gradient(v_profile)
    a_lon = dv / np.maximum(dt_seg, 1e-6)

    drive_energy = 0.0   # energy drawn from battery (driving)
    regen_energy = 0.0   # energy returned to battery (braking)
    segment_power = np.zeros(n)

    for i in range(n):
        v = v_profile[i]
        a = a_lon[i]

        F_drag = 0.5 * p.rho * p.C_d * p.A_front * v**2
        F_roll = p.C_roll * p.mass * g
        F_traction = p.mass * a + F_drag + F_roll
        P_mech = F_traction * v

        if P_mech >= 0:
            P_elec = P_mech / p.eta_motor
            drive_energy += P_elec * dt_seg[i]
        else:
            P_regen = abs(P_mech) * p.eta_regen
            regen_energy += P_regen * dt_seg[i]

        segment_power[i] = P_mech

    net_energy = drive_energy - regen_energy
    lap_time = np.sum(dt_seg)

    return {
        'drive_energy_kJ': drive_energy / 1000,
        'regen_energy_kJ': regen_energy / 1000,
        'net_energy_kJ': net_energy / 1000,
        'energy_per_m': net_energy / np.sum(ds),
        'lap_time_s': lap_time,
        'avg_speed_kmh': np.sum(ds) / lap_time * 3.6,
        'max_speed_kmh': np.max(v_profile) * 3.6,
        'segment_power': segment_power,
        'segment_time': dt_seg,
    }


# ── Full pipeline ────────────────────────────────────────────────────────

def optimize_racing_line(track, n_stations=200, solver='both', obj='curvature'):
    """
    Full racing line optimization pipeline.

    Args:
        solver: 'custom' (Adam+BFGS, decoupled),
                'scipy'  (SLSQP, decoupled),
                'both'   (custom + scipy, decoupled),
                'scp'    (joint path+speed SCP — always minimizes lap time)
        obj: 'curvature' (min sum kappa^2),
             'laptime'   (min lap time),
             'energy'    (min net energy)
             (ignored when solver='scp')
    """
    from car import CarParams

    # subsample
    n_full = len(track.centerline)
    idx = np.linspace(0, n_full - 1, n_stations, dtype=int)
    centerline = track.centerline[idx]
    normals = track.normals[idx]
    widths = track.widths[idx]

    car = CarParams()
    results = {'objective': obj}

    # Curvature warm-start for laptime/energy decoupled solvers and SCP
    if obj in ('laptime', 'energy') or solver == 'scp':
        print(f"    [pipeline] warm-starting from curvature solution...")
        alpha0, _ = solve_scipy(centerline, normals, widths,
                                 alpha0=np.zeros(n_stations),
                                 obj='curvature', car_params=car)
    else:
        alpha0 = np.zeros(n_stations)

    # Decoupled solvers (path only, then velocity profile separately)
    if solver in ('custom', 'both'):
        alpha_c, hist_c = solve_custom(centerline, normals, widths, alpha0.copy(),
                                        obj=obj, car_params=car)
        results['alpha_custom'] = alpha_c
        results['history_custom'] = hist_c

    if solver in ('scipy', 'both'):
        alpha_s, res_s = solve_scipy(centerline, normals, widths, alpha0.copy(),
                                      obj=obj, car_params=car)
        results['alpha_scipy'] = alpha_s
        results['result_scipy'] = res_s

    # SCP: joint path+speed optimizer (always minimizes lap time)
    if solver == 'scp':
        print(f"    [pipeline] running joint SCP optimizer...")
        alpha_scp, v_scp, hist_scp = solve_scp(
            centerline, normals, widths, car, alpha0=alpha0)
        results['alpha_scp'] = alpha_scp
        results['v_scp_raw'] = v_scp
        results['history_scp'] = hist_scp

    # interpolate to full resolution
    from scipy.interpolate import interp1d
    t_sub = np.linspace(0, 1, n_stations)
    t_full = np.linspace(0, 1, n_full)

    smooth_size = max(3, n_full // n_stations)

    for key in ['custom', 'scipy', 'scp']:
        alpha_key = f'alpha_{key}'
        if alpha_key not in results:
            continue

        alpha_sub = results[alpha_key]

        # interpolate and clamp alpha to full resolution
        alpha_full = interp1d(t_sub, alpha_sub, kind='cubic',
                              fill_value='extrapolate')(t_full)
        alpha_full = np.clip(alpha_full, -track.widths, track.widths)
        alpha_full = uniform_filter1d(alpha_full, size=smooth_size, mode='wrap')

        raceline = alpha_to_raceline(alpha_full, track.centerline, track.normals)
        kappa = compute_curvature_from_path(raceline)

        # SCP: also interpolate the jointly-optimized velocity profile
        if key == 'scp' and 'v_scp_raw' in results:
            v_sub = results['v_scp_raw']
            v_profile = interp1d(t_sub, v_sub, kind='cubic',
                                 fill_value='extrapolate')(t_full)
            v_profile = np.clip(v_profile, 1.0, car.v_max)
            v_profile = uniform_filter1d(v_profile, size=smooth_size, mode='wrap')
        else:
            v_profile = compute_velocity_profile(raceline, kappa, car)

        energy = compute_energy(raceline, v_profile, car)

        results[f'raceline_{key}'] = raceline
        results[f'curvature_{key}'] = kappa
        results[f'velocity_{key}'] = v_profile
        results[f'energy_{key}'] = energy
        results[f'alpha_full_{key}'] = alpha_full

    # centerline baseline
    kappa_center = compute_curvature_from_path(track.centerline)
    v_center = compute_velocity_profile(track.centerline, kappa_center, car)
    e_center = compute_energy(track.centerline, v_center, car)
    results['curvature_center'] = kappa_center
    results['velocity_center'] = v_center
    results['energy_center'] = e_center
    results['centerline'] = track.centerline

    # subsampled curvatures for comparison table
    rl_sub_center = centerline
    results['curvature_sub_center'] = compute_curvature_from_path(rl_sub_center)
    for key in ['custom', 'scipy', 'scp']:
        ak = f'alpha_{key}'
        if ak in results:
            rl_sub = alpha_to_raceline(results[ak], centerline, normals)
            results[f'curvature_sub_{key}'] = compute_curvature_from_path(rl_sub)

    return results

