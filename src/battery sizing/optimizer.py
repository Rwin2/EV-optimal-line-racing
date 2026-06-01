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
        print(f"           LP: obj_d={obj_decrease:.4f}  max_viol={max_viol:.4f}  "
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


# ── SCP Pareto: speed-only optimization with time+energy objective ───────

def solve_scp_pareto(kappa, ds, car_params, v0, w_time=1.0, w_energy=1.0,
                     T_ref=1.0, E_ref=1.0, rho=5.0, eps=1e-3, max_iters=15):
    """
    Speed-only SCP for the Pareto front: optimize v on a FIXED raceline.

    Same constraints as solve_scp (cornering, accel, braking), same Simplex
    LP solver, but the objective is the weighted time+energy tradeoff:

        J = w_time * T(v)/T_ref  +  w_energy * E(v)/E_ref

    where T = Σ(ds_i/v_i) and E is the net electrical energy (drive - regen).

    Args:
        kappa: (n,) curvature at each station (from vanilla SCP)
        ds:    (n,) arc-length increments (from vanilla SCP raceline)
        car_params: CarParams
        v0:    (n,) initial speed profile (from vanilla SCP)
        w_time, w_energy: Pareto weights
        T_ref, E_ref: normalization constants
    Returns:
        v_opt: (n,) optimized speed profile
    """
    from simplex import solve_lp
    import time as _time

    n = len(kappa)
    p = car_params
    g = 9.81
    rho_init = rho
    a_lat = p.mu * g * 0.85
    a_lon = min(p.F_drive_max / p.mass, p.mu * g * 0.6)
    a_brk = min(p.F_brake_max / p.mass, p.mu * g * 0.9)

    v = v0.copy()
    abs_kappa = np.abs(kappa)
    J_prev = np.inf

    for it in range(max_iters):
        v_next = np.roll(v, -1)
        v_safe = np.maximum(v, 1e-3)

        # ── Current objectives ──
        dt_seg = ds / v_safe
        T = np.sum(dt_seg)

        # Energy model (same as compute_energy)
        dv = np.roll(v, -1) - np.roll(v, 1)  # centered differences
        a_long = dv / (2.0 * np.maximum(dt_seg, 1e-6))
        F_drag = 0.5 * p.rho * p.C_d * p.A_front * v**2
        F_roll = p.C_roll * p.mass * g
        F_tract = p.mass * a_long + F_drag + F_roll
        P_mech = F_tract * v
        P_drive = np.maximum(P_mech, 0.0)
        P_regen = np.minimum(P_mech, 0.0)
        E = np.sum((P_drive / p.eta_motor + P_regen * p.eta_regen) * dt_seg)

        J = w_time * T / T_ref + w_energy * E / E_ref

        dJ = abs(J_prev - J)
        if it > 0 and dJ < eps:
            break
        J_prev = J

        # ── Linearized objective gradient: ∂J/∂v ──
        # ∂T/∂v_i = -ds_i / v_i²
        dT_dv = -ds / v_safe**2

        # ∂E/∂v_i (numerical, small n so cheap)
        eps_fd = 1e-3
        dE_dv = np.zeros(n)
        for i in range(n):
            vp = v.copy(); vp[i] += eps_fd
            vp_safe = np.maximum(vp, 1e-3)
            dt_p = ds / vp_safe
            dv_p = np.roll(vp, -1) - np.roll(vp, 1)
            a_p = dv_p / (2.0 * np.maximum(dt_p, 1e-6))
            Fd_p = 0.5 * p.rho * p.C_d * p.A_front * vp**2
            Fr_p = p.C_roll * p.mass * g
            Pm_p = (p.mass * a_p + Fd_p + Fr_p) * vp
            Pd_p = np.maximum(Pm_p, 0.0)
            Pr_p = np.minimum(Pm_p, 0.0)
            E_p = np.sum((Pd_p / p.eta_motor + Pr_p * p.eta_regen) * dt_p)
            dE_dv[i] = (E_p - E) / eps_fd

        c_obj = w_time / T_ref * dT_dv + w_energy / E_ref * dE_dv

        # ── Linearized constraints (v-only, α is fixed) ──
        # Same as vanilla SCP but without α columns

        # Cornering: c_c_i = a_lat - v_i² |κ_i| >= 0
        dc_c_dv = np.diag(-2.0 * v * abs_kappa)
        c_c0 = a_lat - v**2 * abs_kappa

        # Acceleration: c_a_i = v_i² + 2*a_lon*ds_i - v_{i+1}² >= 0
        dc_a_dv = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dc_a_dv[i, i]   =  2.0 * v[i]
            dc_a_dv[i, ip1] = -2.0 * v_next[i]
        c_a0 = v**2 + 2.0 * a_lon * ds - v_next**2

        # Braking: c_b_i = v_{i+1}² + 2*a_brk*ds_i - v_i² >= 0
        dc_b_dv = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dc_b_dv[i, i]   = -2.0 * v[i]
            dc_b_dv[i, ip1] =  2.0 * v_next[i]
        c_b0 = v_next**2 + 2.0 * a_brk * ds - v**2

        # Stack: -J·Δv ≤ c0  (constraint c >= 0 → -∂c/∂v · Δv ≤ c0)
        A_ub = np.vstack([-dc_c_dv, -dc_a_dv, -dc_b_dv])
        b_ub = np.concatenate([c_c0, c_a0, c_b0])

        # Bounds: trust region ∩ variable bounds
        lb_v = np.maximum(-rho, 1.0 - v)
        ub_v = np.minimum( rho, p.v_max - v)

        # Solve LP with Simplex
        delta = solve_lp(c_obj, A_ub, b_ub, lb_v, ub_v)

        max_viol = float(np.max(np.maximum(A_ub @ delta - b_ub, 0)))
        if max_viol > 0.5:
            delta *= min(0.5 / max(max_viol, 1e-6), 1.0)
            rho = max(rho * 0.5, 0.1)

        v_new = np.clip(v + delta, 1.0, p.v_max)

        # Evaluate actual objective
        dt_new = ds / np.maximum(v_new, 1e-3)
        T_new = np.sum(dt_new)
        dv_new = np.roll(v_new, -1) - np.roll(v_new, 1)
        a_new = dv_new / (2.0 * np.maximum(dt_new, 1e-6))
        Fd_new = 0.5 * p.rho * p.C_d * p.A_front * v_new**2
        Pm_new = (p.mass * a_new + Fd_new + p.C_roll * p.mass * g) * v_new
        E_new = np.sum((np.maximum(Pm_new, 0) / p.eta_motor +
                        np.minimum(Pm_new, 0) * p.eta_regen) * dt_new)
        J_new = w_time * T_new / T_ref + w_energy * E_new / E_ref

        if J_new < J:
            v = v_new
            rho = min(rho * 1.2, rho_init)
        else:
            v = v_new
            rho = max(rho * 0.5, 0.1)

    return v


# ── Joint line+speed Pareto: energy-weighted SCP ─────────────────────────

def solve_scp_pareto_joint(centerline, normals, widths, car_params, alpha0, v0,
                            w_energy=1.0, T_ref=1.0, E_ref=1.0,
                            rho=2.0, eps=5e-3, max_iters=12):
    """
    Joint path+speed SCP with combined time+energy objective:
        J = T(α,v) / T_ref  +  w_energy * E(α,v) / E_ref

    Both the racing line (α) and speed (v) are jointly optimized.
    Warm-started from (alpha0, v0) — typically the min-time solution.
    Higher w_energy pushes towards energy-saving lines (tighter cornering,
    lower speed). Used in a sweep to build the joint Pareto front.

    Returns: alpha (n,), v (n,), T_final (float), E_final (float)
    """
    from simplex import solve_lp

    n    = len(centerline)
    p    = car_params
    grav = 9.81
    a_lat = p.mu * grav * 0.85
    a_lon = min(p.F_drive_max / p.mass, p.mu * grav * 0.6)
    a_brk = min(p.F_brake_max / p.mass, p.mu * grav * 0.9)
    rho_init = rho

    alpha, v = alpha0.copy(), v0.copy()
    J_prev = np.inf

    def _energy(v_, ds_):
        """Net electrical energy in Wh (consistent with compute_lap_energy_time)."""
        v_s = np.maximum(v_, 1e-3)
        dt  = ds_ / v_s
        dv  = np.roll(v_, -1) - np.roll(v_, 1)
        a   = dv / (2.0 * np.maximum(dt, 1e-6))
        F_d = 0.5 * p.rho * p.C_d * p.A_front * v_**2
        F_t = p.mass * a + F_d + p.C_roll * p.mass * grav
        P_m = F_t * v_
        J   = float(np.sum((np.maximum(P_m, 0) / p.eta_motor
                             + np.minimum(P_m, 0) * p.eta_regen) * dt))
        return J / 3600.0  # J → Wh

    for it in range(max_iters):
        path  = alpha_to_raceline(alpha, centerline, normals)
        dpath = np.diff(path, axis=0, append=path[0:1])
        ds    = np.linalg.norm(dpath, axis=1)
        v_next = np.roll(v, -1)
        v_s    = np.maximum(v, 1e-3)

        kappa, dkappa_da = _jax_curvature_jacobian(alpha, centerline, normals)

        T = float(np.sum(ds / v_s))
        E = _energy(v, ds)
        J = T / T_ref + w_energy * E / E_ref
        if it > 0 and abs(J_prev - J) < eps:
            break
        J_prev = J

        # ∂ds/∂α — banded (same as solve_scp)
        dds_da = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dds_da[i, i]   = -np.dot(dpath[i], normals[i])   / ds[i]
            dds_da[i, ip1] +=  np.dot(dpath[i], normals[ip1]) / ds[i]

        # Time gradients ∂T/∂α and ∂T/∂v
        c_T_alpha = dds_da.T @ (1.0 / v_s)
        c_T_v     = -ds / v_s**2

        # Energy gradient ∂E/∂α via ∂E/∂ds: dE[Wh]/dds_i = (F_t/eta) / 3600
        # F_t/eta gives dE in J/m; divide by 3600 to get Wh/m (consistent with E_ref in Wh)
        dt_s   = ds / v_s
        dv_c   = np.roll(v, -1) - np.roll(v, 1)
        a_long = dv_c / (2.0 * np.maximum(dt_s, 1e-6))
        F_drag = 0.5 * p.rho * p.C_d * p.A_front * v**2
        F_t    = p.mass * a_long + F_drag + p.C_roll * p.mass * grav
        dE_dds = np.where(F_t >= 0, F_t / p.eta_motor, F_t * p.eta_regen)
        c_E_alpha = (dds_da.T @ dE_dds) / 3600.0  # J/m → Wh/m per unit alpha

        # Energy gradient ∂E/∂v — finite differences
        c_E_v  = np.zeros(n)
        eps_fd = 1e-3
        for i in range(n):
            vp = v.copy(); vp[i] += eps_fd
            c_E_v[i] = (_energy(vp, ds) - E) / eps_fd

        c_obj = np.concatenate([
            c_T_alpha / T_ref + w_energy * c_E_alpha / E_ref,
            c_T_v     / T_ref + w_energy * c_E_v     / E_ref,
        ])

        # Physics constraints — identical to solve_scp
        sgn_k   = np.where(kappa >= 0, 1.0, -1.0)
        dc_c_da = -(v**2)[:, None] * sgn_k[:, None] * dkappa_da
        dc_c_dv = np.diag(-2.0 * v * np.abs(kappa))
        c_c0    = a_lat - v**2 * np.abs(kappa)

        dc_a_da = 2.0 * a_lon * dds_da
        dc_a_dv = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dc_a_dv[i, i]   =  2.0 * v[i]
            dc_a_dv[i, ip1] = -2.0 * v_next[i]
        c_a0 = v**2 + 2.0 * a_lon * ds - v_next**2

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

        lb_scp = np.concatenate([
            np.maximum(-rho,     -widths - alpha),
            np.maximum(-rho*3.0,  1.0    - v),
        ])
        ub_scp = np.concatenate([
            np.minimum( rho,      widths - alpha),
            np.minimum( rho*3.0,  p.v_max - v),
        ])

        delta     = solve_lp(c_obj, A_ub, b_ub, lb_scp, ub_scp)
        max_viol  = float(np.max(np.maximum(A_ub @ delta - b_ub, 0)))
        if max_viol > 0.5:
            delta *= min(0.5 / max(max_viol, 1e-6), 1.0)
            rho = max(rho * 0.5, 0.1)

        alpha_new = np.clip(alpha + delta[:n], -widths, widths)
        v_new     = np.clip(v     + delta[n:],  1.0, p.v_max)

        path_new = alpha_to_raceline(alpha_new, centerline, normals)
        ds_new   = np.linalg.norm(np.diff(path_new, axis=0, append=path_new[0:1]), axis=1)
        J_new    = (float(np.sum(ds_new / np.maximum(v_new, 1e-3))) / T_ref
                    + w_energy * _energy(v_new, ds_new) / E_ref)

        alpha, v = alpha_new, v_new
        rho = min(rho * 1.2, rho_init) if J_new < J else max(rho * 0.5, 0.1)

    path = alpha_to_raceline(alpha, centerline, normals)
    ds   = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
    return alpha, v, float(np.sum(ds / np.maximum(v, 1e-3))), _energy(v, ds)


def compute_joint_pareto(centerline, normals, widths, car_params,
                          alpha_time, v_time, w_max=8.0, n_pts=20,
                          rho=2.0, eps=5e-3, max_iters=12):
    """
    Build joint line+speed Pareto front by sweeping w_energy in [0, w_max].

    Warm-starts each Pareto point from the previous solution (continuation).
    Returns arrays sorted by T ascending for use as interpolation input.

    Returns:
        param_arr: normalized line-choice parameter in [0, 1]  (0=fastest, 1=most efficient)
        T_arr:     lap time at each Pareto point (s)
        E_arr:     lap energy at each Pareto point (Wh)
    """
    p    = car_params
    grav = 9.81

    # Reference values from min-time solution
    path0 = alpha_to_raceline(alpha_time, centerline, normals)
    ds0   = np.linalg.norm(np.diff(path0, axis=0, append=path0[0:1]), axis=1)
    v_s0  = np.maximum(v_time, 1e-3)
    T_ref = float(np.sum(ds0 / v_s0))

    dt0  = ds0 / v_s0
    dv0  = np.roll(v_time, -1) - np.roll(v_time, 1)
    a0   = dv0 / (2 * np.maximum(dt0, 1e-6))
    F_d0 = 0.5 * p.rho * p.C_d * p.A_front * v_time**2
    F_t0 = p.mass * a0 + F_d0 + p.C_roll * p.mass * grav
    P_m0 = F_t0 * v_time
    E_ref = float(np.sum((np.maximum(P_m0, 0) / p.eta_motor
                          + np.minimum(P_m0, 0) * p.eta_regen) * dt0)) / 3600.0  # J → Wh

    w_arr  = np.concatenate([[0.0], np.geomspace(1e-2, w_max, n_pts - 1)])
    T_list = [T_ref]
    E_list = [E_ref]
    alpha_c, v_c = alpha_time.copy(), v_time.copy()

    n_sweep = len(w_arr) - 1  # number of points after w=0
    for i, w in enumerate(w_arr[1:], 1):
        print(f"    w={w:.3f}  ({i}/{n_sweep})", end='  ', flush=True)
        alpha_c, v_c, T_i, E_i = solve_scp_pareto_joint(
            centerline, normals, widths, car_params, alpha_c, v_c,
            w_energy=w, T_ref=T_ref, E_ref=E_ref,
            rho=rho, eps=eps, max_iters=max_iters)
        T_list.append(T_i)
        E_list.append(E_i)
        print(f"T={T_i:.2f}s  E={E_i:.1f}Wh  TxE={T_i*E_i:.1f}")

    T_arr = np.array(T_list)
    E_arr = np.array(E_list)
    # Sort by T ascending
    idx   = np.argsort(T_arr)
    T_arr, E_arr = T_arr[idx], E_arr[idx]
    # Keep only non-dominated points: scan T ascending, keep iff E reaches a new minimum
    nd_idx = []
    min_E  = np.inf
    for i in range(len(T_arr)):
        if E_arr[i] < min_E:
            min_E = E_arr[i]
            nd_idx.append(i)
    T_arr    = T_arr[nd_idx]
    E_arr    = E_arr[nd_idx]
    n_kept   = len(nd_idx)
    if n_kept < n_pts:
        print(f"  [Pareto] {n_pts - n_kept} dominated point(s) removed, {n_kept} kept.")
    param_arr = np.linspace(0.0, 1.0, n_kept)
    return param_arr, T_arr, E_arr


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

    # Warm-start: minimize curvature with scipy SLSQP to get a good initial path
    print(f"    [pipeline] warm-starting from curvature solution...")
    half_w = widths
    bounds = [(-hw, hw) for hw in half_w]

    def _curv_obj(alpha):
        path = alpha_to_raceline(alpha, centerline, normals)
        kappa = compute_curvature_from_path(path)
        dalpha = np.diff(alpha, append=alpha[0])
        return np.sum(kappa**2) + 1e-5 * np.sum(dalpha**2)

    from scipy.optimize import minimize as scipy_minimize
    res_warm = scipy_minimize(_curv_obj, np.zeros(n_stations), method='SLSQP',
                              bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
    alpha0 = res_warm.x

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

