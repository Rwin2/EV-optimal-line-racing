"""
Racing line optimizer for EV optimal line racing.

Stage 1: Path Optimization — minimize total squared curvature
Stage 2: Velocity Profile — maximize speed subject to friction/power limits
Stage 3: Energy Analysis — compute EV energy consumption and regen

Two path solvers:
  1. Custom: Projected BFGS (Alg 6.6 with bound projection)
  2. Library: scipy.optimize.minimize with SLSQP

References:
  - Kochenderfer & Wheeler, Algorithms for Optimization, 2nd ed.
    - BFGS: Algorithm 6.6, eq (6.26)
    - Augmented Lagrangian: Algorithm 10.3, eq (10.37)-(10.38)
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


# ── Objective and gradient ───────────────────────────────────────────────

def objective(alpha, centerline, normals):
    """f(alpha) = sum of squared curvature + tiny smoothness regularizer."""
    path = alpha_to_raceline(alpha, centerline, normals)
    kappa = compute_curvature_from_path(path)
    f_curv = np.sum(kappa**2)
    dalpha = np.diff(alpha, append=alpha[0])
    f_smooth = 1e-5 * np.sum(dalpha**2)
    return f_curv + f_smooth


def gradient(alpha, centerline, normals, eps=1e-5):
    """Finite-difference gradient of the objective."""
    n = len(alpha)
    f0 = objective(alpha, centerline, normals)
    grad = np.zeros(n)
    for i in range(n):
        alpha_p = alpha.copy()
        alpha_p[i] += eps
        grad[i] = (objective(alpha_p, centerline, normals) - f0) / eps
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


# ── Augmented Lagrangian + BFGS (Algorithm 10.3) — kept for reference ────

def _backtracking_line_search(f, x, d, g, alpha0=1.0, c1=1e-4, rho=0.5):
    """Backtracking line search satisfying Armijo condition."""
    a = alpha0
    fx = f(x)
    slope = g @ d
    for _ in range(30):
        if f(x + a * d) <= fx + c1 * a * slope:
            return a
        a *= rho
    return a


def bfgs_unconstrained(f, grad_f, x0, max_iter=200, tol=1e-6):
    """Standard BFGS (Alg 6.6) for unconstrained subproblems."""
    n = len(x0)
    x = x0.copy()
    Q = np.eye(n)
    g = grad_f(x)

    for k in range(1, max_iter + 1):
        if np.linalg.norm(g) < tol:
            break
        d = -Q @ g
        step = _backtracking_line_search(f, x, d, g)
        x_new = x + step * d
        g_new = grad_f(x_new)

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

    return x


def augmented_lagrangian_bfgs(f, grad_f, constraints, jac_c, x0, lb, ub,
                               rho0=1.0, gamma=2.0, k_outer=20, bfgs_iter=200):
    """
    Augmented Lagrangian (Alg 10.3) with projected BFGS inner solver.
    Handles inequality constraints c(x) <= 0, plus box bounds.
    """
    x = np.clip(x0, lb, ub)
    cx = constraints(x)
    m = len(cx)
    lam = np.zeros(m)
    rho = rho0
    history = []

    for outer in range(k_outer):
        def L(z):
            fz = f(z)
            cz = constraints(z)
            act = np.maximum(lam + rho * cz, 0.0)
            return fz + np.sum(act**2) / (2.0 * rho)

        def grad_L(z):
            gf = grad_f(z)
            cz = constraints(z)
            J = jac_c(z)
            act = np.maximum(lam + rho * cz, 0.0)
            return gf + J.T @ act

        x, _ = projected_bfgs(L, grad_L, x, lb, ub, max_iter=bfgs_iter, tol=1e-7)

        cx = constraints(x)
        lam = np.maximum(lam + rho * cx, 0.0)
        rho = min(rho * gamma, 1e6)

        fval = f(x)
        max_viol = np.max(np.maximum(cx, 0.0))
        history.append((outer, fval, max_viol, rho))

        if max_viol < 1e-6 and outer > 2:
            break

    return x, history


# ── Constraint helpers ───────────────────────────────────────────────────

def make_track_constraints(widths):
    """
    c(alpha) <= 0 where:
      c_{2i}   =  alpha_i - w_i/2
      c_{2i+1} = -alpha_i - w_i/2
    """
    half_w = widths / 2.0
    n = len(widths)

    def constraints(alpha):
        upper = alpha - half_w
        lower = -alpha - half_w
        return np.concatenate([upper, lower])

    def jacobian(alpha):
        # analytical: top block = +I, bottom block = -I
        J = np.zeros((2 * n, n))
        J[:n, :] = np.eye(n)
        J[n:, :] = -np.eye(n)
        return J

    return constraints, jacobian


# ── Solvers ──────────────────────────────────────────────────────────────

def solve_scipy(centerline, normals, widths, alpha0=None):
    """Solve racing line optimization using scipy SLSQP."""
    n = len(centerline)
    if alpha0 is None:
        alpha0 = np.zeros(n)

    half_w = widths / 2.0
    bounds = [(-hw, hw) for hw in half_w]

    def f(alpha):
        return objective(alpha, centerline, normals)

    def g(alpha):
        return gradient(alpha, centerline, normals)

    result = scipy_minimize(f, alpha0, jac=g, method='SLSQP',
                            bounds=bounds,
                            options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
    return result.x, result


def solve_custom(centerline, normals, widths, alpha0=None):
    """
    Two-phase custom solver:
      Phase 1: Projected Adam (robust global convergence)
      Phase 2: Projected BFGS (fast local refinement)
    """
    n = len(centerline)
    if alpha0 is None:
        alpha0 = np.zeros(n)

    half_w = widths / 2.0
    lb = -half_w
    ub = half_w

    def f(alpha):
        return objective(alpha, centerline, normals)

    def g(alpha):
        return gradient(alpha, centerline, normals)

    # Phase 1: Vanilla Adam — run until convergence, no budget limit
    alpha_adam, hist_adam = projected_adam(f, g, alpha0, lb, ub,
                                          max_iter=3000, lr=0.5)

    # Phase 2: BFGS refinement from Adam solution
    alpha_opt, hist_bfgs = projected_bfgs(f, g, alpha_adam, lb, ub,
                                           max_iter=500, tol=1e-10)

    # merge histories
    offset = hist_adam[-1][0] if hist_adam else 0
    history = hist_adam + [(h[0] + offset, h[1]) for h in hist_bfgs]

    return alpha_opt, history


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

    # Step 2: power limit
    # P = F_drive * v, at high speed: F_drive = P_max/v
    # drag + rolling at speed v: F_resist = 0.5*rho*Cd*A*v^2 + Crr*m*g
    # net: P_max/v - F_resist >= m*a => P_max >= v*(m*a + F_resist)
    # for max speed: P_max = v * (0.5*rho*Cd*A*v^2 + Crr*m*g)
    # solve cubic for v_power_limit... approximate:
    v_power = np.full(n, p.v_max)
    for i in range(n):
        # find max v where P_max >= v * resistance(v)
        for v_try in np.linspace(p.v_max, 5.0, 50):
            F_drag = 0.5 * p.rho * p.C_d * p.A_front * v_try**2
            F_roll = p.C_roll * p.mass * g
            P_needed = v_try * (F_drag + F_roll)
            if P_needed <= p.P_max:
                v_power[i] = v_try
                break

    v_profile = np.minimum(v_corner, v_power)

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

def optimize_racing_line(track, n_stations=200, solver='both'):
    """
    Full racing line optimization pipeline:
      1. Subsample track
      2. Optimize path (minimize curvature)
      3. Compute velocity profile
      4. Compute energy consumption
    """
    from car import CarParams

    # subsample
    n_full = len(track.centerline)
    idx = np.linspace(0, n_full - 1, n_stations, dtype=int)
    centerline = track.centerline[idx]
    normals = track.normals[idx]
    widths = track.widths[idx]
    alpha0 = np.zeros(n_stations)

    results = {}

    # solve
    if solver in ('custom', 'both'):
        alpha_c, hist_c = solve_custom(centerline, normals, widths, alpha0.copy())
        results['alpha_custom'] = alpha_c
        results['history_custom'] = hist_c

    if solver in ('scipy', 'both'):
        alpha_s, res_s = solve_scipy(centerline, normals, widths, alpha0.copy())
        results['alpha_scipy'] = alpha_s
        results['result_scipy'] = res_s

    # interpolate to full resolution
    from scipy.interpolate import interp1d
    t_sub = np.linspace(0, 1, n_stations)
    t_full = np.linspace(0, 1, n_full)

    car = CarParams()

    for key in ['custom', 'scipy']:
        alpha_key = f'alpha_{key}'
        if alpha_key not in results:
            continue

        alpha_sub = results[alpha_key]

        # interpolate and clamp
        alpha_full = interp1d(t_sub, alpha_sub, kind='cubic',
                              fill_value='extrapolate')(t_full)
        alpha_full = np.clip(alpha_full, -track.widths / 2, track.widths / 2)
        # smooth proportional to upsampling ratio to avoid interpolation artifacts
        smooth_size = max(3, n_full // n_stations)
        alpha_full = uniform_filter1d(alpha_full, size=smooth_size, mode='wrap')

        raceline = alpha_to_raceline(alpha_full, track.centerline, track.normals)
        kappa = compute_curvature_from_path(raceline)
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
    for key in ['custom', 'scipy']:
        ak = f'alpha_{key}'
        if ak in results:
            rl_sub = alpha_to_raceline(results[ak], centerline, normals)
            results[f'curvature_sub_{key}'] = compute_curvature_from_path(rl_sub)

    return results


# ── Comparison report ────────────────────────────────────────────────────

def print_comparison(track, results):
    """Print comparison table for all solvers."""
    print(f"\n{'='*70}")
    print(f"  {track.name} — Optimization Results")
    print(f"  Track length: {track.length:.0f} m")
    print(f"{'='*70}")

    has_custom = 'alpha_custom' in results
    has_scipy = 'alpha_scipy' in results

    # path metrics
    print(f"\n  PATH OPTIMIZATION (minimize curvature)")
    print(f"  {'Metric':<28} {'Centerline':>12}", end='')
    if has_custom: print(f" {'Custom':>12}", end='')
    if has_scipy:  print(f" {'Scipy':>12}", end='')
    print()
    print(f"  {'─'*64}")

    kc = results['curvature_sub_center']
    row = lambda name, vals: _print_row(name, vals, has_custom, has_scipy)

    vals = [np.sum(kc**2)]
    if has_custom: vals.append(np.sum(results['curvature_sub_custom']**2))
    if has_scipy:  vals.append(np.sum(results['curvature_sub_scipy']**2))
    row('Sum kappa^2', vals)

    vals = [np.max(np.abs(kc))]
    if has_custom: vals.append(np.max(np.abs(results['curvature_sub_custom'])))
    if has_scipy:  vals.append(np.max(np.abs(results['curvature_sub_scipy'])))
    row('Max |kappa|', vals)

    # velocity metrics
    print(f"\n  VELOCITY PROFILE")
    vc = results['velocity_center']
    vals = [np.mean(vc) * 3.6]
    if has_custom: vals.append(np.mean(results['velocity_custom']) * 3.6)
    if has_scipy:  vals.append(np.mean(results['velocity_scipy']) * 3.6)
    row('Avg speed (km/h)', vals)

    vals = [np.max(vc) * 3.6]
    if has_custom: vals.append(np.max(results['velocity_custom']) * 3.6)
    if has_scipy:  vals.append(np.max(results['velocity_scipy']) * 3.6)
    row('Max speed (km/h)', vals)

    vals = [results['energy_center']['lap_time_s']]
    if has_custom: vals.append(results['energy_custom']['lap_time_s'])
    if has_scipy:  vals.append(results['energy_scipy']['lap_time_s'])
    row('Lap time (s)', vals)

    # energy metrics
    print(f"\n  ENERGY (EV)")
    vals = [results['energy_center']['drive_energy_kJ']]
    if has_custom: vals.append(results['energy_custom']['drive_energy_kJ'])
    if has_scipy:  vals.append(results['energy_scipy']['drive_energy_kJ'])
    row('Drive energy (kJ)', vals)

    vals = [results['energy_center']['regen_energy_kJ']]
    if has_custom: vals.append(results['energy_custom']['regen_energy_kJ'])
    if has_scipy:  vals.append(results['energy_scipy']['regen_energy_kJ'])
    row('Regen recovered (kJ)', vals)

    vals = [results['energy_center']['net_energy_kJ']]
    if has_custom: vals.append(results['energy_custom']['net_energy_kJ'])
    if has_scipy:  vals.append(results['energy_scipy']['net_energy_kJ'])
    row('Net energy (kJ)', vals)

    vals = [results['energy_center']['energy_per_m']]
    if has_custom: vals.append(results['energy_custom']['energy_per_m'])
    if has_scipy:  vals.append(results['energy_scipy']['energy_per_m'])
    row('Net energy/meter (J/m)', vals)

    # solver agreement
    if has_custom and has_scipy:
        ac = results['alpha_custom']
        asc = results['alpha_scipy']
        print(f"\n  SOLVER AGREEMENT")
        print(f"    Mean |alpha_diff|  = {np.mean(np.abs(ac - asc)):.4f}")
        print(f"    Max  |alpha_diff|  = {np.max(np.abs(ac - asc)):.4f}")
        print(f"    Correlation        = {np.corrcoef(ac, asc)[0,1]:.6f}")


def _print_row(name, vals, has_custom, has_scipy):
    print(f"  {name:<28} {vals[0]:>12.4f}", end='')
    i = 1
    if has_custom: print(f" {vals[i]:>12.4f}", end=''); i += 1
    if has_scipy:  print(f" {vals[i]:>12.4f}", end=''); i += 1
    print()


# ── Visualization ────────────────────────────────────────────────────────

def plot_full_analysis(track, results, output_path=None):
    """Generate comprehensive 4-panel figure."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    has_custom = 'raceline_custom' in results
    has_scipy = 'raceline_scipy' in results

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='#111')
    fig.suptitle(f'{track.name} — Racing Line Optimization',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    # ── Panel 1: Track with racing lines ──
    ax = axes[0, 0]
    ax.set_facecolor('#1a3a15')
    road = np.vstack([track.left_boundary, track.right_boundary[::-1],
                      track.left_boundary[0:1]])
    ax.add_patch(Polygon(road, closed=True, fc='#3a3a3a', ec='none', zorder=1))
    ax.plot(track.left_boundary[:, 0], track.left_boundary[:, 1],
            'w-', lw=0.8, alpha=0.5, zorder=3)
    ax.plot(track.right_boundary[:, 0], track.right_boundary[:, 1],
            'w-', lw=0.8, alpha=0.5, zorder=3)

    cl = results['centerline']
    ax.plot(cl[:, 0], cl[:, 1], '-', color='#ffdd00', lw=2.5,
            label='Centerline', zorder=4, alpha=0.9)
    if has_custom:
        rl = results['raceline_custom']
        ax.plot(rl[:, 0], rl[:, 1], '-', color='#ff3333', lw=1.8,
                label='Custom (Adam+BFGS)', zorder=5)
    if has_scipy:
        rl = results['raceline_scipy']
        ax.plot(rl[:, 0], rl[:, 1], '--', color='#33ccff', lw=1.8,
                label='Scipy (SLSQP)', zorder=6, alpha=0.9)

    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9, facecolor='#222',
              edgecolor='#555', labelcolor='white')
    ax.set_title('Racing Lines', color='white', fontsize=13, fontweight='bold')
    ax.axis('off')

    # ── Panel 2: Velocity heatmap on racing line ──
    ax2 = axes[0, 1]
    ax2.set_facecolor('#1a3a15')
    ax2.add_patch(Polygon(road.copy(), closed=True, fc='#3a3a3a', ec='none', zorder=1))
    ax2.plot(track.left_boundary[:, 0], track.left_boundary[:, 1],
             'w-', lw=0.5, alpha=0.3, zorder=3)
    ax2.plot(track.right_boundary[:, 0], track.right_boundary[:, 1],
             'w-', lw=0.5, alpha=0.3, zorder=3)

    # pick best available solver for heatmap
    if has_scipy:
        rl_heat = results['raceline_scipy']
        v_heat = results['velocity_scipy']
        heat_label = 'Scipy'
    elif has_custom:
        rl_heat = results['raceline_custom']
        v_heat = results['velocity_custom']
        heat_label = 'Custom'
    else:
        rl_heat = cl
        v_heat = results['velocity_center']
        heat_label = 'Center'

    # colored line segments by velocity
    points = rl_heat.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='RdYlGn', linewidth=3, zorder=5)
    lc.set_array(v_heat[:-1] * 3.6)  # km/h
    ax2.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label('Speed (km/h)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    ax2.set_aspect('equal')
    ax2.set_title(f'Velocity Heatmap ({heat_label})', color='white',
                  fontsize=13, fontweight='bold')
    ax2.axis('off')

    # ── Panel 3: Speed profile comparison ──
    ax3 = axes[1, 0]
    ax3.set_facecolor('#1a1a1a')
    n = len(results['velocity_center'])
    s = np.linspace(0, 100, n)

    ax3.plot(s, results['velocity_center'] * 3.6, '-', color='#ffdd00',
             lw=1.2, label='Centerline', alpha=0.7)
    if has_custom:
        ax3.plot(s, results['velocity_custom'] * 3.6, '-', color='#ff3333',
                 lw=1.5, label='Custom')
    if has_scipy:
        ax3.plot(s, results['velocity_scipy'] * 3.6, '--', color='#33ccff',
                 lw=1.5, label='Scipy', alpha=0.9)

    ax3.set_xlabel('Track position (%)', color='white', fontsize=10)
    ax3.set_ylabel('Speed (km/h)', color='white', fontsize=10)
    ax3.set_title('Speed Profile', color='white', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, facecolor='#222', edgecolor='#555', labelcolor='white')
    ax3.tick_params(colors='#aaa')
    for spine in ax3.spines.values():
        spine.set_color('#555')

    # ── Panel 4: Energy breakdown ──
    ax4 = axes[1, 1]
    ax4.set_facecolor('#1a1a1a')

    labels = ['Centerline']
    drive_e = [results['energy_center']['drive_energy_kJ']]
    regen_e = [results['energy_center']['regen_energy_kJ']]
    net_e = [results['energy_center']['net_energy_kJ']]
    lap_t = [results['energy_center']['lap_time_s']]

    if has_custom:
        labels.append('Custom')
        drive_e.append(results['energy_custom']['drive_energy_kJ'])
        regen_e.append(results['energy_custom']['regen_energy_kJ'])
        net_e.append(results['energy_custom']['net_energy_kJ'])
        lap_t.append(results['energy_custom']['lap_time_s'])
    if has_scipy:
        labels.append('Scipy')
        drive_e.append(results['energy_scipy']['drive_energy_kJ'])
        regen_e.append(results['energy_scipy']['regen_energy_kJ'])
        net_e.append(results['energy_scipy']['net_energy_kJ'])
        lap_t.append(results['energy_scipy']['lap_time_s'])

    x_pos = np.arange(len(labels))
    bar_w = 0.25
    colors_drive = ['#ffdd00', '#ff3333', '#33ccff'][:len(labels)]
    colors_regen = ['#88aa00', '#aa5522', '#2299aa'][:len(labels)]
    colors_net = ['#ccaa00', '#cc2222', '#2288cc'][:len(labels)]

    bars1 = ax4.bar(x_pos - bar_w, drive_e, bar_w, label='Drive',
                    color=colors_drive, alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = ax4.bar(x_pos, regen_e, bar_w, label='Regen',
                    color=colors_regen, alpha=0.8, edgecolor='white', linewidth=0.5)
    bars3 = ax4.bar(x_pos + bar_w, net_e, bar_w, label='Net',
                    color=colors_net, alpha=0.9, edgecolor='white', linewidth=0.5)

    # lap time annotations on net bars
    for i, (b, t) in enumerate(zip(bars3, lap_t)):
        ax4.text(b.get_x() + b.get_width()/2, max(b.get_height(), 0) + 5,
                 f'{t:.1f}s', ha='center', va='bottom', color='white',
                 fontsize=8, fontweight='bold')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, color='white', fontsize=10)
    ax4.set_ylabel('Energy (kJ)', color='white', fontsize=10)
    ax4.set_title('Energy Breakdown (lap time annotated)',
                  color='white', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9, facecolor='#222', edgecolor='#555', labelcolor='white')
    ax4.tick_params(colors='#aaa')
    for spine in ax4.spines.values():
        spine.set_color('#555')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        fig.savefig(output_path, facecolor=fig.get_facecolor(),
                    dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_convergence(results, output_path=None):
    """Plot BFGS convergence history."""
    import matplotlib.pyplot as plt

    if 'history_custom' not in results:
        return

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#111')
    ax.set_facecolor('#1a1a1a')

    hist = results['history_custom']
    iters = [h[0] for h in hist]
    fvals = [h[1] for h in hist]
    ax.semilogy(iters, fvals, '-', color='#ff3333', lw=2, label='Adam + BFGS')

    if 'result_scipy' in results:
        # scipy only gives final value
        ax.axhline(results['result_scipy'].fun, color='#33ccff', ls='--',
                   lw=1.5, label=f'Scipy final = {results["result_scipy"].fun:.6f}')

    ax.set_xlabel('Iteration', color='white', fontsize=11)
    ax.set_ylabel('Objective f(alpha)', color='white', fontsize=11)
    ax.set_title('Convergence: Custom Solver', color='white',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#222', edgecolor='#555', labelcolor='white')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_color('#555')

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, facecolor=fig.get_facecolor(),
                    dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_multi_track_summary(all_tracks, all_results, output_path=None):
    """Summary figure: 3 tracks side-by-side with key metrics."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import LineCollection

    n_tracks = len(all_tracks)
    fig, axes = plt.subplots(2, n_tracks, figsize=(7 * n_tracks, 12), facecolor='#111')
    fig.suptitle('Multi-Track Racing Line Optimization — Summary',
                 color='white', fontsize=18, fontweight='bold', y=0.98)

    for col, (track, results) in enumerate(zip(all_tracks, all_results)):
        # top row: racing lines with velocity heatmap
        ax = axes[0, col]
        ax.set_facecolor('#1a3a15')
        road = np.vstack([track.left_boundary, track.right_boundary[::-1],
                          track.left_boundary[0:1]])
        ax.add_patch(Polygon(road, closed=True, fc='#3a3a3a', ec='none', zorder=1))
        ax.plot(track.left_boundary[:, 0], track.left_boundary[:, 1],
                'w-', lw=0.5, alpha=0.4, zorder=3)
        ax.plot(track.right_boundary[:, 0], track.right_boundary[:, 1],
                'w-', lw=0.5, alpha=0.4, zorder=3)

        # centerline
        cl = results['centerline']
        ax.plot(cl[:, 0], cl[:, 1], '-', color='#ffdd00', lw=1.5,
                alpha=0.6, zorder=4)

        # optimized line colored by velocity
        rl = results.get('raceline_scipy', results.get('raceline_custom', cl))
        v = results.get('velocity_scipy', results.get('velocity_custom',
                         results['velocity_center']))
        points = rl.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='RdYlGn', linewidth=2.5, zorder=5)
        lc.set_array(v[:-1] * 3.6)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01, shrink=0.8)
        cbar.set_label('km/h', color='#aaa', fontsize=8)
        cbar.ax.tick_params(labelsize=7, colors='#aaa')

        # metrics text box
        e = results.get('energy_scipy', results.get('energy_custom',
                         results['energy_center']))
        kc = results['curvature_sub_center']
        ks = results.get('curvature_sub_scipy', results.get('curvature_sub_custom', kc))
        curv_reduction = (1 - np.sum(ks**2) / np.sum(kc**2)) * 100
        info = (f"Lap: {e['lap_time_s']:.1f}s | Net E: {e['net_energy_kJ']:.0f} kJ\n"
                f"Curv. reduction: {curv_reduction:.0f}% | Corr: "
                f"{np.corrcoef(results['alpha_custom'], results['alpha_scipy'])[0,1]:.3f}")
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7.5,
                color='white', va='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='#000000aa', ec='#555'))

        ax.set_aspect('equal')
        ax.set_title(track.name, color='white', fontsize=13, fontweight='bold')
        ax.axis('off')

        # bottom row: speed profiles
        ax2 = axes[1, col]
        ax2.set_facecolor('#1a1a1a')
        n = len(results['velocity_center'])
        s = np.linspace(0, 100, n)
        ax2.plot(s, results['velocity_center'] * 3.6, '-', color='#ffdd00',
                 lw=1.0, label='Center', alpha=0.6)
        if 'velocity_custom' in results:
            ax2.plot(s, results['velocity_custom'] * 3.6, '-', color='#ff3333',
                     lw=1.2, label='Custom')
        if 'velocity_scipy' in results:
            ax2.plot(s, results['velocity_scipy'] * 3.6, '--', color='#33ccff',
                     lw=1.2, label='Scipy', alpha=0.8)
        ax2.set_xlabel('Position (%)', color='white', fontsize=9)
        ax2.set_ylabel('Speed (km/h)', color='white', fontsize=9)
        ax2.legend(fontsize=7, facecolor='#222', edgecolor='#555', labelcolor='white')
        ax2.tick_params(colors='#aaa', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color('#555')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        fig.savefig(output_path, facecolor=fig.get_facecolor(),
                    dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close(fig)


def curvature_comparison(track, results, n_stations=200, output_path=None):
    """
    Compare cumulated curvature: centerline vs optimized lines.

    Uses the subsampled resolution (n_stations) where the optimization
    was actually performed, to avoid interpolation artifacts.
    """
    import matplotlib.pyplot as plt

    # subsample to optimization resolution
    n_full = len(track.centerline)
    idx = np.linspace(0, n_full - 1, n_stations, dtype=int)
    cl = track.centerline[idx]
    normals = track.normals[idx]
    widths = track.widths[idx]
    n = n_stations
    s_norm = np.linspace(0, 100, n)

    # centerline curvature at this resolution
    kappa_c = compute_curvature_from_path(cl)

    cum_center = np.cumsum(kappa_c**2)
    total_center = cum_center[-1]

    lines_kappa = [
        ('Centerline', kappa_c, '#ffdd00', '-', 2.5),
    ]
    lines_cum = [
        ('Centerline', cum_center, '#ffdd00', '-', 2.5),
    ]

    totals = {
        'Centerline': total_center,
    }

    # numerical solutions at subsampled resolution
    for alpha_key, label, color, ls in [
        ('alpha_custom', 'Custom (Adam+BFGS)', '#ff3333', '-'),
        ('alpha_scipy', 'Scipy (SLSQP)', '#33ccff', '--')
    ]:
        if alpha_key not in results:
            continue
        alpha = results[alpha_key]
        rl = alpha_to_raceline(alpha, cl, normals)
        kappa = compute_curvature_from_path(rl)
        cum = np.cumsum(kappa**2)
        lines_kappa.append((label, kappa, color, ls, 1.5))
        lines_cum.append((label.split(' ')[0], cum, color, ls, 1.5))
        totals[label.split(' ')[0]] = cum[-1]

    # print comparison
    print(f"\n  CURVATURE COMPARISON (Sum kappa^2 at {n_stations} stations)")
    print(f"  {'Method':<25} {'Sum kappa^2':>14} {'vs center':>12}")
    print(f"  {'─'*51}")
    for name, total in totals.items():
        pct_center = (1 - total / total_center) * 100
        print(f"  {name:<25} {total:>14.6f} {pct_center:>+11.1f}%")

    numericals = {k: v for k, v in totals.items()
                  if k != 'Centerline'}
    if numericals:
        best_num = min(numericals.values())
        reduction = (1 - best_num / total_center) * 100
        print(f"\n  Curvature reduction: {reduction:.1f}% vs centerline")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#111')
    fig.suptitle(f'{track.name} — Curvature Comparison ({n_stations} stations)',
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    ax1.set_facecolor('#1a1a1a')
    for name, kappa, color, ls, lw in lines_kappa:
        ax1.plot(s_norm, np.abs(kappa), ls, color=color, lw=lw, label=name, alpha=0.85)
    ax1.set_xlabel('Track position (%)', color='white', fontsize=10)
    ax1.set_ylabel('|Curvature|', color='white', fontsize=10)
    ax1.set_title('Pointwise |Curvature|', color='white', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#222', edgecolor='#555', labelcolor='white')
    ax1.tick_params(colors='#aaa')
    for spine in ax1.spines.values():
        spine.set_color('#555')

    ax2.set_facecolor('#1a1a1a')
    for name, cum, color, ls, lw in lines_cum:
        ax2.plot(s_norm, cum, ls, color=color, lw=lw, label=name, alpha=0.85)
    ax2.set_xlabel('Track position (%)', color='white', fontsize=10)
    ax2.set_ylabel('Cumulated Sum(kappa^2)', color='white', fontsize=10)
    ax2.set_title('Cumulated Squared Curvature', color='white', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, facecolor='#222', edgecolor='#555', labelcolor='white')
    ax2.tick_params(colors='#aaa')
    for spine in ax2.spines.values():
        spine.set_color('#555')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        fig.savefig(output_path, facecolor=fig.get_facecolor(),
                    dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close(fig)


# ── Theoretical bound validation (circular track) ────────────────────────

def theoretical_bound_validation(track, results, n_stations=200, output_path=None):
    """
    On a CIRCULAR track (constant curvature), the exact analytical solution
    for min sum(kappa^2) is known:

        Optimal path = inner boundary at radius R_inner = R - w/2
        kappa_optimal = 1 / R_inner  (constant everywhere)
        Sum(kappa^2) = N * kappa_optimal^2

    This validates our numerical solvers against ground truth.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # Get track geometry
    R = np.mean(np.linalg.norm(track.centerline, axis=1))  # average radius
    w = np.mean(track.widths)  # track half-width (center to boundary)
    # The optimizer uses bounds [-w/2, w/2] (see solve_custom/solve_scipy)
    # so the max inward offset is w/2, giving inner radius R - w/2
    alpha_max = w / 2.0  # optimizer's max offset

    # Exact analytical solution within optimizer bounds
    R_inner = R - alpha_max
    kappa_exact = 1.0 / R_inner  # curvature of inner circle

    # At optimization resolution
    n_full = len(track.centerline)
    idx = np.linspace(0, n_full - 1, n_stations, dtype=int)
    cl = track.centerline[idx]
    normals = track.normals[idx]
    widths = track.widths[idx]

    kappa_center = compute_curvature_from_path(cl)
    sum_kappa2_center = np.sum(kappa_center**2)

    # Exact analytical: alpha = -w/2 everywhere (inner boundary of optimizer range)
    alpha_exact_arr = -widths / 2.0
    rl_exact = alpha_to_raceline(alpha_exact_arr, cl, normals)
    kappa_exact_numerical = compute_curvature_from_path(rl_exact)
    sum_kappa2_exact = np.sum(kappa_exact_numerical**2)
    # Also compute the theoretical value for reference
    sum_kappa2_theory = n_stations * kappa_exact**2

    print(f"\n  THEORETICAL VALIDATION — Circular Track")
    print(f"  Center radius R = {R:.1f} m, track half-width w = {w:.1f} m")
    print(f"  Optimizer max offset = w/2 = {alpha_max:.1f} m")
    print(f"  Optimal inner radius = R - w/2 = {R_inner:.1f} m")
    print(f"  Exact kappa = 1/{R_inner:.1f} = {kappa_exact:.6f}")
    print(f"  {'─'*60}")
    print(f"  {'Method':<25} {'Sum kappa^2':>14} {'vs exact':>12} {'% gap':>10}")
    print(f"  {'─'*60}")

    print(f"  Theoretical N*kappa^2 = {sum_kappa2_theory:.6f} (if curvature were exactly 1/R)")
    print(f"  Numerical (same path) = {sum_kappa2_exact:.6f}")
    print()

    rows = [
        ('Exact (alpha=-w/2)', sum_kappa2_exact),
        ('Centerline', sum_kappa2_center),
    ]

    for alpha_key, label in [('alpha_custom', 'Custom (Adam+BFGS)'),
                              ('alpha_scipy', 'Scipy (SLSQP)')]:
        if alpha_key in results:
            alpha = results[alpha_key]
            rl = alpha_to_raceline(alpha, cl, normals)
            kappa = compute_curvature_from_path(rl)
            rows.append((label, np.sum(kappa**2)))

    for name, val in rows:
        gap = (val / sum_kappa2_exact - 1) * 100
        print(f"  {name:<25} {val:>14.6f} {val - sum_kappa2_exact:>+12.6f} {gap:>+9.2f}%")

    # Verify that optimal alpha should be -w/2 (push to inside of curve)
    # For a circle centered at origin traced CCW, normals point outward
    # so alpha = -w/2 means move inward (to optimizer's bound)
    alpha_exact = -widths / 2.0  # inner boundary of optimizer's range

    if 'alpha_custom' in results:
        alpha_c = results['alpha_custom']
        mean_diff = np.mean(np.abs(alpha_c - alpha_exact))
        print(f"\n  Custom alpha vs exact (alpha=-w/2):")
        print(f"    Mean |diff| = {mean_diff:.4f} (should be ~0)")
        print(f"    Mean alpha_custom = {np.mean(alpha_c):.4f} (exact = {np.mean(alpha_exact):.4f})")

    if 'alpha_scipy' in results:
        alpha_s = results['alpha_scipy']
        mean_diff = np.mean(np.abs(alpha_s - alpha_exact))
        print(f"  Scipy alpha vs exact (alpha=-w/2):")
        print(f"    Mean |diff| = {mean_diff:.4f} (should be ~0)")
        print(f"    Mean alpha_scipy = {np.mean(alpha_s):.4f} (exact = {np.mean(alpha_exact):.4f})")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#111')
    fig.suptitle(f'Circular Track — Theoretical Bound Validation (R={R:.0f}m, w={w:.0f}m)',
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    # Panel 1: Track with lines
    ax = axes[0]
    ax.set_facecolor('#1a3a15')
    road = np.vstack([track.left_boundary, track.right_boundary[::-1],
                      track.left_boundary[0:1]])
    ax.add_patch(Polygon(road, closed=True, fc='#3a3a3a', ec='none', zorder=1))
    ax.plot(track.left_boundary[:, 0], track.left_boundary[:, 1],
            'w-', lw=0.8, alpha=0.5, zorder=3)
    ax.plot(track.right_boundary[:, 0], track.right_boundary[:, 1],
            'w-', lw=0.8, alpha=0.5, zorder=3)

    # Centerline
    ax.plot(track.centerline[:, 0], track.centerline[:, 1], '-',
            color='#ffdd00', lw=2, label='Centerline', zorder=4, alpha=0.8)

    # Exact inner circle (at optimizer's max offset)
    t_plot = np.linspace(0, 2*np.pi, 500)
    ax.plot(R_inner * np.cos(t_plot), R_inner * np.sin(t_plot), '-',
            color='#00ff88', lw=2.5, label=f'Exact optimal (R={R_inner:.1f}m)',
            zorder=7)

    if 'raceline_custom' in results:
        rl = results['raceline_custom']
        ax.plot(rl[:, 0], rl[:, 1], '-', color='#ff3333', lw=1.5,
                label='Custom', zorder=5)
    if 'raceline_scipy' in results:
        rl = results['raceline_scipy']
        ax.plot(rl[:, 0], rl[:, 1], '--', color='#33ccff', lw=1.5,
                label='Scipy', zorder=6, alpha=0.9)

    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8, facecolor='#222',
              edgecolor='#555', labelcolor='white')
    ax.set_title('Racing Lines vs Exact Solution', color='white',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 2: Alpha comparison
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')
    s_norm = np.linspace(0, 100, n_stations)
    ax2.axhline(-alpha_max, color='#00ff88', ls='-', lw=2.5,
                label=f'Exact alpha = {-alpha_max:.1f}', alpha=0.9)
    if 'alpha_custom' in results:
        ax2.plot(s_norm, results['alpha_custom'], '-', color='#ff3333',
                 lw=1.5, label='Custom', alpha=0.8)
    if 'alpha_scipy' in results:
        ax2.plot(s_norm, results['alpha_scipy'], '--', color='#33ccff',
                 lw=1.5, label='Scipy', alpha=0.8)
    ax2.axhline(0, color='#ffdd00', ls=':', lw=1, alpha=0.4, label='Centerline (alpha=0)')
    ax2.axhline(alpha_max, color='white', ls=':', lw=0.5, alpha=0.3)
    ax2.axhline(-alpha_max, color='white', ls=':', lw=0.5, alpha=0.3)
    ax2.set_xlabel('Track position (%)', color='white', fontsize=10)
    ax2.set_ylabel('Lateral offset alpha (m)', color='white', fontsize=10)
    ax2.set_title('Lateral Offset vs Exact', color='white', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, facecolor='#222', edgecolor='#555', labelcolor='white')
    ax2.tick_params(colors='#aaa')
    for spine in ax2.spines.values():
        spine.set_color('#555')

    # Panel 3: Bar chart — Sum kappa^2
    ax3 = axes[2]
    ax3.set_facecolor('#1a1a1a')
    bar_names = [r[0] for r in rows]
    bar_vals = [r[1] for r in rows]
    colors = ['#00ff88', '#88cc44', '#ffdd00', '#ff3333', '#33ccff'][:len(rows)]
    bars = ax3.barh(range(len(rows)), bar_vals, color=colors, edgecolor='white',
                    linewidth=0.5, alpha=0.85)
    ax3.set_yticks(range(len(rows)))
    ax3.set_yticklabels(bar_names, color='white', fontsize=9)
    ax3.set_xlabel('Sum(kappa^2)', color='white', fontsize=10)
    ax3.set_title('Objective Value Comparison', color='white', fontsize=12, fontweight='bold')

    # annotate with gap %
    for i, (bar, (name, val)) in enumerate(zip(bars, rows)):
        gap = (val / sum_kappa2_exact - 1) * 100
        label = f'{val:.4f} ({gap:+.1f}%)' if abs(gap) > 0.01 else f'{val:.4f} (exact)'
        ax3.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                 label, va='center', color='white', fontsize=8)

    ax3.tick_params(colors='#aaa')
    for spine in ax3.spines.values():
        spine.set_color('#555')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        fig.savefig(output_path, facecolor=fig.get_facecolor(),
                    dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from track import get_track

    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ── Part 0: Theoretical validation on circular track ──
    print(f"\n{'#'*70}")
    print(f"  THEORETICAL VALIDATION: Circular Track")
    print(f"{'#'*70}")
    circle_track = get_track('circle')
    circle_results = optimize_racing_line(circle_track, n_stations=200, solver='both')
    print_comparison(circle_track, circle_results)
    theoretical_bound_validation(circle_track, circle_results,
                                  output_path=os.path.join(fig_dir, 'theoretical_validation.png'))
    curvature_comparison(circle_track, circle_results,
                          output_path=os.path.join(fig_dir, 'analytical_circle.png'))

    # ── Part 1: Optimization on all tracks ──
    track_names = ['oval', 'monaco', 'hairpin', 'complex']
    all_tracks = []
    all_results = []

    for track_name in track_names:
        print(f"\n{'#'*70}")
        print(f"  TRACK: {track_name}")
        print(f"{'#'*70}")

        track = get_track(track_name)
        results = optimize_racing_line(track, n_stations=200, solver='both')
        print_comparison(track, results)

        # analytical comparison
        curvature_comparison(track, results,
                              output_path=os.path.join(fig_dir, f'analytical_{track_name}.png'))

        if track_name != 'oval':  # full analysis for non-trivial tracks
            plot_full_analysis(track, results,
                               output_path=os.path.join(fig_dir, f'analysis_{track_name}.png'))
        plot_convergence(results,
                         output_path=os.path.join(fig_dir, f'convergence_{track_name}.png'))

        all_tracks.append(track)
        all_results.append(results)

    # multi-track summary (skip oval for the summary)
    plot_multi_track_summary(all_tracks[1:], all_results[1:],
                             output_path=os.path.join(fig_dir, 'summary_all_tracks.png'))
    print("\nAll tracks processed.")
