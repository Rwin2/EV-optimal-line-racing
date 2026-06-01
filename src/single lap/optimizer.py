"""
Racing line optimizer for EV optimal line racing.

Core functions:
  - alpha_to_raceline: lateral offset parameterization
  - compute_curvature_from_path: periodic curvature computation
  - compute_velocity_profile: forward-backward speed integration
  - compute_energy: EV energy consumption with regen

References:
  Kochenderfer & Wheeler, Algorithms for Optimization, 2nd ed.
"""

import numpy as np


def alpha_to_raceline(alpha, centerline, normals):
    """Convert lateral offsets to a 2D racing line."""
    return centerline + alpha[:, np.newaxis] * normals


def compute_curvature_from_path(path):
    """Compute curvature at each point of a closed 2D path (periodic).

    Uses periodic finite differences to avoid boundary artifacts
    that np.gradient introduces on non-periodic data.
    """
    n = len(path)
    pad = 3
    x = np.concatenate([path[-pad:, 0], path[:, 0], path[:pad, 0]])
    y = np.concatenate([path[-pad:, 1], path[:, 1], path[:pad, 1]])

    dx = np.gradient(x, edge_order=2)
    dy = np.gradient(y, edge_order=2)
    ddx = np.gradient(dx, edge_order=2)
    ddy = np.gradient(dy, edge_order=2)

    dx = dx[pad:pad+n]
    dy = dy[pad:pad+n]
    ddx = ddx[pad:pad+n]
    ddy = ddy[pad:pad+n]

    speed = np.sqrt(dx**2 + dy**2)
    speed = np.maximum(speed, 1e-10)
    kappa = (dx * ddy - dy * ddx) / speed**3
    return kappa


def compute_velocity_profile(path, kappa, car_params, ds=None):
    """
    Compute optimal velocity at each point of a racing line.

    Uses forward-backward integration:
    1. Cornering limit: v_max = sqrt(a_lat_max / |kappa|)
    2. Forward pass: acceleration-limited
    3. Backward pass: braking-limited
    4. Power limit: v <= P_max / F_drive
    """
    n = len(path)
    p = car_params

    if ds is None:
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)

    g = 9.81
    a_lat_max = p.mu * g * 0.85
    a_lon_max = min(p.F_drive_max / p.mass, p.mu * g * 0.6)
    a_brake = min(p.F_brake_max / p.mass, p.mu * g * 0.9)

    abs_kappa = np.abs(kappa)
    abs_kappa = np.maximum(abs_kappa, 1e-6)
    v_corner = np.sqrt(a_lat_max / abs_kappa)
    v_corner = np.minimum(v_corner, p.v_max)

    F_roll_const = p.C_roll * p.mass * g
    v_candidates = np.linspace(p.v_max, 5.0, 50)
    P_needed = v_candidates * (0.5 * p.rho * p.C_d * p.A_front * v_candidates**2 + F_roll_const)
    valid = P_needed <= p.P_max
    v_power_limit = v_candidates[np.argmax(valid)] if np.any(valid) else 5.0

    v_profile = np.minimum(v_corner, v_power_limit)

    for i in range(1, 2 * n):
        idx = i % n
        prev = (i - 1) % n
        v_max_accel = np.sqrt(max(v_profile[prev]**2 + 2 * a_lon_max * ds[prev], 0))
        v_profile[idx] = min(v_profile[idx], v_max_accel)

    for i in range(2 * n - 2, -1, -1):
        idx = i % n
        nxt = (i + 1) % n
        v_max_brake = np.sqrt(max(v_profile[nxt]**2 + 2 * a_brake * ds[idx], 0))
        v_profile[idx] = min(v_profile[idx], v_max_brake)

    return v_profile


def compute_energy(path, v_profile, car_params, ds=None):
    """
    Compute energy consumption along a racing line with given velocity profile.

    Returns dict with drive_energy_kJ, regen_energy_kJ, net_energy_kJ,
    energy_per_m, lap_time_s, segment_power, segment_time.
    """
    n = len(path)
    p = car_params
    g = 9.81

    if ds is None:
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)

    dt_seg = ds / np.maximum(v_profile, 0.5)

    dv = np.gradient(v_profile)
    a_lon = dv / np.maximum(dt_seg, 1e-6)

    drive_energy = 0.0
    regen_energy = 0.0
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


# ── SCP solver (joint path+speed) ─────────────────────────────────────

def _jax_curvature_jacobian(alpha, centerline, normals):
    """Compute curvature and its Jacobian w.r.t. alpha using JAX autodiff."""
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)

        def _curvature_jax(a):
            path = jnp.array(centerline) + a[:, None] * jnp.array(normals)
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
        jac = np.array(jax.jacobian(_curvature_jax)(a_jax))
        return kappa, jac
    except ImportError:
        # FD fallback
        kappa0 = compute_curvature_from_path(alpha_to_raceline(alpha, centerline, normals))
        n = len(alpha)
        jac = np.zeros((n, n))
        eps = 1e-5
        for j in range(n):
            a_p = alpha.copy(); a_p[j] += eps
            k_p = compute_curvature_from_path(alpha_to_raceline(a_p, centerline, normals))
            jac[:, j] = (k_p - kappa0) / eps
        return kappa0, jac


def solve_scp(centerline, normals, widths, car_params, alpha0=None,
              rho=3.0, eps=1e-2, max_iters=10, step_callback=None):
    """
    Joint path+speed optimization via Sequential Convex Programming.

    At each iteration, linearize the nonlinear constraints around the current
    iterate and solve an LP sub-problem with the Simplex algorithm (Ch 12).
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
        path = alpha_to_raceline(alpha, centerline, normals)
        dpath = np.diff(path, axis=0, append=path[0:1])
        ds = np.linalg.norm(dpath, axis=1)
        v_next = np.roll(v, -1)

        kappa, dkappa_da = _jax_curvature_jacobian(alpha, centerline, normals)

        J = np.sum(ds / np.maximum(v, 1e-3))
        history.append(J)
        if step_callback is not None:
            step_callback(it, alpha, v)
        dJ = abs(J_prev - J)
        print(f"    [SCP] iter {it:2d}: J={J:.4f}  dJ={dJ:.5f}")
        if it > 0 and dJ < eps:
            print(f"    [SCP] converged after {it} iterations.")
            break
        J_prev = J

        # Constraint Jacobians
        dds_da = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dds_da[i, i]   = -np.dot(dpath[i], normals[i])   / ds[i]
            dds_da[i, ip1] +=  np.dot(dpath[i], normals[ip1]) / ds[i]

        sgn_k = np.where(kappa >= 0, 1.0, -1.0)

        c_obj = np.concatenate([
            dds_da.T @ (1.0 / np.maximum(v, 1e-3)),
            -ds / np.maximum(v, 1e-3)**2,
        ])

        # Cornering
        dc_c_da = -(v**2)[:, None] * sgn_k[:, None] * dkappa_da
        dc_c_dv = np.diag(-2.0 * v * np.abs(kappa))
        c_c0 = a_lat - v**2 * np.abs(kappa)

        # Acceleration
        dc_a_da = 2.0 * a_lon * dds_da
        dc_a_dv = np.zeros((n, n))
        for i in range(n):
            ip1 = (i + 1) % n
            dc_a_dv[i, i]   =  2.0 * v[i]
            dc_a_dv[i, ip1] = -2.0 * v_next[i]
        c_a0 = v**2 + 2.0 * a_lon * ds - v_next**2

        # Braking
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
            np.maximum(-rho,      -half_w - alpha),
            np.maximum(-rho*3.0,   1.0    - v),
        ])
        ub_scp = np.concatenate([
            np.minimum( rho,       half_w - alpha),
            np.minimum( rho*3.0,   p.v_max - v),
        ])

        t_lp = time.time()
        delta = solve_lp(c_obj, A_ub, b_ub, lb_scp, ub_scp)
        t_lp = time.time() - t_lp

        max_viol = float(np.max(np.maximum(A_ub @ delta - b_ub, 0)))
        obj_decrease = float(c_obj @ delta)
        print(f"           LP: obj_d={obj_decrease:.4f}  max_viol={max_viol:.4f}  "
              f"rho={rho:.2f}  time={t_lp:.1f}s  total={time.time()-t_iter:.1f}s")

        if max_viol > 0.5:
            damping = min(0.5 / max(max_viol, 1e-6), 1.0)
            delta *= damping
            rho = max(rho * 0.5, 0.1)
            print(f"    [SCP] damping step by {damping:.2f}, rho -> {rho:.2f}")

        alpha_new = np.clip(alpha + delta[:n], -half_w, half_w)
        v_new     = np.clip(v     + delta[n:],  1.0, p.v_max)

        path_new = alpha_to_raceline(alpha_new, centerline, normals)
        ds_new = np.linalg.norm(np.diff(path_new, axis=0, append=path_new[0:1]), axis=1)
        J_new = np.sum(ds_new / np.maximum(v_new, 1e-3))

        if J_new < J:
            alpha, v = alpha_new, v_new
            rho = min(rho * 1.2, rho_init)
        else:
            alpha, v = alpha_new, v_new
            rho = max(rho * 0.5, 0.1)
            print(f"    [SCP] J increased ({J:.4f} -> {J_new:.4f}), rho -> {rho:.2f}")

    return alpha, v, history
