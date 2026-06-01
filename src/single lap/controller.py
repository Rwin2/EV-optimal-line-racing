"""
Racing controller: TV-LQR trajectory tracking.

Two-phase approach:
  Phase 1: Generate dynamically feasible trajectory using Stanley tracker
           → (s_bar, u_bar) consistent with bicycle model dynamics
  Phase 2: Linearize + Riccati backward pass → TV-LQR gains K[k]

Online: u[k] = u_bar[k] + K[k] @ (s[k] - s_bar[k])
        with spatial indexing (find nearest reference point by position)
"""

import numpy as np
from car import CarState, CarParams


class ILQRController:
    """
    TV-LQR trajectory tracking for the bicycle model.

    State:   s = [x, y, psi, vx, vy, omega]   (6D)
    Control: u = [delta, F_drive]              (2D)
    """

    def __init__(self, track, racing_line, v_profile, params=None, dt=0.02,
                 Q=None, R=None):
        self.track = track
        self.p = params or CarParams()
        self.dt = dt
        self.racing_line = racing_line
        self.v_profile = v_profile

        self.Q = Q if Q is not None else np.diag([5., 5., 10., 1., 0.5, 0.5])
        self.R = R if R is not None else np.diag([10., 1e-5])

        # Compute heading from raceline
        dx = np.roll(racing_line[:, 0], -1) - np.roll(racing_line[:, 0], 1)
        dy = np.roll(racing_line[:, 1], -1) - np.roll(racing_line[:, 1], 1)
        self.rl_headings = np.arctan2(dy, dx)

        # Phase 1: Stanley → dynamically feasible (s_bar, u_bar)
        print(f"  [iLQR] Phase 1: generating dynamic trajectory...")
        self.s_bar, self.u_bar, self.N = self._generate_dynamic_trajectory()
        avg_err, max_err = self._tracking_error()
        print(f"    avg_err={avg_err:.2f}m, max_err={max_err:.2f}m, "
              f"N={self.N}, T={self.N * dt:.1f}s")

        # Phase 2: Riccati backward pass → K[k]
        print(f"  [iLQR] Phase 2: computing TV-LQR gains...")
        self.K = self._riccati_backward()
        print(f"  [iLQR] Done. K_max={np.max(np.abs(self.K)):.1f}")

        # Spatial index for online control
        self.ref_xy = self.s_bar[:self.N, :2].copy()
        self._prev_idx = 0

    # ── Discrete dynamics (RK4) ──────────────────────────────────────────

    def _fd(self, s6, u2):
        p = self.p
        delta = float(np.clip(u2[0], -p.delta_max, p.delta_max))
        F     = float(np.clip(u2[1], -p.F_brake_max, p.F_drive_max))
        dt    = self.dt

        def deriv(s):
            _, _, psi, vx, vy, om = s
            vx = max(vx, 0.5)
            alpha_f = delta - np.arctan2(vy + p.l_f * om, vx)
            alpha_r = -np.arctan2(vy - p.l_r * om, vx)
            F_lim   = p.mu * p.mass * 9.81 * 0.5
            Fyf = np.clip(p.C_f * alpha_f, -F_lim, F_lim)
            Fyr = np.clip(p.C_r * alpha_r, -F_lim, F_lim)
            Fd  = min(F, p.P_max / vx) if F > 0 else F
            F_drag = 0.5 * p.rho * p.C_d * p.A_front * vx**2
            F_roll = p.C_roll * p.mass * 9.81
            return np.array([
                vx * np.cos(psi) - vy * np.sin(psi),
                vx * np.sin(psi) + vy * np.cos(psi),
                om,
                (Fd - F_drag - F_roll + p.mass*vy*om - Fyf*np.sin(delta)) / p.mass,
                (Fyf*np.cos(delta) + Fyr - p.mass*vx*om) / p.mass,
                (p.l_f*Fyf*np.cos(delta) - p.l_r*Fyr) / p.I_z,
            ])

        k1 = deriv(s6)
        k2 = deriv(s6 + 0.5*dt*k1)
        k3 = deriv(s6 + 0.5*dt*k2)
        k4 = deriv(s6 + dt*k3)
        s_next = s6 + (dt / 6.) * (k1 + 2*k2 + 2*k3 + k4)
        s_next[3] = max(s_next[3], 0.1)
        return s_next

    # ── FD Jacobians ─────────────────────────────────────────────────────

    def _linearize(self, s, u, eps=1e-4):
        f0 = self._fd(s, u)
        n, m = len(s), len(u)
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        for i in range(n):
            sp = s.copy(); sp[i] += eps
            df = self._fd(sp, u) - f0
            df[2] = (df[2] + np.pi) % (2*np.pi) - np.pi
            A[:, i] = df / eps
        for j in range(m):
            up = u.copy(); up[j] += eps
            df = self._fd(s, up) - f0
            df[2] = (df[2] + np.pi) % (2*np.pi) - np.pi
            B[:, j] = df / eps
        return A, B

    # ── Phase 1: Stanley initialization ──────────────────────────────────

    def _find_nearest_rl(self, pos):
        dists = np.linalg.norm(self.racing_line - pos, axis=1)
        return np.argmin(dists)

    def _generate_dynamic_trajectory(self):
        p = self.p
        rl = self.racing_line
        n_rl = len(rl)

        ds = np.linalg.norm(np.diff(rl, axis=0, append=rl[0:1]), axis=1)
        v_safe = np.maximum(self.v_profile, 1.0)
        T_est = np.sum(ds / v_safe) * 1.3
        N = int(T_est / self.dt)

        s_bar = np.zeros((N + 1, 6))
        u_bar = np.zeros((N, 2))

        s_bar[0] = [rl[0, 0], rl[0, 1], self.rl_headings[0],
                    self.v_profile[0], 0.0, 0.0]

        rl_idx = 0
        lap_step = None

        for k in range(N):
            pos = s_bar[k, :2]
            psi = s_bar[k, 2]
            vx = max(s_bar[k, 3], 0.5)

            rl_idx = self._find_nearest_rl(pos)

            to_car = pos - rl[rl_idx]
            tangent = rl[(rl_idx + 1) % n_rl] - rl[(rl_idx - 1) % n_rl]
            tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
            normal = np.array([-tangent[1], tangent[0]])
            e_ct = np.dot(to_car, normal)

            e_psi = (psi - self.rl_headings[rl_idx] + np.pi) % (2*np.pi) - np.pi

            # Stanley steering with lookahead
            delta = -1.5 * e_psi - np.arctan2(5.0 * e_ct, vx + 1.0)
            look_idx = (rl_idx + max(3, int(vx * 0.3))) % n_rl
            look_err = (psi - self.rl_headings[look_idx] + np.pi) % (2*np.pi) - np.pi
            delta -= 0.3 * look_err
            delta = np.clip(delta, -p.delta_max, p.delta_max)

            # Adaptive speed
            pos_err = np.linalg.norm(to_car)
            err_scale = 1.0 / (1.0 + 0.5 * pos_err)
            v_target = self.v_profile[rl_idx] * err_scale
            F = 3000.0 * (v_target - vx) - 500.0 * s_bar[k, 5]
            F = np.clip(F, -p.F_brake_max, p.F_drive_max)

            u_bar[k] = [delta, F]
            s_bar[k + 1] = self._fd(s_bar[k], u_bar[k])

            if k > 100:
                prev_rl_idx = self._find_nearest_rl(s_bar[k-1, :2])
                if prev_rl_idx > n_rl * 0.75 and rl_idx < n_rl * 0.25:
                    lap_step = k + 1
                    break

        if lap_step is None:
            lap_step = N

        return s_bar[:lap_step + 1], u_bar[:lap_step], lap_step

    def _tracking_error(self):
        errs = [np.linalg.norm(self.s_bar[k, :2] - self.racing_line[self._find_nearest_rl(self.s_bar[k, :2])])
                for k in range(self.N)]
        return np.mean(errs), np.max(errs)

    # ── Phase 2: Riccati backward pass → K[k] ───────────────────────────

    def _riccati_backward(self):
        N = self.N
        K = np.zeros((N, 2, 6))
        P = self.Q.copy()

        for k in range(N - 1, -1, -1):
            A, B = self._linearize(self.s_bar[k], self.u_bar[k])
            M = self.R + B.T @ P @ B + 1e-3 * np.eye(2)
            K[k] = -np.linalg.inv(M) @ B.T @ P @ A
            P = self.Q + A.T @ P @ (A + B @ K[k])
            P = 0.5 * (P + P.T)

        return K

    # ── Online control ───────────────────────────────────────────────────

    def _find_nearest_ref(self, pos):
        window = 100
        N = self.N
        indices = np.arange(self._prev_idx - 10, self._prev_idx + window) % N
        dists = np.linalg.norm(self.ref_xy[indices] - pos, axis=1)
        best_idx = indices[np.argmin(dists)]
        self._prev_idx = best_idx
        return best_idx

    def control(self, state: CarState):
        """
        Online: u[k] = u_bar[k] + K[k] @ (s[k] - s_bar[k])
        """
        pos = np.array([state.x, state.y])
        k = self._find_nearest_ref(pos)

        s = np.array([state.x, state.y, state.psi, state.vx, state.vy, state.omega])
        s_err = s - self.s_bar[k]
        s_err[2] = (s_err[2] + np.pi) % (2*np.pi) - np.pi

        du = self.K[k] @ s_err
        u = self.u_bar[k] + du
        delta   = float(np.clip(u[0], -self.p.delta_max,   self.p.delta_max))
        F_drive = float(np.clip(u[1], -self.p.F_brake_max, self.p.F_drive_max))

        return delta, F_drive


def generate_racing_line(track, mode='min_laptime'):
    """
    Generate a racing line for the given track.

    Parameters
    ----------
    track : Track
    mode  : 'min_laptime' — time-optimal line via SCP (default)
            'center'      — track centerline

    Returns
    -------
    racing_line : ndarray (N, 2)
    v_profile   : ndarray (N,)
    """
    from scipy.optimize import minimize as scipy_minimize
    from scipy.interpolate import interp1d
    from scipy.ndimage import uniform_filter1d
    from optimizer import (alpha_to_raceline, compute_curvature_from_path,
                           compute_velocity_profile, solve_scp)
    from car import CarParams

    n_full = len(track.centerline)
    car    = CarParams()

    if mode == 'center':
        racing_line = track.centerline.copy()
        kappa = compute_curvature_from_path(racing_line)
        ds    = np.linalg.norm(np.diff(racing_line, axis=0, append=racing_line[0:1]), axis=1)
        v     = compute_velocity_profile(racing_line, kappa, car, ds)
        return racing_line, v

    # mode == 'min_laptime': SCP at reduced resolution, interpolated to full track
    n_scp = min(80, n_full)
    idx   = np.linspace(0, n_full - 1, n_scp, dtype=int)
    c_sub = track.centerline[idx]
    n_sub = track.normals[idx]
    w_sub = track.widths[idx]

    bounds = [(-w, w) for w in w_sub]
    def curv_obj(a):
        path = alpha_to_raceline(a, c_sub, n_sub)
        k    = compute_curvature_from_path(path)
        da   = np.diff(a, append=a[0])
        return float(np.sum(k**2) + 1e-5 * np.sum(da**2))
    res_w = scipy_minimize(curv_obj, np.zeros(n_scp), method='SLSQP',
                           bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})

    alpha, _, _ = solve_scp(c_sub, n_sub, w_sub, car, alpha0=res_w.x)

    t_sub      = np.linspace(0, 1, n_scp)
    t_full     = np.linspace(0, 1, n_full)
    alpha_full = interp1d(t_sub, alpha, kind='cubic',
                          fill_value='extrapolate')(t_full)
    alpha_full = np.clip(alpha_full, -track.widths, track.widths)
    alpha_full = uniform_filter1d(alpha_full, size=max(3, n_full // n_scp), mode='wrap')
    racing_line = alpha_to_raceline(alpha_full, track.centerline, track.normals)

    kappa = compute_curvature_from_path(racing_line)
    ds    = np.linalg.norm(np.diff(racing_line, axis=0, append=racing_line[0:1]), axis=1)
    v     = compute_velocity_profile(racing_line, kappa, car, ds)
    return racing_line, v
