"""
Racing controller: Stanley + TV-LQR trajectory tracking.

Architecture:
  Layer 1 — Stanley feedforward: heading PD + cross-track error + lookahead.
            Reactive, works at any error level, naturally periodic.
  Layer 2 — TV-LQR feedback: linearize bicycle dynamics at each raceline point,
            backward Riccati (periodic, 3 laps), small correction for tighter tracking.

Online:
  1. Find nearest raceline point (spatial, periodic).
  2. Compute Stanley feedforward (steering + speed).
  3. Add TV-LQR correction: du = K[i] @ (s - s_ref[i]), scaled by proximity.
"""

import numpy as np
from car import CarState, CarParams
from optimizer import compute_curvature_from_path, compute_velocity_profile


class SpatialTVLQRController:
    """
    Stanley + TV-LQR trajectory tracking.

    State:   s = [x, y, psi, vx, vy, omega]   (6D)
    Control: u = [delta, F_drive]              (2D)
    """

    def __init__(self, track, racing_line, v_profile, params=None, dt=0.02,
                 Q=None, R=None):
        self.track = track
        self.p = params or CarParams()
        self.dt = dt
        self.racing_line = racing_line
        self.v_profile = np.maximum(v_profile, 1.0)
        self.n_rl = len(racing_line)

        self.Q = Q if Q is not None else np.diag([10., 10., 20., 3., 1., 1.])
        self.R = R if R is not None else np.diag([0.5, 1e-6])

        print(f"  [TV-LQR] Building spatial reference ({self.n_rl} pts)...")
        self._compute_geometry()
        self._build_reference()

        print(f"  [TV-LQR] Linearizing + Riccati (periodic, dt={dt})...")
        self._linearize_all()
        self._compute_gains()

        self.s_ref_arr = self.s_ref
        self._prev_idx = 0
        print(f"  [TV-LQR] Ready.")

    # ── Raceline geometry ─────────────────────────────────────────────────

    def _compute_geometry(self):
        rl = self.racing_line
        n = self.n_rl
        self.ds = np.linalg.norm(
            np.diff(rl, axis=0, append=rl[0:1]), axis=1)
        self.ds = np.maximum(self.ds, 1e-6)

        dx = np.zeros(n)
        dy = np.zeros(n)
        for i in range(n):
            vec = rl[(i + 1) % n] - rl[(i - 1) % n]
            dx[i], dy[i] = vec
        self.headings = np.arctan2(dy, dx)

        self.curvature = np.zeros(n)
        for i in range(n):
            dtheta = self.headings[(i + 1) % n] - self.headings[(i - 1) % n]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            ds2 = self.ds[i] + self.ds[(i - 1) % n]
            self.curvature[i] = dtheta / max(ds2, 1e-6)

    def _build_reference(self):
        """Reference state at each raceline point with dynamically-feasible speed."""
        n = self.n_rl
        p = self.p

        kappa = compute_curvature_from_path(self.racing_line)

        # 95% grip margin on top of optimizer's 85%
        dyn_params = CarParams(
            mass=p.mass, L=p.L, l_f=p.l_f, l_r=p.l_r,
            mu=p.mu * 0.95,
            C_d=p.C_d, A_front=p.A_front, rho=p.rho, C_roll=p.C_roll,
            P_max=p.P_max, F_drive_max=p.F_drive_max,
            F_brake_max=p.F_brake_max, v_max=p.v_max,
            C_f=p.C_f, C_r=p.C_r, I_z=p.I_z, width=p.width,
            length=p.length, eta_motor=p.eta_motor, eta_regen=p.eta_regen,
            Q_batt=p.Q_batt, V_nom=p.V_nom, delta_max=p.delta_max,
            SOC_min=p.SOC_min,
        )
        v_feasible = compute_velocity_profile(self.racing_line, kappa, dyn_params)
        self.v_profile_dyn = np.minimum(self.v_profile, v_feasible)

        self.s_ref = np.zeros((n, 6))
        for i in range(n):
            v = self.v_profile_dyn[i]
            self.s_ref[i] = [
                self.racing_line[i, 0], self.racing_line[i, 1],
                self.headings[i], v, 0.0, v * self.curvature[i],
            ]

    # ── Discrete dynamics (RK4, bicycle model) ────────────────────────────

    def _fd(self, s6, u2, dt):
        p = self.p
        delta = float(np.clip(u2[0], -p.delta_max, p.delta_max))
        F = float(np.clip(u2[1], -p.F_brake_max, p.F_drive_max))

        def deriv(s):
            _, _, psi, vx, vy, om = s
            vx = max(vx, 0.5)
            alpha_f = delta - np.arctan2(vy + p.l_f * om, vx)
            alpha_r = -np.arctan2(vy - p.l_r * om, vx)
            F_lim = p.mu * p.mass * 9.81 * 0.5
            Fyf = np.clip(p.C_f * alpha_f, -F_lim, F_lim)
            Fyr = np.clip(p.C_r * alpha_r, -F_lim, F_lim)
            Fd = min(F, p.P_max / vx) if F > 0 else F
            F_drag = 0.5 * p.rho * p.C_d * p.A_front * vx**2
            F_roll = p.C_roll * p.mass * 9.81
            return np.array([
                vx * np.cos(psi) - vy * np.sin(psi),
                vx * np.sin(psi) + vy * np.cos(psi),
                om,
                (Fd - F_drag - F_roll + p.mass * vy * om
                 - Fyf * np.sin(delta)) / p.mass,
                (Fyf * np.cos(delta) + Fyr
                 - p.mass * vx * om) / p.mass,
                (p.l_f * Fyf * np.cos(delta)
                 - p.l_r * Fyr) / p.I_z,
            ])

        k1 = deriv(s6)
        k2 = deriv(s6 + 0.5 * dt * k1)
        k3 = deriv(s6 + 0.5 * dt * k2)
        k4 = deriv(s6 + dt * k3)
        s_next = s6 + (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        s_next[3] = max(s_next[3], 0.1)
        return s_next

    # ── FD Jacobians ──────────────────────────────────────────────────────

    def _linearize(self, s, u):
        dt = self.dt
        ns, nu = 6, 2
        A = np.zeros((ns, ns))
        B = np.zeros((ns, nu))
        for i in range(ns):
            eps = max(abs(s[i]) * 1e-4, 1e-5)
            sp = s.copy(); sp[i] += eps
            sm = s.copy(); sm[i] -= eps
            df = self._fd(sp, u, dt) - self._fd(sm, u, dt)
            df[2] = (df[2] + np.pi) % (2 * np.pi) - np.pi
            A[:, i] = df / (2 * eps)
        for j in range(nu):
            eps = max(abs(u[j]) * 1e-4, 1e-5)
            up = u.copy(); up[j] += eps
            um = u.copy(); um[j] -= eps
            df = self._fd(s, up, dt) - self._fd(s, um, dt)
            df[2] = (df[2] + np.pi) % (2 * np.pi) - np.pi
            B[:, j] = df / (2 * eps)
        return A, B

    def _linearize_all(self):
        n = self.n_rl
        self.A_mats = np.zeros((n, 6, 6))
        self.B_mats = np.zeros((n, 6, 2))
        self.u_ff = np.zeros((n, 2))
        p = self.p
        g = 9.81
        for i in range(n):
            v = self.v_profile[i]
            delta = np.arctan(p.L * self.curvature[i])
            delta = np.clip(delta, -p.delta_max, p.delta_max)
            v_next = self.v_profile[(i + 1) % n]
            a_lon = (v_next**2 - v**2) / (2 * self.ds[i])
            F_drag = 0.5 * p.rho * p.C_d * p.A_front * v**2
            F_roll = p.C_roll * p.mass * g
            F = p.mass * a_lon + F_drag + F_roll
            F = np.clip(F, -p.F_brake_max, p.F_drive_max)
            self.u_ff[i] = [delta, F]
        for i in range(n):
            self.A_mats[i], self.B_mats[i] = self._linearize(
                self.s_ref[i], self.u_ff[i])

    # ── Periodic Riccati ──────────────────────────────────────────────────

    def _compute_gains(self):
        n = self.n_rl
        Q, R = self.Q, self.R
        reg = 0.1
        self.K = np.zeros((n, 2, 6))
        P = Q.copy()

        for lap in range(3):
            for i in range(n - 1, -1, -1):
                Ai, Bi = self.A_mats[i], self.B_mats[i]
                Quu = R + Bi.T @ P @ Bi + reg * np.eye(2)
                Qux = Bi.T @ P @ Ai
                self.K[i] = -np.linalg.solve(Quu, Qux)
                P_new = Q + Ai.T @ P @ (Ai + Bi @ self.K[i])
                P_new = 0.5 * (P_new + P_new.T)
                eigvals, eigvecs = np.linalg.eigh(P_new)
                eigvals = np.clip(eigvals, 0, 1e6)
                P = eigvecs @ np.diag(eigvals) @ eigvecs.T

        max_sr = 0.0
        unstable = 0
        self.stable_mask = np.ones(n, dtype=bool)
        for i in range(n):
            Acl = self.A_mats[i] + self.B_mats[i] @ self.K[i]
            sr = np.max(np.abs(np.linalg.eigvals(Acl)))
            if sr > 1.0:
                unstable += 1
                self.K[i] = np.zeros((2, 6))
                self.stable_mask[i] = False
            max_sr = max(max_sr, sr)
        print(f"    Max spectral radius: {max_sr:.4f}, "
              f"stable: {n - unstable}/{n} (zeroed {unstable} unstable gains)")

    # ── Online control ────────────────────────────────────────────────────

    def _find_nearest_rl(self, pos):
        n = self.n_rl
        indices = np.arange(self._prev_idx - 20,
                            self._prev_idx + 80) % n
        dists = np.linalg.norm(self.racing_line[indices] - pos, axis=1)
        best_idx = indices[np.argmin(dists)]
        self._prev_idx = best_idx
        return best_idx

    def _stanley_control(self, state, rl_idx):
        """Stanley feedforward: steering + speed control."""
        p = self.p
        rl = self.racing_line
        n = self.n_rl
        vx = max(state.vx, 0.5)
        pos = np.array([state.x, state.y])

        to_car = pos - rl[rl_idx]
        tangent = rl[(rl_idx + 1) % n] - rl[(rl_idx - 1) % n]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
        normal = np.array([-tangent[1], tangent[0]])
        e_ct = np.dot(to_car, normal)

        e_psi = (state.psi - self.headings[rl_idx] + np.pi) % (2 * np.pi) - np.pi

        delta = -1.5 * e_psi - np.arctan2(5.0 * e_ct, vx + 1.0)
        look_idx = (rl_idx + max(3, int(vx * 0.3))) % n
        look_err = (state.psi - self.headings[look_idx] + np.pi) % (2 * np.pi) - np.pi
        delta -= 0.3 * look_err
        delta = np.clip(delta, -p.delta_max, p.delta_max)

        # Speed: target from dynamically-feasible profile with lookahead
        look_ahead_pts = max(8, int(vx * 0.5))
        v_target = self.v_profile_dyn[rl_idx]
        for offset in range(1, look_ahead_pts):
            j = (rl_idx + offset) % n
            v_target = min(v_target, self.v_profile_dyn[j])

        pos_err = np.linalg.norm(to_car)
        v_target *= 1.0 / (1.0 + 0.1 * pos_err)

        F = 3000.0 * (v_target - vx) - 500.0 * state.omega
        F = np.clip(F, -p.F_brake_max, p.F_drive_max)

        return np.array([delta, F])

    def control(self, state: CarState):
        """
        Online control:
          Base: Stanley feedforward (robust, periodic, reactive).
          Refinement: TV-LQR correction scaled by proximity to reference.
        """
        pos = np.array([state.x, state.y])
        i = self._find_nearest_rl(pos)

        u_stanley = self._stanley_control(state, i)

        s = np.array([state.x, state.y, state.psi,
                      state.vx, state.vy, state.omega])
        s_err = s - self.s_ref[i]
        s_err[2] = (s_err[2] + np.pi) % (2 * np.pi) - np.pi

        du_lqr = self.K[i] @ s_err

        pos_err = np.linalg.norm(s_err[:2])
        lqr_scale = 0.15 * max(0.0, 1.0 - pos_err / 5.0)

        u = u_stanley + lqr_scale * du_lqr

        delta = float(np.clip(u[0], -self.p.delta_max, self.p.delta_max))
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


# Backward-compatible alias
ILQRController = SpatialTVLQRController
