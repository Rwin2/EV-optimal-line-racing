"""
Racing controllers.
Each controller takes the current car state and returns (delta, F_drive).

Controllers:
  PurePursuitController — geometric path follower + PD speed control (baseline)
  ILQRController        — TV-LQR tracking controller (mirrors HW2 cartpole_balance.py)
                          OFFLINE: linearize bicycle dynamics along SCP reference,
                                   backward Riccati recursion → time-varying gains K[k]
                          ONLINE:  u[k] = u_ref[k] + K[k] @ (s[k] - s_ref[k])
"""

import numpy as np
from car import CarState, CarParams


class PurePursuitController:
    """
    Pure pursuit path follower with speed profiling.
    Base controller that follows a given racing line (defaults to centerline).
    """

    def __init__(self, track, racing_line=None, params=None,
                 lookahead=15.0, speed_mode="curvature", v_profile=None):
        self.track = track
        self.racing_line = racing_line if racing_line is not None else track.centerline
        self.p = params or CarParams()
        self.lookahead = lookahead
        self.speed_mode = speed_mode
        # Use jointly-optimized speed profile if provided (from SCP),
        # otherwise recompute from the racing line geometry.
        if v_profile is not None:
            self.target_speeds = np.asarray(v_profile)
        else:
            self._compute_speed_profile()

    def _compute_speed_profile(self):
        """Compute target speed at each point based on mode."""
        n = len(self.racing_line)
        # Curvature of racing line
        dx = np.gradient(self.racing_line[:, 0])
        dy = np.gradient(self.racing_line[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        speed = np.sqrt(dx**2 + dy**2)
        speed[speed < 1e-10] = 1e-10
        curvature = np.abs(dx * ddy - dy * ddx) / speed**3

        if self.speed_mode == "constant":
            self.target_speeds = np.full(n, 25.0)  # ~90 km/h
        elif self.speed_mode == "aggressive":
            # Max speed everywhere, limited by curvature
            max_lat_acc = self.p.mu * 9.81 * 0.95
            v_corner = np.sqrt(max_lat_acc / (curvature + 1e-6))
            self.target_speeds = np.clip(v_corner, 10.0, self.p.v_max)
        elif self.speed_mode == "curvature":
            # Moderate: use 70% of max lateral grip
            max_lat_acc = self.p.mu * 9.81 * 0.70
            v_corner = np.sqrt(max_lat_acc / (curvature + 1e-6))
            self.target_speeds = np.clip(v_corner, 8.0, self.p.v_max * 0.85)
        elif self.speed_mode == "energy_saving":
            # Conservative: 50% grip, lower top speed
            max_lat_acc = self.p.mu * 9.81 * 0.50
            v_corner = np.sqrt(max_lat_acc / (curvature + 1e-6))
            self.target_speeds = np.clip(v_corner, 8.0, 40.0)
        else:
            self.target_speeds = np.full(n, 20.0)

        # Forward-backward pass to enforce acceleration/braking limits
        self.target_speeds = self._smooth_speed_profile(self.target_speeds)

    def _smooth_speed_profile(self, speeds):
        """Forward-backward smoothing to respect acceleration limits."""
        n = len(speeds)
        ds = np.linalg.norm(np.diff(self.racing_line, axis=0, append=self.racing_line[0:1]), axis=1)
        a_max = 5.0   # max longitudinal acceleration (m/s^2)
        a_brake = 8.0  # max braking deceleration (m/s^2)

        # Forward pass (acceleration limited)
        for i in range(1, n):
            v_max_next = np.sqrt(speeds[i-1]**2 + 2 * a_max * ds[i-1])
            speeds[i] = min(speeds[i], v_max_next)

        # Backward pass (braking limited) - wrap around
        for i in range(n-2, -1, -1):
            v_max_prev = np.sqrt(speeds[i+1]**2 + 2 * a_brake * ds[i])
            speeds[i] = min(speeds[i], v_max_prev)

        return speeds

    def control(self, state: CarState):
        """Compute steering and drive force."""
        pos = state.position()

        # Find nearest point on racing line
        dists = np.linalg.norm(self.racing_line - pos, axis=1)
        nearest_idx = np.argmin(dists)

        # Lookahead point
        n = len(self.racing_line)
        cumul = 0.0
        lookahead_idx = nearest_idx
        for i in range(1, n):
            idx = (nearest_idx + i) % n
            prev_idx = (nearest_idx + i - 1) % n
            cumul += np.linalg.norm(self.racing_line[idx] - self.racing_line[prev_idx])
            if cumul >= self.lookahead:
                lookahead_idx = idx
                break

        target = self.racing_line[lookahead_idx]

        # Pure pursuit steering
        dx = target[0] - state.x
        dy = target[1] - state.y
        angle_to_target = np.arctan2(dy, dx)
        heading_error = angle_to_target - state.psi
        # Normalize to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi

        L_d = np.linalg.norm(target - pos)
        L_d = max(L_d, 1.0)
        delta = np.arctan2(2.0 * self.p.L * np.sin(heading_error), L_d)
        delta = np.clip(delta, -self.p.delta_max, self.p.delta_max)

        # Speed control (PD)
        v_target = self.target_speeds[nearest_idx]
        v_error = v_target - state.vx
        Kp = 2000.0
        Kd = 500.0
        F_drive = Kp * v_error - Kd * state.omega  # damping on yaw
        F_drive = np.clip(F_drive, -self.p.F_brake_max, self.p.F_drive_max)

        return delta, F_drive


class ILQRController:
    """
    Time-Varying LQR (TV-LQR) trajectory tracking controller.

    Mirrors cartpole_balance.py from HW2 (Optimal-learned-based-control),
    adapted to the 6-DOF bicycle model.

    Architecture:
      OFFLINE (once, before the race):
        1. Build time-indexed reference trajectory (s_ref, u_ref) from SCP output
        2. Linearize discrete bicycle dynamics A[k], B[k] at each reference point
           via finite differences  [like linearize() in cartpole_swingup.py]
        3. TV-LQR backward Riccati recursion → gain matrices K[k]
           [like ricatti_recursion() in cartpole_balance.py]

      ONLINE (every dt = 0.01 s):
        u[k] = u_ref[k] + K[k] @ (s[k] - s_ref[k])
        [like simulate() in cartpole_balance.py]

    State:   s = [x, y, psi, vx, vy, omega]   (6D — SOC excluded from control)
    Control: u = [delta, F_drive]              (2D)
    """

    def __init__(self, track, racing_line, v_profile, params=None, dt=0.02,
                 Q=None, R=None, QN=None):
        self.track = track
        self.p = params or CarParams()
        self.dt = dt  # must match simulator dt

        # Cost matrices (6 states, 2 controls)
        # Q weights: [x(m), y(m), psi(rad), vx(m/s), vy(m/s), omega(rad/s)]
        # R weights: [delta(rad), F_drive(N)] — F_drive has large magnitude so tiny weight
        self.Q  = Q  if Q  is not None else np.diag([100., 100., 20., 5., 1., 1.])
        self.R  = R  if R  is not None else np.diag([10., 1e-6])
        self.QN = QN if QN is not None else 10. * self.Q

        # Build time-indexed reference trajectory from SCP output
        self.s_ref, self.u_ref, self.t_ref = self._build_reference(racing_line, v_profile)
        self.N = len(self.t_ref)

        # Offline: iLQR backward pass (like ilqr() in HW2 cartpole_swingup.py)
        print(f"  [iLQR] Computing gains offline ({self.N} timesteps, dt={self.dt})...")
        self.Y, self.y = self._ilqr_backward_pass()
        print(f"  [iLQR] Done. Reference lap: {self.t_ref[-1]:.1f}s")

        self._step = 0  # simulation timestep counter

    # ── Discrete dynamics (RK4, no heading wrap) ─────────────────────────

    def _discrete_dynamics(self, s6, u2):
        """
        RK4-discretized bicycle dynamics. psi is NOT wrapped so FD Jacobians
        are smooth across the 0/2pi boundary.
        s6 = [x, y, psi, vx, vy, omega],  u2 = [delta, F_drive]
        """
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

    # ── FD Jacobians (like linearize() in HW2 cartpole_swingup.py) ───────

    def _linearize(self, s, u, eps=1e-4):
        """
        Finite-difference Jacobians of discrete dynamics at (s, u).
        Returns A (6,6) = df/ds  and  B (6,2) = df/du.
        """
        f0 = self._discrete_dynamics(s, u)
        n, m = len(s), len(u)
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        for i in range(n):
            sp = s.copy(); sp[i] += eps
            df = self._discrete_dynamics(sp, u) - f0
            df[2] = (df[2] + np.pi) % (2*np.pi) - np.pi  # wrap psi diff
            A[:, i] = df / eps
        for j in range(m):
            up = u.copy(); up[j] += eps
            df = self._discrete_dynamics(s, up) - f0
            df[2] = (df[2] + np.pi) % (2*np.pi) - np.pi
            B[:, j] = df / eps
        return A, B

    # ── Reference trajectory builder ──────────────────────────────────────

    def _build_reference(self, racing_line, v_profile):
        """
        Convert spatially-indexed (racing_line, v_profile) from SCP to a
        time-indexed reference (s_ref, u_ref, t_ref) at dt resolution.
        Mirrors reference() in HW2 cartpole_balance.py.
        """
        from scipy.interpolate import interp1d
        from optimizer import compute_curvature_from_path
        p = self.p

        # Arc-length and time at each spatial point
        diff = np.vstack([np.diff(racing_line, axis=0),
                          racing_line[0:1] - racing_line[-1:]])  # close loop
        ds     = np.linalg.norm(diff, axis=1)
        v_safe = np.maximum(v_profile, 0.5)
        t_pts  = np.concatenate([[0.], np.cumsum(ds[:-1] / v_safe[:-1])])
        T_lap  = t_pts[-1] + ds[-1] / v_safe[-1]

        t_ref = np.arange(0., T_lap, self.dt)

        # Heading: centered differences (periodic)
        dx_cl = np.roll(racing_line[:, 0], -1) - np.roll(racing_line[:, 0], 1)
        dy_cl = np.roll(racing_line[:, 1], -1) - np.roll(racing_line[:, 1], 1)
        psi_pts   = np.arctan2(dy_cl, dx_cl)
        kappa_pts = compute_curvature_from_path(racing_line)
        omega_pts = v_safe * kappa_pts  # quasi-steady yaw rate

        def _interp(arr):
            t_ext = np.append(t_pts, T_lap)
            a_ext = np.append(arr, arr[0])
            return interp1d(t_ext, a_ext, kind='linear',
                            bounds_error=False, fill_value=(arr[0], arr[-1]))(t_ref)

        s_ref = np.stack([
            _interp(racing_line[:, 0]),
            _interp(racing_line[:, 1]),
            _interp(psi_pts),
            _interp(v_safe),
            np.zeros(len(t_ref)),   # vy ≈ 0 (quasi-steady)
            _interp(omega_pts),
        ], axis=1)

        # Reference control: Ackermann steering + feedforward drive force
        kappa_r = _interp(kappa_pts)
        delta_r = np.clip(np.arctan(p.L * kappa_r), -p.delta_max, p.delta_max)
        dvx_r   = np.gradient(s_ref[:, 3], self.dt)
        F_drag  = 0.5 * p.rho * p.C_d * p.A_front * s_ref[:, 3]**2
        F_roll  = p.C_roll * p.mass * 9.81
        Fd_r    = np.clip(p.mass * dvx_r + F_drag + F_roll,
                          -p.F_brake_max, p.F_drive_max)
        u_ref = np.stack([delta_r, Fd_r], axis=1)

        return s_ref, u_ref, t_ref

    # ── iLQR backward pass (like ilqr() in HW2 cartpole_swingup.py) ─────

    def _ilqr_backward_pass(self):
        """
        iLQR backward pass — computes gain matrices Y[k] and offset vectors y[k].

        Exactly mirrors HW2 cartpole_swingup.py backward pass:
          P = QN
          p = QN @ (s_bar[-1] - s_goal)        # here s_goal = s_ref[-1] (loop back)
          for k in range(N-1, -1, -1):
              inv = inv(R + B[k]^T P B[k])
              Y[k] = -inv @ B[k]^T @ P @ A[k]
              y[k] = -inv @ (R @ u_bar[k] + B[k]^T @ p)
              p = Q @ (s_bar[k] - s_goal) + A[k]^T @ (p + P @ B[k] @ y[k])
              P = Q + A[k]^T @ P @ (A[k] + B[k] @ Y[k])

        Returns (Y, y):
          Y: (N, 2, 6) gain matrices
          y: (N, 2) offset vectors
        """
        N = self.N
        n, m = 6, 2
        Y = np.zeros((N, m, n))
        y = np.zeros((N, m))

        # s_goal = final reference state (the trajectory is periodic)
        s_goal = self.s_ref[-1]

        P = self.QN.copy()
        p = self.QN @ (self.s_ref[-1] - s_goal)  # = 0 for periodic traj

        for k in range(N - 1, -1, -1):
            A, B = self._linearize(self.s_ref[k], self.u_ref[k])
            inv_k  = np.linalg.inv(self.R + B.T @ P @ B)
            Y[k]   = -inv_k @ B.T @ P @ A
            y[k]   = -inv_k @ (self.R @ self.u_ref[k] + B.T @ p)
            p      = self.Q @ (self.s_ref[k] - s_goal) + A.T @ (p + P @ B @ y[k])
            P      = self.Q + A.T @ P @ (A + B @ Y[k])

        return Y, y

    # ── Online control (like simulate() in HW2 cartpole_swingup.py) ──────

    def control(self, state: CarState):
        """
        Closed-loop iLQR policy (matches HW2 cartpole_swingup.py line 132):
          du[k] = Y[k] @ (s[k] - s_ref[k]) + y[k]
          u[k]  = u_ref[k] + du[k]
        """
        k = self._step % self.N

        s = np.array([state.x, state.y, state.psi, state.vx, state.vy, state.omega])
        s_err    = s - self.s_ref[k]
        s_err[2] = (s_err[2] + np.pi) % (2*np.pi) - np.pi  # wrap heading error

        du = self.Y[k] @ s_err + self.y[k]
        u = self.u_ref[k] + du
        delta   = float(np.clip(u[0], -self.p.delta_max,   self.p.delta_max))
        F_drive = float(np.clip(u[1], -self.p.F_brake_max, self.p.F_drive_max))

        self._step += 1
        return delta, F_drive


def generate_racing_line(track, mode="center"):
    """
    Generate a racing line on the given track.

    Returns (raceline, v_profile_or_None).

    - "center":      centerline, no optimization → (centerline, None)
    - "min_laptime": SCP joint path+speed optimizer → (raceline, v_profile)
                     v_profile is the jointly-optimized speed at each point;
                     passed to PurePursuitController so it uses the SCP speeds
                     instead of recomputing from curvature.
    """
    if mode == "center":
        return track.centerline.copy(), None
    elif mode == "min_laptime":
        from optimizer import optimize_racing_line
        results = optimize_racing_line(track, n_stations=80, solver='scp')
        return results['raceline_scp'], results['velocity_scp']
    else:
        return track.centerline.copy(), None
