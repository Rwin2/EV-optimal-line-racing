"""
Racing controllers / baselines.
Each controller takes the current car state and track, and returns (delta, F_drive).
"""

import numpy as np
from car import CarState, CarParams


class PurePursuitController:
    """
    Pure pursuit path follower with speed profiling.
    Base controller that follows a given racing line (defaults to centerline).
    """

    def __init__(self, track, racing_line=None, params=None,
                 lookahead=15.0, speed_mode="curvature"):
        self.track = track
        self.racing_line = racing_line if racing_line is not None else track.centerline
        self.p = params or CarParams()
        self.lookahead = lookahead
        self.speed_mode = speed_mode
        # Precompute target speeds
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


def generate_racing_line(track, mode="center"):
    """
    Generate a racing line on the given track.
    - "center": follow centerline
    - "minimum_curvature": smooth out corners by using track width
    """
    if mode == "center":
        return track.centerline.copy()
    elif mode == "minimum_curvature":
        from optimizer import optimize_racing_line
        results = optimize_racing_line(track, n_stations=150, solver='custom')
        return results['raceline_custom']
    else:
        return track.centerline.copy()
