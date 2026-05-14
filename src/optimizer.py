"""
Speed profile optimizer using CasADi.

Optimizes a scalar speed profile along a fixed racing line while respecting:
- lateral grip limits,
- longitudinal acceleration/braking limits,
- speed limits,
- simple EV energy consumption model.

Objective:
    J = w_time * T + w_energy * E
"""

import os
import sys

import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(__file__))
from car import CarParams


def _compute_curvature(racing_line):
    """Compute absolute curvature for a closed racing line."""
    x = racing_line[:, 0]
    y = racing_line[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    ds = np.sqrt(dx**2 + dy**2)
    ds = np.maximum(ds, 1e-6)

    curvature = np.abs(dx * ddy - dy * ddx) / (ds**3)
    curvature = np.maximum(curvature, 0.0)

    return curvature


def _downsample_racing_line(racing_line, n_target=150):
    """Downsample racing line for optimization."""
    n = len(racing_line)

    if n_target >= n:
        return racing_line.copy(), np.arange(n)

    indices = np.linspace(0, n - 1, n_target, dtype=int)
    return racing_line[indices], indices


class SpeedProfileOptimizer:
    """
    CasADi-based speed profile optimizer.
    """

    def __init__(
        self,
        racing_line,
        params=None,
        v_min=5.0,
        v_max=None,
        a_max=5.0,
        a_brake=8.0,
        mu=None,
        g=9.81,
        n_points=150,
    ):

        self.racing_line_full = np.asarray(racing_line, dtype=float)

        self.p = params or CarParams()

        self.racing_line, self.downsample_indices = (
            _downsample_racing_line(
                self.racing_line_full,
                n_target=n_points
            )
        )

        self.n = len(self.racing_line)

        if self.n < 2:
            raise ValueError("Racing line too short.")

        self.curvature = _compute_curvature(self.racing_line)

        self.ds = np.linalg.norm(
            np.diff(
                self.racing_line,
                axis=0,
                append=self.racing_line[0:1]
            ),
            axis=1
        )

        self.v_min = float(v_min)
        self.v_max = float(v_max if v_max is not None else self.p.v_max)

        self.a_max = float(a_max)
        self.a_brake = float(a_brake)

        self.mu = float(mu if mu is not None else self.p.mu)

        self.g = float(g)

    def _smooth_initial_profile(self, speeds):
        """Forward/backward smoothing."""
        v = np.asarray(speeds, dtype=float).copy()

        v = np.clip(v, self.v_min, self.v_max)

        # Forward pass
        for i in range(1, self.n):
            vmax_next = np.sqrt(
                v[i - 1] ** 2 + 2.0 * self.a_max * self.ds[i - 1]
            )
            v[i] = min(v[i], vmax_next)

        # Backward pass
        for i in range(self.n - 2, -1, -1):
            vmax_prev = np.sqrt(
                v[i + 1] ** 2 + 2.0 * self.a_brake * self.ds[i]
            )
            v[i] = min(v[i], vmax_prev)

        return np.clip(v, self.v_min, self.v_max)

    def optimize(
        self,
        w_time=1.0,
        w_energy=1.0,
        T_ref=100.0,
        E_ref=1e5,
        initial_guess=None,
    ):
        """
        Solve optimal speed profile.
        """

        opti = ca.Opti()

        eps = 1e-3

        # Decision variable
        v = opti.variable(self.n)

        # Periodic shift
        v_next = ca.vertcat(v[1:], v[0])

        ds_cas = ca.DM(self.ds)

        # Average speed on segment
        v_avg = 0.5 * (v + v_next)

        # Segment traversal time
        dt = ds_cas / (v_avg + eps)

        # Total lap time
        T = ca.sum1(dt)

        # Longitudinal acceleration
        a_long = (
            (v_next**2 - v**2)
            / (2.0 * ds_cas + eps)
        )

        # Aerodynamic drag
        drag = (
            0.5
            * self.p.rho
            * self.p.C_d
            * self.p.A_front
            * v_avg**2
        )

        # Rolling resistance
        rolling = (
            self.p.C_roll
            * self.p.mass
            * self.g
        )

        # Total longitudinal force
        F_long = (
            self.p.mass * a_long
            + drag
            + rolling
        )

        # Mechanical power
        P_mech = F_long * v_avg

        # Smooth positive/negative split for energy
        eps_energy = 1e-1
        P_abs = ca.sqrt(P_mech**2 + eps_energy**2)
        P_drive = 0.5 * (P_mech + P_abs)
        P_brake = 0.5 * (P_mech - P_abs)

        # EV energy model
        E = ca.sum1(
            (
                P_drive / self.p.eta_motor
                + P_brake * self.p.eta_regen
            )
            * dt
        )

        # Normalized objective
        J = (
            w_time * (T / T_ref)
            + w_energy * (E / E_ref)
        )

        opti.minimize(J)

        # --------------------------------------------------
        # Constraints
        # --------------------------------------------------

        # Speed limits
        opti.subject_to(v >= self.v_min)
        opti.subject_to(v <= self.v_max)

        # Lateral grip
        max_v2 = (
            self.mu * self.g
            / (self.curvature + eps)
        )

        opti.subject_to(
            v**2 <= ca.DM(max_v2)
        )

        # Longitudinal accel
        opti.subject_to(
            v_next**2 - v**2
            <= 2.0 * self.a_max * ds_cas
        )

        # Longitudinal braking
        opti.subject_to(
            v**2 - v_next**2
            <= 2.0 * self.a_brake * ds_cas
        )

        # --------------------------------------------------
        # Initial guess
        # --------------------------------------------------

        if initial_guess is None:

            v0 = np.sqrt(
                self.mu * self.g
                / (self.curvature + 1e-3)
            )

            v0 = np.clip(
                v0,
                self.v_min,
                self.v_max
            )

        else:

            v0 = np.asarray(initial_guess)
            if v0.ndim != 1:
                raise ValueError("initial_guess must be a 1D speed array")
            if v0.shape[0] == len(self.racing_line_full):
                v0 = v0[self.downsample_indices]
            elif v0.shape[0] != self.n:
                raise ValueError(
                    f"initial_guess length must be {self.n} or {len(self.racing_line_full)}"
                )

        v0 = self._smooth_initial_profile(v0)

        opti.set_initial(v, v0)

        # --------------------------------------------------
        # Solver
        # --------------------------------------------------

        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 15000,
                "ipopt.tol": 1e-4,
                "ipopt.acceptable_tol": 1e-3,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.linear_solver": "mumps",
                "ipopt.warm_start_init_point": "yes",
                "ipopt.warm_start_bound_push": 1e-6,
                "ipopt.warm_start_mult_bound_push": 1e-6,
                "ipopt.warm_start_slack_bound_push": 1e-6,
            }
        )

        sol = opti.solve()

        v_opt = np.asarray(
            sol.value(v)
        ).flatten()

        # --------------------------------------------------
        # Interpolate back to full resolution
        # --------------------------------------------------

        n_full = len(self.racing_line_full)

        v_full = np.interp(
            np.arange(n_full),
            self.downsample_indices,
            v_opt,
            period=n_full
        )

        v_full = np.clip(
            v_full,
            self.v_min,
            self.v_max
        )

        return v_full

    def baseline_speed_profile(self):
        """Compute a baseline feasible speed profile using curvature limits."""
        eps = 1e-6
        v0 = np.sqrt(
            self.mu * self.g
            / np.maximum(self.curvature, eps)
        )
        v0 = np.clip(v0, self.v_min, self.v_max)
        v_smooth = self._smooth_initial_profile(v0)

        n_full = len(self.racing_line_full)
        v_full = np.interp(
            np.arange(n_full),
            self.downsample_indices,
            v_smooth,
            period=n_full
        )
        return np.clip(v_full, self.v_min, self.v_max)

    def compute_pareto_frontier(
        self,
        n_points=15,
        w_energy_min=1e-4,
        w_energy_max=1e2,
        T_ref=100.0,
        E_ref=1e5,
    ):
        """Compute Pareto frontier for trade-off between lap time and energy."""
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")
        if w_energy_min <= 0 or w_energy_max <= 0:
            raise ValueError("w_energy_min and w_energy_max must be positive.")
        if w_energy_min >= w_energy_max:
            raise ValueError("w_energy_min must be smaller than w_energy_max.")

        # Try to get at least n_points by adaptively narrowing the range
        max_attempts = 3
        current_w_max = w_energy_max
        frontier = None

        for attempt in range(max_attempts):
            w_energy_values = np.logspace(
                np.log10(w_energy_min),
                np.log10(current_w_max),
                n_points,
            )

            temp_frontier = {
                'w_energy': [],
                'lap_time': [],
                'energy': [],
                'profiles': [],
            }

            baseline_profile = self.baseline_speed_profile()
            v_prev = baseline_profile

            for w_energy in w_energy_values:
                v_opt = None
                for initial_guess in (v_prev, baseline_profile):
                    try:
                        v_opt = self.optimize(
                            w_time=1.0,
                            w_energy=w_energy,
                            T_ref=T_ref,
                            E_ref=E_ref,
                            initial_guess=initial_guess,
                        )
                        break
                    except RuntimeError:
                        continue

                if v_opt is None:
                    print(
                        f"Warning: optimizer failed for w_energy={w_energy:.2e} and was skipped."
                    )
                    continue

                metrics = self.compute_metrics(v_opt)

                temp_frontier['w_energy'].append(w_energy)
                temp_frontier['lap_time'].append(metrics['lap_time_s'])
                temp_frontier['energy'].append(metrics['energy_J'])
                temp_frontier['profiles'].append(v_opt)
                v_prev = v_opt

            if len(temp_frontier['lap_time']) >= n_points:
                frontier = temp_frontier
                break
            elif len(temp_frontier['lap_time']) >= 2:
                # If we have at least 2 points but fewer than requested, use them
                frontier = temp_frontier
                break
            else:
                # Narrow the range and try again
                current_w_max /= 10.0
                if current_w_max <= w_energy_min:
                    break

        if frontier is None or len(frontier['lap_time']) < 2:
            raise RuntimeError(
                "Pareto frontier produced fewer than 2 valid points. "
                "Try narrowing the weight range or reducing n_points."
            )

        order = np.argsort(frontier['lap_time'])
        frontier['w_energy'] = [frontier['w_energy'][i] for i in order]
        frontier['lap_time'] = [frontier['lap_time'][i] for i in order]
        frontier['energy'] = [frontier['energy'][i] for i in order]
        frontier['profiles'] = [frontier['profiles'][i] for i in order]

        return frontier

    def compute_metrics(self, v_profile):
        """
        Compute lap time and energy.
        """

        eps = 1e-3

        v = np.asarray(v_profile)

        ds = np.linalg.norm(
            np.diff(
                self.racing_line_full,
                axis=0,
                append=self.racing_line_full[0:1]
            ),
            axis=1
        )

        v_next = np.roll(v, -1)

        v_avg = 0.5 * (v + v_next)

        dt = ds / (v_avg + eps)

        T = np.sum(dt)

        a_long = (
            (v_next**2 - v**2)
            / (2.0 * ds + eps)
        )

        drag = (
            0.5
            * self.p.rho
            * self.p.C_d
            * self.p.A_front
            * v_avg**2
        )

        rolling = (
            self.p.C_roll
            * self.p.mass
            * self.g
        )

        F_long = (
            self.p.mass * a_long
            + drag
            + rolling
        )

        P_mech = F_long * v_avg

        P_drive = np.maximum(P_mech, 0)

        P_brake = np.minimum(P_mech, 0)

        E = np.sum(
            (
                P_drive / self.p.eta_motor
                + P_brake * self.p.eta_regen
            )
            * dt
        )

        return {
            "lap_time_s": float(T),
            "energy_J": float(E),
        }