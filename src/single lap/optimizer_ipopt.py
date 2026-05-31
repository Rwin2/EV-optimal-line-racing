"""
Speed profile optimizer using CasADi.

Optimizes a scalar speed profile along a fixed racing line while respecting:
- lateral grip limits,
- longitudinal acceleration/braking limits,
- speed limits,
- simple EV energy consumption model.

Objective (normalized adaptively):
    J = w_time * (T / T_ref) + w_energy * (E / E_ref)

where T_ref and E_ref are estimated from the baseline profile so that
both terms are O(1) regardless of track length or car parameters.
"""

import os
import sys

import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(__file__))
from car import CarParams


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _compute_curvature(racing_line: np.ndarray) -> np.ndarray:
    """Absolute curvature (1/m) for a closed racing line."""
    x = racing_line[:, 0]
    y = racing_line[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    ds = np.hypot(dx, dy)
    ds = np.maximum(ds, 1e-6)

    curvature = np.abs(dx * ddy - dy * ddx) / ds ** 3
    return np.maximum(curvature, 0.0)


def _arc_lengths(racing_line: np.ndarray) -> np.ndarray:
    """Segment arc-lengths ds[i] = ||p[i+1] - p[i]|| (closed, so last→first)."""
    diff = np.diff(racing_line, axis=0, append=racing_line[0:1])
    return np.linalg.norm(diff, axis=1)


def _resample_uniform(racing_line: np.ndarray, n_target: int):
    """
    Re-sample a closed racing line to n_target *uniformly spaced* points.

    Returns (resampled_line, original_arc_fractions) where
    original_arc_fractions are the cumulative arc-length fractions at which
    the optimized speeds must be evaluated to interpolate back to the full
    resolution.
    """
    ds = _arc_lengths(racing_line)
    s_full = np.concatenate([[0.0], np.cumsum(ds[:-1])])
    total = ds.sum()

    s_uniform = np.linspace(0.0, total, n_target, endpoint=False)

    x_r = np.interp(s_uniform, s_full, racing_line[:, 0], period=total)
    y_r = np.interp(s_uniform, s_full, racing_line[:, 1], period=total)
    resampled = np.column_stack([x_r, y_r])

    return resampled, s_full, s_uniform, total


# ---------------------------------------------------------------------------
# Speed-profile smoothing (feasibility)
# ---------------------------------------------------------------------------

def _smooth_speed_profile(
    v: np.ndarray,
    ds: np.ndarray,
    v_min: float,
    v_max: float,
    a_max: float,
    a_brake: float,
    n_passes: int = 3,
) -> np.ndarray:
    """
    Forward/backward smoothing to enforce acceleration/braking limits.
    Multiple passes handle the wrap-around (closed circuit) correctly.
    """
    v = np.clip(v, v_min, v_max).copy()
    n = len(v)

    for _ in range(n_passes):
        # Forward pass  (acceleration limited)
        for i in range(1, n):
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_max * ds[i - 1]))

        # Wrap: last→first
        v[0] = min(v[0], np.sqrt(v[-1] ** 2 + 2.0 * a_max * ds[-1]))

        # Backward pass (braking limited)
        for i in range(n - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_brake * ds[i]))

        # Wrap: first→last
        v[-1] = min(v[-1], np.sqrt(v[0] ** 2 + 2.0 * a_brake * ds[-1]))

    return np.clip(v, v_min, v_max)


# ---------------------------------------------------------------------------
# Pareto-front pruning
# ---------------------------------------------------------------------------

def _prune_dominated(lap_times, energies, profiles, w_energies):
    """Remove dominated solutions from a candidate Pareto set.

    A point (t, e) is dominated if there exists another point (t', e') with
    t' <= t AND e' <= e (with at least one strict inequality).
    """
    n = len(lap_times)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i:
                continue
            if lap_times[j] <= lap_times[i] and energies[j] <= energies[i]:
                if lap_times[j] < lap_times[i] or energies[j] < energies[i]:
                    dominated = True
                    break
        if not dominated:
            keep.append(i)

    return (
        [lap_times[i] for i in keep],
        [energies[i] for i in keep],
        [profiles[i] for i in keep],
        [w_energies[i] for i in keep],
    )


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class SpeedProfileOptimizer:
    """
    CasADi-based speed profile optimizer for a fixed racing line.

    The optimizer works on a uniformly-resampled version of the racing line
    (n_points nodes) for numerical conditioning, then interpolates the
    solution back to the original resolution.
    """

    def __init__(
        self,
        racing_line: np.ndarray,
        params: CarParams = None,
        v_min: float = 5.0,
        v_max: float = None,
        a_max: float = 5.0,
        a_brake: float = 8.0,
        mu: float = None,
        grip_fraction: float = 0.85,
        g: float = 9.81,
        n_points: int = None,
        curvature: np.ndarray = None,
    ):
        self.racing_line_full = np.asarray(racing_line, dtype=float)
        if self.racing_line_full.ndim != 2 or self.racing_line_full.shape[1] != 2:
            raise ValueError("racing_line must be an (N, 2) array.")

        self.p = params or CarParams()
        self.g = float(g)
        self.v_min = float(v_min)
        self.v_max = float(v_max if v_max is not None else self.p.v_max)
        self.a_max = float(a_max)
        self.a_brake = float(a_brake)
        self.mu = float(mu if mu is not None else self.p.mu)
        self.grip_fraction = float(grip_fraction)

        if n_points is not None and n_points != len(racing_line):
            # Uniformly-resampled line for the optimizer
            (
                self.racing_line,
                self._s_full,
                self._s_uniform,
                self._total_length,
            ) = _resample_uniform(self.racing_line_full, n_points)
            self._resampled = True
        else:
            # Work directly at full resolution (no resampling mismatch)
            self.racing_line = self.racing_line_full
            ds_tmp = _arc_lengths(self.racing_line_full)
            self._s_full = np.concatenate([[0.0], np.cumsum(ds_tmp[:-1])])
            self._s_uniform = self._s_full
            self._total_length = ds_tmp.sum()
            self._resampled = False

        self.n = len(self.racing_line)

        self.ds = _arc_lengths(self.racing_line)
        # Use externally-provided curvature for consistency with SCP
        if curvature is not None:
            if self._resampled:
                self.curvature = np.interp(
                    self._s_uniform, self._s_full,
                    np.abs(curvature), period=self._total_length)
            else:
                self.curvature = np.abs(curvature)
        else:
            self.curvature = _compute_curvature(self.racing_line)

        # Pre-compute the full-resolution ds for metrics
        self._ds_full = _arc_lengths(self.racing_line_full)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_full_resolution(self, v_opt: np.ndarray) -> np.ndarray:
        """Interpolate optimizer solution (uniform grid) → original resolution."""
        if not self._resampled:
            return np.clip(v_opt, self.v_min, self.v_max)
        v_full = np.interp(
            self._s_full,
            self._s_uniform,
            v_opt,
            period=self._total_length,
        )
        return np.clip(v_full, self.v_min, self.v_max)

    def _initial_guess(self, override=None) -> np.ndarray:
        """Return a feasible initial guess on the uniform grid."""
        if override is not None:
            arr = np.asarray(override, dtype=float).flatten()
            n_full = len(self.racing_line_full)
            if arr.shape[0] == n_full:
                # Subsample from full resolution
                arr = np.interp(
                    self._s_uniform, self._s_full, arr, period=self._total_length
                )
            elif arr.shape[0] != self.n:
                raise ValueError(
                    f"initial_guess must have length {self.n} (optimizer grid) "
                    f"or {n_full} (full racing line)."
                )
        else:
            eps = 1e-6
            arr = np.sqrt(
                self.mu * self.g / np.maximum(self.curvature, eps)
            )

        return _smooth_speed_profile(
            arr, self.ds, self.v_min, self.v_max, self.a_max, self.a_brake
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def baseline_speed_profile(self) -> np.ndarray:
        """
        Curvature-limited baseline speed profile at full resolution.
        Useful as a reference and warm-start seed.
        """
        eps = 1e-6
        v0 = np.sqrt(self.grip_fraction * self.mu * self.g / np.maximum(self.curvature, eps))
        v0 = _smooth_speed_profile(
            v0, self.ds, self.v_min, self.v_max, self.a_max, self.a_brake
        )
        return self._to_full_resolution(v0)

    def optimize(
        self,
        w_time: float = 1.0,
        w_energy: float = 1.0,
        T_ref: float = None,
        E_ref: float = None,
        initial_guess: np.ndarray = None,
    ) -> np.ndarray:
        """
        Solve for the optimal speed profile.

        Parameters
        ----------
        w_time, w_energy : scalar weights (both positive).
        T_ref, E_ref     : normalization constants (auto-estimated from baseline
                           if not provided — recommended).
        initial_guess    : 1-D speed array at optimizer or full resolution.

        Returns
        -------
        v_full : (N_full,) optimal speed profile at original resolution.
        """
        if w_time <= 0 or w_energy < 0:
            raise ValueError("w_time must be > 0 and w_energy >= 0.")

        # Auto-calibrate normalization constants from baseline
        if T_ref is None or E_ref is None:
            v_base = self.baseline_speed_profile()
            m = self.compute_metrics(v_base)
            T_ref = T_ref or max(m["lap_time_s"], 1.0)
            E_ref = E_ref or max(abs(m["energy_J"]), 1.0)

        v0 = self._initial_guess(initial_guess)

        opti = ca.Opti()

        # ------------------------------------------------------------------
        # Decision variable
        # ------------------------------------------------------------------
        v = opti.variable(self.n)

        # Periodic successor
        v_next = ca.vertcat(v[1:], v[0])

        ds_cas = ca.DM(self.ds)
        eps_v = 1e-3  # avoid division by zero in dt

        # Average speed on each segment
        v_avg = 0.5 * (v + v_next)

        # Traversal time per segment
        dt_seg = ds_cas / (v_avg + eps_v)

        # Total lap time
        T = ca.sum1(dt_seg)

        # ------------------------------------------------------------------
        # Longitudinal dynamics
        # ------------------------------------------------------------------
        a_long = (v_next ** 2 - v ** 2) / (2.0 * ds_cas + eps_v)

        # Resistive forces (drag + rolling)
        F_drag = 0.5 * self.p.rho * self.p.C_d * self.p.A_front * v_avg ** 2
        F_roll = self.p.C_roll * self.p.mass * self.g

        # Net traction force
        F_long = self.p.mass * a_long + F_drag + F_roll

        # Mechanical power per segment
        P_mech = F_long * v_avg

        # EV energy model — use CasADi smooth max/min for differentiability
        # ca.fmax/fmin are exact (not smooth-abs), which is fine for interior-point.
        P_drive = ca.fmax(P_mech, 0.0)
        P_regen = ca.fmin(P_mech, 0.0)

        E = ca.sum1(
            (P_drive / self.p.eta_motor + P_regen * self.p.eta_regen) * dt_seg
        )

        # ------------------------------------------------------------------
        # Objective (normalized so both terms are ~ O(1))
        # ------------------------------------------------------------------
        J = w_time * (T / T_ref) + w_energy * (E / E_ref)
        opti.minimize(J)

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------

        # Speed bounds
        opti.subject_to(v >= self.v_min)
        opti.subject_to(v <= self.v_max)

        # Lateral grip: v² ≤ grip_fraction * mu*g / kappa
        eps_kappa = 1e-4
        v2_max_lat = ca.DM(
            self.grip_fraction * self.mu * self.g / (self.curvature + eps_kappa)
        )
        opti.subject_to(v ** 2 <= v2_max_lat)

        # Longitudinal acceleration
        opti.subject_to(
            v_next ** 2 - v ** 2 <= 2.0 * self.a_max * ds_cas
        )

        # Longitudinal braking
        opti.subject_to(
            v ** 2 - v_next ** 2 <= 2.0 * self.a_brake * ds_cas
        )

        # ------------------------------------------------------------------
        # Initial guess & solver
        # ------------------------------------------------------------------
        opti.set_initial(v, v0)

        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 10000,
                "ipopt.tol": 1e-5,
                "ipopt.acceptable_tol": 1e-3,
                "ipopt.acceptable_iter": 10,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.linear_solver": "mumps",
                "ipopt.warm_start_init_point": "yes",
                "ipopt.warm_start_bound_push": 1e-6,
                "ipopt.warm_start_mult_bound_push": 1e-6,
                "ipopt.warm_start_slack_bound_push": 1e-6,
                "ipopt.hessian_approximation": "limited-memory",  # faster for large N
            },
        )

        sol = opti.solve()

        v_opt = np.asarray(sol.value(v)).flatten()
        return self._to_full_resolution(v_opt)

    def compute_metrics(self, v_profile: np.ndarray) -> dict:
        """
        Compute lap time (s) and net electrical energy (J) for a given
        speed profile at full resolution.

        Uses the same energy model as the optimizer (exact fmax/fmin split,
        not smooth-abs) to keep metrics consistent with the objective.
        """
        v = np.asarray(v_profile, dtype=float)
        n = len(v)
        if n != len(self.racing_line_full):
            raise ValueError(
                f"v_profile length {n} != full racing line length "
                f"{len(self.racing_line_full)}."
            )

        ds = self._ds_full
        v_next = np.roll(v, -1)
        v_avg = 0.5 * (v + v_next)

        eps_v = 1e-3
        dt_seg = ds / (v_avg + eps_v)
        T = dt_seg.sum()

        a_long = (v_next ** 2 - v ** 2) / (2.0 * ds + eps_v)

        F_drag = 0.5 * self.p.rho * self.p.C_d * self.p.A_front * v_avg ** 2
        F_roll = self.p.C_roll * self.p.mass * self.g
        F_long = self.p.mass * a_long + F_drag + F_roll

        P_mech = F_long * v_avg
        P_drive = np.maximum(P_mech, 0.0)
        P_regen = np.minimum(P_mech, 0.0)

        E = np.sum(
            (P_drive / self.p.eta_motor + P_regen * self.p.eta_regen) * dt_seg
        )

        return {
            "lap_time_s": float(T),
            "energy_J": float(E),
            "energy_kWh": float(E / 3.6e6),
            "avg_speed_ms": float(v.mean()),
            "max_speed_ms": float(v.max()),
        }

    def compute_pareto_frontier(
        self,
        n_points: int = 15,
        w_energy_min: float = 1e-4,
        w_energy_max: float = 1e2,
        T_ref: float = None,
        E_ref: float = None,
        prune_dominated: bool = True,
        verbose: bool = True,
        v_seed: np.ndarray = None,
    ) -> dict:
        """
        Sweep w_energy ∈ [w_energy_min, w_energy_max] on a log scale and
        collect the resulting (lap_time, energy) trade-off curve.

        Strategy
        --------
        - Normalization (T_ref, E_ref) is estimated once from the baseline so
          the two objective terms are always comparably scaled.
        - Each solve is warm-started from the *previous* solution (chain) and,
          on failure, retried from the baseline.
        - Dominated solutions are filtered out to return a true Pareto front.

        Returns
        -------
        dict with keys:
            'w_energy'  : list of weights used
            'lap_time'  : list of lap times (s), sorted ascending
            'energy'    : list of net energies (J)
            'profiles'  : list of speed profile arrays
        """
        # Validate inputs
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")
        if w_energy_min <= 0 or w_energy_max <= 0:
            raise ValueError("Weight bounds must be positive.")
        if w_energy_min >= w_energy_max:
            raise ValueError("w_energy_min must be < w_energy_max.")

        # Calibrate normalization from seed (SCP profile) or baseline
        v_base = v_seed if v_seed is not None else self.baseline_speed_profile()
        base_metrics = self.compute_metrics(v_base)
        T_ref = T_ref or max(base_metrics["lap_time_s"], 1.0)
        E_ref = E_ref or max(abs(base_metrics["energy_J"]), 1.0)

        if verbose:
            print(
                f"  Normalization: T_ref={T_ref:.1f}s  "
                f"E_ref={E_ref/1e6:.3f} MJ"
            )

        weights = np.logspace(
            np.log10(w_energy_min),
            np.log10(w_energy_max),
            n_points,
        )

        lap_times, energies, profiles, w_used = [], [], [], []
        v_prev = v_base  # warm-start seed from SCP

        for i, we in enumerate(weights):
            if verbose:
                print(
                    f"  [{i+1:2d}/{n_points}] w_energy={we:.2e} ...",
                    end=" ",
                    flush=True,
                )

            v_opt = None
            # Try chained warm-start first, then baseline cold-start
            for guess in (v_prev, v_base):
                try:
                    v_opt = self.optimize(
                        w_time=1.0,
                        w_energy=we,
                        T_ref=T_ref,
                        E_ref=E_ref,
                        initial_guess=guess,
                    )
                    break
                except Exception:
                    continue

            if v_opt is None:
                if verbose:
                    print("FAILED (skipped)")
                continue

            m = self.compute_metrics(v_opt)

            if verbose:
                print(
                    f"T={m['lap_time_s']:.1f}s  "
                    f"E={m['energy_J']/1e6:.3f}MJ"
                )

            lap_times.append(m["lap_time_s"])
            energies.append(m["energy_J"])
            profiles.append(v_opt)
            w_used.append(we)
            v_prev = v_opt  # update warm-start seed

        if len(lap_times) < 2:
            raise RuntimeError(
                f"Pareto frontier has only {len(lap_times)} valid point(s). "
                "Try widening the weight range or reducing n_points."
            )

        # Sort by ascending lap time
        order = np.argsort(lap_times)
        lap_times = [lap_times[i] for i in order]
        energies = [energies[i] for i in order]
        profiles = [profiles[i] for i in order]
        w_used = [w_used[i] for i in order]

        # Filter dominated solutions
        if prune_dominated and len(lap_times) > 2:
            lap_times, energies, profiles, w_used = _prune_dominated(
                lap_times, energies, profiles, w_used
            )
            if verbose:
                print(
                    f"  → {len(lap_times)} non-dominated points "
                    "after pruning."
                )

        return {
            "w_energy": w_used,
            "lap_time": lap_times,
            "energy": energies,
            "profiles": profiles,
        }