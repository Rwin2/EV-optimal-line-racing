"""
Speed profile optimizer using CasADi.

Optimizes a scalar speed profile along a fixed racing line while respecting
lateral grip, longitudinal acceleration/braking, and speed limits.
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
    speed = np.sqrt(dx**2 + dy**2)
    speed = np.maximum(speed, 1e-6)
    curvature = np.abs(dx * ddy - dy * ddx) / (speed**3)
    curvature = np.maximum(curvature, 0.0)
    return curvature


def _downsample_racing_line(racing_line, n_target=50):
    """Downsample a racing line to reduce problem size."""
    n = len(racing_line)
    indices = np.linspace(0, n - 1, n_target, dtype=int)
    return racing_line[indices], indices


class SpeedProfileOptimizer:
    """Optimize a longitudinal speed profile for a fixed racing line."""

    def __init__(self, racing_line, params=None, v_min=5.0, v_max=None,
                 a_max=5.0, a_brake=8.0, mu=None, g=9.81,
                 regen_efficiency=None, motor_efficiency=None, n_points=50):
        self.racing_line_full = np.asarray(racing_line, dtype=float)
        self.p = params or CarParams()
        
        # Downsample the racing line to reduce solver complexity
        self.racing_line, self.downsample_indices = _downsample_racing_line(
            self.racing_line_full, n_target=n_points)
        
        self.n = len(self.racing_line)
        self.curvature = _compute_curvature(self.racing_line)
        self.ds = np.linalg.norm(
            np.diff(self.racing_line, axis=0, append=self.racing_line[0:1]), axis=1)
        self.v_min = float(v_min)
        self.v_max = float(v_max if v_max is not None else self.p.v_max)
        self.a_max = float(a_max)
        self.a_brake = float(a_brake)
        self.mu = float(mu if mu is not None else self.p.mu)
        self.g = float(g)
        self.eta_regen = float(regen_efficiency if regen_efficiency is not None else self.p.eta_regen)
        self.eta_motor = float(motor_efficiency if motor_efficiency is not None else self.p.eta_motor)

        if self.n < 2:
            raise ValueError("Racing line must contain at least two points.")

    def _smooth_initial_profile(self, speeds):
        """Enforce simple acceleration/braking limits on an initial speed guess."""
        v = np.asarray(speeds, dtype=float).copy()
        v = np.clip(v, self.v_min, self.v_max)
        n = self.n

        # Forward acceleration pass
        for i in range(1, n):
            v_max_next = np.sqrt(v[i-1]**2 + 2.0 * self.a_max * self.ds[i-1])
            v[i] = min(v[i], v_max_next)

        # Backward braking pass
        for i in range(n-2, -1, -1):
            v_max_prev = np.sqrt(v[i+1]**2 + 2.0 * self.a_brake * self.ds[i])
            v[i] = min(v[i], v_max_prev)

        # Close the loop between last and first point
        v_last_allowed = np.sqrt(v[0]**2 + 2.0 * self.a_max * self.ds[-1])
        v[-1] = min(v[-1], v_last_allowed)
        v_first_allowed = np.sqrt(v[-1]**2 + 2.0 * self.a_brake * self.ds[-1])
        v[0] = min(v[0], v_first_allowed)

        return np.clip(v, self.v_min, self.v_max)

    def optimize(self, w_time=1.0, w_energy=1e-3, initial_guess=None):
        """Solve for the optimal speed profile and return it as a NumPy array."""
        opti = ca.Opti()

        # Decision variable: speed at each racing line point
        v = opti.variable(self.n)
        eps = 1e-2

        # Simple time integration using trapezoidal rule
        v_next = ca.vertcat(v[1:], v[0])
        v_avg = 0.5 * (v + v_next)
        dt = self.ds / (v_avg + eps)
        T = ca.sum1(dt)

        # Simple energy model: power = drag force * velocity
        # E = sum(P * dt) where P = (m*a + drag)*v
        a_long = (v_next**2 - v**2) / (2.0 * ca.vertcat(self.ds[1:], self.ds[0]) + eps)
        drag = 0.5 * self.p.rho * self.p.C_d * self.p.A_front * v_avg**2
        F_long = self.p.mass * a_long + drag
        P = F_long * v_avg
        # Energy with simplified regen: just sum power (positive = consumption)
        E = ca.sum1(P * dt)

        # Objective: weighted sum of lap time and energy
        J = w_time * T + w_energy * E
        opti.minimize(J)

        # Speed limits
        opti.subject_to(v >= self.v_min)
        opti.subject_to(v <= self.v_max)

        # Lateral grip limits: v^2 * curvature <= mu * g
        max_v2 = self.mu * self.g / (self.curvature + eps)
        opti.subject_to(v**2 <= ca.DM(max_v2))

        # Longitudinal acceleration / braking constraints
        ds_cas = ca.DM(self.ds)
        v_sq = v**2
        opti.subject_to(v_sq[1:] - v_sq[:-1] <= 2.0 * self.a_max * ds_cas[:-1])
        opti.subject_to(v_sq[:-1] - v_sq[1:] <= 2.0 * self.a_brake * ds_cas[:-1])
        # Wrap-around for closed loop
        opti.subject_to(v_sq[0] - v_sq[-1] <= 2.0 * self.a_brake * ds_cas[-1])
        opti.subject_to(v_sq[-1] - v_sq[0] <= 2.0 * self.a_max * ds_cas[-1])

        # Initial guess: curvature-limited profile with smoothing
        if initial_guess is None:
            v0 = np.sqrt(np.maximum(self.mu * self.g / (self.curvature + 1e-6), 0.0))
            v0 = np.clip(v0, self.v_min, self.v_max)
        else:
            v0 = np.asarray(initial_guess, dtype=float)
            if v0.shape != (self.n,):
                raise ValueError("initial_guess must have shape (N,)")
            v0 = np.clip(v0, self.v_min, self.v_max)

        v0 = self._smooth_initial_profile(v0)
        opti.set_initial(v, v0)

        # Solver configuration with relaxed tolerances
        opti.solver("ipopt", {
            'ipopt.print_level': 0,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-1,
            'ipopt.max_iter': 3000,
            'ipopt.acceptable_iter': 20,
            'ipopt.mu_strategy': 'adaptive',
        })

        solution = opti.solve()
        v_opt = np.asarray(solution.value(v)).flatten()
        
        # Interpolate back to full resolution
        n_full = len(self.racing_line_full)
        downsample_indices = np.asarray(self.downsample_indices, dtype=float)
        v_full = np.interp(np.arange(n_full), downsample_indices, v_opt, period=n_full)
        v_full = np.clip(v_full, self.v_min, self.v_max)
        
        return v_full

    def compute_metrics(self, v_profile):
        """Compute lap time and energy metrics for a given speed profile."""
        v = np.asarray(v_profile, dtype=float)
        eps = 1e-2
        
        # Ensure full resolution
        if len(v) != len(self.racing_line_full):
            v = np.interp(np.arange(len(self.racing_line_full)), 
                          np.linspace(0, len(self.racing_line_full)-1, len(v)), v)
        
        # Compute segment lengths and times
        ds = np.linalg.norm(
            np.diff(self.racing_line_full, axis=0, append=self.racing_line_full[0:1]), axis=1)
        
        v_next = np.roll(v, -1)
        v_avg = 0.5 * (v + v_next)
        dt = ds / (v_avg + eps)
        lap_time = np.sum(dt)
        
        # Energy: P = (m*a + drag)*v
        a_long = (v_next**2 - v**2) / (2.0 * ds + eps)
        drag = 0.5 * self.p.rho * self.p.C_d * self.p.A_front * v_avg**2
        F_long = self.p.mass * a_long + drag
        P = F_long * v_avg
        energy = np.sum(P * dt)
        
        return {'lap_time_s': float(lap_time), 'energy_J': float(energy)}

    def compute_pareto_frontier(self, n_points=20):
        """Compute Pareto frontier by solving for multiple weight combinations.
        
        Returns:
            dict with 'weights', 'lap_times', 'energies', 'profiles'
        """
        import matplotlib.pyplot as plt
        
        results = {
            'w_energy': [],
            'lap_time': [],
            'energy': [],
            'profiles': [],
        }
        
        # Generate weights from pure time optimization to pure energy optimization
        for i in range(n_points):
            alpha = i / (n_points - 1)  # 0 to 1
            w_time = 1.0 - alpha
            w_energy = alpha * 1e2  # Scale energy weight for balance
            
            print(f"  [{i+1}/{n_points}] w_time={w_time:.2f}, w_energy={w_energy:.2e}")
            
            try:
                v_opt = self.optimize(w_time=w_time, w_energy=w_energy)
                metrics = self.compute_metrics(v_opt)
                
                results['w_energy'].append(w_energy)
                results['lap_time'].append(metrics['lap_time_s'])
                results['energy'].append(metrics['energy_J'])
                results['profiles'].append(v_opt)
            except Exception as e:
                print(f"    Failed: {e}")
        
        return results

    def plot_pareto_frontier(self, frontier_results, output_path=None):
        """Plot the Pareto frontier."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(frontier_results['lap_time'], frontier_results['energy'], 
                'o-', linewidth=2, markersize=8, color='#ff3333')
        ax.set_xlabel('Lap Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy (J)', fontsize=12, fontweight='bold')
        ax.set_title('Pareto Frontier: Time vs Energy Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150)
            print(f"Saved: {output_path}")
        else:
            plt.show()
        plt.close(fig)
