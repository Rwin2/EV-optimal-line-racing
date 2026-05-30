"""
3-car race simulation with optional per-step grip perturbation.

Cars:
  1. Baseline  — follows centerline, curvature-based speed
  2. Nominal   — follows optimized raceline + speed profile (deterministic μ)
  3. Robust    — follows robust raceline + speed profile (SAA, uncertain μ)

All cars use iLQR trajectory tracking.

Outputs: GIF + race metrics.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))

from track import get_track
from car import BicycleModel, CarState, CarParams
from controller import ILQRController
from optimizer import (alpha_to_raceline, compute_curvature_from_path,
                       compute_velocity_profile)
from simulator import simulate_race, render_video
from scipy.interpolate import interp1d

TRACK = "monaco"
N_STATIONS = 40
MU_SIGMA = 0.10
DT = 0.02
MAX_TIME = 90.0
FPS = 15
SEED = 123

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJ_ROOT, "figures")
CACHE_DIR = os.path.join(PROJ_ROOT, "cache")


def interpolate_to_full(alpha_sub, v_sub, trk, n_sub):
    """Interpolate subsampled (α, v) back to the full-resolution track."""
    n_full = len(trk.centerline)
    t_sub = np.linspace(0, 1, n_sub)
    t_full = np.linspace(0, 1, n_full)

    alpha_full = interp1d(t_sub, alpha_sub, kind='cubic',
                          fill_value='extrapolate')(t_full)
    alpha_full = np.clip(alpha_full, -trk.widths, trk.widths)

    v_full = interp1d(t_sub, v_sub, kind='cubic',
                      fill_value='extrapolate')(t_full)
    v_full = np.clip(v_full, 1.0, 70.0)

    raceline = alpha_to_raceline(alpha_full, trk.centerline, trk.normals)
    return raceline, v_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-perturb', action='store_true',
                        help='Run with constant mu (no stochastic perturbation)')
    args = parser.parse_args()
    perturb = not args.no_perturb

    mode_str = "Grip Perturbation" if perturb else "Deterministic (no perturbation)"
    print("=" * 60)
    print(f"  3-Car Race Simulation — {mode_str}")
    print("=" * 60)

    # Load optimization results
    cache_path = os.path.join(CACHE_DIR, 'optimization_results.npz')
    if not os.path.exists(cache_path):
        print("ERROR: Run run_analysis.py first to generate optimization_results.npz")
        return

    data = np.load(cache_path)
    alpha_nom = data['alpha_nom']
    v_nom = data['v_nom']
    alpha_rob = data['alpha_rob']
    v_rob = data['v_rob']

    # Track
    trk = get_track(TRACK)
    print(f"Track: {trk.name} | {trk.length:.0f}m\n")

    # Interpolate to full resolution
    print("[1] Interpolating optimized solutions to full resolution...")
    rl_nom, v_nom_full = interpolate_to_full(alpha_nom, v_nom, trk, N_STATIONS)
    rl_rob, v_rob_full = interpolate_to_full(alpha_rob, v_rob, trk, N_STATIONS)
    rl_center = trk.centerline.copy()

    # Baseline speed profile: conservative curvature-based
    kappa_center = compute_curvature_from_path(rl_center)
    v_center = compute_velocity_profile(rl_center, kappa_center, CarParams())
    # Scale down for conservative baseline
    v_center = np.clip(v_center * 0.75, 5.0, CarParams().v_max * 0.85)

    print(f"    Nominal raceline: {len(rl_nom)} pts")
    print(f"    Robust raceline:  {len(rl_rob)} pts")
    print(f"    Baseline (center): {len(rl_center)} pts")

    # Controllers — all iLQR
    print("\n[2] Setting up iLQR controllers...")
    params = CarParams()

    print("    Computing iLQR gains for Baseline...")
    ctrl_baseline = ILQRController(
        trk, racing_line=rl_center, v_profile=v_center, params=params, dt=DT)

    print("    Computing iLQR gains for Nominal...")
    ctrl_nominal = ILQRController(
        trk, racing_line=rl_nom, v_profile=v_nom_full, params=params, dt=DT)

    print("    Computing iLQR gains for Robust...")
    ctrl_robust = ILQRController(
        trk, racing_line=rl_rob, v_profile=v_rob_full, params=params, dt=DT)

    controllers = [ctrl_baseline, ctrl_nominal, ctrl_robust]
    car_names = ['Baseline', 'Nominal', 'Robust (SAA)']
    car_colors = ['#3399ff', '#ff3333', '#33cc66']

    # Car models
    if perturb:
        rng = np.random.default_rng(SEED)
        max_steps = int(MAX_TIME / DT)
        mu_sequence = rng.normal(params.mu, MU_SIGMA, size=max_steps)
        mu_sequence = np.clip(mu_sequence, 0.3, 1.5)

        class SharedMuBicycleModel(BicycleModel):
            def __init__(self, params, mu_seq):
                super().__init__(CarParams(**{f.name: getattr(params, f.name)
                                 for f in params.__dataclass_fields__.values()}))
                self.mu_seq = mu_seq
                self._step_count = 0

            def step(self, state, delta, F_drive, dt=0.01):
                idx = min(self._step_count, len(self.mu_seq) - 1)
                self.p.mu = self.mu_seq[idx]
                self._step_count += 1
                return super().step(state, delta, F_drive, dt)

        car_models = [SharedMuBicycleModel(params, mu_sequence) for _ in range(3)]
    else:
        car_models = [
            BicycleModel(CarParams(**{f.name: getattr(params, f.name)
                         for f in params.__dataclass_fields__.values()}))
            for _ in range(3)
        ]

    # Initial states: start each car on its reference trajectory
    initial_states = []
    for i, ctrl in enumerate(controllers):
        s0 = ctrl.s_bar[0]
        initial_states.append(CarState(
            x=s0[0], y=s0[1], psi=s0[2], vx=s0[3], SOC=1.0))

    # Simulate
    mu_info = f"μ perturbed per step, σ={MU_SIGMA}" if perturb else "μ=1.0 (deterministic)"
    print(f"\n[3] Simulating ({MAX_TIME}s, dt={DT}, {mu_info})...")
    sim_data = simulate_race(trk, controllers, car_models, initial_states,
                              dt=DT, max_time=MAX_TIME, stop_after_laps=1)
    sim_time = sim_data['n_steps'] * DT
    print(f"    {sim_data['n_steps']} steps, {sim_time:.1f}s simulated")

    # Results
    print("\n[4] Results:")
    print(f"  {'Car':<16} {'Dist (m)':>9} {'Avg (km/h)':>11} {'Max (km/h)':>11} {'E used (%)':>11} {'Laps':>5}")
    print(f"  {'─'*65}")
    for i in range(3):
        m = sim_data['metrics'][i]
        print(f"  {car_names[i]:<16} {m['distance_m']:>8.0f} {m['avg_speed_kmh']:>10.1f} "
              f"{m['max_speed_kmh']:>10.1f} {m['energy_used_pct']:>10.2f} {m['laps_completed']:>5d}")
        if m['lap_times_s']:
            print(f"  {'':16} Lap times: {', '.join(f'{t:.1f}s' for t in m['lap_times_s'])}")

    # Render GIF
    print("\n[5] Rendering GIF...")
    racing_lines_display = [
        (rl_center, car_colors[0], car_names[0]),
        (rl_nom, car_colors[1], car_names[1]),
        (rl_rob, car_colors[2], car_names[2]),
    ]

    os.makedirs(FIG_DIR, exist_ok=True)
    suffix = '_deterministic' if not perturb else ''
    gif_path = os.path.join(FIG_DIR, f'race_simulation{suffix}.gif')
    render_video(trk, sim_data, car_names, car_colors,
                 output_path=gif_path, racing_lines=racing_lines_display,
                 fps=FPS, fmt='gif')

    print(f"\nDone! GIF saved to: {gif_path}")


if __name__ == '__main__':
    main()
