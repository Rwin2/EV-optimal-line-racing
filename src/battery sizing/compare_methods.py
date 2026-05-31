"""
Pareto method comparison: grip_fraction proxy vs joint SCP.

Runs both methods on a single circuit and produces a 2-panel figure showing:
  Left : T vs E Pareto fronts overlaid (grip in red, SCP in blue)
  Right: T*E product curves (shows where the KKT optimum lives)

Usage:
    python src/compare_methods.py --track complex --Q-batt 36.9
    python src/compare_methods.py --track monza   --Q-batt 26.0
    python src/compare_methods.py --track hairpin --Q-batt 26.0
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

_this = os.path.dirname(os.path.abspath(__file__))
_sl   = os.path.join(os.path.dirname(_this), 'single lap')
sys.path.insert(0, _sl)
sys.path.insert(0, _this)

from car import CarParams
from track import get_track
from controller import generate_racing_line
from race_strategy import (
    compute_scp_pareto, compute_strategy_pareto, analytical_optimal
)


def run_comparison(track_name, Q_kWh, n_pts_grip=30, n_pts_scp=20,
                   w_max=8.0, T_max_pct=None, output_path=None):
    track  = get_track(track_name)
    n_laps = max(20, round(65_000 / track.length))
    params = CarParams(Q_batt=Q_kWh)

    print(f"Track: {track_name}  |  {n_laps} laps  |  Q={Q_kWh} kWh  |  mass={params.mass:.0f} kg")

    # ── Grip proxy Pareto ─────────────────────────────────────────────────────
    print("\n[1/2] Grip proxy Pareto...")
    racing_line, _ = generate_racing_line(track, mode='min_laptime')
    g_arr, T_grip, E_grip = compute_strategy_pareto(
        racing_line, params, g_min=0.50, g_max=0.90, n_pts=n_pts_grip)

    # ── Joint SCP Pareto ──────────────────────────────────────────────────────
    print("\n[2/2] Joint SCP Pareto...")
    param_arr, T_scp, E_scp = compute_scp_pareto(
        track, params, n_pts=n_pts_scp, w_max=w_max)

    # ── Analytical optima ─────────────────────────────────────────────────────
    T_max = None
    if T_max_pct is not None:
        T_max_grip = T_grip.min() * (1.0 + T_max_pct / 100.0)
        T_max_scp  = T_scp.min()  * (1.0 + T_max_pct / 100.0)
    else:
        T_max_grip = T_max_scp = None

    ana_grip = analytical_optimal(g_arr,    T_grip, E_grip, Q_kWh, n_laps, T_max=T_max_grip)
    ana_scp  = analytical_optimal(param_arr, T_scp,  E_scp,  Q_kWh, n_laps, T_max=T_max_scp)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Pareto Method Comparison  |  {track.name}  |  {n_laps} laps  |  Q={Q_kWh:.1f} kWh"
        + (f"  |  T_max=+{T_max_pct:.0f}%" if T_max_pct else ""),
        fontsize=12, fontweight='bold'
    )

    C_grip = '#e74c3c'
    C_scp  = '#2980b9'
    C_opt  = '#f39c12'

    # Panel 0: T vs E Pareto overlay
    ax = axes[0]
    ax.plot(T_grip, E_grip, 'o-', color=C_grip, ms=5, lw=1.5, label='Grip proxy')
    ax.plot(T_scp,  E_scp,  's-', color=C_scp,  ms=5, lw=1.5, label='Joint SCP')
    ax.scatter(ana_grip['T_star'], ana_grip['E_star'], s=140, marker='D',
               color=C_grip, zorder=5, edgecolors='k', lw=0.5,
               label=f'Grip opt* (g*={ana_grip["g_star"]:.3f})')
    ax.scatter(ana_scp['T_star'],  ana_scp['E_star'],  s=140, marker='D',
               color=C_scp,  zorder=5, edgecolors='k', lw=0.5,
               label=f'SCP opt* (idx*={ana_scp["g_star"]:.3f})')
    if T_max_pct:
        ax.axvline(T_max_grip, color=C_grip, ls=':', lw=1.2,
                   label=f'T_max grip ({T_max_grip:.1f}s)')
        ax.axvline(T_max_scp, color=C_scp, ls=':', lw=1.2,
                   label=f'T_max scp ({T_max_scp:.1f}s)')
    ax.set_xlabel('Lap time T  [s]')
    ax.set_ylabel('Energy E  [Wh/lap]')
    ax.set_title('Pareto fronts: T vs E', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, ls='--')

    # Panel 1: T*E product
    ax = axes[1]
    ax.plot(g_arr,    T_grip * E_grip, 'o-', color=C_grip, ms=5, lw=1.5, label='Grip proxy')
    ax.plot(param_arr, T_scp  * E_scp,  's-', color=C_scp,  ms=5, lw=1.5, label='Joint SCP')
    ax.axvline(ana_grip['g_star'], color=C_grip, ls='--', lw=1.5,
               label=f'Grip g*={ana_grip["g_star"]:.3f}')
    ax.axvline(ana_scp['g_star'],  color=C_scp,  ls='--', lw=1.5,
               label=f'SCP idx*={ana_scp["g_star"]:.3f}')
    ax.scatter(ana_grip['g_star'], ana_grip['TE_min'], s=120, color=C_grip, zorder=5)
    ax.scatter(ana_scp['g_star'],  ana_scp['TE_min'],  s=120, color=C_scp,  zorder=5)
    ax.set_xlabel('Pareto parameter (0=fastest, 1=efficient)')
    ax.set_ylabel('T x E  [s*Wh]')
    ax.set_title('KKT product minimiser', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, ls='--')

    # Summary text
    T_spread_grip = 100 * (T_grip.max() - T_grip.min()) / T_grip.min()
    T_spread_scp  = 100 * (T_scp.max()  - T_scp.min())  / T_scp.min()
    E_spread_grip = 100 * (E_grip.max() - E_grip.min()) / E_grip.min()
    E_spread_scp  = 100 * (E_scp.max()  - E_scp.min())  / E_scp.min()
    txt = (
        f"Grip: T spread={T_spread_grip:.0f}%  E spread={E_spread_grip:.0f}%\n"
        f"SCP:  T spread={T_spread_scp:.0f}%   E spread={E_spread_scp:.0f}%\n"
        f"\nGrip p*={ana_grip['p_star']:.3f}  feasible={ana_grip['feasible']}\n"
        f"SCP  p*={ana_scp['p_star']:.3f}  feasible={ana_scp['feasible']}"
    )
    fig.text(0.5, 0.02, txt, ha='center', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.9))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)

    return g_arr, T_grip, E_grip, param_arr, T_scp, E_scp, ana_grip, ana_scp


def main():
    parser = argparse.ArgumentParser(description='Compare grip vs SCP Pareto methods')
    parser.add_argument('--track',   default='complex')
    parser.add_argument('--Q-batt',  type=float, default=36.9, dest='Q_kWh')
    parser.add_argument('--T-max-pct', type=float, default=None, dest='T_max_pct')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    from pathlib import Path
    proj_root = Path(__file__).parent.parent.parent
    fig_dir   = proj_root / 'figures' / args.track
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'pareto_comparison.png'
    out = str(fig_dir / suffix) if not args.no_plot else None

    run_comparison(args.track, args.Q_kWh, T_max_pct=args.T_max_pct, output_path=out)
    print("\nDone.")


if __name__ == '__main__':
    main()
