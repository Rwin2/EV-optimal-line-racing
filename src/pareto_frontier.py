"""
Compute and visualize Pareto frontier for speed profile optimization.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from track import get_track
from optimizer import SpeedProfileOptimizer


def main():
    print("=" * 70)
    print("  Pareto Frontier: Time vs Energy Trade-off")
    print("=" * 70)
    
    # Load track
    print("\n[1/3] Loading track...")
    trk = get_track('complex')
    print(f"  {trk.name} | {len(trk.centerline)} points | {trk.length:.0f} m")
    
    # Create optimizer
    print("\n[2/3] Computing Pareto frontier (10 points / safer weight range)...")
    opt = SpeedProfileOptimizer(trk.centerline, a_max=4.0, a_brake=8.0, n_points=40)
    frontier = opt.compute_pareto_frontier(
        n_points=10,
        w_energy_min=1e-4,
        w_energy_max=1e-1,
    )
    
    # Print results table
    print("\n[3/3] Pareto Frontier Results:")
    print(f"  {'w_energy':<14} {'Lap Time (s)':<16} {'Energy (J)':<16} {'Trade-off':<12}")
    print(f"  {'─'*60}")
    
    for i, (we, t, e) in enumerate(zip(frontier['w_energy'], frontier['lap_time'], frontier['energy'])):
        # Compute trade-off improvement: % change from extremes
        if i == 0:
            t_ref, e_ref = t, e
        pct_t = 100 * (t - t_ref) / t_ref if t_ref > 0 else 0
        pct_e = 100 * (e - e_ref) / e_ref if e_ref > 0 else 0
        trade = f"+{pct_t:.1f}% / {pct_e:.1f}%"
        print(f"  {we:<14.2e} {t:<16.2f} {e:<16.0f} {trade:<12}")
    
    # Plot Pareto frontier
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    
    # Main Pareto plot
    ax1.plot(frontier['lap_time'], frontier['energy'], 
            'o-', linewidth=2.5, markersize=8, color='#ff3333', 
            label='Pareto Frontier', zorder=3)
    ax1.scatter([frontier['lap_time'][0]], [frontier['energy'][0]], 
               s=200, color='#00aa00', marker='*', zorder=4, label='Max Speed (Min Time)')
    ax1.scatter([frontier['lap_time'][-1]], [frontier['energy'][-1]], 
               s=200, color='#0066ff', marker='s', zorder=4, label='Min Energy')
    
    ax1.set_xlabel('Lap Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy Consumption (J)', fontsize=12, fontweight='bold')
    ax1.set_title('Pareto Frontier: Time vs Energy Trade-off', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    
    # Efficiency frontier: Energy/Time ratio
    efficiency = np.array(frontier['energy']) / np.array(frontier['lap_time'])
    ax2.plot(frontier['w_energy'], efficiency, 'o-', linewidth=2.5, markersize=8, 
            color='#9933ff', label='Energy Efficiency')
    ax2.set_xlabel('Optimization Weight: w_energy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Energy / Time (J/s)', fontsize=12, fontweight='bold')
    ax2.set_title('Energy Efficiency along Frontier', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    
    fig.tight_layout()
    
    # Save Pareto figure
    proj_root = os.path.dirname(os.path.dirname(__file__))
    fig_dir = os.path.join(proj_root, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    output_path = os.path.join(fig_dir, 'pareto_frontier.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close(fig)

    # Compare with a baseline speed profile
    baseline = opt.baseline_speed_profile()
    fastest = frontier['profiles'][0]
    efficient = frontier['profiles'][-1]

    baseline_metrics = opt.compute_metrics(baseline)
    fastest_metrics = opt.compute_metrics(fastest)
    efficient_metrics = opt.compute_metrics(efficient)

    fig2, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    x = np.arange(len(baseline))
    ax.plot(x, baseline, label='Baseline curvature-limited', color='#888888', linewidth=2)
    ax.plot(x, fastest, label=f'Pareto min time ({fastest_metrics["lap_time_s"]:.1f}s)', color='#d62728', linewidth=2)
    ax.plot(x, efficient, label=f'Pareto min energy ({efficient_metrics["energy_J"]:.0f}J)', color='#1f77b4', linewidth=2)
    ax.set_xlabel('Track segment index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Speed Profile Comparison: Baseline vs Pareto Optima', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    fig2.tight_layout()

    output_path2 = os.path.join(fig_dir, 'pareto_speed_profiles.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path2}")
    plt.close(fig2)

    # Summary
    print(f"\n  Summary:")
    print(f"    Time range: {frontier['lap_time'][0]:.1f}s — {frontier['lap_time'][-1]:.1f}s")
    print(f"    Energy range: {frontier['energy'][-1]:.0f}J — {frontier['energy'][0]:.0f}J")
    print(f"    Max speed compromise: {(frontier['lap_time'][-1] / frontier['lap_time'][0] - 1) * 100:.1f}% slower for")
    print(f"                          {(1 - frontier['energy'][-1] / frontier['energy'][0]) * 100:.1f}% energy savings")


if __name__ == '__main__':
    main()
