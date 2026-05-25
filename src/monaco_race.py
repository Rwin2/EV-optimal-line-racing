"""
Monaco E-Prix — 51-lap race simulation and battery sizing.

Pipeline:
  1. Battery sizing sweep  → find optimal Q_batt for 51 laps
  2. Multi-lap simulation  → simulate the full race with the optimal battery
  3. Results & figures

Usage:
    python src/monaco_race.py                    # full pipeline
    python src/monaco_race.py --sizing-only      # only battery sizing sweep
    python src/monaco_race.py --sim-only         # only race simulation (uses default Q_batt)
    python src/monaco_race.py --Q-batt 45        # override battery size
    python src/monaco_race.py --laps 33          # change number of laps
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from car import CarParams
from track import get_track
from controller import generate_racing_line
from battery_sizing import sweep_battery_size, find_optimal_Q, plot_results


# ---------------------------------------------------------------------------
# Multi-lap race simulation (lightweight — no video, energy & time tracking)
# ---------------------------------------------------------------------------

def simulate_race_multilap(track, racing_line, params, n_laps,
                            grip_fraction=0.85, noise_std=0.02):
    """
    Simulate n_laps using the point-mass lap time model (Phase 1 physics).
    Each lap gets a small random perturbation (+/- noise_std) to simulate
    driving variability and track conditions.

    Returns dict with per-lap times, SoC, energy.
    """
    from battery_sizing import compute_lap_energy_time

    T_lap_base, E_lap_base, E_drive_base, E_regen_base = compute_lap_energy_time(
        racing_line, params, grip_fraction)

    rng = np.random.default_rng(seed=42)
    lap_times = []
    lap_soc_end = []
    lap_energy_Wh = []
    soc = 1.0

    print(f"\n  Simulating {n_laps} laps (point-mass model, noise={noise_std:.0%})...")
    print(f"  Baseline: T_lap={T_lap_base:.1f}s | E_lap={E_lap_base:.0f} Wh")

    for lap in range(1, n_laps + 1):
        noise = 1.0 + rng.normal(0.0, noise_std)
        T_lap = T_lap_base * max(noise, 0.8)
        E_lap = E_lap_base * max(noise, 0.8)

        energy_fraction = E_lap / (params.Q_batt * 1000.0)
        soc_new = soc - energy_fraction
        soc_new = max(soc_new, 0.0)

        lap_times.append(T_lap)
        lap_soc_end.append(soc_new)
        lap_energy_Wh.append(E_lap)

        print(f"    Lap {lap:2d}/{n_laps} | time={T_lap:.1f}s | SoC={soc_new:.3f} | "
              f"E={E_lap:.0f} Wh")

        soc = soc_new
        if soc <= params.SOC_min:
            print(f"  !! Battery depleted at lap {lap} — race stopped.")
            break

    return {
        'lap_times_s': lap_times,
        'lap_soc_end': lap_soc_end,
        'lap_energy_Wh': lap_energy_Wh,
        'laps_completed': len(lap_times),
        'total_time_s': sum(lap_times),
        'total_time_min': sum(lap_times) / 60.0,
        'final_SoC': lap_soc_end[-1] if lap_soc_end else soc,
        'total_energy_Wh': sum(lap_energy_Wh),
        'total_energy_kWh': sum(lap_energy_Wh) / 1000.0,
        'track_length_m': track.length,
    }


# ---------------------------------------------------------------------------
# Results plotting
# ---------------------------------------------------------------------------

def plot_race_results(sim_results, params, n_laps, track_name, output_path=None):
    laps_done = sim_results['laps_completed']
    laps      = np.arange(1, laps_done + 1)
    times     = np.array(sim_results['lap_times_s'])
    soc       = np.array(sim_results['lap_soc_end'])
    energy    = np.array(sim_results['lap_energy_Wh'])
    cum_time  = np.cumsum(times) / 60.0  # minutes

    finished  = laps_done >= n_laps
    status_str = f"FINISHED ({laps_done}/{n_laps} laps)" if finished \
                 else f"DNF — battery — {laps_done}/{n_laps} laps"

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"Monaco E-Prix — {track_name}\n"
        f"Q_batt = {params.Q_batt:.1f} kWh  |  mass = {params.mass:.0f} kg  |  {status_str}",
        fontsize=12, fontweight='bold')
    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ── (0,0) Lap time + cumulative time ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(laps, times, 'o-', color='#3498db', ms=4, lw=1.5, label='Lap time')
    ax.axhline(np.mean(times), color='#e74c3c', ls='--', lw=1.2,
               label=f'avg {np.mean(times):.1f} s')
    ax2 = ax.twinx()
    ax2.plot(laps, cum_time, 's--', color='#9b59b6', ms=3, lw=1, alpha=0.7,
             label='Cumulative time')
    ax2.set_ylabel('Cumulative time (min)', color='#9b59b6', fontsize=8)
    ax.set_xlabel('Lap'); ax.set_ylabel('Lap time (s)')
    ax.set_title('Lap time evolution', fontsize=10)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── (0,1) SoC over race ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    soc_pct = soc * 100
    colors_soc = ['#2ecc71' if s > 30 else '#f39c12' if s > 10 else '#e74c3c'
                  for s in soc_pct]
    ax.bar(laps, soc_pct, color=colors_soc, alpha=0.85, width=0.8)
    ax.plot(laps, soc_pct, 'k-', lw=1.0, alpha=0.5)
    ax.axhline(5.0, color='#e74c3c', ls='--', lw=1.2, label='SOC_min = 5%')
    ax.set_xlabel('Lap'); ax.set_ylabel('State of Charge (%)')
    ax.set_title('Battery SoC over race', fontsize=10)
    ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # ── (0,2) Cumulative energy consumed ─────────────────────────────────
    cum_energy_kWh = np.cumsum(energy) / 1000.0
    ax = fig.add_subplot(gs[0, 2])
    ax.fill_between(laps, cum_energy_kWh, alpha=0.3, color='#e67e22')
    ax.plot(laps, cum_energy_kWh, 'o-', color='#e67e22', ms=3, lw=1.5,
            label='Energy used')
    # Show battery limit
    E_avail = params.Q_batt * (1 - params.SOC_min)
    ax.axhline(E_avail, color='#e74c3c', ls='--', lw=1.2,
               label=f'E available = {E_avail:.1f} kWh')
    # Project to n_laps if race unfinished
    if not finished and laps_done > 1:
        avg_e = np.mean(energy) / 1000.0
        laps_proj = np.arange(laps_done, n_laps + 1)
        e_proj = cum_energy_kWh[-1] + avg_e * (laps_proj - laps_done)
        ax.plot(laps_proj, e_proj, '--', color='#e67e22', alpha=0.4, lw=1,
                label='Projected (avg)')
    ax.set_xlabel('Lap'); ax.set_ylabel('Cumulative energy (kWh)')
    ax.set_title('Cumulative energy consumption', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── (1,0) Energy per lap ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(laps, energy, color='#e67e22', alpha=0.8, width=0.8)
    ax.axhline(np.mean(energy), color='#9b59b6', ls='--', lw=1.2,
               label=f'avg {np.mean(energy):.0f} Wh/lap')
    ax.set_xlabel('Lap'); ax.set_ylabel('Energy (Wh)')
    ax.set_title('Energy per lap', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # ── (1,1) Race pace histogram ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(times, bins=min(15, laps_done // 2 + 1), color='#3498db', alpha=0.75,
            edgecolor='white', lw=0.5)
    ax.axvline(np.mean(times), color='#e74c3c', ls='--', lw=1.2,
               label=f'mean = {np.mean(times):.1f} s')
    ax.axvline(np.median(times), color='#f39c12', ls=':', lw=1.2,
               label=f'median = {np.median(times):.1f} s')
    ax.set_xlabel('Lap time (s)'); ax.set_ylabel('Count')
    ax.set_title('Lap time distribution', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # ── (1,2) Summary text box ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    total_dist_km = laps_done * (sim_results.get('track_length_m', 1288)) / 1000.0
    summary = (
        f"RACE SUMMARY\n"
        f"{'─'*28}\n"
        f"Laps completed  {laps_done:>6} / {n_laps}\n"
        f"Total time      {sim_results['total_time_min']:>8.2f} min\n"
        f"Avg lap time    {np.mean(times):>8.2f} s\n"
        f"Best lap        {np.min(times):>8.2f} s\n"
        f"Worst lap       {np.max(times):>8.2f} s\n"
        f"{'─'*28}\n"
        f"Final SoC       {sim_results['final_SoC']*100:>7.1f} %\n"
        f"Energy used     {sim_results['total_energy_kWh']:>6.2f} kWh\n"
        f"                {sim_results['total_energy_kWh']/params.Q_batt*100:>6.1f} % of batt\n"
        f"Avg Wh/lap      {np.mean(energy):>8.0f} Wh\n"
        f"{'─'*28}\n"
        f"Q_batt          {params.Q_batt:>6.1f} kWh\n"
        f"Car mass        {params.mass:>6.0f} kg\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Monaco E-Prix — 51-lap race simulation')
    parser.add_argument('--track', default='complex', help='Track name')
    parser.add_argument('--laps', type=int, default=0,
                        help='Number of race laps (0 = auto: target 65 km)')
    parser.add_argument('--Q-batt', type=float, default=None, dest='Q_batt',
                        help='Override battery size (kWh). If not set, optimized automatically.')
    parser.add_argument('--Q-min', type=float, default=15.0, dest='Q_min')
    parser.add_argument('--Q-max', type=float, default=80.0, dest='Q_max')
    parser.add_argument('--sizing-only', action='store_true',
                        help='Only run battery sizing sweep, skip simulation')
    parser.add_argument('--sim-only', action='store_true',
                        help='Only run race simulation (use --Q-batt or default 40 kWh)')
    parser.add_argument('--dt', type=float, default=0.05, help='Simulation time step (s)')
    parser.add_argument('--line', default='min_laptime',
                        choices=['min_laptime', 'center'], help='Racing line')
    args = parser.parse_args()

    proj_root = Path(__file__).parent.parent

    print("=" * 65)
    print("  EV Race Simulation — Optimal Line Racing")
    print("=" * 65)

    # Load track first so we can auto-compute laps
    print(f"\n[Track] Loading '{args.track}'...")
    track = get_track(args.track)
    racing_line, v_profile = generate_racing_line(track, mode=args.line)

    if args.laps == 0:
        args.laps = max(20, round(65_000 / track.length))
        print(f"  Auto laps: {args.laps} ({args.laps * track.length / 1000:.1f} km)")

    fig_dir = proj_root / 'figures' / args.track
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Track : {args.track}  |  Laps : {args.laps}")
    print(f"  Length : {track.length:.0f} m/lap")
    print(f"  Total  : {args.laps * track.length / 1000:.1f} km over {args.laps} laps")

    # ── Step 1: Battery sizing ───────────────────────────────────────────
    Q_optimal = args.Q_batt  # may be None → auto

    if not args.sim_only:
        print(f"\n[1/2] Battery sizing sweep ({args.Q_min}–{args.Q_max} kWh)...")
        sizing_results = sweep_battery_size(
            racing_line, n_laps=args.laps,
            Q_min=args.Q_min, Q_max=args.Q_max, n_pts=40)

        Q_min_r, Q_opt = find_optimal_Q(sizing_results)

        print("\n" + "─" * 65)
        if Q_opt is None:
            print("  No feasible Q_batt found. Increase --Q-max.")
            if Q_optimal is None:
                Q_optimal = 40.0
                print(f"  Defaulting to Q_batt = {Q_optimal} kWh.")
        else:
            print(f"  Minimum feasible Q_batt : {Q_min_r['Q_batt_kWh']:.1f} kWh")
            print(f"  Mass at Q*              : {Q_min_r['mass_kg']:.0f} kg")
            print(f"  Predicted lap time      : {Q_min_r['T_lap_s']:.1f} s")
            print(f"  Predicted race time     : {Q_min_r['T_total_min']:.1f} min")
            if Q_optimal is None:
                Q_optimal = round(Q_min_r['Q_batt_kWh'], 1)
                print(f"  Selected Q_batt (optimal)      : {Q_optimal:.1f} kWh")

        plot_results(
            sizing_results, args.laps, track.name,
            output_path=str(fig_dir / f'battery_sizing_{args.laps}laps.png'))

        if args.sizing_only:
            print("\nDone (sizing only).")
            return

    if Q_optimal is None:
        Q_optimal = 40.0

    # ── Step 2: Race simulation ──────────────────────────────────────────
    print(f"\n[2/2] Race simulation with Q_batt = {Q_optimal:.1f} kWh...")
    params = CarParams(Q_batt=Q_optimal)
    print(f"  Car mass : {params.mass:.0f} kg")

    sim = simulate_race_multilap(
        track, racing_line, params, n_laps=args.laps)

    # Summary
    print("\n" + "=" * 65)
    print("  RACE SUMMARY")
    print("─" * 65)
    print(f"  Laps completed   : {sim['laps_completed']} / {args.laps}")
    print(f"  Total race time  : {sim['total_time_min']:.2f} min  ({sim['total_time_s']:.1f} s)")
    print(f"  Avg lap time     : {np.mean(sim['lap_times_s']):.2f} s")
    print(f"  Best lap time    : {np.min(sim['lap_times_s']):.2f} s")
    print(f"  Final SoC        : {sim['final_SoC']:.3f}  ({sim['final_SoC']*100:.1f}%)")
    print(f"  Total energy     : {sim['total_energy_kWh']:.2f} kWh "
          f"({sim['total_energy_kWh']/Q_optimal*100:.1f}% of battery)")
    print("=" * 65)

    # Save results JSON
    out_json = fig_dir / f'race_{args.laps}laps.json'
    with open(out_json, 'w') as f:
        json.dump({
            'track': args.track,
            'n_laps': args.laps,
            'Q_batt_kWh': Q_optimal,
            'mass_kg': params.mass,
            **sim,
        }, f, indent=2)
    print(f"\n  Results saved: {out_json.name}")

    # Plot
    plot_race_results(
        sim, params, args.laps, track.name,
        output_path=str(fig_dir / f'race_{args.laps}laps.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()