"""
Battery sizing optimization for multi-lap EV racing.

Uses a fast point-mass lap time simulation (no CasADi) to sweep over battery
capacity values and find the optimal Q_batt for a given race (number of laps).

Physics: same model as optimizer_ipopt.py but in numpy — suitable for sweeps.

Usage:
    python src/battery_sizing.py                        # default: 51 laps, complex track
    python src/battery_sizing.py --laps 33 --track complex
    python src/battery_sizing.py --Q-min 20 --Q-max 80 --n 40
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_this = os.path.dirname(os.path.abspath(__file__))
_sl   = os.path.join(os.path.dirname(_this), 'single lap')
sys.path.insert(0, _sl)
sys.path.insert(0, _this)

from car import CarParams
from track import get_track
from controller import generate_racing_line
from optimizer_ipopt import _compute_curvature, _arc_lengths


# ---------------------------------------------------------------------------
# Phase 1 — Motor efficiency map
# ---------------------------------------------------------------------------

def _eta_motor_map(P_norm):
    """
    Parabolic motor efficiency map η(P/P_max).

    Peak efficiency η_peak = 0.95 at 60 % of rated power (typical PMSM).
    Drops to η_min = 0.80 at no-load and near max power.

    Calibration: parabola y = η_peak - k*(x - x_opt)^2
    with k chosen so that η(0) = η_min.
    """
    eta_peak = 0.95
    eta_min  = 0.80
    x_opt    = 0.60
    k = (eta_peak - eta_min) / x_opt ** 2
    return np.clip(eta_peak - k * (np.asarray(P_norm) - x_opt) ** 2,
                   eta_min, eta_peak)


# ---------------------------------------------------------------------------
# Point-mass lap time simulation
# ---------------------------------------------------------------------------

def _speed_profile(ds, kappa, params, grip_fraction=0.85, g=9.81, n_iters=4):
    """
    Forward-backward pass speed profile (classic point-mass LTS).

    All acceleration/braking limits are derived from CarParams — no external
    tuning constants.

    Limits applied:
    - Lateral grip:   v² ≤ grip_fraction * μ * g / κ
    - Acceleration:   a ≤ min(F_drive_max/mass, P_max/(mass*v))  [force + power]
    - Braking:        a ≤ min(F_brake_max/mass, μ*g)             [force + adhesion]
    """
    n = len(ds)
    eps_kappa = 1e-4
    eps_v     = 0.5

    # Braking limit: force-limited or adhesion-limited (whichever is lower)
    a_brake = min(params.F_brake_max / params.mass, params.mu * g)

    # Lateral grip ceiling
    v_lat = np.sqrt(grip_fraction * params.mu * g / (kappa + eps_kappa))
    v = np.minimum(v_lat, params.v_max)

    for _ in range(n_iters):
        # Forward pass: traction force limit AND power limit (speed-dependent)
        for i in range(n):
            j = (i + 1) % n
            a_force = params.F_drive_max / params.mass
            a_power = params.P_max / (params.mass * max(v[i], eps_v))
            a_eff   = min(a_force, a_power)
            v_fwd   = np.sqrt(v[i] ** 2 + 2.0 * a_eff * ds[i])
            v[j]    = min(v[j], v_fwd)

        # Backward pass: braking limit
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v_bwd  = np.sqrt(v[j] ** 2 + 2.0 * a_brake * ds[i])
            v[i]   = min(v[i], v_bwd)

    return v


def compute_lap_energy_time(racing_line, params, grip_fraction=0.85):
    """
    Fast lap time and energy computation for a fixed racing line and car params.

    Phase 1 improvements vs intermediate model:
    - a_max and a_brake derived from F_drive_max, F_brake_max, P_max, mass
      → lap time now increases with battery mass
    - Motor efficiency η(P) from parabolic map instead of constant 0.92
      → heavier cars that draw more power pay a higher efficiency penalty

    Returns
    -------
    T_lap      : float — lap time (s)
    E_net_Wh   : float — net electrical energy consumed per lap (Wh)
    E_drive_Wh : float — gross energy drawn from battery (Wh)
    E_regen_Wh : float — energy recovered through regen braking (Wh)
    """
    kappa = _compute_curvature(racing_line)
    ds    = _arc_lengths(racing_line)
    v     = _speed_profile(ds, kappa, params, grip_fraction)

    v_next = np.roll(v, -1)
    v_avg  = np.maximum(0.5 * (v + v_next), 1e-3)
    dt     = ds / v_avg
    T_lap  = dt.sum()

    # Longitudinal force balance
    a_long = (v_next ** 2 - v ** 2) / (2.0 * ds + 1e-6)
    F_drag = 0.5 * params.rho * params.C_d * params.A_front * v_avg ** 2
    F_roll = params.C_roll * params.mass * 9.81
    F_long = params.mass * a_long + F_drag + F_roll

    P_mech  = F_long * v_avg
    P_drive = np.maximum(P_mech, 0.0)
    P_regen = np.minimum(P_mech, 0.0)

    # Motor efficiency map: η varies with normalised power level
    eta_drive = _eta_motor_map(P_drive / params.P_max)

    E_drive_Wh = np.sum((P_drive / eta_drive) * dt) / 3600.0
    E_regen_Wh = np.sum((-P_regen * params.eta_regen) * dt) / 3600.0
    E_net_Wh   = E_drive_Wh - E_regen_Wh

    return T_lap, E_net_Wh, E_drive_Wh, E_regen_Wh


# ---------------------------------------------------------------------------
# Battery sizing sweep
# ---------------------------------------------------------------------------

def sweep_battery_size(racing_line, n_laps=51,
                        Q_min=20.0, Q_max=80.0, n_pts=40, grip_fraction=0.85):
    """
    Sweep battery capacity Q_batt and compute feasibility + lap time.

    For each Q_batt:
    - mass = m_chassis + Q_batt / e_spec  (heavier battery → slower lap)
    - a_max, a_brake derived from CarParams (F_drive_max, F_brake_max, P_max)
    - E_available = Q_batt * (1 - SOC_min) * 1000  Wh
    - Feasible iff E_available >= n_laps * E_lap

    Returns list of result dicts.
    """
    Q_values = np.linspace(Q_min, Q_max, n_pts)
    results = []

    print(f"\n{'Q_batt':>8} {'mass':>6} {'T_lap':>7} {'E_lap':>8} "
          f"{'E_avail':>8} {'SoC_end':>8} {'status':>10}")
    print("─" * 65)

    for Q in Q_values:
        params = CarParams(Q_batt=Q)
        T_lap, E_lap, E_drive, E_regen = compute_lap_energy_time(
            racing_line, params, grip_fraction)

        E_total = n_laps * E_lap
        E_available = Q * 1000.0 * (1.0 - params.SOC_min)
        SoC_final = 1.0 - E_total / (Q * 1000.0)
        SoC_final = max(SoC_final, 0.0)
        feasible = E_available >= E_total
        T_total = n_laps * T_lap
        m_batt = Q / params.e_spec  # battery mass (kg)

        results.append({
            'Q_batt_kWh': Q,
            'mass_kg': params.mass,
            'm_batt_kg': m_batt,
            'T_lap_s': T_lap,
            'E_lap_Wh': E_lap,
            'E_drive_Wh': E_drive,
            'E_regen_Wh': E_regen,
            'E_total_Wh': E_total,
            'E_available_Wh': E_available,
            'SoC_final': SoC_final,
            'feasible': feasible,
            'T_total_s': T_total,
            'T_total_min': T_total / 60.0,
        })

        status = "feasible" if feasible else "INFEASIBLE"
        print(f"  {Q:6.1f} kWh  {params.mass:5.0f} kg  {T_lap:6.1f}s  "
              f"{E_lap:7.0f} Wh  {E_available:7.0f} Wh  {SoC_final:6.3f}  {status}")

    return results


# ---------------------------------------------------------------------------
# Optimal Q_batt selection
# ---------------------------------------------------------------------------

def find_optimal_Q(results):
    """
    Among feasible configurations, find:
    - Q_min_feasible : smallest battery that completes the race
    - Q_optimal      : lightest feasible → fastest lap time (in this model,
                       lap time increases with mass, so smallest feasible = fastest)
    """
    feasible = [r for r in results if r['feasible']]
    if not feasible:
        return None, None

    Q_min_feasible = min(feasible, key=lambda r: r['Q_batt_kWh'])
    Q_optimal = min(feasible, key=lambda r: r['T_lap_s'])  # same as Q_min_feasible here

    return Q_min_feasible, Q_optimal


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, n_laps, track_name, output_path=None):
    Q          = np.array([r['Q_batt_kWh']          for r in results])
    mass       = np.array([r['mass_kg']              for r in results])
    m_batt     = np.array([r['m_batt_kg']            for r in results])
    T_total    = np.array([r['T_total_min']          for r in results])
    E_lap      = np.array([r['E_lap_Wh']             for r in results])
    E_drive    = np.array([r['E_drive_Wh']           for r in results])
    E_regen    = np.array([r['E_regen_Wh']           for r in results])
    E_total    = np.array([r['E_total_Wh']  / 1000.0 for r in results])
    E_avail    = np.array([r['E_available_Wh']/1000.0 for r in results])
    SoC_final  = np.array([r['SoC_final']            for r in results])
    feasible   = np.array([r['feasible']             for r in results])

    Q_min_feasible = Q[feasible].min() if feasible.any() else None
    C_ok  = '#2ecc71'
    C_bad = '#e74c3c'
    C_star = '#f39c12'

    def _vline(ax):
        if Q_min_feasible is not None:
            ax.axvline(Q_min_feasible, color=C_star, ls='--', lw=1.4,
                       label=f'Q* = {Q_min_feasible:.1f} kWh')

    def _scatter(ax, y):
        ax.scatter(Q[feasible],  y[feasible],  c=C_ok,  s=20, zorder=3, label='feasible')
        ax.scatter(Q[~feasible], y[~feasible], c=C_bad, s=20, zorder=3, label='infeasible')

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"Battery sizing — {track_name} — {n_laps} laps",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ── (0,0) Race time ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _scatter(ax, T_total)
    _vline(ax)
    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel('Total race time (min)')
    ax.set_title(f'Race time — {n_laps} laps', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── (0,1) Energy budget: required vs available ───────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(Q, E_total,  'o-', ms=4, color='#e67e22', label=f'E required ({n_laps} laps)')
    ax.plot(Q, E_avail,  's--', ms=4, color=C_ok,     label='E available (95% of Q)')
    if Q_min_feasible is not None:
        ax.axvline(Q_min_feasible, color=C_star, ls='--', lw=1.4, label=f'Q* = {Q_min_feasible:.1f} kWh')
    ax.fill_between(Q, E_total, E_avail,
                    where=E_avail >= E_total, alpha=0.15, color=C_ok,   label='margin')
    ax.fill_between(Q, E_total, E_avail,
                    where=E_avail < E_total,  alpha=0.15, color=C_bad,  label='deficit')
    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Energy budget', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── (0,2) SoC at race end ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    _scatter(ax, SoC_final * 100)
    _vline(ax)
    ax.axhline(5.0, color='#3498db', ls=':', lw=1.2, label='SOC_min = 5%')
    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel('Final SoC (%)')
    ax.set_title('SoC at race end', fontsize=10)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── (1,0) Energy margin ───────────────────────────────────────────────
    margin_kWh = E_avail - E_total
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(Q[feasible],   margin_kWh[feasible],
           width=(Q[1]-Q[0])*0.8, color=C_ok,  alpha=0.8, label='feasible')
    ax.bar(Q[~feasible],  margin_kWh[~feasible],
           width=(Q[1]-Q[0])*0.8, color=C_bad, alpha=0.8, label='infeasible (deficit)')
    ax.axhline(0, color='black', lw=0.8)
    _vline(ax)
    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel('Energy margin (kWh)')
    ax.set_title('Safety margin  (E_available - E_required)', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # ── (1,1) Drive vs regen energy per lap ──────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    # Show drive energy as positive bars, regen as negative bars below zero
    ax.bar(Q, E_drive, width=(Q[1]-Q[0])*0.8, color='#e67e22', alpha=0.75,
           label='Drive (from battery)')
    ax.bar(Q, -E_regen, width=(Q[1]-Q[0])*0.8, color='#2ecc71', alpha=0.75,
           label='Regen recovery (returned)')
    ax.plot(Q, E_lap, 'k-', lw=1.5, label='Net = Drive - Regen')
    ax.axhline(0, color='black', lw=0.6)
    _vline(ax)
    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel('Energy per lap (Wh)')
    ax.set_title('Drive / regen breakdown per lap', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # ── (1,2) Battery mass fraction ───────────────────────────────────────
    batt_frac = m_batt / mass * 100  # %
    ax = fig.add_subplot(gs[1, 2])
    ax2 = ax.twinx()
    _scatter(ax, mass)
    _vline(ax)
    ax2.plot(Q, batt_frac, 'k--', lw=1.2, label='Battery mass fraction')
    ax2.set_ylabel('Battery mass fraction (%)', color='black')
    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel('Total car mass (kg)', color='#555')
    ax.set_title('Mass breakdown', fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Battery sizing for multi-lap EV racing')
    parser.add_argument('--track', default='complex', help='Track name')
    parser.add_argument('--laps', type=int, default=0,
                        help='Number of race laps (0 = auto: target 65 km)')
    parser.add_argument('--Q-min', type=float, default=15.0, dest='Q_min',
                        help='Min battery capacity to sweep (kWh)')
    parser.add_argument('--Q-max', type=float, default=80.0, dest='Q_max',
                        help='Max battery capacity to sweep (kWh)')
    parser.add_argument('--n', type=int, default=40, help='Number of sweep points')
    parser.add_argument('--line', default='min_laptime',
                        choices=['min_laptime', 'center'], help='Racing line mode')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()

    from pathlib import Path
    proj_root = Path(__file__).parent.parent.parent

    print("=" * 65)
    print("  Battery Sizing Optimizer — EV Race")
    print("=" * 65)

    print(f"\n[1/3] Loading track and racing line...")
    track = get_track(args.track)
    racing_line, _ = generate_racing_line(track, mode=args.line)

    if args.laps == 0:
        args.laps = max(20, round(65_000 / track.length))
        print(f"  Auto laps: {args.laps} ({args.laps * track.length / 1000:.1f} km)")
    print(f"  Track : {args.track}  ({track.length:.0f} m/lap)")
    print(f"  Laps  : {args.laps}  ({args.laps * track.length / 1000:.1f} km total)")
    print(f"  Sweep : {args.Q_min}–{args.Q_max} kWh ({args.n} points)")

    print(f"\n[2/3] Sweeping Q_batt ({args.Q_min}–{args.Q_max} kWh)...")
    results = sweep_battery_size(
        racing_line, n_laps=args.laps,
        Q_min=args.Q_min, Q_max=args.Q_max, n_pts=args.n)

    Q_min_r, Q_opt = find_optimal_Q(results)

    print("\n" + "=" * 65)
    if Q_opt is None:
        print("  No feasible battery size found in the sweep range.")
        print("  Increase --Q-max or reduce --laps.")
    else:
        print(f"  Minimum feasible Q_batt : {Q_min_r['Q_batt_kWh']:.1f} kWh")
        print(f"  Car mass at Q*          : {Q_min_r['mass_kg']:.0f} kg")
        print(f"  Lap time at Q*          : {Q_min_r['T_lap_s']:.1f} s")
        print(f"  Final SoC at Q*         : {Q_min_r['SoC_final']:.3f}")
        print(f"  Total race time at Q*   : {Q_min_r['T_total_min']:.1f} min")

    if not args.no_plot:
        print(f"\n[3/3] Plotting...")
        fig_dir = proj_root / 'figures' / args.track
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = str(fig_dir / f'battery_sizing_{args.laps}laps.png')
        plot_results(results, args.laps, track.name, output_path=out)

    print("\nDone.")


if __name__ == '__main__':
    main()