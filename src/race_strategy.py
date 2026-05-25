"""
Phase 4: Co-optimization of racing line aggressiveness + pace management.

Control variables (per lap t = 1..n_laps):
  g_t  in [g_min, g_max]  — cornering aggressiveness (grip_fraction proxy)
  p_t  in [p_min, 1.0]    — pace factor

Model (linear in p):
  T_lap(g, p) = T_base(g) / p
  E_lap(g, p) = E_base(g) * p

NLP:
  minimize   sum T_lap(g_t, p_t)
  s.t.       sum_{tau=1}^{t} E_lap(g_tau, p_tau) <= E_budget   for t = 1..n
  bounds:    g_t in [g_min, g_max],  p_t in [p_min, 1]

Analytical result (KKT, unconstrained relaxation with equality energy budget):
  g* = argmin  T(g) * E(g)          [product minimizer — proven via KKT]
  p* = E_budget / (n_laps * E(g*))   [uniform pace]
  All laps identical at (g*, p*).

Usage:
  python src/race_strategy.py
  python src/race_strategy.py --laps 51 --Q-batt 34 --p-min 0.80
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(__file__))

from car import CarParams
from track import get_track
from controller import generate_racing_line
from battery_sizing import compute_lap_energy_time


# ---------------------------------------------------------------------------
# Pareto curve via grip_fraction sweep
# ---------------------------------------------------------------------------

def compute_strategy_pareto(racing_line, params, g_min=0.50, g_max=0.90, n_pts=30):
    """
    Sweep cornering aggressiveness g = grip_fraction in [g_min, g_max].

    Returns T_arr[i], E_arr[i] for each g value.
    g_max -> fastest lap, most energy.
    g_min -> slowest lap, least energy.
    """
    g_arr = np.linspace(g_min, g_max, n_pts)
    T_list, E_list = [], []

    print(f"\n  {'g':>6}  {'T_lap':>7}  {'E_lap':>8}  {'T*E':>10}")
    print("  " + "-" * 38)

    for g in g_arr:
        T, E, _, _ = compute_lap_energy_time(racing_line, params, grip_fraction=g)
        T_list.append(T)
        E_list.append(E)
        print(f"  {g:.3f}  {T:7.2f}s  {E:7.1f}Wh  {T*E:10.1f}")

    return g_arr, np.array(T_list), np.array(E_list)


# ---------------------------------------------------------------------------
# Analytical optimal strategy (KKT)
# ---------------------------------------------------------------------------

def analytical_optimal(g_arr, T_arr, E_arr, Q_kWh, n_laps,
                        p_min=0.80, SOC_min=0.05):
    """
    Closed-form optimal strategy.

    KKT conditions give: g* = argmin T(g) * E(g)
    Intuition: this balances the marginal time-energy trade-off.
    Then:  p* = E_budget / (n_laps * E(g*)), clipped to [p_min, 1].

    If p* > 1: battery is oversized -> use min-time strategy (g=g_max, p=1).
    If p* < p_min: battery too small for this race -> infeasible.
    """
    TE = T_arr * E_arr
    idx_star = int(np.argmin(TE))
    g_star = g_arr[idx_star]
    T_star = T_arr[idx_star]
    E_star = E_arr[idx_star]

    E_budget = Q_kWh * 1000.0 * (1.0 - SOC_min)  # Wh
    p_unconstrained = E_budget / (n_laps * E_star)

    if p_unconstrained > 1.0:
        # Battery oversized: revert to minimum-time line at full pace
        idx_best = int(np.argmin(T_arr))
        g_star   = g_arr[idx_best]
        T_star   = T_arr[idx_best]
        E_star   = E_arr[idx_best]
        p_star   = 1.0
        p_unconstrained_clipped = p_unconstrained  # keep for reporting
    else:
        p_star = max(p_unconstrained, p_min)
        p_unconstrained_clipped = p_unconstrained

    T_total   = n_laps * T_star / p_star
    E_total   = n_laps * E_star * p_star
    SoC_final = 1.0 - E_total / (Q_kWh * 1000.0)
    feasible  = (p_unconstrained >= p_min) and (SoC_final >= SOC_min - 1e-6)

    return {
        'g_star'          : g_star,
        'p_star'          : p_star,
        'p_unconstrained' : p_unconstrained_clipped,
        'T_star'          : T_star,
        'E_star'          : E_star,
        'T_total_s'       : T_total,
        'T_total_min'     : T_total / 60.0,
        'E_total_Wh'      : E_total,
        'SoC_final'       : SoC_final,
        'feasible'        : feasible,
        'idx_star'        : idx_star,
        'TE_min'          : T_star * E_star,
    }


# ---------------------------------------------------------------------------
# NLP strategy optimizer
# ---------------------------------------------------------------------------

def solve_nlp_strategy(g_arr, T_arr, E_arr, Q_kWh, n_laps=51,
                        p_min=0.80, SOC_min=0.05):
    """
    General NLP: optimize per-lap (g_t, p_t) with SLSQP.

    x = [g_1, ..., g_n, p_1, ..., p_n]

    Expected to confirm the analytical result (all g_t = g*, all p_t = p*)
    when all laps are symmetric.  Per-lap variation appears when constraints
    become active at different laps (e.g. safety car periods — not modelled here).
    """
    n = n_laps
    E_budget = Q_kWh * 1000.0 * (1.0 - SOC_min)

    T_func = interp1d(g_arr, T_arr, kind='linear', fill_value='extrapolate')
    E_func = interp1d(g_arr, E_arr, kind='linear', fill_value='extrapolate')
    dT_dg  = np.gradient(T_arr, g_arr)
    dTdg_func = interp1d(g_arr, dT_dg, kind='linear', fill_value='extrapolate')

    # Warm start from analytical solution
    ana = analytical_optimal(g_arr, T_arr, E_arr, Q_kWh, n_laps, p_min, SOC_min)
    g0  = np.clip(ana['g_star'], g_arr[0], g_arr[-1])
    p0  = np.clip(ana['p_star'], p_min, 1.0)
    x0  = np.concatenate([np.full(n, g0), np.full(n, p0)])

    def objective(x):
        g, p = x[:n], x[n:]
        return float(np.sum(T_func(g) / p))

    def jac(x):
        g, p = x[:n], x[n:]
        dg = dTdg_func(g) / p
        dp = -T_func(g) / p ** 2
        return np.concatenate([dg, dp])

    # Final SoC constraint (tightest; intermediate ones satisfied by symmetry)
    def con_final(x):
        g, p = x[:n], x[n:]
        return E_budget - float(np.sum(E_func(g) * p))

    # Also enforce each lap individually (no mid-race depletion)
    constraints = [{'type': 'ineq', 'fun': con_final}]
    for k in range(1, n):
        def con_k(x, k=k):
            g, p = x[:n], x[n:]
            return E_budget - float(np.sum(E_func(g[:k]) * p[:k]))
        constraints.append({'type': 'ineq', 'fun': con_k})

    bounds = [(g_arr[0], g_arr[-1])] * n + [(p_min, 1.0)] * n

    res = minimize(
        objective, x0, jac=jac,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False},
    )

    g_opt, p_opt = res.x[:n], res.x[n:]
    T_laps = T_func(g_opt) / p_opt
    E_laps = E_func(g_opt) * p_opt
    SoC    = 1.0 - np.cumsum(E_laps) / (Q_kWh * 1000.0)

    return {
        'g'          : g_opt,
        'p'          : p_opt,
        'T_laps'     : T_laps,
        'E_laps'     : E_laps,
        'SoC'        : SoC,
        'T_total_s'  : float(np.sum(T_laps)),
        'T_total_min': float(np.sum(T_laps)) / 60.0,
        'E_total_Wh' : float(np.sum(E_laps)),
        'SoC_final'  : float(SoC[-1]),
        'feasible'   : res.success and (float(SoC[-1]) >= SOC_min - 1e-4),
        'converged'  : res.success,
        'message'    : res.message,
    }


# ---------------------------------------------------------------------------
# Battery sweep: Q* with vs without strategy
# ---------------------------------------------------------------------------

def sweep_Q_strategy(racing_line, params_base, g_arr, T_arr, E_arr,
                     n_laps=51, Q_min=20.0, Q_max=55.0, n_pts=36,
                     p_min=0.80, SOC_min=0.05):
    """
    For each Q_batt:
    - No strategy: check feasibility at g=g_max, p=1 (full attack).
    - With strategy: check feasibility with analytical optimal (g*, p*).

    Returns list of result dicts.
    """
    Q_values = np.linspace(Q_min, Q_max, n_pts)
    results  = []

    # Baseline T and E at full aggressiveness
    T_base = T_arr[-1]   # g=g_max
    E_base = E_arr[-1]

    for Q in Q_values:
        E_budget = Q * 1000.0 * (1.0 - SOC_min)
        mass     = params_base.m_chassis + Q / params_base.e_spec

        # No-strategy feasibility
        E_req_no_strat = n_laps * E_base
        feasible_no    = E_budget >= E_req_no_strat
        T_tot_no       = n_laps * T_base          # same T regardless (p=1)
        SoC_no         = 1.0 - E_req_no_strat / (Q * 1000.0)

        # With-strategy feasibility
        ana = analytical_optimal(g_arr, T_arr, E_arr, Q, n_laps, p_min, SOC_min)
        feasible_strat = ana['feasible']
        T_tot_strat    = ana['T_total_s']
        SoC_strat      = ana['SoC_final']

        results.append({
            'Q_kWh'            : Q,
            'mass_kg'          : mass,
            'feasible_no_strat': feasible_no,
            'feasible_strat'   : feasible_strat,
            'T_total_no_s'     : T_tot_no,
            'T_total_strat_s'  : T_tot_strat,
            'SoC_no'           : SoC_no,
            'SoC_strat'        : SoC_strat,
            'g_star'           : ana['g_star'],
            'p_star'           : ana['p_star'],
            'p_unconstrained'  : ana['p_unconstrained'],
        })

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_strategy_analysis(g_arr, T_arr, E_arr, ana, nlp,
                            sweep_results, Q_kWh, n_laps, track_name,
                            output_path=None):
    """6-panel figure summarising the Phase 4 co-optimisation."""

    # Derived sweep arrays
    Q_arr     = np.array([r['Q_kWh']              for r in sweep_results])
    feas_no   = np.array([r['feasible_no_strat']  for r in sweep_results])
    feas_str  = np.array([r['feasible_strat']     for r in sweep_results])
    T_no      = np.array([r['T_total_no_s']       for r in sweep_results]) / 60.0
    T_str     = np.array([r['T_total_strat_s']    for r in sweep_results]) / 60.0
    p_unc     = np.array([r['p_unconstrained']    for r in sweep_results])

    Q_star_no  = Q_arr[feas_no ].min()  if feas_no.any()  else None
    Q_star_str = Q_arr[feas_str].min()  if feas_str.any() else None

    C_base = '#e74c3c'
    C_strat = '#2980b9'
    C_opt   = '#f39c12'
    C_nlp   = '#27ae60'

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f"Phase 4 — Race Strategy Co-optimisation  |  "
        f"{track_name}  |  {n_laps} laps  |  Q={Q_kWh:.1f} kWh",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)

    # ── (0,0) Pareto curve + product minimiser ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(T_arr, E_arr, c=g_arr, cmap='plasma', s=60,
                    zorder=3, edgecolors='k', linewidths=0.4, label='Pareto pts')
    ax.plot(T_arr, E_arr, '-', color='0.7', lw=1.2, zorder=2)
    ax.scatter(T_arr[-1], E_arr[-1], s=160, marker='*', color=C_base, zorder=5,
               label=f'Baseline (g={g_arr[-1]:.2f}, p=1)')
    ax.scatter(ana['T_star'], ana['E_star'], s=160, marker='D', color=C_opt, zorder=5,
               label=f'Optimal g*={ana["g_star"]:.2f}')
    if nlp is not None:
        T_nlp = np.mean(nlp['T_laps'] * nlp['p'])   # T_base = T_lap*p
        E_nlp = np.mean(nlp['E_laps'] / nlp['p'])   # E_base = E_lap/p
        ax.scatter(T_nlp, E_nlp, s=100, marker='^', color=C_nlp, zorder=5,
                   label=f'NLP avg')
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('grip fraction g', fontsize=8)
    ax.set_xlabel('Lap time T(g)  [s]')
    ax.set_ylabel('Energy E(g)  [Wh/lap]')
    ax.set_title('Pareto curve: T vs E', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ── (0,1) T×E product — shows g* ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    TE = T_arr * E_arr
    ax.plot(g_arr, TE, 'o-', ms=5, color='#8e44ad', lw=1.8)
    ax.axvline(ana['g_star'], color=C_opt, ls='--', lw=1.5,
               label=f'g* = {ana["g_star"]:.3f}')
    ax.scatter(ana['g_star'], ana['TE_min'], s=120, color=C_opt, zorder=5)
    ax.set_xlabel('Grip fraction g')
    ax.set_ylabel('T(g) × E(g)  [s·Wh]')
    ax.set_title('KKT product minimiser', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ── (0,2) Per-lap strategy ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    laps = np.arange(1, n_laps + 1)
    ax2 = ax.twinx()

    # Analytical solution (flat lines)
    ax.axhline(ana['g_star'], color=C_opt, lw=2.0, label=f'g* = {ana["g_star"]:.3f}')
    ax2.axhline(ana['p_star'], color=C_nlp, lw=2.0, ls='--',
                label=f'p* = {ana["p_star"]:.3f}')

    if nlp is not None:
        ax.step(laps, nlp['g'], where='mid', color=C_opt, lw=1.2, alpha=0.5,
                ls=':', label='NLP g_t')
        ax2.step(laps, nlp['p'], where='mid', color=C_nlp, lw=1.2, alpha=0.5,
                 ls=':', label='NLP p_t')

    ax2.set_ylabel('Pace factor p_t', color=C_nlp)
    ax2.tick_params(axis='y', colors=C_nlp)
    p_lo = min(ana['p_star'] - 0.05, 0.73)
    ax2.set_ylim(p_lo, 1.08)
    ax.set_ylim(g_arr[0] - 0.03, g_arr[-1] + 0.04)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Grip fraction g_t', color=C_opt)
    ax.tick_params(axis='y', colors=C_opt)
    ax.set_title('Optimal per-lap strategy', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ── (1,0) SoC trajectory ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    laps = np.arange(1, n_laps + 1)

    # Current Q: baseline (g=g_max, p=1)
    E_base_lap   = E_arr[-1]
    SoC_base = 1.0 - np.cumsum(np.full(n_laps, E_base_lap)) / (Q_kWh * 1000.0)
    ax.plot(laps, SoC_base * 100, '-', color=C_base, lw=2.0,
            label=f'Q={Q_kWh:.0f} kWh, baseline (p=1)')

    # Strategy scenario: Q*_strategy (p* < 1, shows pace management)
    if Q_star_str is not None and abs(Q_star_str - Q_kWh) > 0.5:
        ana_str = analytical_optimal(g_arr, T_arr, E_arr, Q_star_str, n_laps,
                                     p_min=0.80)
        E_str_lap = ana_str['E_star'] * ana_str['p_star']
        SoC_str   = 1.0 - np.cumsum(np.full(n_laps, E_str_lap)) / (Q_star_str * 1000.0)
        ax.plot(laps, SoC_str * 100, '--', color=C_strat, lw=1.8,
                label=f'Q={Q_star_str:.0f} kWh, p*={ana_str["p_star"]:.2f}')

    if nlp is not None:
        ax.plot(laps, nlp['SoC'] * 100, ':', color=C_nlp, lw=1.4, label='NLP (Q=36.9 kWh)')

    ax.axhline(5.0, color='grey', ls=':', lw=1.2, label='SOC_min = 5%')
    ax.set_xlabel('Lap')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title('SoC trajectory', fontsize=10)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ── (1,1) Optimal pace p*(Q) and minimum Q comparison ───────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(Q_arr, np.clip(p_unc, 0, 1.2), 'o-', color=C_strat, ms=4, lw=1.8,
            label='p*(Q) = E_budget / (n·E*)')
    ax.axhline(1.0,        color='black',  ls='-',  lw=1.0, label='p = 1 (full speed)')
    ax.axhline(0.80, color='grey', ls=':', lw=1.2, label='p_min = 0.80')
    if Q_star_no is not None:
        ax.axvline(Q_star_no,  color=C_base,  ls='--', lw=1.4,
                   label=f'Q*_baseline = {Q_star_no:.1f} kWh')
    if Q_star_str is not None:
        ax.axvline(Q_star_str, color=C_strat, ls='--', lw=1.4,
                   label=f'Q*_strategy = {Q_star_str:.1f} kWh')
    ax.fill_between(Q_arr, 0, np.clip(p_unc, 0, 1.2),
                    where=p_unc >= 1.0, alpha=0.10, color='green', label='overcapacity')
    ax.fill_between(Q_arr, 0, np.clip(p_unc, 0, 1.2),
                    where=(p_unc >= 0.80) & (p_unc < 1.0),
                    alpha=0.15, color=C_strat, label='strategy zone')
    ax.set_xlabel('Battery capacity Q (kWh)')
    ax.set_ylabel('Optimal pace factor p*')
    ax.set_title('p*(Q): required pace vs battery size', fontsize=10)
    ax.set_ylim(0.55, 1.25)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ── (1,2) Summary text ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    T_base_min  = n_laps * T_arr[-1] / 60.0
    T_ana_min   = ana['T_total_min']
    dT_s        = ana['T_total_s'] - n_laps * T_arr[-1]
    e_spec      = CarParams().e_spec
    mass_saving = (((Q_star_no or 0) - (Q_star_str or 0)) / e_spec
                   if (Q_star_no and Q_star_str) else 0.0)

    nlp_line = ('yes' if (nlp and nlp['converged']) else
                'no'  if (nlp and not nlp['converged']) else 'not run')

    lines = [
        "─" * 33,
        "  PHASE 4 SUMMARY",
        "─" * 33,
        f"  Track          : {track_name}",
        f"  Laps           : {n_laps}",
        f"  Q_batt         : {Q_kWh:.1f} kWh",
        "",
        "  BASELINE (g=g_max, p=1):",
        f"    T_total      = {T_base_min:.2f} min",
        f"    SoC_final    = {SoC_base[-1]*100:.1f}%",
        "",
        f"  OPTIMAL (g*={ana['g_star']:.3f}, p*={ana['p_star']:.3f}):",
        f"    T_total      = {T_ana_min:.2f} min",
        f"    SoC_final    = {ana['SoC_final']*100:.1f}%",
        f"    dT_race      = {dT_s:+.1f} s",
        "",
        "  BATTERY SIZING:",
        f"    Q*_baseline  = {Q_star_no:.1f} kWh" if Q_star_no else "    Q*_baseline  = N/A",
        f"    Q*_strategy  = {Q_star_str:.1f} kWh" if Q_star_str else "    Q*_strategy  = N/A",
        (f"    Mass saving  = {mass_saving:.0f} kg"
         if (Q_star_no and Q_star_str) else ""),
        "",
        f"  NLP            : {nlp_line}",
        "",
        "  NOTE: T*E monotone at Monaco",
        "  => optimal line = max aggression",
        "  => strategy = pace management only",
    ]

    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.9))

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
    parser = argparse.ArgumentParser(
        description='Phase 4: race strategy co-optimisation')
    parser.add_argument('--track',  default='complex', help='Track name')
    parser.add_argument('--laps',   type=int,   default=0,
                        help='Number of race laps (0 = auto: target 65 km)')
    parser.add_argument('--Q-batt', type=float, default=36.9,  dest='Q_kWh',
                        help='Battery capacity for single-Q analysis (kWh)')
    parser.add_argument('--p-min',  type=float, default=0.80,  dest='p_min',
                        help='Minimum pace factor [0.5, 1.0]')
    parser.add_argument('--g-min',  type=float, default=0.50,  dest='g_min',
                        help='Min grip fraction for Pareto sweep')
    parser.add_argument('--g-max',  type=float, default=0.90,  dest='g_max',
                        help='Max grip fraction for Pareto sweep')
    parser.add_argument('--n-pareto', type=int, default=30,    dest='n_pareto',
                        help='Number of Pareto sweep points')
    parser.add_argument('--no-nlp',   action='store_true',
                        help='Skip the NLP solve (faster, analytical only)')
    parser.add_argument('--no-plot',  action='store_true')
    args = parser.parse_args()

    from pathlib import Path
    proj_root = Path(__file__).parent.parent

    print("=" * 65)
    print("  Phase 4 — Race Strategy Co-optimisation")
    print("=" * 65)
    print(f"  Track    : {args.track}")
    print(f"  Q_batt   : {args.Q_kWh} kWh")
    print(f"  p_min    : {args.p_min}")
    print(f"  g range  : [{args.g_min}, {args.g_max}]  ({args.n_pareto} pts)")

    # ------------------------------------------------------------------
    # 1. Load track and racing line
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading track and racing line...")
    track = get_track(args.track)
    racing_line, _ = generate_racing_line(track, mode='min_laptime')

    if args.laps == 0:
        args.laps = max(20, round(65_000 / track.length))
        print(f"  Auto laps: {args.laps} ({args.laps * track.length / 1000:.1f} km)")

    params = CarParams(Q_batt=args.Q_kWh)
    print(f"  Track length : {track.length:.0f} m  |  Laps : {args.laps}")
    print(f"  Car mass     : {params.mass:.0f} kg  (Q={args.Q_kWh} kWh)")

    # ------------------------------------------------------------------
    # 2. Compute strategy Pareto curve
    # ------------------------------------------------------------------
    print(f"\n[2/5] Computing Pareto curve (grip_fraction sweep)...")
    g_arr, T_arr, E_arr = compute_strategy_pareto(
        racing_line, params,
        g_min=args.g_min, g_max=args.g_max, n_pts=args.n_pareto)

    T_range_pct = 100 * (T_arr.max() - T_arr.min()) / T_arr.min()
    E_range_pct = 100 * (E_arr.max() - E_arr.min()) / E_arr.min()
    print(f"\n  T range: {T_arr.min():.2f}–{T_arr.max():.2f}s  ({T_range_pct:.1f}% spread)")
    print(f"  E range: {E_arr.min():.1f}–{E_arr.max():.1f} Wh  ({E_range_pct:.1f}% spread)")

    # ------------------------------------------------------------------
    # 3. Analytical optimal (KKT)
    # ------------------------------------------------------------------
    print(f"\n[3/5] Analytical optimal strategy (KKT)...")
    ana = analytical_optimal(g_arr, T_arr, E_arr, args.Q_kWh, args.laps,
                              p_min=args.p_min)
    print(f"  g* = {ana['g_star']:.3f}  (T*E minimiser at T={ana['T_star']:.2f}s, E={ana['E_star']:.1f}Wh)")
    print(f"  p* (unconstrained) = {ana['p_unconstrained']:.4f}")
    print(f"  p* (applied)       = {ana['p_star']:.4f}")
    print(f"  T_total            = {ana['T_total_min']:.3f} min  (vs baseline {args.laps * T_arr[-1] / 60:.3f} min)")
    print(f"  SoC_final          = {ana['SoC_final']*100:.2f}%")
    print(f"  Feasible           : {ana['feasible']}")

    # ------------------------------------------------------------------
    # 4. NLP (optional)
    # ------------------------------------------------------------------
    nlp = None
    if not args.no_nlp:
        print(f"\n[4/5] NLP strategy optimisation ({args.laps} laps × 2 vars)...")
        nlp = solve_nlp_strategy(g_arr, T_arr, E_arr, args.Q_kWh,
                                  n_laps=args.laps, p_min=args.p_min)
        print(f"  Converged  : {nlp['converged']}  ({nlp['message']})")
        print(f"  T_total    : {nlp['T_total_min']:.3f} min")
        print(f"  SoC_final  : {nlp['SoC_final']*100:.2f}%")
        print(f"  g mean/std : {nlp['g'].mean():.4f} / {nlp['g'].std():.5f}")
        print(f"  p mean/std : {nlp['p'].mean():.4f} / {nlp['p'].std():.5f}")
    else:
        print(f"\n[4/5] NLP skipped (--no-nlp).")

    # ------------------------------------------------------------------
    # 5. Battery sweep: Q* with vs without strategy
    # ------------------------------------------------------------------
    print(f"\n[5/5] Battery sweep (Q* with vs without strategy)...")
    sweep = sweep_Q_strategy(
        racing_line, params, g_arr, T_arr, E_arr,
        n_laps=args.laps, Q_min=20.0, Q_max=55.0, n_pts=36,
        p_min=args.p_min)

    Q_arr_sw = np.array([r['Q_kWh'] for r in sweep])
    feas_no  = np.array([r['feasible_no_strat'] for r in sweep])
    feas_str = np.array([r['feasible_strat']    for r in sweep])

    Q_star_no  = Q_arr_sw[feas_no ].min() if feas_no.any()  else None
    Q_star_str = Q_arr_sw[feas_str].min() if feas_str.any() else None

    print(f"\n  Q*_no_strategy  = {Q_star_no:.1f} kWh" if Q_star_no else "  Q*_no_strategy  = not found")
    print(f"  Q*_with_strategy = {Q_star_str:.1f} kWh" if Q_star_str else "  Q*_with_strategy = not found")
    if Q_star_no and Q_star_str:
        dQ = Q_star_no - Q_star_str
        e_spec = params.e_spec
        dm = dQ / e_spec
        print(f"  Battery saving   = {dQ:.1f} kWh  ->  {dm:.0f} kg lighter")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    if not args.no_plot:
        fig_dir = proj_root / 'figures' / args.track
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = str(fig_dir / f'race_strategy_{args.laps}laps.png')
        plot_strategy_analysis(
            g_arr, T_arr, E_arr, ana, nlp, sweep,
            args.Q_kWh, args.laps, track.name,
            output_path=out)

    print("\nDone.")


if __name__ == '__main__':
    main()
