"""
Generate report-ready 2-panel figures for the battery sizing and race strategy sections.
Saves to figures/report/

  battery_sizing_report.png     — Energy budget + SoC final vs Q
  race_strategy_report.png      — Pareto T vs E + p*(Q) battery reduction
  circuit_comparison_report.png — SoC trajectories + energy/lap bar chart

Run:  python "src/battery sizing/report_figures.py"
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_this = os.path.dirname(os.path.abspath(__file__))
_sl   = os.path.join(os.path.dirname(_this), 'single lap')
sys.path.insert(0, _sl)
sys.path.insert(0, _this)

from car import CarParams
from track import get_track
from battery_sizing import sweep_battery_size, find_optimal_Q
from race_strategy import compute_strategy_pareto, analytical_optimal, sweep_Q_strategy

PROJ_ROOT = os.path.dirname(os.path.dirname(_this))
FIG_DIR   = os.path.join(PROJ_ROOT, 'figures', 'report')
os.makedirs(FIG_DIR, exist_ok=True)

C_OK    = '#2ecc71'
C_BAD   = '#e74c3c'
C_STAR  = '#f39c12'
C_STRAT = '#2980b9'
FS_AX   = 11
FS_TK   = 10
FS_LEG  = 9
FS_TI   = 11


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Battery Sizing
# ─────────────────────────────────────────────────────────────────────────────

def fig_battery_sizing(track_name='complex', n_laps=51):
    print(f"\n[1/3] Battery sizing  (track={track_name}, {n_laps} laps)...")
    track = get_track(track_name)

    from controller import generate_racing_line
    racing_line, _ = generate_racing_line(track, mode='min_laptime')

    results = sweep_battery_size(racing_line, n_laps=n_laps,
                                  Q_min=15.0, Q_max=80.0, n_pts=40)
    Q_min_r, _ = find_optimal_Q(results)
    Q_star      = Q_min_r['Q_batt_kWh'] if Q_min_r else None

    Q       = np.array([r['Q_batt_kWh']          for r in results])
    E_req   = np.array([r['E_total_Wh']  / 1000.0 for r in results])
    E_avail = np.array([r['E_available_Wh']/1000.0 for r in results])
    SoC_f   = np.array([r['SoC_final'] * 100       for r in results])
    ok      = np.array([r['feasible']              for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.30)

    # ── Left: energy budget ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(Q, E_req,   'o-', ms=4, color='#e67e22', lw=2,
            label=f'Required ({n_laps} laps)')
    ax.plot(Q, E_avail, 's--', ms=4, color=C_OK, lw=2,
            label='Available (95 % of $Q$)')
    ax.fill_between(Q, E_req, E_avail,
                    where=E_avail >= E_req, alpha=0.15, color=C_OK,  label='Margin')
    ax.fill_between(Q, E_req, E_avail,
                    where=E_avail <  E_req, alpha=0.15, color=C_BAD, label='Deficit')
    if Q_star:
        ax.axvline(Q_star, color=C_STAR, ls='--', lw=2.0,
                   label=f'$Q^* = {Q_star:.1f}$ kWh')
    ax.set_xlabel('Battery capacity (kWh)', fontsize=FS_AX)
    ax.set_ylabel('Energy (kWh)',            fontsize=FS_AX)
    ax.set_title('Energy budget',            fontsize=FS_TI, fontweight='bold')
    ax.legend(fontsize=FS_LEG)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=FS_TK)

    # ── Right: SoC at race end ─────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(Q[ok],  SoC_f[ok],  c=C_OK,  s=35, zorder=3, label='Feasible')
    ax.scatter(Q[~ok], SoC_f[~ok], c=C_BAD, s=35, zorder=3, label='Infeasible')
    ax.axhline(5.0, color='#3498db', ls=':', lw=1.8,
               label='$\\mathrm{SoC}_{\\min} = 5\\%$')
    if Q_star:
        ax.axvline(Q_star, color=C_STAR, ls='--', lw=2.0,
                   label=f'$Q^* = {Q_star:.1f}$ kWh')
    ax.set_xlabel('Battery capacity (kWh)', fontsize=FS_AX)
    ax.set_ylabel('Final SoC (%)',          fontsize=FS_AX)
    ax.set_title('State of charge at race end', fontsize=FS_TI, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=FS_LEG)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=FS_TK)

    out = os.path.join(FIG_DIR, 'battery_sizing_report.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}  (Q* = {Q_star:.1f} kWh)" if Q_star else f"  -> {out}")

    return racing_line  # reused in fig 2


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Race Strategy
# ─────────────────────────────────────────────────────────────────────────────

def _load_circuit_records():
    CIRCUITS = [
        ('complex', 'Grand Prix',  '#e74c3c', 'race_51laps.json'),
        ('monza',   'Monza-Style', '#3498db', 'race_58laps.json'),
        ('hairpin', 'Hairpin',     '#2ecc71', 'race_51laps.json'),
    ]
    records = []
    for name, label, color, fname in CIRCUITS:
        path = os.path.join(PROJ_ROOT, 'figures', name, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping")
            continue
        with open(path) as f:
            d = json.load(f)
        records.append({
            'label':     label,
            'color':     color,
            'energy_wh': np.array(d['lap_energy_Wh']),
            'Q_kWh':     d['Q_batt_kWh'],
            'mass_kg':   d['mass_kg'],
        })
    return records


def fig_race_strategy(racing_line, Q_kWh=36.9, n_laps=51,
                       T_max_pct=15.0, p_min=0.80):
    print(f"\n[2/3] Race strategy  (Q={Q_kWh} kWh, {n_laps} laps, "
          f"T_max=+{T_max_pct:.0f}%)...")

    params = CarParams(Q_batt=Q_kWh)
    g_arr, T_arr, E_arr = compute_strategy_pareto(
        racing_line, params, g_min=0.50, g_max=0.90, n_pts=30)

    T_max = T_arr.min() * (1.0 + T_max_pct / 100.0)
    ana   = analytical_optimal(g_arr, T_arr, E_arr, Q_kWh, n_laps,
                                p_min=p_min, T_max=T_max)

    sweep = sweep_Q_strategy(params, g_arr, T_arr, E_arr,
                              n_laps=n_laps, Q_min=4.0, Q_max=55.0, n_pts=52,
                              p_min=p_min, T_max=T_max)
    Q_sw  = np.array([r['Q_kWh']            for r in sweep])
    p_unc = np.array([r['p_unconstrained']   for r in sweep])
    fno   = np.array([r['feasible_no_strat'] for r in sweep])
    fstr  = np.array([r['feasible_strat']    for r in sweep])
    Qn    = float(Q_sw[fno ].min()) if fno.any()  else None
    Qs    = float(Q_sw[fstr].min()) if fstr.any() else None

    circuit_records = _load_circuit_records()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.subplots_adjust(wspace=0.36)

    # ── Left: Pareto T vs E ──────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(T_arr, E_arr, c=g_arr, cmap='plasma',
                    s=60, zorder=3, edgecolors='k', linewidths=0.4)
    ax.plot(T_arr, E_arr, '-', color='0.75', lw=1.2, zorder=2)
    i_fast = int(np.argmin(T_arr))
    kkt_at_fastest = (ana['idx_star'] == i_fast)
    ax.scatter(T_arr[i_fast], E_arr[i_fast], s=160, marker='*',
               color=C_BAD, zorder=5,
               label='Fastest ($p=1$)' + (' = KKT opt.' if kkt_at_fastest else ''))
    if not kkt_at_fastest:
        ax.scatter(ana['T_star'], ana['E_star'], s=130, marker='D',
                   color=C_STAR, zorder=5,
                   label=f'KKT opt. ($g^*={ana["g_star"]:.2f}$)')
    ax.axvline(T_max, color='grey', ls=':', lw=1.6,
               label=f'$T_{{\\rm max}}$ (+{T_max_pct:.0f}%)')
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('Grip fraction $g$', fontsize=FS_LEG)
    ax.set_xlabel('Lap time $T$ (s)',        fontsize=FS_AX)
    ax.set_ylabel('Net energy $E$ (Wh/lap)', fontsize=FS_AX)
    ax.set_title('Time–energy Pareto front',  fontsize=FS_TI, fontweight='bold')
    ax.legend(fontsize=FS_LEG)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=FS_TK)

    # ── Middle: p*(Q) and Q* reduction ───────────────────────────────────
    ax = axes[1]
    p_plot = np.clip(p_unc, 0.0, 1.35)
    ax.plot(Q_sw, p_plot, 'o-', color=C_STRAT, ms=4, lw=2, label='$p^*(Q)$')
    ax.axhline(1.0,   color='black', ls='-',  lw=1.2, label='Full pace ($p=1$)')
    ax.axhline(p_min, color='grey',  ls=':',  lw=1.6,
               label=f'$p_{{\\rm min}}={p_min:.2f}$')
    if Qn:
        ax.axvline(Qn, color=C_BAD,   ls='--', lw=2.0,
                   label=f'$Q^*$ no strategy = {Qn:.0f} kWh')
    if Qs:
        ax.axvline(Qs, color=C_STRAT, ls='--', lw=2.0,
                   label=f'$Q^*$ with strategy = {Qs:.0f} kWh')
    ax.fill_between(Q_sw, 0, p_plot, where=p_unc >= 1.0,
                    alpha=0.08, color='green')
    ax.fill_between(Q_sw, 0, p_plot,
                    where=(p_unc >= p_min) & (p_unc < 1.0),
                    alpha=0.12, color=C_STRAT)
    ax.set_xlabel('Battery capacity $Q$ (kWh)',  fontsize=FS_AX)
    ax.set_ylabel('Optimal pace factor $p^*$',   fontsize=FS_AX)
    ax.set_title('Required pace vs battery size', fontsize=FS_TI, fontweight='bold')
    ax.legend(fontsize=FS_LEG, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=FS_TK)
    ax.set_ylim(max(0.0, float(p_plot.min()) - 0.05), 1.42)

    # ── Right: energy consumption by circuit ─────────────────────────────
    ax = axes[2]
    if circuit_records:
        xs     = np.arange(len(circuit_records))
        means  = [np.mean(r['energy_wh']) for r in circuit_records]
        colors = [r['color']              for r in circuit_records]
        labels = [r['label']              for r in circuit_records]
        bars   = ax.bar(xs, means, color=colors, alpha=0.85,
                        edgecolor='white', linewidth=1.2, width=0.5)
        for bar, r, m in zip(bars, circuit_records, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 5.0,
                    f'{m:.0f} Wh\n$Q^*={r["Q_kWh"]:.0f}$ kWh\n{r["mass_kg"]:.0f} kg',
                    ha='center', va='bottom', fontsize=8.5, linespacing=1.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=FS_TK)
        ax.set_ylim(0, max(means) * 1.32)
    ax.set_ylabel('Mean net energy per lap (Wh)', fontsize=FS_AX)
    ax.set_title('Energy consumption by circuit',  fontsize=FS_TI, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(labelsize=FS_TK)

    out = os.path.join(FIG_DIR, 'race_strategy_report.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if Qn and Qs:
        e_spec = CarParams().e_spec
        print(f"  -> {out}  "
              f"(Q*_baseline={Qn:.0f} kWh, Q*_strategy={Qs:.0f} kWh, "
              f"saving={(Qn-Qs)/e_spec:.0f} kg)")
    else:
        print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Circuit Comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_circuit_comparison():
    print("\n[3/3] Circuit comparison...")

    CIRCUITS = [
        ('complex', 'Grand Prix',  '#e74c3c', 'race_51laps.json'),
        ('monza',   'Monza-Style', '#3498db', 'race_58laps.json'),
        ('hairpin', 'Hairpin',     '#2ecc71', 'race_51laps.json'),
    ]

    records = []
    for name, label, color, fname in CIRCUITS:
        path = os.path.join(PROJ_ROOT, 'figures', name, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping")
            continue
        with open(path) as f:
            d = json.load(f)
        records.append({
            'label':     label,
            'color':     color,
            'laps_done': d['laps_completed'],
            'soc':       np.array(d['lap_soc_end']) * 100.0,
            'energy_wh': np.array(d['lap_energy_Wh']),
            'Q_kWh':     d['Q_batt_kWh'],
            'mass_kg':   d['mass_kg'],
        })

    if not records:
        print("  No JSON data found — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.30)

    # ── Left: SoC trajectories ─────────────────────────────────────────────
    ax = axes[0]
    for r in records:
        xs = np.linspace(0.0, 1.0, r['laps_done'])
        ax.plot(xs, r['soc'], '-', color=r['color'], lw=2.2, label=r['label'])
    ax.axhline(5.0, color='black', ls=':', lw=1.6,
               label='$\\mathrm{SoC}_{\\min} = 5\\%$')
    ax.set_xlabel('Race progress (fraction of laps)', fontsize=FS_AX)
    ax.set_ylabel('State of Charge (%)',              fontsize=FS_AX)
    ax.set_title('Battery SoC over race',             fontsize=FS_TI, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 103)
    ax.legend(fontsize=FS_LEG)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=FS_TK)

    # ── Right: mean energy per lap ─────────────────────────────────────────
    ax = axes[1]
    xs     = np.arange(len(records))
    means  = [np.mean(r['energy_wh']) for r in records]
    colors = [r['color']              for r in records]
    labels = [r['label']              for r in records]
    bars   = ax.bar(xs, means, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=1.2, width=0.5)
    for bar, r, m in zip(bars, records, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 5.0,
                f'{m:.0f} Wh\n$Q^*={r["Q_kWh"]:.0f}$ kWh\n{r["mass_kg"]:.0f} kg',
                ha='center', va='bottom', fontsize=8.5, linespacing=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FS_TK)
    ax.set_ylabel('Mean net energy per lap (Wh)', fontsize=FS_AX)
    ax.set_title('Energy consumption by circuit',  fontsize=FS_TI, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.32)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(labelsize=FS_TK)

    out = os.path.join(FIG_DIR, 'circuit_comparison_report.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Report Figures — Battery Sizing & Race Strategy")
    print("=" * 55)

    racing_line = fig_battery_sizing(track_name='complex', n_laps=51)
    fig_race_strategy(racing_line, Q_kWh=36.9, n_laps=51,
                       T_max_pct=15.0, p_min=0.80)
    fig_circuit_comparison()

    print(f"\nDone — figures saved to figures/report/")


if __name__ == '__main__':
    main()
