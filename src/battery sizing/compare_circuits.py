"""
Cross-circuit comparison figure.

Reads race JSON results from figures/<track>/ and produces a single
6-panel comparison figure saved to figures/comparison_circuits.png.

Usage:
    python src/compare_circuits.py
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_this = os.path.dirname(os.path.abspath(__file__))
_sl   = os.path.join(os.path.dirname(_this), 'single lap')
sys.path.insert(0, _sl)
sys.path.insert(0, _this)

from track import get_track
from car import CarParams
from battery_sizing import compute_lap_energy_time, _speed_profile
from optimizer_ipopt import _compute_curvature, _arc_lengths
from controller import generate_racing_line

# ---------------------------------------------------------------------------
# Circuit registry
# ---------------------------------------------------------------------------

CIRCUITS = [
    {
        'key'      : 'complex',
        'label'    : 'Monaco\n(Grand Prix Circuit)',
        'color'    : '#e74c3c',
        'json'     : 'complex/race_51laps.json',
        'n_laps'   : 51,
    },
    {
        'key'      : 'monza',
        'label'    : 'Monza\n(Monza-Style)',
        'color'    : '#2980b9',
        'json'     : 'monza/race_58laps.json',
        'n_laps'   : 58,
    },
    {
        'key'      : 'hairpin',
        'label'    : 'Hairpin\n(Hairpin & Chicane)',
        'color'    : '#27ae60',
        'json'     : 'hairpin/race_51laps.json',
        'n_laps'   : 51,
    },
]


def load_results(proj_root):
    fig_dir = os.path.join(proj_root, 'figures')
    for c in CIRCUITS:
        path = os.path.join(fig_dir, c['json'])
        with open(path) as f:
            c['data'] = json.load(f)
    return CIRCUITS


def load_speed_profiles(circuits):
    """Compute normalised speed profile along each racing line."""
    for c in circuits:
        track = get_track(c['key'])
        racing_line, _ = generate_racing_line(track, mode='min_laptime')
        params = CarParams(Q_batt=c['data']['Q_batt_kWh'])
        kappa = _compute_curvature(racing_line)
        ds    = _arc_lengths(racing_line)
        v     = _speed_profile(ds, kappa, params, grip_fraction=0.85)
        s     = np.concatenate([[0], np.cumsum(ds[:-1])])
        s_norm = s / s[-1]  # normalised arc-length [0, 1]
        c['v_profile'] = v
        c['s_norm']    = s_norm
        c['kappa']     = kappa
        c['track_length'] = track.length
        print(f"  {c['key']:10} loaded  (len={track.length:.0f}m)")
    return circuits


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(circuits, output_path=None):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        'Circuit Comparison — EV Race Simulation  (~65 km race distance)',
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36)

    # ── (0,0) SoC trajectory ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for c in circuits:
        d     = c['data']
        laps  = np.arange(1, len(d['lap_soc_end']) + 1)
        soc   = np.array(d['lap_soc_end']) * 100
        ax.plot(laps / c['n_laps'], soc, '-', color=c['color'],
                lw=2.0, label=c['label'].replace('\n', ' '))
    ax.axhline(5.0, color='grey', ls=':', lw=1.2, label='SoC_min = 5%')
    ax.set_xlabel('Race progress (fraction of laps)')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title('SoC trajectory', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ── (0,1) Lap time distribution ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    positions = [1, 2, 3]
    data_bp   = [np.array(c['data']['lap_times_s']) for c in circuits]
    colors_bp = [c['color'] for c in circuits]
    bp = ax.boxplot(data_bp, positions=positions, widths=0.5, patch_artist=True,
                    medianprops=dict(color='white', lw=2),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2))
    for patch, col in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    # Overlay individual laps as scatter
    for i, (c, pos) in enumerate(zip(circuits, positions)):
        laps = np.array(c['data']['lap_times_s'])
        ax.scatter(np.full(len(laps), pos) + np.random.default_rng(i).uniform(-0.12, 0.12, len(laps)),
                   laps, s=8, color=c['color'], alpha=0.5, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([c['label'] for c in circuits], fontsize=7)
    ax.set_ylabel('Lap time (s)')
    ax.set_title('Lap time distribution', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # ── (0,2) Speed profile comparison ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for c in circuits:
        ax.plot(c['s_norm'], c['v_profile'] * 3.6,
                '-', color=c['color'], lw=1.4, alpha=0.85,
                label=c['label'].replace('\n', ' '))
    ax.set_xlabel('Normalised arc-length')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Speed profile (optimal line)', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ── (1,0) Energy per lap — mean breakdown ───────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    labels  = [c['label'] for c in circuits]
    e_mean  = [np.mean(c['data']['lap_energy_Wh']) for c in circuits]
    e_std   = [np.std(c['data']['lap_energy_Wh'])  for c in circuits]
    x = np.arange(len(circuits))
    bars = ax.bar(x, e_mean, yerr=e_std, color=[c['color'] for c in circuits],
                  alpha=0.8, width=0.5, capsize=5,
                  error_kw=dict(lw=1.2, capthick=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Net energy per lap (Wh)')
    ax.set_title('Mean energy consumption per lap', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels on bars
    for bar, mean, std in zip(bars, e_mean, e_std):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 8,
                f'{mean:.0f} Wh', ha='center', va='bottom', fontsize=8)

    # ── (1,1) Key metrics bar comparison ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    metrics = {
        'Race time\n(min)'     : [c['data']['total_time_min'] for c in circuits],
        'Q* battery\n(kWh)'    : [c['data']['Q_batt_kWh']     for c in circuits],
        'Car mass\n(kg / 10)'  : [c['data']['mass_kg'] / 10   for c in circuits],
    }
    x      = np.arange(len(circuits))
    n_met  = len(metrics)
    width  = 0.22
    offsets = np.linspace(-(n_met - 1) * width / 2, (n_met - 1) * width / 2, n_met)
    cmap_m = ['#8e44ad', '#e67e22', '#16a085']
    for (label, vals), offset, col in zip(metrics.items(), offsets, cmap_m):
        ax.bar(x + offset, vals, width=width, label=label, color=col, alpha=0.80)
    ax.set_xticks(x)
    ax.set_xticklabels([c['label'] for c in circuits], fontsize=7)
    ax.set_title('Race metrics comparison', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # ── (1,2) Summary table ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    col_headers = ['', 'Monaco', 'Monza', 'Hairpin']
    row_data = [
        ['Length (m)']     + [f"{c['track_length']:.0f}" for c in circuits],
        ['Laps']           + [str(c['data']['n_laps']) for c in circuits],
        ['Race dist. (km)']+ [f"{c['data']['n_laps']*c['track_length']/1000:.1f}" for c in circuits],
        ['Q* (kWh)']       + [f"{c['data']['Q_batt_kWh']:.1f}" for c in circuits],
        ['Mass (kg)']      + [f"{c['data']['mass_kg']:.0f}" for c in circuits],
        ['T_lap avg (s)']  + [f"{np.mean(c['data']['lap_times_s']):.1f}" for c in circuits],
        ['E/lap avg (Wh)'] + [f"{np.mean(c['data']['lap_energy_Wh']):.0f}" for c in circuits],
        ['Race time (min)']+ [f"{c['data']['total_time_min']:.1f}" for c in circuits],
        ['SoC final (%)']  + [f"{c['data']['final_SoC']*100:.1f}" for c in circuits],
    ]

    table = ax.table(
        cellText=row_data,
        colLabels=col_headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Colour header row and first column
    header_color = '#2c3e50'
    for j in range(len(col_headers)):
        table[0, j].set_facecolor(header_color)
        table[0, j].set_text_props(color='white', fontweight='bold')
    circuit_colors = ['white'] + [c['color'] for c in circuits]
    for i in range(1, len(row_data) + 1):
        for j, col in enumerate(circuit_colors):
            if j > 0:
                table[i, j].set_facecolor(col + '22')  # light tint
        table[i, 0].set_text_props(fontweight='bold')
        table[i, 0].set_facecolor('#ecf0f1')

    ax.set_title('Summary', fontsize=10, pad=10)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from pathlib import Path
    proj_root = str(Path(__file__).parent.parent.parent)

    print("=" * 60)
    print("  Circuit Comparison Figure")
    print("=" * 60)

    print("\n[1/3] Loading race results...")
    circuits = load_results(proj_root)

    print("\n[2/3] Computing speed profiles...")
    circuits = load_speed_profiles(circuits)

    print("\n[3/3] Plotting...")
    out = os.path.join(proj_root, 'figures', 'comparison_circuits.png')
    plot_comparison(circuits, output_path=out)

    print("\nDone.")


if __name__ == '__main__':
    main()
