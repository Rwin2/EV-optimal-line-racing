"""
EV Racing Simulator
====================
Professional racing simulator with:
- Realistic track rendering (asphalt, curbs, grass)
- Multiple car simulation with different strategies
- Video/GIF output of races
- Telemetry overlay (speed, SOC, lap time)
- Race results saved to races/ directory with timestamps

Usage:
    python simulator.py                          # default race
    python simulator.py --track complex          # choose track
    python simulator.py --controllers center aggressive eco  # pick strategies
    python simulator.py --time 30 --gif          # 30s race, GIF output
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as pe

_this = os.path.dirname(os.path.abspath(__file__))
_sl   = os.path.join(os.path.dirname(_this), 'single lap')
sys.path.insert(0, _sl)
sys.path.insert(0, _this)

from track import get_track, Track, TRACK_GENERATORS
from car import BicycleModel, CarState, CarParams
from controller import PurePursuitController, ILQRController, generate_racing_line

# ── Color palette ────────────────────────────────────────────────────────
ASPHALT      = '#3a3a3a'
ASPHALT_DARK = '#2c2c2c'
GRASS        = '#2d5a27'
GRASS_LIGHT  = '#3a7a30'
CURB_RED     = '#cc2222'
CURB_WHITE   = '#eeeeee'
LINE_WHITE   = '#cccccc'
BACKGROUND   = '#1a3a15'

RACE_STRATEGIES = {
    'center':     {'name': 'Center-Line',   'line': 'center',       'speed': 'curvature',    'color': '#3399ff', 'ctrl': 'pursuit'},
    'optimal':    {'name': 'Min-LapTime',   'line': 'min_laptime',  'speed': 'aggressive',   'color': '#ff3333', 'ctrl': 'pursuit'},
    'aggressive': {'name': 'Aggressive',    'line': 'min_laptime',  'speed': 'aggressive',   'color': '#ffcc00', 'ctrl': 'pursuit'},
    'eco':        {'name': 'Eco-Save',      'line': 'center',       'speed': 'energy_saving','color': '#33cc66', 'ctrl': 'pursuit'},
    'constant':   {'name': 'Constant-Speed','line': 'center',       'speed': 'constant',     'color': '#cc66ff', 'ctrl': 'pursuit'},
    'ilqr':       {'name': 'TV-LQR',        'line': 'min_laptime',  'speed': 'aggressive',   'color': '#00ff88', 'ctrl': 'ilqr'},
}


# ── Track drawing ────────────────────────────────────────────────────────

def draw_track(ax, track: Track, show_racing_lines=None):
    """Render the track with asphalt, grass, curbs, and road markings."""
    ax.set_facecolor(BACKGROUND)
    cx, cy = track.centerline[:, 0], track.centerline[:, 1]
    pad = 50
    ax.set_xlim(cx.min() - pad, cx.max() + pad)
    ax.set_ylim(cy.min() - pad, cy.max() + pad)

    # Grass background
    ax.add_patch(patches.Rectangle(
        (cx.min()-pad, cy.min()-pad), np.ptp(cx)+2*pad, np.ptp(cy)+2*pad,
        facecolor=GRASS, zorder=0))

    # Grass texture
    rng = np.random.RandomState(42)
    gx = rng.uniform(cx.min()-pad, cx.max()+pad, 1500)
    gy = rng.uniform(cy.min()-pad, cy.max()+pad, 1500)
    ax.scatter(gx, gy, s=0.3, c=GRASS_LIGHT, alpha=0.25, zorder=0.5, marker='.')

    # Runoff area
    w_extra = track.widths[:, np.newaxis] + 5
    runoff_l = track.centerline + track.normals * w_extra
    runoff_r = track.centerline - track.normals * w_extra
    runoff = np.vstack([runoff_l, runoff_r[::-1], runoff_l[0:1]])
    ax.add_patch(patches.Polygon(runoff, closed=True, fc='#555555', ec='none', alpha=0.35, zorder=1))

    # Asphalt
    road = np.vstack([track.left_boundary, track.right_boundary[::-1], track.left_boundary[0:1]])
    ax.add_patch(patches.Polygon(road, closed=True, fc=ASPHALT, ec='none', zorder=2))

    # Curbs (alternating red/white)
    n = len(track.centerline)
    seg = max(1, n // 40)
    for i in range(0, n, seg):
        end = min(i + seg, n)
        c = CURB_RED if (i // seg) % 2 == 0 else CURB_WHITE
        for bnd in [track.left_boundary, track.right_boundary]:
            s = bnd[i:end]
            if len(s) >= 2:
                ax.plot(s[:, 0], s[:, 1], color=c, lw=3.0, solid_capstyle='butt', zorder=3)

    # Dashed center line
    for i in range(0, n, max(1, n//60)):
        end = min(i + max(1, n//120), n)
        if (i // max(1, n//60)) % 2 == 0:
            s = track.centerline[i:end]
            ax.plot(s[:, 0], s[:, 1], color=LINE_WHITE, lw=0.7, alpha=0.25, zorder=3.5)

    # Checkered start/finish
    sf_l, sf_r = track.left_boundary[0], track.right_boundary[0]
    for j in range(8):
        p1 = sf_l + (j/8)*(sf_r - sf_l)
        p2 = sf_l + ((j+1)/8)*(sf_r - sf_l)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='white' if j%2==0 else 'black', lw=3, zorder=4)

    # Racing lines
    if show_racing_lines:
        for rl, color, name in show_racing_lines:
            ax.plot(rl[:, 0], rl[:, 1], color=color, lw=1.5, alpha=0.6,
                    ls='--', zorder=4.5, label=name)

    ax.set_aspect('equal')
    ax.axis('off')


def draw_car(ax, state, color, label=None):
    """Draw car as a rounded rectangle with heading."""
    w, h = 1.8, 4.8
    body = patches.FancyBboxPatch((-h/2, -w/2), h, w, boxstyle="round,pad=0.3",
                                   fc=color, ec='white', lw=0.8, zorder=10)
    t = Affine2D().rotate(state.psi).translate(state.x, state.y) + ax.transData
    body.set_transform(t)
    ax.add_patch(body)
    # Windshield
    ws = patches.FancyBboxPatch((h/6, -w/2.5), h/4, w/1.25, boxstyle="round,pad=0.1",
                                 fc='#111', ec='none', zorder=10.1)
    ws.set_transform(t)
    ax.add_patch(ws)
    if label:
        ax.text(state.x, state.y + w + 3, label, color=color, fontsize=6.5,
                ha='center', va='bottom', fontweight='bold', zorder=12,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])


# ── Simulation ───────────────────────────────────────────────────────────

def simulate_race(track, controllers, car_models, initial_states, dt=0.02, max_time=60.0,
                  stop_after_laps=1):
    """Run race. Stops when first car completes stop_after_laps laps (or max_time).
    Returns state arrays per car + metrics."""
    n_cars = len(controllers)
    max_steps = int(max_time / dt)

    # Dynamic arrays (we may stop early)
    positions_list = [[] for _ in range(n_cars)]
    headings_list  = [[] for _ in range(n_cars)]
    speeds_list    = [[] for _ in range(n_cars)]
    socs_list      = [[] for _ in range(n_cars)]

    current = list(initial_states)

    # Lap detection: track arc-length progress around the circuit
    start_pos = track.centerline[0]
    n_track = len(track.centerline)
    prev_idx = [0] * n_cars
    lap_counts = [0] * n_cars
    lap_times = [[] for _ in range(n_cars)]
    lap_start_step = [0] * n_cars
    # Don't count laps during first few seconds (startup)
    warmup_steps = int(2.0 / dt)

    actual_steps = 0
    race_done = False

    for step in range(max_steps):
        if race_done:
            break

        for i in range(n_cars):
            s = current[i]
            positions_list[i].append([s.x, s.y])
            headings_list[i].append(s.psi)
            speeds_list[i].append(s.vx)
            socs_list[i].append(s.SOC)

            delta, F_drive = controllers[i].control(s)
            current[i] = car_models[i].step(s, delta, F_drive, dt)

            # Lap detection: find nearest centerline index
            if step > warmup_steps:
                dists = np.sqrt((track.centerline[:, 0] - s.x)**2 +
                                (track.centerline[:, 1] - s.y)**2)
                curr_idx = np.argmin(dists)
                # Detect crossing from high index back to low (completing a lap)
                if prev_idx[i] > n_track * 0.75 and curr_idx < n_track * 0.25:
                    lap_counts[i] += 1
                    lap_time = (step - lap_start_step[i]) * dt
                    lap_times[i].append(lap_time)
                    lap_start_step[i] = step
                    print(f"    {controllers[i].__class__.__name__} car {i}: Lap {lap_counts[i]} in {lap_time:.1f}s")
                    if lap_counts[i] >= stop_after_laps:
                        race_done = True
                prev_idx[i] = curr_idx

        actual_steps = step + 1

    # Convert to arrays
    positions = [np.array(p) for p in positions_list]
    headings = [np.array(h) for h in headings_list]
    speeds = [np.array(s) for s in speeds_list]
    socs = [np.array(s) for s in socs_list]

    # Compute metrics
    metrics = []
    for i in range(n_cars):
        dist = np.sum(np.linalg.norm(np.diff(positions[i], axis=0), axis=1))
        metrics.append({
            'distance_m': float(dist),
            'avg_speed_kmh': float(np.mean(speeds[i]) * 3.6),
            'max_speed_kmh': float(np.max(speeds[i]) * 3.6),
            'final_SOC': float(socs[i][-1]),
            'energy_used_pct': float((1.0 - socs[i][-1]) * 100),
            'avg_speed_ms': float(np.mean(speeds[i])),
            'laps_completed': lap_counts[i],
            'lap_times_s': lap_times[i],
        })

    return {
        'positions': positions,
        'headings': headings,
        'speeds': speeds,
        'socs': socs,
        'metrics': metrics,
        'dt': dt,
        'n_steps': actual_steps,
    }


# ── Rendering ────────────────────────────────────────────────────────────

def render_video(track, sim_data, car_names, car_colors,
                 output_path, racing_lines=None, fps=20, fmt='mp4'):
    """Render race video. Uses pre-rendered background for speed."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    positions = sim_data['positions']
    headings = sim_data['headings']
    speeds = sim_data['speeds']
    socs = sim_data['socs']
    dt = sim_data['dt']
    n_steps = sim_data['n_steps']
    n_cars = len(positions)

    # Subsample frames
    frame_skip = max(1, int(1.0 / (fps * dt)))
    frame_indices = list(range(0, n_steps, frame_skip))
    n_frames = len(frame_indices)

    # Create figure
    fig = plt.figure(figsize=(14, 9), dpi=80, facecolor='#111111')
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.02)
    ax_track = fig.add_subplot(gs[0, 0])
    ax_telem = fig.add_subplot(gs[0, 1])

    fig.suptitle(track.name, color='white', fontsize=14,
                 fontweight='bold', fontfamily='monospace', y=0.97)

    # Pre-render track background as image (HUGE speedup)
    print("  Pre-rendering track background...")
    draw_track(ax_track, track, show_racing_lines=racing_lines)

    # Save track background as raster
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(ax_track.bbox)

    # Dynamic elements (cars, telemetry) - will be updated each frame
    car_artists = []
    label_artists = []

    def animate(frame_num):
        idx = frame_indices[frame_num]
        t = idx * dt

        # Restore track background
        fig.canvas.restore_region(bg)

        # Clear old car patches
        for a in car_artists + label_artists:
            a.remove()
        car_artists.clear()
        label_artists.clear()

        # Draw cars
        for i in range(n_cars):
            px, py = positions[i][idx]
            psi = headings[i][idx]
            st = CarState(x=px, y=py, psi=psi)
            w, h = 1.8, 4.8
            body = patches.FancyBboxPatch((-h/2, -w/2), h, w, boxstyle="round,pad=0.3",
                                           fc=car_colors[i], ec='white', lw=0.8, zorder=10)
            tr = Affine2D().rotate(psi).translate(px, py) + ax_track.transData
            body.set_transform(tr)
            ax_track.add_patch(body)
            car_artists.append(body)

            ws = patches.FancyBboxPatch((h/6, -w/2.5), h/4, w/1.25, boxstyle="round,pad=0.1",
                                         fc='#111', ec='none', zorder=10.1)
            ws.set_transform(tr)
            ax_track.add_patch(ws)
            car_artists.append(ws)

            txt = ax_track.text(px, py + w + 3, car_names[i], color=car_colors[i],
                                fontsize=6, ha='center', va='bottom', fontweight='bold',
                                zorder=12, path_effects=[pe.withStroke(linewidth=2, foreground='black')])
            label_artists.append(txt)

        # Telemetry panel
        ax_telem.clear()
        ax_telem.set_facecolor('#1a1a1a')
        ax_telem.set_xlim(0, 1); ax_telem.set_ylim(0, 1)
        ax_telem.axis('off')

        ax_telem.text(0.5, 0.95, f'T: {t:.1f}s', color='white', fontsize=11,
                      ha='center', fontweight='bold', transform=ax_telem.transAxes,
                      fontfamily='monospace')

        for i in range(n_cars):
            y = 0.82 - i * 0.20
            spd = speeds[i][idx] * 3.6
            soc = socs[i][idx]
            ax_telem.text(0.05, y, car_names[i], color=car_colors[i], fontsize=8,
                          fontweight='bold', transform=ax_telem.transAxes, fontfamily='monospace')
            ax_telem.text(0.05, y-0.06, f'{spd:5.0f} km/h', color='#aaa', fontsize=7,
                          transform=ax_telem.transAxes, fontfamily='monospace')
            # SOC bar
            bx, by, bw, bh = 0.05, y-0.12, 0.85, 0.03
            ax_telem.add_patch(patches.Rectangle((bx, by), bw, bh, fc='#333',
                               ec='#555', lw=0.5, transform=ax_telem.transAxes, zorder=5))
            sc = '#33cc33' if soc > 0.3 else '#ffcc00' if soc > 0.1 else '#ff3333'
            ax_telem.add_patch(patches.Rectangle((bx, by), bw*soc, bh, fc=sc,
                               ec='none', transform=ax_telem.transAxes, zorder=6))
            ax_telem.text(bx+bw+0.02, by+bh/2, f'{soc*100:.0f}%', color='#aaa',
                          fontsize=6, va='center', transform=ax_telem.transAxes, fontfamily='monospace')

        if frame_num % 50 == 0:
            print(f"  Frame {frame_num}/{n_frames}")

        return car_artists + label_artists

    print(f"  Rendering {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False)

    if fmt == 'mp4':
        writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    else:
        writer = PillowWriter(fps=fps)

    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def render_static_frame(track, states, car_names, car_colors,
                        racing_lines=None, output_path='frame.png'):
    """Render a single static frame."""
    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='#111111')
    ax = fig.add_subplot(111)

    draw_track(ax, track, show_racing_lines=racing_lines)

    for i, state in enumerate(states):
        draw_car(ax, state, car_colors[i], label=car_names[i])

    ax.set_title(track.name, color='white', fontsize=14,
                 fontweight='bold', fontfamily='monospace', pad=10)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Output management ────────────────────────────────────────────────────

def create_race_output_dir(proj_root):
    """Create timestamped race output directory."""
    races_dir = proj_root / 'races'
    races_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    race_dir = races_dir / f'race_{timestamp}'
    race_dir.mkdir()
    (race_dir / 'video').mkdir()
    (race_dir / 'results').mkdir()
    return race_dir


def save_race_results(race_dir, car_names, controller_keys, metrics, track_name, sim_time):
    """Save per-controller results and overall benchmark."""
    results_dir = race_dir / 'results'

    # Per-controller results
    for i, key in enumerate(controller_keys):
        ctrl_dir = results_dir / key
        ctrl_dir.mkdir(exist_ok=True)
        with open(ctrl_dir / 'metrics.json', 'w') as f:
            json.dump({
                'controller': key,
                'name': car_names[i],
                'track': track_name,
                'sim_time_s': sim_time,
                **metrics[i]
            }, f, indent=2)

    # Overall benchmark
    benchmark = {
        'track': track_name,
        'sim_time_s': sim_time,
        'timestamp': datetime.now().isoformat(),
        'controllers': {}
    }
    for i, key in enumerate(controller_keys):
        benchmark['controllers'][key] = {
            'name': car_names[i],
            **metrics[i]
        }

    # Determine rankings
    by_dist = sorted(range(len(metrics)), key=lambda i: -metrics[i]['distance_m'])
    by_energy = sorted(range(len(metrics)), key=lambda i: metrics[i]['energy_used_pct'])
    benchmark['rankings'] = {
        'fastest_by_distance': controller_keys[by_dist[0]],
        'most_efficient': controller_keys[by_energy[0]],
    }

    with open(results_dir / 'benchmark.json', 'w') as f:
        json.dump(benchmark, f, indent=2)

    # Human-readable summary
    with open(results_dir / 'benchmark.md', 'w') as f:
        f.write(f'# Race Benchmark: {track_name}\n\n')
        f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
        f.write(f'**Sim time:** {sim_time}s\n\n')
        f.write(f'| Controller | Distance (m) | Avg Speed (km/h) | Max Speed (km/h) | Energy Used (%) | Final SOC |\n')
        f.write(f'|------------|-------------|-------------------|-------------------|-----------------|----------|\n')
        for i, key in enumerate(controller_keys):
            m = metrics[i]
            f.write(f"| {car_names[i]} | {m['distance_m']:.0f} | {m['avg_speed_kmh']:.1f} | "
                    f"{m['max_speed_kmh']:.1f} | {m['energy_used_pct']:.2f} | {m['final_SOC']:.3f} |\n")
        f.write(f'\n**Fastest:** {car_names[by_dist[0]]}\n')
        f.write(f'**Most efficient:** {car_names[by_energy[0]]}\n')


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='EV Racing Simulator')
    parser.add_argument('--track', default='complex', choices=list(TRACK_GENERATORS.keys()),
                        help='Track to race on')
    parser.add_argument('--strategies', nargs='+', default=['center', 'optimal', 'aggressive', 'eco'],
                        choices=list(RACE_STRATEGIES.keys()), help='Race strategies to compare')
    parser.add_argument('--time', type=float, default=120.0, help='Max simulation time (seconds)')
    parser.add_argument('--laps', type=int, default=1, help='Stop after N laps')
    parser.add_argument('--dt', type=float, default=0.02, help='Time step')
    parser.add_argument('--gif', action='store_true', help='Output GIF instead of MP4')
    parser.add_argument('--fps', type=int, default=15, help='Video FPS')
    parser.add_argument('--no-video', action='store_true', help='Skip video, only save results')
    parser.add_argument('--figure-only', action='store_true', help='Only generate static figure')
    args = parser.parse_args()

    proj_root = Path(__file__).parent.parent.parent

    print("=" * 60)
    print("  EV Optimal Line Racing Simulator")
    print("=" * 60)

    # Track
    print(f"\n[1/5] Generating track: {args.track}")
    track = get_track(args.track)
    print(f"  {track.name} | {track.length:.0f} m | {len(track.centerline)} pts")

    # Race strategies
    print(f"\n[2/5] Setting up {len(args.strategies)} race strategies...")
    tangent = track.centerline[1] - track.centerline[0]
    heading = np.arctan2(tangent[1], tangent[0])
    params = CarParams()

    controllers, car_models, initial_states = [], [], []
    car_names, car_colors, racing_lines, ctrl_keys = [], [], [], []

    line_cache = {}  # avoid recomputing the same racing line twice
    for i, key in enumerate(args.strategies):
        cfg = RACE_STRATEGIES[key]
        if cfg['line'] not in line_cache:
            line_cache[cfg['line']] = generate_racing_line(track, mode=cfg['line'])
        rl, v_profile = line_cache[cfg['line']]
        if cfg.get('ctrl') == 'ilqr':
            ctrl = ILQRController(track, racing_line=rl, v_profile=v_profile, params=params)
        else:
            ctrl = PurePursuitController(track, racing_line=rl, params=params,
                                         speed_mode=cfg['speed'], v_profile=v_profile)
        controllers.append(ctrl)
        car_models.append(BicycleModel(params))
        start_pos = track.centerline[0] + (i - len(args.strategies)/2) * 2.5 * track.normals[0]
        initial_states.append(CarState(x=start_pos[0], y=start_pos[1], psi=heading, vx=12.0, SOC=1.0))
        car_names.append(cfg['name'])
        car_colors.append(cfg['color'])
        racing_lines.append(rl)
        ctrl_keys.append(key)
        print(f"  [{cfg['name']}] line={cfg['line']}, speed_tracking={cfg['speed']}")

    # Static figure only mode
    if args.figure_only:
        fig_dir = proj_root / 'figures'
        fig_dir.mkdir(exist_ok=True)
        states = [CarState(x=track.centerline[len(track.centerline)//4, 0],
                           y=track.centerline[len(track.centerline)//4, 1],
                           psi=heading)]
        center_rl, _ = generate_racing_line(track, mode="center")
        render_static_frame(track, states, ['EV'], ['#3399ff'],
                            racing_lines=[(center_rl, '#3399ff', 'Racing Line')],
                            output_path=str(fig_dir / 'track_with_raceline.png'))
        print("\nDone! Figure only.")
        return

    # Simulate
    print(f"\n[3/5] Simulating (max {args.time}s, stop after {args.laps} lap(s))...")
    sim_data = simulate_race(track, controllers, car_models, initial_states,
                             dt=args.dt, max_time=args.time, stop_after_laps=args.laps)
    sim_time_actual = sim_data['n_steps'] * args.dt
    print(f"  {sim_data['n_steps']} steps x {len(controllers)} cars ({sim_time_actual:.1f}s)")

    # Results
    print("\n[4/5] Results:")
    print(f"  {'Controller':<16} {'Dist (m)':>9} {'Avg (km/h)':>11} {'Max (km/h)':>11} {'E used (%)':>11} {'SOC':>6}")
    print(f"  {'─'*70}")
    for i, key in enumerate(ctrl_keys):
        m = sim_data['metrics'][i]
        print(f"  {car_names[i]:<16} {m['distance_m']:>8.0f} {m['avg_speed_kmh']:>10.1f} "
              f"{m['max_speed_kmh']:>10.1f} {m['energy_used_pct']:>10.2f} {m['final_SOC']:>6.3f}")

    # Save outputs
    race_dir = create_race_output_dir(proj_root)
    save_race_results(race_dir, car_names, ctrl_keys, sim_data['metrics'], track.name, args.time)
    print(f"\n[5/5] Saving outputs to {race_dir.name}/")

    # Static frame
    fig_dir = proj_root / 'figures'
    fig_dir.mkdir(exist_ok=True)
    frame_idx = sim_data['n_steps'] // 4
    frame_states = [CarState(x=sim_data['positions'][i][frame_idx, 0],
                             y=sim_data['positions'][i][frame_idx, 1],
                             psi=sim_data['headings'][i][frame_idx])
                    for i in range(len(ctrl_keys))]
    rl_display = [(racing_lines[i], car_colors[i], car_names[i]) for i in range(len(ctrl_keys))]
    render_static_frame(track, frame_states, car_names, car_colors,
                        racing_lines=rl_display,
                        output_path=str(fig_dir / 'track_with_raceline.png'))

    # Video/GIF
    if not args.no_video:
        ext = 'gif' if args.gif else 'mp4'
        vid_path = str(race_dir / 'video' / f'race.{ext}')
        render_video(track, sim_data, car_names, car_colors,
                     output_path=vid_path, racing_lines=rl_display,
                     fps=args.fps, fmt=ext)

    print(f"\nDone! Results in: {race_dir}")


if __name__ == '__main__':
    main()
