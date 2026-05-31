"""
EV Racing Simulator
====================
Racing simulator with:
- Realistic track rendering (asphalt, curbs, grass)
- Multiple car simulation with different strategies
- Video/GIF output of races
- Telemetry overlay (speed, SOC, lap time)
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(__file__))

from track import Track
from car import CarState

# ── Color palette ────────────────────────────────────────────────────────
ASPHALT      = '#3a3a3a'
ASPHALT_DARK = '#2c2c2c'
GRASS        = '#2d5a27'
GRASS_LIGHT  = '#3a7a30'
CURB_RED     = '#cc2222'
CURB_WHITE   = '#eeeeee'
LINE_WHITE   = '#cccccc'
BACKGROUND   = '#1a3a15'



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


# ── Simulation ───────────────────────────────────────────────────────

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

    # Lap detection: track nearest centerline index, detect wrap-around
    n_track = len(track.centerline)
    prev_idx = [0] * n_cars
    lap_counts = [0] * n_cars
    lap_times = [[] for _ in range(n_cars)]
    lap_start_step = [0] * n_cars
    warmup_steps = int(3.0 / dt)
    # Track progress: index must advance past 50% before wrapping counts
    past_halfway = [False] * n_cars

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

            # Lap detection: nearest centerline index wraps from high to low
            if step > warmup_steps:
                dists = np.sqrt((track.centerline[:, 0] - s.x)**2 +
                                (track.centerline[:, 1] - s.y)**2)
                curr_idx = np.argmin(dists)
                if curr_idx > n_track * 0.5:
                    past_halfway[i] = True
                if past_halfway[i] and prev_idx[i] > n_track * 0.7 and curr_idx < n_track * 0.3:
                    lap_counts[i] += 1
                    lap_time = (step - lap_start_step[i]) * dt
                    lap_times[i].append(lap_time)
                    lap_start_step[i] = step
                    past_halfway[i] = False
                    print(f"    {controllers[i].__class__.__name__} car {i}: Lap {lap_counts[i]} in {lap_time:.1f}s")
                    if all(lc >= stop_after_laps for lc in lap_counts):
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


# ── Rendering ────────────────────────────────────────────────────────

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
