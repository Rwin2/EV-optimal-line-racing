"""Generate a 4-panel circuit overview figure for the appendix."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_this = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_this), 'single lap'))
from track import get_track

CIRCUITS = [
    ('complex', 'Grand Prix Circuit (complex)', 1288, 51, 'Straights, hairpin, chicane'),
    ('monza',   'Monza-Style Circuit (monza)',  1115, 58, 'High-speed straights, chicanes'),
    ('hairpin', 'Hairpin Circuit (hairpin)',     1269, 51, 'Tight hairpin, slow sections'),
    ('monaco',  'Monaco Circuit (monaco)',        903, 72, 'Sharp corners, short lap'),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.subplots_adjust(hspace=0.35, wspace=0.25)

for ax, (name, label, length, laps, desc) in zip(axes.flat, CIRCUITS):
    t = get_track(name)
    cl = t.centerline
    lb = t.left_boundary
    rb = t.right_boundary

    ax.fill(np.append(lb[:, 0], rb[::-1, 0]),
            np.append(lb[:, 1], rb[::-1, 1]),
            color='#dcdcdc', zorder=1)
    ax.plot(lb[:, 0], lb[:, 1], 'k-', lw=0.8, zorder=2)
    ax.plot(rb[:, 0], rb[:, 1], 'k-', lw=0.8, zorder=2)
    ax.plot(cl[:, 0], cl[:, 1], '--', color='#e74c3c', lw=1.2, zorder=3)

    # Start/finish marker
    ax.plot(cl[0, 0], cl[0, 1], 's', color='#2ecc71', ms=7, zorder=4)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{label}\n{length} m · {laps} laps · {desc}',
                 fontsize=9, pad=6, linespacing=1.5)

PROJ_ROOT = os.path.dirname(os.path.dirname(_this))
out = os.path.join(PROJ_ROOT, 'figures', 'report', 'circuits_overview.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved -> {out}')
