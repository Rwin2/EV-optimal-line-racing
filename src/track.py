"""
Track generation module.
Generates closed 2D racing circuits with inner/outer boundaries.
Supports multiple track shapes: oval, figure-8, complex circuits with hairpins/chicanes.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass


@dataclass
class Track:
    """A closed 2D racing circuit."""
    centerline: np.ndarray    # (N, 2) center points
    left_boundary: np.ndarray  # (N, 2) left boundary
    right_boundary: np.ndarray # (N, 2) right boundary
    widths: np.ndarray         # (N,) half-width at each point
    normals: np.ndarray        # (N, 2) outward normal at each point
    s: np.ndarray              # (N,) arc-length parameter
    length: float              # total track length
    name: str = "Track"

    def get_curvature(self):
        """Compute signed curvature at each centerline point."""
        dx = np.gradient(self.centerline[:, 0])
        dy = np.gradient(self.centerline[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        return curvature

    def point_in_track(self, x, y):
        """Check if a point is inside track boundaries (approximate)."""
        # Find nearest centerline point
        dists = np.sqrt((self.centerline[:, 0] - x)**2 + (self.centerline[:, 1] - y)**2)
        idx = np.argmin(dists)
        # Project onto normal direction
        to_point = np.array([x - self.centerline[idx, 0], y - self.centerline[idx, 1]])
        normal_dist = abs(np.dot(to_point, self.normals[idx]))
        return normal_dist <= self.widths[idx]

    def nearest_centerline_info(self, x, y):
        """Get nearest centerline index, lateral offset, and arc-length."""
        dists = np.sqrt((self.centerline[:, 0] - x)**2 + (self.centerline[:, 1] - y)**2)
        idx = np.argmin(dists)
        to_point = np.array([x - self.centerline[idx, 0], y - self.centerline[idx, 1]])
        lateral = np.dot(to_point, self.normals[idx])
        return idx, lateral, self.s[idx]


def _smooth_closed_curve(control_points, n_points=500):
    """Create a smooth closed curve from control points using periodic cubic spline."""
    # Close the curve
    pts = np.vstack([control_points, control_points[0:1]])
    t = np.zeros(len(pts))
    for i in range(1, len(pts)):
        t[i] = t[i-1] + np.linalg.norm(pts[i] - pts[i-1])
    t /= t[-1]  # normalize to [0, 1]

    cs_x = CubicSpline(t, pts[:, 0], bc_type='periodic')
    cs_y = CubicSpline(t, pts[:, 1], bc_type='periodic')

    t_fine = np.linspace(0, 1, n_points, endpoint=False)
    x = cs_x(t_fine)
    y = cs_y(t_fine)
    return np.column_stack([x, y])


def _compute_track_geometry(centerline, half_width):
    """Compute normals, boundaries, arc-length from a centerline."""
    n = len(centerline)
    # Tangent vectors (periodic)
    tangents = np.zeros_like(centerline)
    for i in range(n):
        tangents[i] = centerline[(i+1) % n] - centerline[(i-1) % n]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    tangents = tangents / norms

    # Outward normals (rotate tangent 90 degrees left)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Handle variable or constant width
    if isinstance(half_width, (int, float)):
        widths = np.full(n, half_width)
    else:
        widths = half_width

    left_boundary = centerline + normals * widths[:, np.newaxis]
    right_boundary = centerline - normals * widths[:, np.newaxis]

    # Arc-length
    ds = np.linalg.norm(np.diff(centerline, axis=0, append=centerline[0:1]), axis=1)
    s = np.cumsum(ds)
    s = np.roll(s, 1)
    s[0] = 0.0
    length = s[-1] + ds[-1]

    return left_boundary, right_boundary, widths, normals, s, length


def generate_oval(n_points=500, length=400, width=150, track_width=12):
    """Generate an oval/elliptical circuit."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = length/2 * np.cos(t)
    y = width/2 * np.sin(t)
    centerline = np.column_stack([x, y])
    left, right, widths, normals, s, total_len = _compute_track_geometry(centerline, track_width)
    return Track(centerline, left, right, widths, normals, s, total_len, "Oval")


def generate_figure_eight(n_points=500, scale=200, track_width=10):
    """Generate a figure-eight shaped track (lemniscate of Bernoulli)."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    denom = 1 + np.sin(t)**2
    x = scale * np.cos(t) / denom
    y = scale * np.sin(t) * np.cos(t) / denom
    centerline = np.column_stack([x, y])
    left, right, widths, normals, s, total_len = _compute_track_geometry(centerline, track_width)
    return Track(centerline, left, right, widths, normals, s, total_len, "Figure Eight")


def generate_complex_circuit(n_points=600, track_width=10, seed=42):
    """Generate a complex circuit with varying features (straights, hairpins, chicanes)."""
    control_points = np.array([
        [0, 0],
        [80, 10],
        [160, 5],
        [220, -20],
        [260, -80],
        [240, -150],
        [180, -180],
        [120, -160],
        [80, -200],
        [40, -260],
        [-20, -280],
        [-80, -240],
        [-100, -180],
        [-120, -120],
        [-160, -80],
        [-200, -40],
        [-180, 20],
        [-120, 50],
        [-60, 30],
    ])
    centerline = _smooth_closed_curve(control_points, n_points)
    # Vary track width: narrower in tight corners
    curvature = np.zeros(n_points)
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    speed = np.sqrt(dx**2 + dy**2)
    speed[speed < 1e-10] = 1e-10
    curvature = np.abs(dx * ddy - dy * ddx) / speed**3
    # Width varies: wider on straights, narrower on curves
    max_curv = np.percentile(curvature, 95)
    curv_norm = np.clip(curvature / max_curv, 0, 1)
    half_widths = track_width * (1.0 - 0.3 * curv_norm)

    left, right, widths, normals, s, total_len = _compute_track_geometry(centerline, half_widths)
    return Track(centerline, left, right, widths, normals, s, total_len, "Grand Prix Circuit")


def generate_hairpin_chicane(n_points=600, track_width=14):
    """Track with exaggerated hairpins and chicanes: long straights into very tight turns."""
    control_points = np.array([
        # long straight
        [0, 0],
        [100, 0],
        [200, 0],
        # tight hairpin right
        [240, -10],
        [250, -40],
        [230, -60],
        [200, -50],
        # straight back
        [150, -55],
        [50, -60],
        # chicane (quick S)
        [20, -70],
        [0, -100],
        [20, -130],
        # another straight
        [50, -140],
        [150, -145],
        # very tight hairpin left
        [190, -155],
        [200, -190],
        [180, -210],
        [150, -200],
        # long return straight
        [100, -195],
        [0, -180],
        [-50, -150],
        # wide sweeper back to start
        [-70, -100],
        [-60, -50],
        [-40, -20],
    ])
    centerline = _smooth_closed_curve(control_points, n_points)
    left, right, widths, normals, s, total_len = _compute_track_geometry(centerline, track_width)
    return Track(centerline, left, right, widths, normals, s, total_len, "Hairpin & Chicane Circuit")


def generate_monaco_style(n_points=800, track_width=18):
    """Sharp-cornered circuit: long straights into tight 90-degree bends, wide track."""
    control_points = np.array([
        # bottom straight
        [0, 0],
        [100, 0],
        [200, 0],
        # sharp right turn
        [250, 0],
        [260, -10],
        [260, -30],
        # right straight going up
        [260, -80],
        [260, -140],
        # sharp right + chicane at top
        [260, -180],
        [250, -195],
        [230, -200],
        [200, -195],
        [180, -200],
        [160, -195],
        # top straight going left
        [120, -190],
        [60, -190],
        # sharp left turn down
        [10, -190],
        [0, -180],
        [0, -160],
        # left straight going down
        [0, -120],
        [0, -60],
        # back to start
        [0, -20],
    ])
    centerline = _smooth_closed_curve(control_points, n_points)
    left, right, widths, normals, s, total_len = _compute_track_geometry(centerline, track_width)
    return Track(centerline, left, right, widths, normals, s, total_len, "Sharp Corner Circuit")


def generate_monza_style(n_points=600, track_width=11):
    """Generate a Monza-inspired circuit: long straights with tight chicanes."""
    control_points = np.array([
        [0, 0],
        [100, 5],
        [200, 0],
        [280, -10],
        [300, -40],
        [280, -70],
        [250, -80],
        [220, -70],
        [200, -85],
        [210, -120],
        [250, -150],
        [260, -200],
        [230, -240],
        [180, -250],
        [120, -230],
        [60, -240],
        [20, -220],
        [0, -180],
        [-20, -120],
        [-40, -60],
        [-30, -20],
    ])
    centerline = _smooth_closed_curve(control_points, n_points)
    left, right, widths, normals, s, total_len = _compute_track_geometry(centerline, track_width)
    return Track(centerline, left, right, widths, normals, s, total_len, "Monza-Style Circuit")


# Registry of available tracks
TRACK_GENERATORS = {
    "oval": generate_oval,
    "figure_eight": generate_figure_eight,
    "complex": generate_complex_circuit,
    "hairpin": generate_hairpin_chicane,
    "monaco": generate_monaco_style,
    "monza": generate_monza_style,
}


def get_track(name="complex", **kwargs):
    """Get a track by name."""
    if name not in TRACK_GENERATORS:
        raise ValueError(f"Unknown track: {name}. Available: {list(TRACK_GENERATORS.keys())}")
    return TRACK_GENERATORS[name](**kwargs)
