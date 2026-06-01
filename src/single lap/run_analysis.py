"""
Single-lap analysis pipeline:
  1. Vanilla SCP → optimal raceline (time only)
  2. Convergence comparison: SCP vs scipy SLSQP on the same objective
  3. Generate convergence figure
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from track import get_track
from car import CarParams
from optimizer import (alpha_to_raceline, compute_curvature_from_path,
                       compute_velocity_profile, solve_scp)

TRACK = "monza"
N_STATIONS = 80   # SCP working resolution
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run_vanilla_scp(trk):
    """Run SCP at N_STATIONS and return per-iteration objective history."""
    n_full = len(trk.centerline)
    idx = np.linspace(0, n_full - 1, N_STATIONS, dtype=int)
    centerline = trk.centerline[idx]
    normals = trk.normals[idx]
    widths = trk.widths[idx]
    car = CarParams()

    # Warm-start from curvature
    from scipy.optimize import minimize as scipy_minimize
    bounds = [(-w, w) for w in widths]
    def curv_obj(a):
        path = alpha_to_raceline(a, centerline, normals)
        k = compute_curvature_from_path(path)
        da = np.diff(a, append=a[0])
        return np.sum(k**2) + 1e-5 * np.sum(da**2)
    res_warm = scipy_minimize(curv_obj, np.zeros(N_STATIONS), method='SLSQP',
                              bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
    alpha0 = res_warm.x

    print("[1] Running vanilla SCP...")
    alpha, v, hist = solve_scp(centerline, normals, widths, car, alpha0=alpha0,
                                rho=3.0, eps=1e-2, max_iters=10)

    rl = alpha_to_raceline(alpha, centerline, normals)
    kappa = compute_curvature_from_path(rl)
    ds = np.linalg.norm(np.diff(rl, axis=0, append=rl[0:1]), axis=1)

    return alpha, v, hist, rl, kappa, ds, centerline, normals, widths, car, alpha0


def run_scipy_comparison(centerline, normals, widths, car, alpha0_scp):
    """Run scipy SLSQP on the same joint time objective for convergence comparison."""
    from scipy.optimize import minimize as scipy_minimize

    print("\n[2] Running scipy SLSQP on same time objective...")
    n = len(centerline)

    history_scipy = []

    def joint_time_obj(x):
        alpha = x[:n]
        v = x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        return float(np.sum(ds / np.maximum(v, 0.5)))

    def scipy_iter_callback(x):
        alpha = x[:n]
        v = x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        history_scipy.append(float(np.sum(ds / np.maximum(v, 0.5))))

    v0 = compute_velocity_profile(
        alpha_to_raceline(alpha0_scp, centerline, normals),
        compute_curvature_from_path(alpha_to_raceline(alpha0_scp, centerline, normals)),
        car)
    x0 = np.concatenate([alpha0_scp, v0])

    bounds_alpha = [(-w, w) for w in widths]
    bounds_v = [(1.0, car.v_max)] * n
    bounds_all = bounds_alpha + bounds_v

    g = 9.81
    a_lat = car.mu * g * 0.85
    a_lon = min(car.F_drive_max / car.mass, car.mu * g * 0.6)
    a_brk = min(car.F_brake_max / car.mass, car.mu * g * 0.9)

    def cornering_constraint(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        kappa = compute_curvature_from_path(path)
        return a_lat - v**2 * np.abs(kappa)

    def accel_constraint(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        v_next = np.roll(v, -1)
        return v**2 + 2*a_lon*ds - v_next**2

    def brake_constraint(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        v_next = np.roll(v, -1)
        return v_next**2 + 2*a_brk*ds - v**2

    constraints = [
        {'type': 'ineq', 'fun': cornering_constraint},
        {'type': 'ineq', 'fun': accel_constraint},
        {'type': 'ineq', 'fun': brake_constraint},
    ]

    t0 = time.time()
    res = scipy_minimize(joint_time_obj, x0, method='SLSQP', bounds=bounds_all,
                         constraints=constraints, callback=scipy_iter_callback,
                         options={'maxiter': 200, 'ftol': 1e-10, 'disp': True})
    dt = time.time() - t0
    print(f"  scipy SLSQP: J={res.fun:.2f}s  iters={res.nit}  time={dt:.1f}s  success={res.success}")

    return history_scipy, res


def plot_convergence(scp_hist, scipy_hist, save_path):
    """Plot SCP vs scipy convergence (discretized objective J at N_STATIONS pts)."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

    ax.plot(range(len(scp_hist)), scp_hist, 'o-', color='#d62728', lw=2, ms=6,
            label=f'SCP + Simplex (final: {scp_hist[-1]:.2f}s)')

    if len(scipy_hist) > 0:
        ax.plot(range(len(scipy_hist)), scipy_hist, 's-', color='#1f77b4', lw=2, ms=4,
                label=f'scipy SLSQP (final: {scipy_hist[-1]:.2f}s)')

    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel(f'Objective J ({N_STATIONS}-pt discretization, s)', fontweight='bold')
    ax.set_title(f'Convergence Comparison — {TRACK.capitalize()} Circuit', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("  Convergence Analysis Pipeline")
    print("=" * 60)

    trk = get_track(TRACK)
    print(f"Track: {trk.name} | {trk.length:.0f}m | {len(trk.centerline)} pts\n")

    alpha, v, hist, rl, kappa, ds, cl, normals, widths, car, alpha0 = run_vanilla_scp(trk)
    scipy_hist, scipy_res = run_scipy_comparison(cl, normals, widths, car, alpha0)

    print("\n[3] Generating figures...")
    plot_convergence(hist, scipy_hist, os.path.join(FIG_DIR, 'convergence_scp_vs_scipy.png'))

    print("\nDone!")


if __name__ == '__main__':
    main()
