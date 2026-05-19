"""
Full analysis pipeline:
  1. Vanilla SCP → optimal raceline (time only)
  2. Convergence comparison: SCP vs scipy SLSQP on the same objective
  3. SCP Pareto front: time vs energy tradeoff
  4. Generate all figures
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
                       compute_velocity_profile, compute_energy,
                       solve_scp, solve_scp_pareto)

TRACK = "monaco"
N_STATIONS = 80
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run_vanilla_scp(trk):
    """Run vanilla SCP (time-only) and return subsampled results."""
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

    return alpha, v, hist, rl, kappa, ds, centerline, normals, widths, car


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
        T = np.sum(ds / np.maximum(v, 0.5))
        history_scipy.append(T)
        return T

    # Same warm-start as SCP
    v0 = compute_velocity_profile(
        alpha_to_raceline(alpha0_scp, centerline, normals),
        compute_curvature_from_path(alpha_to_raceline(alpha0_scp, centerline, normals)),
        car)
    x0 = np.concatenate([alpha0_scp, v0])

    half_w = widths
    bounds_alpha = [(-w, w) for w in half_w]
    bounds_v = [(1.0, car.v_max)] * n
    bounds = bounds_alpha + bounds_v

    # Cornering constraint: a_lat - v² |κ| >= 0
    g = 9.81
    a_lat = car.mu * g * 0.85

    def cornering_constraint(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        kappa = compute_curvature_from_path(path)
        return a_lat - v**2 * np.abs(kappa)

    # Accel constraint: v_{i+1}² - v_i² <= 2*a_lon*ds
    a_lon = min(car.F_drive_max / car.mass, car.mu * g * 0.6)
    a_brk = min(car.F_brake_max / car.mass, car.mu * g * 0.9)

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
    res = scipy_minimize(joint_time_obj, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 200, 'ftol': 1e-10, 'disp': True})
    dt = time.time() - t0
    print(f"  scipy SLSQP: T={res.fun:.2f}s  iters={res.nit}  time={dt:.1f}s  success={res.success}")

    return history_scipy, res


def run_scp_pareto(kappa, ds, car, v_scp):
    """Run SCP Pareto sweep."""
    print("\n[3] Running SCP Pareto sweep...")
    T0 = np.sum(ds / np.maximum(v_scp, 0.5))
    e0 = compute_energy(
        np.zeros((len(kappa), 2)),  # dummy path (not used for energy, only v and ds)
        v_scp, car, ds)
    E0 = e0['net_energy_kJ'] * 1000

    weights = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    results = []

    for we in weights:
        v_opt = solve_scp_pareto(kappa, ds, car, v_scp,
                                  w_time=1.0, w_energy=we,
                                  T_ref=T0, E_ref=max(abs(E0), 1.0),
                                  rho=15.0, max_iters=25)
        T = np.sum(ds / np.maximum(v_opt, 0.5))
        # Recompute energy with dummy path
        e = compute_energy(np.zeros((len(kappa), 2)), v_opt, car, ds)
        E = e['net_energy_kJ'] * 1000
        results.append((we, T, E, v_opt))
        print(f"  w_e={we:6.1f}: T={T:.1f}s  E={E/1e6:.3f} MJ")

    return results, T0, E0


def plot_convergence(scp_hist, scipy_hist, save_path):
    """Plot SCP vs scipy convergence."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

    # SCP: hist is a list of J values, one per iteration
    ax.plot(range(len(scp_hist)), scp_hist, 'o-', color='#d62728', lw=2, ms=6,
            label=f'SCP + Simplex (final: {scp_hist[-1]:.2f}s)')

    # scipy: hist is a list of J values, one per function eval
    # Subsample to ~same number of points as SCP for clarity
    n_scipy = len(scipy_hist)
    if n_scipy > 0:
        step = max(1, n_scipy // 50)
        idx = list(range(0, n_scipy, step))
        ax.plot(idx, [scipy_hist[i] for i in idx], 's-', color='#1f77b4', lw=2, ms=4,
                label=f'scipy SLSQP (final: {scipy_hist[-1]:.2f}s)')

    ax.set_xlabel('Iteration / Function evaluation', fontweight='bold')
    ax.set_ylabel('Lap Time (s)', fontweight='bold')
    ax.set_title(f'Convergence Comparison — {TRACK.capitalize()} Circuit', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pareto(pareto_results, T0, E0, save_path):
    """Plot Pareto front."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')

    T_arr = np.array([r[1] for r in pareto_results])
    E_arr = np.array([r[2] for r in pareto_results]) / 1e6
    w_arr = np.array([max(r[0], 1e-6) for r in pareto_results])

    sc = ax.scatter(T_arr[1:], E_arr[1:], c=np.log10(w_arr[1:]),
                    cmap='plasma', s=80, zorder=3, edgecolors='k', linewidths=0.5)
    ax.plot(T_arr, E_arr, '-', color='0.6', lw=1.5, zorder=2)

    # SCP anchor point
    ax.scatter([T_arr[0]], [E_arr[0]], s=200, marker='*', color='#00aa00',
               zorder=4, label=f'SCP min-time ({T_arr[0]:.1f}s)')
    # Most efficient
    ax.scatter([T_arr[-1]], [E_arr[-1]], s=150, marker='s', color='#0066ff',
               zorder=4, label=f'Min energy ({T_arr[-1]:.1f}s)')

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('log10(w_energy)', fontsize=9)
    ax.set_xlabel('Lap Time (s)', fontweight='bold')
    ax.set_ylabel('Energy (MJ)', fontweight='bold')
    ax.set_title('Pareto Frontier: Time vs Energy (SCP + Simplex)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("  Full Analysis Pipeline")
    print("=" * 60)

    trk = get_track(TRACK)
    print(f"Track: {trk.name} | {trk.length:.0f}m | {len(trk.centerline)} pts\n")

    # 1. Vanilla SCP
    alpha, v, hist, rl, kappa, ds, cl, normals, widths, car = run_vanilla_scp(trk)

    # 2. Scipy comparison
    scipy_hist, scipy_res = run_scipy_comparison(cl, normals, widths, car, alpha)

    # 3. SCP Pareto
    pareto, T0, E0 = run_scp_pareto(kappa, ds, car, v)

    # 4. Figures
    print("\n[4] Generating figures...")
    plot_convergence(hist, scipy_hist, os.path.join(FIG_DIR, 'convergence_scp_vs_scipy.png'))
    plot_pareto(pareto, T0, E0, os.path.join(FIG_DIR, 'pareto_scp.png'))

    print("\nDone!")


if __name__ == '__main__':
    main()
