"""
Full analysis pipeline — scipy SLSQP only.

1. Joint (α, v) optimization → minimum lap time
2. Pareto front: J = (1-w)*T/T_ref + w*E/E_ref  (Alg 15.3)
3. Robust Pareto front under uncertain grip μ (Chapter 20 — SAA)
4. Generate all figures

Uses multiprocessing.Pool for parallelism.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.optimize import minimize as scipy_minimize

sys.path.insert(0, os.path.dirname(__file__))

from track import get_track
from car import CarParams
from optimizer import (alpha_to_raceline, compute_curvature_from_path,
                       compute_velocity_profile, compute_energy)

TRACK = "monza"
N_STATIONS = 40
N_PARETO = 10
N_MC = 20           # Monte Carlo samples for SAA
MU_SIGMA = 0.10     # σ for uncertain grip μ ~ N(μ_nom, σ²)
N_WORKERS = 8
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ── Shared helpers ───────────────────────────────────────────────────────

def subsample_track(trk, n):
    n_full = len(trk.centerline)
    idx = np.linspace(0, n_full - 1, n, dtype=int)
    return trk.centerline[idx], trk.normals[idx], trk.widths[idx]


def warm_start_alpha(centerline, normals, widths, n):
    """Minimize curvature to get a good initial path."""
    bounds = [(-w, w) for w in widths]
    def curv_obj(a):
        path = alpha_to_raceline(a, centerline, normals)
        k = compute_curvature_from_path(path)
        da = np.diff(a, append=a[0])
        return np.sum(k**2) + 1e-5 * np.sum(da**2)
    res = scipy_minimize(curv_obj, np.zeros(n), method='SLSQP',
                         bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
    return res.x


def build_constraints(n, centerline, normals, car):
    """Build scipy constraint dicts for joint (α, v) optimization."""
    g = 9.81
    a_lat = car.mu * g * 0.85
    a_lon = min(car.F_drive_max / car.mass, car.mu * g * 0.6)
    a_brk = min(car.F_brake_max / car.mass, car.mu * g * 0.9)

    def cornering(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        kappa = compute_curvature_from_path(path)
        return a_lat - v**2 * np.abs(kappa)

    def accel(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        v_next = np.roll(v, -1)
        return v**2 + 2*a_lon*ds - v_next**2

    def brake(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        v_next = np.roll(v, -1)
        return v_next**2 + 2*a_brk*ds - v**2

    return [
        {'type': 'ineq', 'fun': cornering},
        {'type': 'ineq', 'fun': accel},
        {'type': 'ineq', 'fun': brake},
    ]


def build_constraints_mu(n, centerline, normals, car, mu_val):
    """Build constraints for a specific μ value (for SAA)."""
    g = 9.81
    a_lat = mu_val * g * 0.85
    a_lon = min(car.F_drive_max / car.mass, mu_val * g * 0.6)
    a_brk = min(car.F_brake_max / car.mass, mu_val * g * 0.9)

    def cornering(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        kappa = compute_curvature_from_path(path)
        return a_lat - v**2 * np.abs(kappa)

    def accel(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        v_next = np.roll(v, -1)
        return v**2 + 2*a_lon*ds - v_next**2

    def brake(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        v_next = np.roll(v, -1)
        return v_next**2 + 2*a_brk*ds - v**2

    return [
        {'type': 'ineq', 'fun': cornering},
        {'type': 'ineq', 'fun': accel},
        {'type': 'ineq', 'fun': brake},
    ]


# ── Level 1: Joint (α, v) time-only optimization ────────────────────────

def solve_joint_time(centerline, normals, widths, car, alpha0):
    """Scipy SLSQP joint (α, v) → minimum lap time."""
    n = len(centerline)

    v0 = compute_velocity_profile(
        alpha_to_raceline(alpha0, centerline, normals),
        compute_curvature_from_path(alpha_to_raceline(alpha0, centerline, normals)),
        car)
    x0 = np.concatenate([alpha0, v0])

    bounds = [(-w, w) for w in widths] + [(1.0, car.v_max)] * n

    def time_obj(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        return np.sum(ds / np.maximum(v, 0.5))

    constraints = build_constraints(n, centerline, normals, car)

    res = scipy_minimize(time_obj, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})
    return res


# ── Level 2: Pareto front (nominal) ─────────────────────────────────────

def _solve_pareto_point(args):
    """Solve one Pareto point. Designed for multiprocessing.Pool."""
    w, x0, bounds, n, centerline, normals, widths, car_dict, T_ref, E_ref = args
    car = CarParams(**car_dict)

    def pareto_obj(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        T = np.sum(ds / np.maximum(v, 0.5))
        e = compute_energy(path, v, car, ds)
        E = e['net_energy_kJ'] * 1000
        return (1 - w) * T / T_ref + w * E / E_ref

    constraints = build_constraints(n, centerline, normals, car)
    res = scipy_minimize(pareto_obj, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})

    alpha_opt, v_opt = res.x[:n], res.x[n:]
    path = alpha_to_raceline(alpha_opt, centerline, normals)
    ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
    T = np.sum(ds / np.maximum(v_opt, 0.5))
    e = compute_energy(path, v_opt, car, ds)
    E = e['net_energy_kJ'] * 1000
    return w, T, E, v_opt, alpha_opt


def run_pareto(centerline, normals, widths, car, x0_time, T_ref, E_ref):
    """Pareto sweep with multiprocessing."""
    n = len(centerline)
    bounds = [(-w, w) for w in widths] + [(1.0, car.v_max)] * n
    weights = np.linspace(0, 0.95, N_PARETO)
    car_dict = {f.name: getattr(car, f.name) for f in car.__dataclass_fields__.values()}

    args_list = [(w, x0_time, bounds, n, centerline, normals, widths, car_dict, T_ref, E_ref)
                 for w in weights]

    with Pool(N_WORKERS) as pool:
        results = pool.map(_solve_pareto_point, args_list)

    results.sort(key=lambda r: r[0])
    return results


# ── Level 3: Robust Pareto front (SAA, Chapter 20) ──────────────────────
#
# Uncertain parameter: tire grip μ varies per station along the track.
# Each MC scenario k draws μ_i^(k) ~ N(μ_nom, σ²) independently at every
# station i.  This models realistic grip variation (wet patches, dirt, wear).
#
# SAA formulation (Alg 20.3):
#   Objective:  (1/N) Σ_k  J(x, μ^(k))   — average over scenarios
#   Constraints: feasible for ALL scenarios — cornering at station i
#                must hold for μ_i^(k) for every k.
#                Equivalently: use μ_i_worst = min_k(μ_i^(k)) per station.

def _build_saa_constraints(n, centerline, normals, car, mu_scenarios):
    """Build constraints using per-station worst-case μ across all MC scenarios.

    mu_scenarios: (N_MC, n) — each row is one scenario's μ vector.
    At each station i, we use μ_i_worst = min over scenarios.
    """
    g = 9.81
    mu_worst = np.min(mu_scenarios, axis=0)  # (n,) per-station worst-case

    a_lat = mu_worst * g * 0.85             # (n,) per-station lateral limit
    a_lon_global = min(car.F_drive_max / car.mass, np.min(mu_worst) * g * 0.6)
    a_brk_global = min(car.F_brake_max / car.mass, np.min(mu_worst) * g * 0.9)

    def cornering(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        kappa = compute_curvature_from_path(path)
        return a_lat - v**2 * np.abs(kappa)  # per-station μ

    def accel(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        v_next = np.roll(v, -1)
        return v**2 + 2*a_lon_global*ds - v_next**2

    def brake(x):
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        v_next = np.roll(v, -1)
        return v_next**2 + 2*a_brk_global*ds - v**2

    return [
        {'type': 'ineq', 'fun': cornering},
        {'type': 'ineq', 'fun': accel},
        {'type': 'ineq', 'fun': brake},
    ]


def _solve_robust_pareto_point(args):
    """Solve one robust Pareto point using SAA with per-station uncertain μ."""
    (w, x0, bounds, n, centerline, normals, widths,
     car_dict, T_ref, E_ref, mu_scenarios) = args
    car_nom = CarParams(**car_dict)
    N = len(mu_scenarios)

    def robust_obj(x):
        """SAA objective: average J over all MC scenarios."""
        alpha, v = x[:n], x[n:]
        path = alpha_to_raceline(alpha, centerline, normals)
        ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
        T = np.sum(ds / np.maximum(v, 0.5))
        # Energy depends on car params — use nominal for objective
        e = compute_energy(path, v, car_nom, ds)
        E = e['net_energy_kJ'] * 1000
        # Average over scenarios: T is the same (path/speed don't change),
        # but the "feasibility cost" is embedded in constraints.
        # For the objective, we keep the nominal T and E.
        return (1 - w) * T / T_ref + w * E / E_ref

    constraints = _build_saa_constraints(n, centerline, normals, car_nom, mu_scenarios)

    res = scipy_minimize(robust_obj, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})

    alpha_opt, v_opt = res.x[:n], res.x[n:]
    path = alpha_to_raceline(alpha_opt, centerline, normals)
    ds = np.linalg.norm(np.diff(path, axis=0, append=path[0:1]), axis=1)
    T = np.sum(ds / np.maximum(v_opt, 0.5))
    e = compute_energy(path, v_opt, car_nom, ds)
    E = e['net_energy_kJ'] * 1000
    return w, T, E, v_opt, alpha_opt


def run_robust_pareto(centerline, normals, widths, car, x0_time, T_ref, E_ref, seed=42):
    """Robust Pareto sweep using SAA (Chapter 20) with per-station μ."""
    n = len(centerline)
    bounds = [(-w, w) for w in widths] + [(1.0, car.v_max)] * n
    weights = np.linspace(0, 0.95, N_PARETO)
    car_dict = {f.name: getattr(car, f.name) for f in car.__dataclass_fields__.values()}

    # Draw N_MC scenarios, each with n independent μ values
    rng = np.random.default_rng(seed)
    mu_scenarios = rng.normal(car.mu, MU_SIGMA, size=(N_MC, n))
    mu_scenarios = np.clip(mu_scenarios, 0.3, 1.5)

    args_list = [(w, x0_time, bounds, n, centerline, normals, widths, car_dict,
                  T_ref, E_ref, mu_scenarios) for w in weights]

    with Pool(N_WORKERS) as pool:
        results = pool.map(_solve_robust_pareto_point, args_list)

    results.sort(key=lambda r: r[0])
    return results, mu_scenarios


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_raceline(trk, alpha_nom, alpha_rob, centerline_sub, normals_sub, save_path):
    """Plot nominal vs robust racelines on the track."""
    rl_nom = alpha_to_raceline(alpha_nom, centerline_sub, normals_sub)
    rl_rob = alpha_to_raceline(alpha_rob, centerline_sub, normals_sub)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.plot(trk.left_boundary[:, 0], trk.left_boundary[:, 1], 'k-', lw=1.5)
    ax.plot(trk.right_boundary[:, 0], trk.right_boundary[:, 1], 'k-', lw=1.5)
    ax.plot(trk.centerline[:, 0], trk.centerline[:, 1], '--', color='0.7', lw=0.8, label='Centerline')
    ax.plot(rl_nom[:, 0], rl_nom[:, 1], '-', color='#1f77b4', lw=2.5, label='Nominal raceline')
    ax.plot(rl_rob[:, 0], rl_rob[:, 1], '--', color='#d62728', lw=2.5, label='Robust raceline (SAA)')
    ax.set_aspect('equal')
    ax.set_title(f'{trk.name} — Nominal vs Robust Racing Lines', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pareto_comparison(nom_results, rob_results, save_path):
    """Plot nominal vs robust Pareto fronts."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    T_nom = np.array([r[1] for r in nom_results])
    E_nom = np.array([r[2] / 1e6 for r in nom_results])
    T_rob = np.array([r[1] for r in rob_results])
    E_rob = np.array([r[2] / 1e6 for r in rob_results])

    # Clip to reasonable range (exclude pathological outliers)
    T_max = max(T_nom.max(), np.percentile(T_rob, 90)) * 1.2
    mask_rob = T_rob <= T_max

    ax.plot(T_nom, E_nom, 'o-', color='#1f77b4', lw=2, ms=7, label='Nominal (deterministic)')
    ax.plot(T_rob[mask_rob], E_rob[mask_rob], 's--', color='#d62728', lw=2, ms=7,
            label=f'Robust (SAA, N={N_MC}, σ_μ={MU_SIGMA})')

    ax.set_xlabel('Lap Time (s)', fontweight='bold')
    ax.set_ylabel('Energy (MJ)', fontweight='bold')
    ax.set_title('Pareto Front: Nominal vs Robust Optimization', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_speed_profiles(nom_results, rob_results, ds, save_path):
    """Plot speed profiles for w=0 (pure time) and w=1 (pure energy) for both nom/rob."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    s = np.cumsum(ds)

    for ax, idx, title in [(axes[0], 0, 'Pure Time (w=0)'), (axes[1], -1, 'Pure Energy (w=1)')]:
        v_nom = nom_results[idx][3]
        v_rob = rob_results[idx][3]
        ax.plot(s, v_nom * 3.6, '-', color='#1f77b4', lw=1.5, label='Nominal')
        ax.plot(s, v_rob * 3.6, '--', color='#d62728', lw=1.5, label='Robust')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Speed Profiles: Nominal vs Robust', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EV Racing Line — Scipy-Only Pipeline")
    print("=" * 60)

    trk = get_track(TRACK)
    print(f"Track: {trk.name} | {trk.length:.0f}m | {len(trk.centerline)} pts")

    centerline, normals, widths = subsample_track(trk, N_STATIONS)
    n = N_STATIONS
    car = CarParams()
    print(f"Subsampled to {n} stations. Workers: {N_WORKERS}\n")

    # ── 1. Warm-start: curvature minimization ──
    print("[1] Warm-start: curvature minimization...")
    t0 = time.time()
    alpha0 = warm_start_alpha(centerline, normals, widths, n)
    print(f"    Done in {time.time()-t0:.1f}s\n")

    # ── 2. Joint time optimization ──
    print("[2] Joint (α, v) time optimization (scipy SLSQP)...")
    t0 = time.time()
    res_time = solve_joint_time(centerline, normals, widths, car, alpha0)
    print(f"    T = {res_time.fun:.2f}s  iters={res_time.nit}  success={res_time.success}")
    print(f"    Done in {time.time()-t0:.1f}s\n")

    x0_time = res_time.x
    alpha_opt = x0_time[:n]
    v_opt = x0_time[n:]

    # Compute reference values for Pareto normalization
    path_opt = alpha_to_raceline(alpha_opt, centerline, normals)
    ds_opt = np.linalg.norm(np.diff(path_opt, axis=0, append=path_opt[0:1]), axis=1)
    T_ref = np.sum(ds_opt / np.maximum(v_opt, 0.5))
    e_ref = compute_energy(path_opt, v_opt, car, ds_opt)
    E_ref = max(abs(e_ref['net_energy_kJ'] * 1000), 1.0)

    print(f"    Reference: T_ref={T_ref:.2f}s  E_ref={E_ref/1e6:.4f}MJ\n")

    # ── 3. Nominal Pareto front ──
    print(f"[3] Nominal Pareto front ({N_PARETO} points, {N_WORKERS} workers)...")
    t0 = time.time()
    nom_results = run_pareto(centerline, normals, widths, car, x0_time, T_ref, E_ref)
    print(f"    Done in {time.time()-t0:.1f}s")
    for w, T, E, _, _ in nom_results:
        print(f"    w={w:.2f}: T={T:.1f}s  E={E/1e6:.4f}MJ")
    print()

    # ── 4. Robust Pareto front (SAA) ──
    print(f"[4] Robust Pareto front (SAA, N_MC={N_MC}, σ_μ={MU_SIGMA})...")
    t0 = time.time()
    rob_results, mu_scenarios = run_robust_pareto(centerline, normals, widths, car,
                                                   x0_time, T_ref, E_ref)
    print(f"    Done in {time.time()-t0:.1f}s")
    for w, T, E, _, _ in rob_results:
        print(f"    w={w:.2f}: T={T:.1f}s  E={E/1e6:.4f}MJ")
    print()

    # ── 5. Figures ──
    print("[5] Generating figures...")
    alpha_rob_w0 = rob_results[0][4]  # robust raceline at w=0 (pure time)
    v_rob_w0 = rob_results[0][3]      # robust speed profile at w=0
    tag = TRACK  # include track name in filenames
    plot_raceline(trk, alpha_opt, alpha_rob_w0, centerline, normals,
                  os.path.join(FIG_DIR, f'raceline_nom_vs_robust_{tag}.png'))
    print(f"    Saved: raceline_nom_vs_robust_{tag}.png")

    plot_pareto_comparison(nom_results, rob_results,
                           os.path.join(FIG_DIR, f'pareto_nominal_vs_robust_{tag}.png'))
    print(f"    Saved: pareto_nominal_vs_robust_{tag}.png")

    plot_speed_profiles(nom_results, rob_results, ds_opt,
                        os.path.join(FIG_DIR, f'speed_profiles_nom_vs_robust_{tag}.png'))
    print(f"    Saved: speed_profiles_nom_vs_robust_{tag}.png")

    # ── 6. Save results for simulation ──
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(os.path.join(cache_dir, f'optimization_results_{tag}.npz'),
             centerline=centerline, normals=normals, widths=widths,
             alpha_nom=alpha_opt, v_nom=v_opt,
             alpha_rob=alpha_rob_w0, v_rob=v_rob_w0,
             mu_scenarios=mu_scenarios)
    print(f"    Saved: cache/optimization_results.npz")

    print("\nDone!")


if __name__ == '__main__':
    main()
