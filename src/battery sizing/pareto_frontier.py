"""
Compute and visualize Pareto frontier for speed profile optimization.

Strategy - arc-length adaptive refinement:
  Pass 1 - coarse log-sweep (N_COARSE pts) over a wide w_energy range.
  Pass 2+ - iteratively bisect the longest gap in normalized (T, E) space.

Cache:
  Results saved to cache/<key>.npz + cache/<key>.json.
  Key is an MD5 hash of all run parameters, so changing any parameter
  automatically busts the cache.
  Run with --recompute to force a fresh computation.
"""
import os
import sys
import json
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_this = os.path.dirname(os.path.abspath(__file__))
_sl   = os.path.join(os.path.dirname(_this), 'single lap')
sys.path.insert(0, _sl)
sys.path.insert(0, _this)

from track import get_track
from optimizer_ipopt import SpeedProfileOptimizer
from controller import generate_racing_line


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(track_name, w_min, w_max, n_coarse, n_refine, n_opt_pts, params=None):
    from car import CarParams
    p = params or CarParams()
    car_str = (f"{p.m_chassis}|{p.Q_batt}|{p.e_spec}|{p.P_max}|"
               f"{p.F_drive_max}|{p.F_brake_max}|{p.eta_motor}|{p.eta_regen}|"
               f"{p.C_d}|{p.C_roll}|{p.mu}|{p.rho}|{p.A_front}")
    s = f"{track_name}|{w_min}|{w_max}|{n_coarse}|{n_refine}|{n_opt_pts}|{car_str}"
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _cache_dir(proj_root):
    d = os.path.join(proj_root, "cache")
    os.makedirs(d, exist_ok=True)
    return d


def save_frontier(frontier, cache_dir, key):
    npz_path  = os.path.join(cache_dir, f"{key}.npz")
    json_path = os.path.join(cache_dir, f"{key}.json")
    np.savez_compressed(
        npz_path,
        profiles=np.array(frontier["profiles"], dtype=np.float32),
    )
    with open(json_path, "w") as f:
        json.dump(
            {"w_energy": frontier["w_energy"],
             "lap_time": frontier["lap_time"],
             "energy":   frontier["energy"]},
            f, indent=2,
        )
    print(f"  Cache saved -> {npz_path}")
    print(f"               -> {json_path}")


def load_frontier(cache_dir, key):
    npz_path  = os.path.join(cache_dir, f"{key}.npz")
    json_path = os.path.join(cache_dir, f"{key}.json")
    if not (os.path.exists(npz_path) and os.path.exists(json_path)):
        return None
    profiles = list(np.load(npz_path)["profiles"].astype(np.float64))
    with open(json_path) as f:
        meta = json.load(f)
    return {**meta, "profiles": profiles}


# ---------------------------------------------------------------------------
# Frontier helpers
# ---------------------------------------------------------------------------

def _fmt_time(s):
    m = int(s) // 60
    sec = s - 60 * m
    return f"{m}:{sec:04.1f}" if m else f"{sec:.1f}s"


def _merge_and_prune(w_list, t_list, e_list, p_list):
    order = np.argsort(t_list)
    w_s, t_s, e_s, p_s = ([l[i] for i in order] for l in (w_list, t_list, e_list, p_list))
    keep_w, keep_t, keep_e, keep_p = [w_s[0]], [t_s[0]], [e_s[0]], [p_s[0]]
    e_min = e_s[0]
    for w, t, e, p in zip(w_s[1:], t_s[1:], e_s[1:], p_s[1:]):
        if e < e_min:
            keep_w.append(w); keep_t.append(t)
            keep_e.append(e); keep_p.append(p)
            e_min = e
    return keep_w, keep_t, keep_e, keep_p


def _gap_lengths(t_list, e_list):
    T = np.array(t_list, dtype=float)
    E = np.array(e_list, dtype=float)
    T_n = (T - T.min()) / max(T.max() - T.min(), 1e-12)
    E_n = (E - E.min()) / max(E.max() - E.min(), 1e-12)
    return np.hypot(np.diff(T_n), np.diff(E_n))


def _refine_frontier(opt, w_list, t_list, e_list, p_list,
                     n_refine, T_ref, E_ref, verbose):
    tried_w   = set(w_list)
    blacklist = set()
    MIN_W_RATIO = 1.05

    for step in range(n_refine):
        gaps     = _gap_lengths(t_list, e_list)
        gaps_eff = gaps.copy()
        for i in range(len(gaps)):
            if (round(w_list[i], 12), round(w_list[i+1], 12)) in blacklist:
                gaps_eff[i] = 0.0

        if gaps_eff.max() < 1e-6:
            if verbose:
                print(f"  [refine] All gaps exhausted after {step} steps.")
            break

        idx    = int(np.argmax(gaps_eff))
        w_lo, w_hi = w_list[idx], w_list[idx+1]
        seg_key    = (round(w_lo, 12), round(w_hi, 12))

        if w_hi / w_lo < MIN_W_RATIO:
            blacklist.add(seg_key)
            if verbose:
                print(f"  [refine {step+1:2d}/{n_refine}] gap {idx}<->{idx+1} too narrow -- skipped")
            continue

        # 7 candidates uniformly in log(w) inside (w_lo, w_hi)
        log_lo, log_hi = np.log10(w_lo), np.log10(w_hi)
        fracs = np.linspace(0, 1, 9)[1:-1]
        w_candidates = [
            10 ** (log_lo + f * (log_hi - log_lo))
            for f in fracs
            if 10 ** (log_lo + f * (log_hi - log_lo)) not in tried_w
        ]
        if not w_candidates:
            blacklist.add(seg_key)
            if verbose:
                print(f"  [refine {step+1:2d}/{n_refine}] gap {idx}<->{idx+1} -- no fresh candidates, blacklisted")
            continue

        w_new = w_candidates[len(w_candidates) // 2]
        tried_w.add(w_new)

        if verbose:
            print(f"  [refine {step+1:2d}/{n_refine}] gap {idx}<->{idx+1}  w={w_new:.3e}  (gap={gaps[idx]:.4f}) ...",
                  end=" ", flush=True)

        v_opt = None
        for guess in (p_list[idx], p_list[idx+1]):
            try:
                v_opt = opt.optimize(w_time=1.0, w_energy=w_new,
                                     T_ref=T_ref, E_ref=E_ref, initial_guess=guess)
                break
            except Exception:
                continue

        if v_opt is None:
            blacklist.add(seg_key)
            if verbose:
                print("FAILED -- blacklisted")
            continue

        m        = opt.compute_metrics(v_opt)
        n_before = len(t_list)
        w_list.append(w_new); t_list.append(m["lap_time_s"])
        e_list.append(m["energy_J"]); p_list.append(v_opt)
        w_list, t_list, e_list, p_list = _merge_and_prune(w_list, t_list, e_list, p_list)

        if len(t_list) > n_before:
            blacklist.clear()
            if verbose:
                print(f"T={m['lap_time_s']:.1f}s  E={m['energy_J']/1e6:.3f}MJ  OK")
        else:
            if verbose:
                print(f"T={m['lap_time_s']:.1f}s  E={m['energy_J']/1e6:.3f}MJ  (dominated, retrying gap)")

    return w_list, t_list, e_list, p_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force_recompute=False):
    print("=" * 70)
    print("  Pareto Frontier: Time vs Energy  (arc-length adaptive refinement)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Run parameters -- change any of these to bust the cache
    # ------------------------------------------------------------------
    TRACK_NAME = "monaco"
    W_MIN      = 1e-4
    W_MAX      = 5e0
    N_COARSE   = 12
    N_REFINE   = 20
    N_OPT_PTS  = 150

    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = _cache_dir(proj_root)
    from car import CarParams as _CarParams
    key = _cache_key(TRACK_NAME, W_MIN, W_MAX, N_COARSE, N_REFINE, N_OPT_PTS, _CarParams())

    # ------------------------------------------------------------------
    # 1. Track + SCP racing line
    # ------------------------------------------------------------------
    print("\n[1/5] Loading track...")
    trk = get_track(TRACK_NAME)
    print(f"  {trk.name} | {len(trk.centerline)} pts | {trk.length:.0f} m")

    print("\n[2/5] Computing SCP racing line (offline path optimization)...")
    from optimizer import (optimize_racing_line, alpha_to_raceline,
                           compute_curvature_from_path)
    from car import CarParams

    # Run SCP at 80 stations — get the RAW subsampled data
    scp_results = optimize_racing_line(trk, n_stations=80, solver='scp')

    # Extract the 80-station subsampled raceline (where SCP actually optimized)
    alpha_sub = scp_results['alpha_scp']
    v_sub     = scp_results['v_scp_raw']
    n_full = len(trk.centerline)
    idx_sub = np.linspace(0, n_full - 1, 80, dtype=int)
    centerline_sub = trk.centerline[idx_sub]
    normals_sub    = trk.normals[idx_sub]
    raceline_sub   = alpha_to_raceline(alpha_sub, centerline_sub, normals_sub)
    kappa_sub      = compute_curvature_from_path(raceline_sub)

    print(f"  SCP: {len(raceline_sub)} stations, J={np.sum(np.linalg.norm(np.diff(raceline_sub, axis=0, append=raceline_sub[0:1]), axis=1) / np.maximum(v_sub, 0.5)):.2f}s")

    # Also keep the full-resolution interpolated results for iLQR later
    raceline_full = scp_results['raceline_scp']
    v_full        = scp_results['velocity_scp']

    # Build IPOPT optimizer on the EXACT SAME 80 stations as SCP
    # Same curvature, same grid, same physics
    p = CarParams()
    g = 9.81
    a_lon = min(p.F_drive_max / p.mass, p.mu * g * 0.6)
    a_brk = min(p.F_brake_max / p.mass, p.mu * g * 0.9)

    opt = SpeedProfileOptimizer(
        raceline_sub, a_max=a_lon, a_brake=a_brk,
        n_points=None, curvature=kappa_sub, grip_fraction=0.85,
    )

    scp_metrics = opt.compute_metrics(v_sub)
    print(f"  SCP metrics: T={scp_metrics['lap_time_s']:.1f}s  E={scp_metrics['energy_J']/1e6:.3f} MJ")

    # For the Pareto front, v_scp is the SCP's raw 80-station profile
    v_scp = v_sub

    # ------------------------------------------------------------------
    # 2 & 3. Load from cache or compute
    # ------------------------------------------------------------------
    frontier = None
    if not force_recompute:
        frontier = load_frontier(cache_dir, key)
        if frontier is not None:
            print(f"\n  Loaded {len(frontier['lap_time'])} pts from cache  (key={key})")
            print("  Run with --recompute to force a fresh computation.")

    if frontier is None:
        # SCP profile IS the min-time point — use as anchor and normalization
        v_base = v_scp
        base_m = opt.compute_metrics(v_base)
        T_ref  = max(base_m["lap_time_s"], 1.0)
        E_ref  = max(abs(base_m["energy_J"]), 1.0)

        print(f"\n[3/5] Coarse sweep ({N_COARSE} pts, {W_MIN:.0e} -> {W_MAX:.0e})...")
        coarse  = opt.compute_pareto_frontier(
            n_points=N_COARSE, w_energy_min=W_MIN, w_energy_max=W_MAX,
            T_ref=T_ref, E_ref=E_ref, prune_dominated=True, verbose=True,
            v_seed=v_scp,
        )
        # Prepend the SCP min-time point as anchor of the Pareto front
        w_list  = [0.0] + list(coarse["w_energy"])
        t_list  = [base_m["lap_time_s"]] + list(coarse["lap_time"])
        e_list  = [base_m["energy_J"]] + list(coarse["energy"])
        p_list  = [v_scp] + list(coarse["profiles"])
        print(f"  -> {len(t_list)} non-dominated points after coarse pass")

        print(f"\n[4/5] Adaptive refinement ({N_REFINE} bisections)...")
        w_list, t_list, e_list, p_list = _refine_frontier(
            opt, w_list, t_list, e_list, p_list,
            n_refine=N_REFINE, T_ref=T_ref, E_ref=E_ref, verbose=True,
        )
        frontier = {"w_energy": w_list, "lap_time": t_list,
                    "energy": e_list, "profiles": p_list}
        print(f"  -> {len(t_list)} points on final frontier")
        save_frontier(frontier, cache_dir, key)

    # ------------------------------------------------------------------
    # 4. Print table
    # ------------------------------------------------------------------
    print(f"\n[5/5] Pareto Frontier Results ({len(frontier['lap_time'])} pts):")
    hdr = f"  {'w_energy':<12} {'Time':>10} {'Energy (MJ)':>13} {'DTime':>8} {'DEnergy':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    T0, E0 = frontier["lap_time"][0], frontier["energy"][0]
    for we, t, e in zip(frontier["w_energy"], frontier["lap_time"], frontier["energy"]):
        print(
            f"  {we:<12.2e} {_fmt_time(t):>10} {e/1e6:>13.4f}"
            f"  {100*(t-T0)/T0:>+6.2f}%  {100*(e-E0)/E0:>+7.2f}%"
        )
    t_range  = frontier["lap_time"][-1] - frontier["lap_time"][0]
    e_saving = 100 * (1 - frontier["energy"][-1] / frontier["energy"][0])
    print(f"\n  Trade-off: +{t_range:.1f}s  ->  {e_saving:.1f}% energy saved")

    # ------------------------------------------------------------------
    # 5. Figures
    # ------------------------------------------------------------------
    fig_dir = os.path.join(proj_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    T_arr = np.array(frontier["lap_time"])
    E_arr = np.array(frontier["energy"]) / 1e6

    # Figure 1 -- Pareto front
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    sc = ax.scatter(T_arr, E_arr, c=np.log10(frontier["w_energy"]),
                    cmap="plasma", s=80, zorder=3, edgecolors="k", linewidths=0.5)
    ax.plot(T_arr, E_arr, "-", color="0.6", linewidth=1.5, zorder=2)
    ax.scatter([T_arr[0]],  [E_arr[0]],  s=200, marker="*", color="#00aa00", zorder=4, label="Min time")
    ax.scatter([T_arr[-1]], [E_arr[-1]], s=150, marker="s", color="#0066ff", zorder=4, label="Min energy")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("log10(w_energy)", fontsize=9)
    ax.set_xlabel("Lap Time (s)", fontweight="bold")
    ax.set_ylabel("Energy (MJ)", fontweight="bold")
    ax.set_title("Pareto Frontier: Time vs Energy", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    path1 = os.path.join(fig_dir, "pareto_frontier.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path1}")

    # Figure 2 -- speed profiles
    baseline  = opt.baseline_speed_profile()
    fastest   = frontier["profiles"][0]
    efficient = frontier["profiles"][-1]
    bm = opt.compute_metrics(baseline)
    fm = opt.compute_metrics(fastest)
    em = opt.compute_metrics(efficient)

    n_mid   = min(3, len(frontier["profiles"]) - 2)
    mid_idx = np.round(np.linspace(1, len(frontier["profiles"]) - 2, n_mid)).astype(int)

    fig2, ax3 = plt.subplots(figsize=(13, 5), facecolor="white")
    x = np.arange(len(baseline))
    ax3.plot(x, baseline, color="0.6", lw=1.5, ls="--",
             label=f"Baseline  T={_fmt_time(bm['lap_time_s'])}  E={bm['energy_J']/1e6:.3f} MJ")
    ax3.plot(x, fastest, color="#d62728", lw=2.0,
             label=f"Min time  T={_fmt_time(fm['lap_time_s'])}  E={fm['energy_J']/1e6:.3f} MJ")
    cmap_b = plt.cm.Blues
    for rank, mi in enumerate(mid_idx):
        vp = frontier["profiles"][mi]
        mm = opt.compute_metrics(vp)
        ax3.plot(x, vp, color=cmap_b(0.45 + 0.2 * rank), lw=1.5,
                 label=f"Mid {rank+1}  T={_fmt_time(mm['lap_time_s'])}  E={mm['energy_J']/1e6:.3f} MJ")
    ax3.plot(x, efficient, color="#1f77b4", lw=2.0,
             label=f"Min energy T={_fmt_time(em['lap_time_s'])}  E={em['energy_J']/1e6:.3f} MJ")
    ax3.set_xlabel("Track segment index", fontweight="bold")
    ax3.set_ylabel("Speed (m/s)", fontweight="bold")
    ax3.set_title("Speed Profiles along the Pareto Frontier", fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.legend(fontsize=8, loc="upper right", ncol=2)
    fig2.tight_layout()
    path2 = os.path.join(fig_dir, "pareto_speed_profiles.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    force = "--recompute" in sys.argv
    main(force_recompute=force)