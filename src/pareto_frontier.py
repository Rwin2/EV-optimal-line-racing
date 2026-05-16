"""
Compute and visualize Pareto frontier for speed profile optimization.

Strategy — arc-length adaptive refinement:
  Pass 1 — coarse log-sweep (12 pts) over a wide w_energy range.
  Pass 2+ — iteratively bisect the longest gap on the frontier
             (measured in normalized T-E space) until we have
             enough well-distributed points or budget is exhausted.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))

from track import get_track
from optimizer import SpeedProfileOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(s):
    """Format seconds as m:ss.s"""
    m = int(s) // 60
    sec = s - 60 * m
    return f"{m}:{sec:04.1f}" if m else f"{sec:.1f}s"


def _merge_and_prune(w_list, t_list, e_list, p_list):
    """Sort by lap-time and remove dominated points."""
    order = np.argsort(t_list)
    w_s = [w_list[i] for i in order]
    t_s = [t_list[i] for i in order]
    e_s = [e_list[i] for i in order]
    p_s = [p_list[i] for i in order]

    # Keep only non-dominated (energy must decrease as time increases)
    keep_w, keep_t, keep_e, keep_p = [w_s[0]], [t_s[0]], [e_s[0]], [p_s[0]]
    e_min = e_s[0]
    for w, t, e, p in zip(w_s[1:], t_s[1:], e_s[1:], p_s[1:]):
        if e < e_min:
            keep_w.append(w)
            keep_t.append(t)
            keep_e.append(e)
            keep_p.append(p)
            e_min = e

    return keep_w, keep_t, keep_e, keep_p


def _gap_lengths(t_list, e_list):
    """
    Euclidean distances between consecutive frontier points in
    normalized (T, E) space (each axis scaled to [0, 1]).
    Returns array of length n-1.
    """
    T = np.array(t_list, dtype=float)
    E = np.array(e_list, dtype=float)
    T_n = (T - T.min()) / max(T.max() - T.min(), 1e-12)
    E_n = (E - E.min()) / max(E.max() - E.min(), 1e-12)
    dT = np.diff(T_n)
    dE = np.diff(E_n)
    return np.hypot(dT, dE)


def _refine_frontier(opt, w_list, t_list, e_list, p_list,
                     n_refine, T_ref, E_ref, verbose):
    """
    Add up to n_refine new Pareto points by bisecting the largest gaps
    (in normalised T-E space).

    Bug-fix vs naive bisection
    --------------------------
    If the solver returns a point that is dominated (falls inside an existing
    gap rather than splitting it), the frontier is unchanged and the same gap
    would be picked forever.  We handle this with a *blacklist*:

    - After a failed or non-improving bisection of a segment (w_lo, w_hi),
      that segment is penalised (gap set to 0) so the next iteration picks
      a different gap.
    - The blacklist is rebuilt from scratch whenever the frontier actually
      changes, because new neighbours may now form actionable gaps.
    - Within a blacklisted segment we try a small log-space jitter (×2 and
      ÷2) before giving up, to escape solver local minima.
    """
    tried_w: set = set(w_list)           # all w values ever submitted
    blacklist: set = set()               # (idx_lo_w, idx_hi_w) segment keys
    MIN_W_RATIO = 1.05                   # stop bisecting if gap < 5% in log-w

    for step in range(n_refine):
        gaps = _gap_lengths(t_list, e_list)

        # Apply blacklist — zero out gaps we've already exhausted
        gaps_eff = gaps.copy()
        for i in range(len(gaps)):
            seg_key = (round(w_list[i], 12), round(w_list[i + 1], 12))
            if seg_key in blacklist:
                gaps_eff[i] = 0.0

        if gaps_eff.max() < 1e-6:
            if verbose:
                print(f"  [refine] All gaps exhausted after {step} steps.")
            break

        idx = int(np.argmax(gaps_eff))
        w_lo, w_hi = w_list[idx], w_list[idx + 1]
        seg_key = (round(w_lo, 12), round(w_hi, 12))

        # If the log-w gap is already tiny, bisection won't help
        if w_hi / w_lo < MIN_W_RATIO:
            blacklist.add(seg_key)
            if verbose:
                print(
                    f"  [refine {step+1:2d}/{n_refine}] "
                    f"gap {idx}↔{idx+1} too narrow in w-space — skipped"
                )
            continue

        # Candidates: geometric mean, then jittered alternatives
        w_candidates = [np.sqrt(w_lo * w_hi)]
        for factor in [2.0, 0.5, 4.0, 0.25]:
            w_c = np.sqrt(w_lo * w_hi) * factor
            if w_lo < w_c < w_hi and w_c not in tried_w:
                w_candidates.append(w_c)

        # Filter already-tried values
        w_candidates = [w for w in w_candidates if w not in tried_w]
        if not w_candidates:
            blacklist.add(seg_key)
            if verbose:
                print(
                    f"  [refine {step+1:2d}/{n_refine}] "
                    f"gap {idx}↔{idx+1} — no fresh w candidates, blacklisted"
                )
            continue

        w_new = w_candidates[0]
        tried_w.add(w_new)

        if verbose:
            print(
                f"  [refine {step+1:2d}/{n_refine}] "
                f"gap {idx}↔{idx+1}  w={w_new:.3e}  "
                f"(gap={gaps[idx]:.4f}) ...",
                end=" ", flush=True,
            )

        # Solve — warm-start from both neighbours
        v_opt = None
        for guess in (p_list[idx], p_list[idx + 1]):
            try:
                v_opt = opt.optimize(
                    w_time=1.0, w_energy=w_new,
                    T_ref=T_ref, E_ref=E_ref,
                    initial_guess=guess,
                )
                break
            except Exception:
                continue

        if v_opt is None:
            blacklist.add(seg_key)
            if verbose:
                print("FAILED — blacklisted")
            continue

        m = opt.compute_metrics(v_opt)
        n_before = len(t_list)

        # Tentatively insert and prune
        w_list.append(w_new)
        t_list.append(m["lap_time_s"])
        e_list.append(m["energy_J"])
        p_list.append(v_opt)
        w_list, t_list, e_list, p_list = _merge_and_prune(
            w_list, t_list, e_list, p_list
        )

        if len(t_list) > n_before:
            # Frontier improved — reset blacklist (new neighbours may be splittable)
            blacklist.clear()
            if verbose:
                print(f"T={m['lap_time_s']:.1f}s  E={m['energy_J']/1e6:.3f}MJ  ✓")
        else:
            # Point was dominated — blacklist this segment
            blacklist.add(seg_key)
            if verbose:
                print(
                    f"T={m['lap_time_s']:.1f}s  E={m['energy_J']/1e6:.3f}MJ  "
                    f"(dominated — blacklisted)"
                )

    return w_list, t_list, e_list, p_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Pareto Frontier: Time vs Energy  (arc-length adaptive refinement)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Track
    # ------------------------------------------------------------------
    print("\n[1/4] Loading track...")
    trk = get_track("complex")
    print(f"  {trk.name} | {len(trk.centerline)} pts | {trk.length:.0f} m")

    opt = SpeedProfileOptimizer(trk.centerline, a_max=4.0, a_brake=8.0, n_points=150)

    # Calibrate normalization once from baseline
    v_base = opt.baseline_speed_profile()
    base_m = opt.compute_metrics(v_base)
    T_ref  = max(base_m["lap_time_s"], 1.0)
    E_ref  = max(abs(base_m["energy_J"]), 1.0)

    # ------------------------------------------------------------------
    # 2. Coarse pass — 12 points over a wide range
    # ------------------------------------------------------------------
    W_MIN = 1e-4
    W_MAX = 5e0
    N_COARSE = 12

    print(f"\n[2/4] Coarse sweep ({N_COARSE} pts, {W_MIN:.0e} → {W_MAX:.0e})...")

    coarse = opt.compute_pareto_frontier(
        n_points=N_COARSE,
        w_energy_min=W_MIN,
        w_energy_max=W_MAX,
        T_ref=T_ref, E_ref=E_ref,
        prune_dominated=True,
        verbose=True,
    )

    w_list = list(coarse["w_energy"])
    t_list = list(coarse["lap_time"])
    e_list = list(coarse["energy"])
    p_list = list(coarse["profiles"])

    print(f"  → {len(t_list)} non-dominated points after coarse pass")

    # ------------------------------------------------------------------
    # 3. Adaptive refinement — bisect largest gaps in (T, E) space
    # ------------------------------------------------------------------
    N_REFINE = 20

    print(f"\n[3/4] Adaptive refinement ({N_REFINE} bisections)...")

    w_list, t_list, e_list, p_list = _refine_frontier(
        opt, w_list, t_list, e_list, p_list,
        n_refine=N_REFINE,
        T_ref=T_ref, E_ref=E_ref,
        verbose=True,
    )

    frontier = {
        "w_energy": w_list,
        "lap_time": t_list,
        "energy":   e_list,
        "profiles": p_list,
    }

    print(f"  → {len(t_list)} points on final frontier")

    # ------------------------------------------------------------------
    # 4. Print table
    # ------------------------------------------------------------------
    print(f"\n[4/4] Pareto Frontier Results ({len(frontier['lap_time'])} pts):")
    hdr = f"  {'w_energy':<12} {'Time':>10} {'Energy (MJ)':>13} {'ΔTime':>8} {'ΔEnergy':>9}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    T0 = frontier["lap_time"][0]
    E0 = frontier["energy"][0]

    for we, t, e in zip(frontier["w_energy"], frontier["lap_time"], frontier["energy"]):
        dt_pct = 100 * (t  - T0) / T0
        de_pct = 100 * (e  - E0) / E0
        print(
            f"  {we:<12.2e} {_fmt_time(t):>10} {e/1e6:>13.4f}"
            f"  {dt_pct:>+6.2f}%  {de_pct:>+7.2f}%"
        )

    t_range = frontier["lap_time"][-1] - frontier["lap_time"][0]
    e_saving = 100 * (1 - frontier["energy"][-1] / frontier["energy"][0])
    print(f"\n  Trade-off summary:")
    print(f"    +{t_range:.1f}s lap time  →  {e_saving:.1f}% energy saved")

    # ------------------------------------------------------------------
    # 5. Figures
    # ------------------------------------------------------------------
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir   = os.path.join(proj_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    T_arr = np.array(frontier["lap_time"])
    E_arr = np.array(frontier["energy"]) / 1e6   # MJ

    # -- Figure 1: Pareto front ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    sc = ax.scatter(T_arr, E_arr, c=np.log10(frontier["w_energy"]),
                    cmap="plasma", s=80, zorder=3, edgecolors="k", linewidths=0.5)
    ax.plot(T_arr, E_arr, "-", color="0.6", linewidth=1.5, zorder=2)
    ax.scatter([T_arr[0]],  [E_arr[0]],  s=200, marker="*", color="#00aa00",
               zorder=4, label="Min time")
    ax.scatter([T_arr[-1]], [E_arr[-1]], s=150, marker="s", color="#0066ff",
               zorder=4, label="Min energy")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("log₁₀(w_energy)", fontsize=9)
    ax.set_xlabel("Lap Time (s)", fontweight="bold")
    ax.set_ylabel("Energy (MJ)", fontweight="bold")
    ax.set_title("Pareto Frontier: Time vs Energy", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # -- Figure 1b: efficiency (E/T) vs w_energy -------------------------
    ax2 = axes[1]
    eff = E_arr / T_arr          # MJ/s = MW
    ax2.semilogx(frontier["w_energy"], eff * 1e3, "o-",
                 linewidth=2, markersize=7, color="#9933ff")
    ax2.set_xlabel("w_energy (log scale)", fontweight="bold")
    ax2.set_ylabel("Energy rate (kJ/s)", fontweight="bold")
    ax2.set_title("Energy Efficiency along Frontier", fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--", which="both")

    fig.tight_layout()
    path1 = os.path.join(fig_dir, "pareto_frontier.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Saved: {path1}")

    # -- Figure 2: speed profiles ----------------------------------------
    baseline  = opt.baseline_speed_profile()
    fastest   = frontier["profiles"][0]
    efficient = frontier["profiles"][-1]

    bm = opt.compute_metrics(baseline)
    fm = opt.compute_metrics(fastest)
    em = opt.compute_metrics(efficient)

    # Pick 3 intermediate profiles evenly spaced along the frontier
    n_mid = min(3, len(frontier["profiles"]) - 2)
    mid_idx = np.round(np.linspace(1, len(frontier["profiles"]) - 2, n_mid)).astype(int)

    fig2, ax3 = plt.subplots(figsize=(13, 5), facecolor="white")
    x = np.arange(len(baseline))

    ax3.plot(x, baseline, color="0.6",    lw=1.5, ls="--",
             label=f"Baseline  T={_fmt_time(bm['lap_time_s'])}  E={bm['energy_J']/1e6:.3f} MJ")
    ax3.plot(x, fastest,  color="#d62728", lw=2.0,
             label=f"Min time  T={_fmt_time(fm['lap_time_s'])}  E={fm['energy_J']/1e6:.3f} MJ")

    cmap = plt.cm.Blues
    for rank, mi in enumerate(mid_idx):
        vp = frontier["profiles"][mi]
        mm = opt.compute_metrics(vp)
        color = cmap(0.45 + 0.2 * rank)
        ax3.plot(x, vp, color=color, lw=1.5,
                 label=f"Intermediate {rank+1}  T={_fmt_time(mm['lap_time_s'])}  E={mm['energy_J']/1e6:.3f} MJ")

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
    print(f"✓ Saved: {path2}")


if __name__ == "__main__":
    main()