# EV Optimal Line Racing

**AA222/CS361 Final Project**

Optimal racing line optimization for electric vehicles on closed circuits, with energy-aware speed profiling and closed-loop trajectory tracking.

## Authors
- Erwin Poussi (erwinpi@stanford.edu)
- Matthieu Hautsch (matthaut@stanford.edu)

---

## Project Structure

```
EV-optimal-line-racing/
├── src/
│   ├── track.py            # Track geometry generation (7 circuits)
│   ├── car.py              # Vehicle physics (bicycle model + EV battery)
│   ├── optimizer.py        # SCP joint path+speed optimizer + SCP Pareto
│   ├── simplex.py          # Simplex LP solver (Ch 12, course reader)
│   ├── optimizer_ipopt.py  # IPOPT speed optimizer (CasADi, for benchmarking)
│   ├── controller.py       # Online controllers (Pure Pursuit, iLQR)
│   ├── simulator.py        # Race simulator with video rendering
│   ├── run_analysis.py     # Full analysis: convergence + Pareto figures
│   └── pareto_frontier.py  # IPOPT Pareto frontier (for benchmarking)
├── references/             # Course reader + papers
├── report/                 # LaTeX (proposal, status update)
├── figures/                # Generated figures
└── races/                  # [untracked] Simulation outputs
```

---

## Pipeline

```
┌─────────────────────────────────────────────────────┐
│  STAGE 1: Offline Path + Speed Optimization         │
│                                                     │
│  solve_scp()  [optimizer.py]                        │
│    Variables: x = [α₁..αₙ, v₁..vₙ]                │
│    Objective: min Σ(dsᵢ / vᵢ)  (lap time)          │
│    Constraints: cornering, accel, braking limits    │
│    Method: SCP → linearize → Simplex LP (Ch 12)    │
│    Output: racing line α* + speed profile v*        │
│                                                     │
│  Monaco result: 35.2s → 24.0s (30 iters)           │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: Pareto Front (Time vs Energy)             │
│                                                     │
│  solve_scp_pareto()  [optimizer.py]                 │
│    Fixed path α* from Stage 1                       │
│    Objective: w_t·T/T_ref + w_e·E/E_ref            │
│    Same constraints as Stage 1                      │
│    Sweep w_e ∈ [0, 50] → Pareto front              │
│                                                     │
│  Result: +12s → 80% energy saved (10 Pareto pts)   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3: Online Trajectory Tracking                │
│                                                     │
│  ILQRController  [controller.py]                    │
│    Offline: linearize dynamics → Riccati → Y_k, y_k │
│    Online:  u_k = ū_k + Y_k(s_k - s̄_k) + y_k      │
│                                                     │
│  Result: planned 26.9s → achieved 27.0s             │
└─────────────────────────────────────────────────────┘
```

---

## Running

```bash
cd src/

# Full analysis (convergence + Pareto figures)
python run_analysis.py

# Race simulation
python simulator.py --track monaco --strategies optimal ilqr --no-video

# Pareto frontier (SCP vs IPOPT benchmark)
python pareto_frontier.py --recompute
```

---

## Key Results (Monaco, 903m)

| Metric | Value |
|---|---|
| SCP min-time (offline) | 24.0s (30 iters, Simplex LP) |
| scipy SLSQP (same constraints) | 21.9s (500 iters, 80k func evals) |
| iLQR closed-loop tracking | 27.0s (vs 26.9s planned) |
| Pareto: +3s | 34% energy saved |
| Pareto: +12s | 80% energy saved |

---

## References

- Kochenderfer & Wheeler, *Algorithms for Optimization*, MIT Press, 2019 — Simplex (Ch 12), KKT conditions (Ch 10)
- Xiong, *Racing Line Optimization*, MIT MEng thesis, 2010
- Rajamani, *Vehicle Dynamics and Control*, Springer, 2012 — bicycle model
