# EV Optimal Line Racing

> AA222/CS361 Final Project

Optimal racing line optimization for electric vehicles on closed circuits, with energy-aware speed profiling, closed-loop trajectory tracking, battery sizing, and race strategy co-optimization.

## Authors

- Erwin Poussi (<erwinpi@stanford.edu>)
- Matthieu Hautsch (<matthaut@stanford.edu>)

---

## Project Structure

```text
EV-optimal-line-racing/
├── src/
│   ├── track.py            # Track geometry (7 circuits: oval, complex, monza, ...)
│   ├── car.py              # Vehicle physics: bicycle model + EV battery/motor
│   ├── optimizer.py        # SCP joint path+speed optimizer + SCP Pareto
│   ├── simplex.py          # Simplex LP solver (Ch 12, course reader)
│   ├── optimizer_ipopt.py  # IPOPT speed optimizer (CasADi, for benchmarking)
│   ├── controller.py       # Online controllers (Pure Pursuit, iLQR)
│   ├── simulator.py        # Single-lap race simulator with video rendering
│   ├── run_analysis.py     # Full analysis: convergence + Pareto figures
│   ├── pareto_frontier.py  # IPOPT Pareto frontier (for benchmarking)
│   ├── battery_sizing.py   # Phase 1 — multi-lap battery sizing sweep
│   ├── monaco_race.py      # Phase 1 — 51-lap Monaco E-Prix simulation
│   ├── race_strategy.py    # Phase 4 — race strategy co-optimization
│   └── compare_circuits.py # Cross-circuit comparison figure
├── src_single_lap/         # Backup of original single-lap codebase
├── figures/                # Generated figures
├── references/             # Course reader + papers
├── report/                 # LaTeX (proposal, status update)
└── races/                  # [untracked] Simulation outputs
```

---

## Pipeline

```text
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
│  Grand Prix Circuit: 35.2s → 42.7s (optimal)       │
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
│  Result: +3s → 34% energy saved (per lap)          │
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
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 1: Multi-lap battery sizing + race sim       │
│                                                     │
│  battery_sizing.py                                  │
│    Point-mass LTS (no CasADi) — fast enough         │
│    for sweeps over Q_batt                           │
│    Physics: a_max from F_drive/mass and P_max/v;    │
│             η(P) parabolic motor efficiency map     │
│    mass = m_chassis + Q_batt / e_spec               │
│    Sweep: Q_batt ∈ [15, 80] kWh → feasibility      │
│    Result: Q* = 36.9 kWh, mass = 824 kg            │
│                                                     │
│  monaco_race.py                                     │
│    51-lap E-Prix simulation (point-mass + 2% noise) │
│    Result: 51/51 laps, SoC_final = 7.4%            │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 4: Race strategy co-optimization             │
│                                                     │
│  race_strategy.py                                   │
│    Control: (g_t, p_t) per lap                     │
│      g_t = cornering aggressiveness (grip_fraction) │
│      p_t = pace factor                             │
│    Model: T(g,p) = T_base(g)/p                     │
│           E(g,p) = E_base(g)·p                     │
│                                                     │
│    KKT analytical result:                           │
│      g* = argmin T(g)·E(g)   [product minimizer]   │
│      p* = E_budget / (n·E(g*))                     │
│                                                     │
│    Result (Monaco): T×E monotone → g* = g_max      │
│    → pace management only; line choice decoupled   │
│    Q* with strategy = 30 kWh vs 37 kWh (-35 kg)   │
└─────────────────────────────────────────────────────┘
```

---

## Running

```bash
# Full single-lap analysis (convergence + Pareto figures)
python src/run_analysis.py

# Battery sizing sweep — 51 laps
python src/battery_sizing.py --laps 51 --Q-min 15 --Q-max 80

# 51-lap race simulation at optimal Q*
python src/monaco_race.py --laps 51

# Race strategy co-optimization (Phase 4)
python src/race_strategy.py --laps 51 --Q-batt 36.9

# Run on a different circuit (monza, hairpin, oval, ...)
python src/battery_sizing.py --track monza
python src/race_strategy.py --track monza

# Cross-circuit comparison figure (reads JSON results from figures/<track>/)
python src/compare_circuits.py

# Pareto frontier (SCP vs IPOPT benchmark)
python src/pareto_frontier.py --recompute
```

---

## Key Results

### Single-lap optimization (Grand Prix Circuit, 1288 m)

| Metric | Value |
| --- | --- |
| SCP min-time (offline) | 42.7s (SCP + Simplex LP) |
| scipy SLSQP benchmark | 21.9s (500 iters, 80k func evals) |
| iLQR closed-loop tracking | 27.0s (vs 26.9s planned) |
| Pareto: +3s | 34% energy saved |
| Pareto: +12s | 80% energy saved |

### Phase 1 — Battery sizing & 51-lap race (Formula E Gen3 baseline)

| Metric | Value |
| --- | --- |
| Car model | Formula E Gen3 (m_chassis=640 kg, P_max=300 kW) |
| Motor efficiency | Parabolic η(P): peak 95% at 60% load |
| Q* (minimum feasible) | 36.9 kWh → total mass 824 kg |
| Lap time at Q* | 50.2 s |
| Energy per lap | 677 Wh net (drive 750 Wh – regen 73 Wh) |
| Race result | 51/51 laps, SoC_final = 7.4% |

### Phase 4 — Race strategy co-optimization (Grand Prix Circuit, 51 laps)

| Metric | Value |
| --- | --- |
| KKT result | g* = g_max (T×E product monotone at Monaco) |
| Physical insight | Corner-limited circuits: line choice decoupled from energy |
| Q* with pace strategy (p_min=0.80) | 30.0 kWh vs 37.0 kWh baseline |
| Battery mass saving | 7.0 kWh → **35 kg lighter** |
| NLP confirmation | g=0.9000±0, p=1.0000±0 across all 51 laps |

### Cross-circuit comparison (`figures/comparison_circuits.png`)

| Circuit | Length | Laps | Race dist. | Q* | Avg lap |
| --- | --- | --- | --- | --- | --- |
| Monaco (Grand Prix) | 1288 m | 51 | 65.7 km | 36.9 kWh | 50.2 s |
| Monza-Style | 1115 m | 58 | 64.7 km | 26.0 kWh | 31.2 s |
| Hairpin & Chicane | 1269 m | 51 | 64.7 km | 26.0 kWh | 63.2 s |

### Circuits available

| Name | Description | Length |
| --- | --- | --- |
| `complex` | Grand Prix Circuit (Monaco-type) | 1288 m |
| `monza` | Monza-Style (long straights + chicanes) | 1115 m |
| `hairpin` | Hairpin & Chicane Circuit | 1269 m |
| `monaco` | Sharp Corner Circuit | 903 m |
| `oval` | Oval | 909 m |
| `circle` | Circular Track | 628 m |
| `figure_eight` | Figure Eight | 889 m |

---

## Model Notes

**Point-mass Lap Time Simulator** (`battery_sizing`, `monaco_race`, `race_strategy`):
uses forward-backward passes to enforce grip, force, and power limits. No yaw/tire dynamics — suitable for multi-lap sweeps where bicycle model + pure pursuit is numerically unstable.

**Phase 1 physics**:

- `a_max = min(F_drive_max/mass, P_max/(mass×v))` — force + power limited
- `a_brake = min(F_brake_max/mass, μ×g)` — force + adhesion limited
- Mass coupled to battery: `mass = m_chassis + Q_batt / e_spec`
- Motor efficiency: `η(P/P_max)` parabolic, peak 95% at 60% load

**Phase 4 limitation**: The grip_fraction proxy for "line choice" generates a T×E curve that is monotone for all tested circuits (time is more sensitive to aggressiveness than energy). A non-trivial interior optimum would require the IPOPT Pareto frontier, where the optimizer can independently tune high-speed vs low-speed sections.

---

## References

- Kochenderfer & Wheeler, *Algorithms for Optimization*, MIT Press, 2019 — Simplex (Ch 12), KKT conditions (Ch 10)
- Xiong, *Racing Line Optimization*, MIT MEng thesis, 2010
- Rajamani, *Vehicle Dynamics and Control*, Springer, 2012 — bicycle model
