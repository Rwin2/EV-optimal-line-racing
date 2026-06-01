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
│   ├── single lap/             # Single-lap analysis (SCP optimizer, iLQR, simulator)
│   │   ├── track.py                # Track geometry (7 circuits)
│   │   ├── car.py                  # Vehicle physics: bicycle model + EV battery/motor
│   │   ├── optimizer.py            # SCP joint path+speed optimizer + SCP Pareto
│   │   ├── simplex.py              # Simplex LP solver (Ch 12, course reader)
│   │   ├── optimizer_ipopt.py      # IPOPT speed optimizer (CasADi, for benchmarking)
│   │   ├── controller.py           # Online controllers (Pure Pursuit, iLQR)
│   │   ├── simulator.py            # Single-lap race simulator with video rendering
│   │   ├── run_analysis.py         # Full analysis: convergence + Pareto figures
│   │   └── pareto_frontier.py      # IPOPT Pareto frontier (for benchmarking)
│   └── battery sizing/         # Multi-lap battery sizing + race strategy
│       ├── car.py                  # Extended car model (mass = m_chassis + Q_batt/e_spec)
│       ├── optimizer.py            # SCP + solve_scp_pareto_joint (joint Pareto)
│       ├── pareto_frontier.py      # Car-params-aware cache key
│       ├── simulator.py            # Multi-lap simulator
│       ├── battery_sizing.py       # Phase 1 — battery sizing sweep
│       ├── monaco_race.py          # Phase 1 — multi-lap race simulation
│       ├── race_strategy.py        # Phase 4 — race strategy co-optimization
│       ├── compare_circuits.py     # Cross-circuit comparison figure
│       └── compare_methods.py      # Pareto method comparison: grip proxy vs joint SCP
├── figures/                # Generated figures (per-circuit subfolders)
├── references/             # Course reader + papers
├── report/                 # LaTeX — final conference paper
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
│  Grand Prix Circuit: 35.2s → 38.1s (optimal)       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: Joint SCP Pareto (Time vs Energy)         │
│                                                     │
│  compute_joint_pareto()  [optimizer.py]             │
│    Variables: α (line geometry) + v (speed) jointly │
│    Objective: T/T_ref + w_E·E/E_ref                 │
│    Sweep w_E ∈ [0.01, 8] → Pareto front            │
│    Non-dominated filter removes dominated pts       │
│                                                     │
│  Grand Prix: T spread 127%, E spread 348% per lap  │
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
│  race_strategy.py  --pareto-method scp              │
│    Control: (line_t, p_t) per lap                   │
│      line_t = SCP Pareto operating point            │
│      p_t    = pace factor ∈ [p_min, 1]             │
│    Model: T(line,p) = T_base(line)/p               │
│           E(line,p) = E_base(line)·p               │
│    Mass coupling: E_base ∝ mass (battery weight)   │
│                                                     │
│    KKT analytical result:                           │
│      line* = argmin T(line)·E(line)                 │
│      p*    = E_budget / (n·E(line*))               │
│                                                     │
│    Q* without strategy = 27.0 kWh                  │
│    Q* with strategy    =  5.0 kWh  (-110 kg)       │
│    With T_max=+15%     =  8-11 kWh (~80 kg saved)  │
└─────────────────────────────────────────────────────┘
```

---

## Running

```bash
# Full single-lap analysis (convergence + Pareto figures)
python "src/single lap/run_analysis.py"

# Battery sizing sweep
python "src/battery sizing/battery_sizing.py" --track complex --laps 51 --Q-min 15 --Q-max 80
python "src/battery sizing/battery_sizing.py" --track monaco  --laps 72 --Q-min 20 --Q-max 80

# Multi-lap race simulation at optimal Q*
python "src/battery sizing/monaco_race.py" --track complex --laps 51 --Q-batt 36.9
python "src/battery sizing/monaco_race.py" --track monza   --laps 58 --Q-batt 31.7
python "src/battery sizing/monaco_race.py" --track hairpin --laps 51 --Q-batt 31.7
python "src/battery sizing/monaco_race.py" --track monaco  --laps 72 --Q-batt 38.3

# Phase 4: Race strategy co-optimization (joint SCP Pareto)
python "src/battery sizing/race_strategy.py" --track complex --Q-batt 36.9 --pareto-method scp --no-nlp --T-max-pct 15
python "src/battery sizing/race_strategy.py" --track monza   --Q-batt 31.7 --pareto-method scp --no-nlp --T-max-pct 15
python "src/battery sizing/race_strategy.py" --track hairpin --Q-batt 31.7 --pareto-method scp --no-nlp --T-max-pct 15
python "src/battery sizing/race_strategy.py" --track monaco  --Q-batt 38.3 --pareto-method scp --no-nlp --T-max-pct 15

# Pareto method comparison: grip proxy vs joint SCP
python "src/battery sizing/compare_methods.py" --track complex --Q-batt 36.9 --T-max-pct 15
python "src/battery sizing/compare_methods.py" --track monza   --Q-batt 31.7 --T-max-pct 15
python "src/battery sizing/compare_methods.py" --track hairpin --Q-batt 31.7 --T-max-pct 15
python "src/battery sizing/compare_methods.py" --track monaco  --Q-batt 38.3 --T-max-pct 15

# Cross-circuit comparison figure (reads JSON results from figures/<track>/)
python "src/battery sizing/compare_circuits.py"
```

---

## Key Results

### Single-lap optimization (Grand Prix Circuit, 1288 m)

| Metric | Value |
| --- | --- |
| SCP min-time (offline) | 38.1s (SCP + Simplex LP) |
| scipy SLSQP benchmark | 21.9s (500 iters, 80k func evals) |
| iLQR closed-loop tracking | 27.0s (vs 26.9s planned) |
| Joint SCP Pareto: +3s | 34% energy saved |
| Joint SCP Pareto: T spread | 127% (38s → 86s per lap) |

### Phase 1 — Battery sizing & multi-lap race (Formula E Gen3 baseline)

Car model: m_chassis=640 kg, P_max=300 kW, parabolic η(P): peak 95% at 60% load

| Circuit | Track key | Length | Laps | Race dist. | Q* | Mass | Avg lap | SoC final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grand Prix Circuit | `complex` | 1288 m | 51 | 65.7 km | 36.9 kWh | 824 kg | 51.5 s | 7.4% |
| Monza-Style | `monza` | 1115 m | 58 | 64.7 km | 31.7 kWh | 798 kg | 54.4 s | 5.4% |
| Hairpin & Chicane | `hairpin` | 1269 m | 51 | 64.7 km | 31.7 kWh | 798 kg | 63.2 s | 7.1% |
| Sharp Corner Circuit | `monaco` | 903 m | 72 | 65.0 km | 38.3 kWh | 832 kg | 45.9 s | 8.4% |

### Phase 4 — Race strategy co-optimization (Grand Prix Circuit, 51 laps)

| Metric | Value |
| --- | --- |
| Pareto method | Grip-fraction sweep (g ∈ [0.50, 0.90]) + KKT closed-form |
| KKT analytical result | line\* = argmin T(g)·E(g), p\* = E\_budget/(n·E(line\*)) |
| Mass coupling | E_base ∝ mass_new / mass_ref included in sweep |
| Q* without pace strategy | 37 kWh (824 kg, full effort) |
| Q* with pace strategy (p_min=0.80, T_max=+15%) | 25 kWh → **60 kg lighter** |

### Cross-circuit comparison (`figures/comparison_circuits.png`)

Compares Grand Prix Circuit, Monza-Style, and Hairpin & Chicane at ~65 km race distance.
See Phase 1 results table above for full per-circuit breakdown.

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

**Phase 4 — joint SCP Pareto**:
The joint SCP co-optimizes both path geometry (α) and speed (v) simultaneously for each energy weight w_E, so each Pareto point is a distinct racing line, not just a speed rescaling. The energy gradient w.r.t. α is computed in consistent Wh units (J/m ÷ 3600). A non-dominated filter removes Pareto points where a slower line also consumes more energy. The T_max parameter models a competitive pace constraint (e.g. +15% over fastest lap) and is recommended to prevent unrealistically slow strategies from dominating the battery sizing signal.

---

## References

- Kochenderfer & Wheeler, *Algorithms for Optimization*, MIT Press, 2019 — Simplex (Ch 12), KKT conditions (Ch 10)
- Xiong, *Racing Line Optimization*, MIT MEng thesis, 2010
- Rajamani, *Vehicle Dynamics and Control*, Springer, 2012 — bicycle model
