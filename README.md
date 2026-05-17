# EV Optimal Line Racing

**AA222/CS361 Final Project**

Optimal racing line optimization for electric vehicles on closed circuits. This project formulates and solves the minimum-lap-time path optimization problem while accounting for EV battery energy constraints.

## Authors
- Erwin Poussi (erwinpi@stanford.edu)
- Matthieu Hautsch (matthaut@stanford.edu)

---

## Project Structure

```
EV-optimal-line-racing/
├── src/
│   ├── track.py        # Track geometry generation
│   ├── car.py          # Vehicle physics (bicycle model + EV battery)
│   ├── optimizer.py    # Offline racing line optimizer (SCP + Simplex)
│   ├── simplex.py      # Simplex LP solver (Ch 12, course reader)
│   ├── controller.py   # Online tracking controllers (Pure Pursuit, iLQR)
│   └── simulator.py    # Orchestrator: runs race, renders video/GIF
├── references/         # Papers and course materials
├── report/             # LaTeX report (main.tex, references.bib)
├── figures/            # Generated output figures
├── races/              # [untracked] Race outputs (timestamped)
│   └── race_YYYYMMDD_HHMMSS/
│       ├── video/race.gif
│       └── results/
│           ├── benchmark.json
│           ├── benchmark.md
│           └── <strategy>/metrics.json
└── NOTES.md            # [untracked] Project notes
```

---

## Architecture and Data Flow

The pipeline splits cleanly into **offline planning** (done once before the race) and **online control** (runs every timestep during simulation).

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OFFLINE  (runs once, before the race)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

track.py
  └─ generates Track object:
       centerline (N,2), left/right boundaries,
       normals, half-widths, arc-length

optimizer.py  [the core]
  └─ parameterizes racing line as:
       raceline = centerline + α·normals
     where α ∈ [-w, w] is the lateral offset
  └─ minimizes one of three objectives:
       'laptime'   → min Σ(ds/v)          ← correct objective
       'curvature' → min Σκ²              ← geometric proxy
       'energy'    → min net EV energy
  └─ solver: SCP (linearize → Simplex LP → trust region)
  └─ outputs: raceline (N,2) array

speed profiler  [inside optimizer.py]
  └─ given the raceline, computes target speed
     at each point via forward-backward pass:
       1. cornering limit: v ≤ √(μg / |κ|)
       2. forward pass:    acceleration limited
       3. backward pass:   braking limited
       4. power cap:       v ≤ P_max / F

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ONLINE  (every dt = 0.02 s, during race)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PurePursuitController  [controller.py]
  input:  current car state (x, y, ψ, vx, vy, ω, SOC)
  └─ finds nearest point on raceline
  └─ looks ahead by a fixed distance → target point
  └─ computes steering angle δ (pure pursuit geometry)
  └─ computes drive force F via PD speed error
  output: (δ, F_drive)

ILQRController  [controller.py]
  OFFLINE: linearize bicycle dynamics along SCP reference,
           backward Riccati recursion → gains Y[k], offsets y[k]
  ONLINE:  u[k] = u_ref[k] + Y[k] @ (s[k] - s_ref[k]) + y[k]

BicycleModel.step()  [car.py]
  input:  state, (δ, F_drive), dt
  └─ RK4 integration of 7-state ODE:
       (x, y, ψ, vx, vy, ω, SOC)
  └─ tire slip angles → lateral forces
  └─ friction circle clamping
  └─ power → SOC dynamics
  output: new state

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ORCHESTRATOR  [simulator.py]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each race strategy:
  1. Call optimizer (offline) → raceline + speed profile
  2. Build PurePursuitController with that raceline
  3. Step simulation loop:
       controller.control(state) → (δ, F)
       car.step(state, δ, F, dt) → state
  4. Log metrics (lap time, SOC, energy, speed)
  5. Render video frame by frame → GIF/MP4
```

---

## Available Tracks

| Key | Description |
|---|---|
| `oval` | Simple ellipse |
| `circle` | Perfect circle (used for theoretical validation) |
| `complex` | Grand Prix-style circuit (hairpins, chicanes) |
| `hairpin` | Exaggerated hairpins and tight S-bends |
| `monaco` | Sharp 90-degree corners connected by straights |
| `monza` | Long straights with tight chicanes |
| `figure_eight` | Lemniscate of Bernoulli (self-crossing) |

---

## Setup

```bash
pip install numpy scipy matplotlib pillow
```

For MP4 output (requires ffmpeg):
```bash
brew install ffmpeg   # macOS
```

---

## Running

```bash
cd src/

# Default race on complex circuit
python simulator.py --gif

# Choose track and strategies
python simulator.py --track monaco --controllers center optimal aggressive eco --gif

# MP4 output
python simulator.py --track monza --time 60 --fps 20

# Just save metrics, no video
python simulator.py --no-video

# Run the optimizer standalone (generates figures/)
python optimizer.py
```

---

## Validating the Optimizer

**Quick bounds check (circular track — exact solution known):**
```python
from track import get_track
from optimizer import optimize_racing_line
import numpy as np

track = get_track('circle')
results = optimize_racing_line(track, n_stations=100, solver='scipy', obj='curvature')
alpha = results['alpha_scipy']
w = np.mean(track.widths)
print(f"Track half-width:  {w:.1f} m")
print(f"Max |alpha|:        {np.max(np.abs(alpha)):.2f} m  (should approach {w:.1f})")
print(f"Mean alpha:         {np.mean(alpha):.2f} m  (should be ~ -{w:.1f}, inner boundary)")
```

**Race comparison (the key sanity check):**
```bash
python simulator.py --track monaco --controllers center optimal aggressive --no-video
```
`optimal` (min-laptime path) should post a faster lap time than both `center` and `aggressive`.

---

## Key Design Decisions

- **Frenet parameterization**: racing line expressed as `α(s)` lateral offset from centerline, not (x,y) — this gives a clean 1D optimization variable with natural track boundary constraints
- **Correct objective**: `obj='laptime'` minimizes `Σ(ds/v)` jointly over path and speed; `obj='curvature'` minimizes `Σκ²` which is a geometric proxy (faster to compute, less accurate)
- **Warm-starting**: `laptime` and `energy` objectives warm-start from the `curvature` solution to avoid local minima
- **Friction circle**: tire forces clamped by `√(Fx²+Fy²) ≤ μmg`
- **EV battery model**: power-based SOC dynamics with separate motor efficiency (92%) and regen efficiency (65%)
- **RK4 integration**: 4th-order Runge-Kutta at dt=0.01 s for numerical stability

---

## References

- Kochenderfer & Wheeler, *Algorithms for Optimization*, MIT Press, 2019 — BFGS (Alg 6.6), Adam (Alg 5.8), Augmented Lagrangian (Alg 10.3)
- Xiong, *Racing Line Optimization*, MIT MEng thesis, 2010 — problem formulation, `E = ∫√κ ds` proxy
- van den Eshof, van Kampen & Salazar, *A Computationally Efficient Framework for Free-trajectory Minimum-lap-time Optimization of Racing Cars*, arXiv:2511.13522, 2025 — SCP approach, joint path+speed optimization
- Heilmeier et al., *Minimum Curvature Trajectory Planning and Control for an Autonomous Race Car*, Vehicle System Dynamics, 2020
- Rajamani, *Vehicle Dynamics and Control*, Springer, 2012 — bicycle model
