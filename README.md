# EV Optimal Line Racing

**AA222/CS361 Final Project**

Optimal racing line optimization for electric vehicles on closed circuits. This project formulates and solves a dual optimization problem: minimizing lap time while managing battery energy consumption, using Model Predictive Control (MPC).

## Authors
- Erwin Poussi (erwinpi@stanford.edu)
- Matthieu Hautsch (matthaut@stanford.edu)

## Project Structure
```
EV-optimal-line-racing/
├── src/                    # Source code
│   ├── track.py            # Track generation (oval, complex, monza, figure-8)
│   ├── car.py              # Bicycle model + EV battery/motor dynamics
│   ├── controller.py       # Racing controllers & baselines (pure pursuit)
│   └── simulator.py        # Main entry: simulation engine + rendering
├── references/             # Paper references and course materials
├── report/                 # LaTeX proposal and final report
│   ├── main.tex
│   ├── references.bib
│   └── aa222-jmlr2e.sty
├── figures/                # Generated figures for the report
├── races/                  # [untracked] Race outputs (timestamped)
│   └── race_YYYYMMDD_HHMMSS/
│       ├── video/          # race.mp4 or race.gif
│       └── results/
│           ├── benchmark.json
│           ├── benchmark.md
│           └── <controller>/metrics.json
├── NOTES.md                # [untracked] Project reflexion notes
└── .gitignore
```

## Setup
```bash
pip install numpy scipy matplotlib pillow
```
For MP4 video output, you also need `ffmpeg` installed:
```bash
brew install ffmpeg    # macOS
```

## Running a Race

```bash
cd src/

# Default race: 4 controllers on the complex circuit (40s, GIF output)
python simulator.py --gif

# Choose track and controllers
python simulator.py --track monza --controllers center aggressive eco --gif

# MP4 video (requires ffmpeg)
python simulator.py --track complex --time 60 --fps 20

# Only generate the static figure (for the proposal)
python simulator.py --figure-only

# Skip video, only save metrics
python simulator.py --no-video
```

### Available Tracks
| Track | Description |
|-------|-------------|
| `oval` | Simple elliptical circuit |
| `complex` | Grand Prix-style with hairpins and chicanes |
| `monza` | Monza-inspired: long straights + tight corners |
| `figure_eight` | Lemniscate of Bernoulli shape |

### Available Controllers (Baselines)
| Key | Strategy | Description |
|-----|----------|-------------|
| `center` | Center-Line | Follows track centerline, curvature-based speed |
| `optimal` | Min-Curvature | Smooths corners by using track width |
| `aggressive` | Aggressive | Uses 95% tire grip, max speed everywhere |
| `eco` | Eco-Save | Conservative speed, energy-efficient |
| `constant` | Constant-Speed | Fixed 90 km/h, centerline |

### Race Output Structure
Each run creates a timestamped directory under `races/`:
```
races/race_20260424_143022/
├── video/race.gif              # Visual output
└── results/
    ├── benchmark.json          # All controllers compared
    ├── benchmark.md            # Human-readable summary table
    ├── center/metrics.json     # Per-controller detailed metrics
    ├── aggressive/metrics.json
    └── ...
```

## Code Architecture

```
simulator.py (orchestrator)
    ├── track.py       → generates Track objects (centerline, boundaries, normals)
    ├── car.py         → BicycleModel.step(state, delta, F_drive, dt) via RK4
    ├── controller.py  → PurePursuitController.control(state) → (delta, F_drive)
    └── rendering      → matplotlib-based track + car + telemetry visualization
```

**Simulation loop:**
1. Generate track geometry (closed spline from control points)
2. Compute racing lines (centerline or minimum-curvature heuristic)
3. For each timestep: controller outputs (steering, throttle) → bicycle model integration → update state
4. Save metrics (distance, speed, SOC, energy) per controller
5. Render video with pre-rendered track background + moving cars + live telemetry

## Key Design Decisions

- **Bicycle model**: captures both longitudinal and lateral dynamics (steering, drift, tire forces) while remaining fast enough for real-time optimization
- **RK4 integration**: 4th-order Runge-Kutta for numerical stability at dt=0.02s
- **Friction circle constraint**: tire forces limited by $F_x^2 + F_y^2 \leq (\mu mg)^2$
- **Battery model**: power-based SOC evolution with separate traction/regen efficiencies
- **Speed profiling**: forward-backward pass ensures acceleration/braking limits are respected
