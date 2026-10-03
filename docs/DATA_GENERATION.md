# Generating datasets

`rc_drift_sim.datagen` simulates large batches of episodes in which any variable (car, tire,
surface, tire condition, starting state, driver inputs) can be fixed, randomized or swept, and
exports the time series for training, system identification or analysis.

- [Quick start](#quick-start)
- [The batch spec](#the-batch-spec)
- [Choosing and varying variables](#choosing-and-varying-variables)
- [Driver inputs (maneuvers)](#driver-inputs-maneuvers)
- [Export options](#export-options)
- [What gets written](#what-gets-written)
- [Reproducibility](#reproducibility)
- [Performance](#performance)
- [How it works](#how-it-works)

## Quick start

The easiest way is the web GUI: run `driftsim-gui`, change what you want to vary, press Preview and
Start. It edits exactly the specs described below (Save JSON / Import), so everything here also
applies to it.

From the command line, run one of the ready-made specs in `examples/specs/`:

```bash
driftsim-datagen examples/specs/drift_dataset.json
```

It prints progress and writes `exports/<timestamp>_drift_dataset/`. The same from Python:

```python
from rc_drift_sim import default_spec, run_batch

spec = default_spec()
spec["episodes"] = 1000
spec["params"]["vehicle.mass"] = {"dist": "uniform", "low": 1.4, "high": 1.8}
summary = run_batch(spec)          # returns status, output folder, timing, ...
```

`examples/batch_export.py` is a complete runnable version, including reading the data back.

| example spec | what it makes |
|---|---|
| `drift_dataset.json` | 2,000 drift entries on P-tile with randomized mass, grip, wear, latency and slide inputs |
| `domain_randomization.json` | 5,000 episodes over all 4 tire compounds x 11 surfaces with randomized car, condition and random inputs |
| `sysid_chirp.json` | 120 frequency-sweep steering runs on 3 surfaces at 4 throttle levels (grid) for system identification |
| `understeer_sweep.json` | a 3 x 5 x 4 grid of CG position, steering and throttle for steady-state cornering |

## The batch spec

A spec is a JSON object. Everything is optional except what you want to change:

```json
{
  "name": "my_dataset",
  "seed": 0,
  "episodes": 1000,
  "duration_s": 4.0,
  "params":   { "vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8} },
  "maneuver": { "type": "random", "params": {} },
  "export":   { "formats": ["npz"], "signals": ["pose", "velocity", "derived", "actions"] },
  "run":      { "workers": 0, "device": "cpu" }
}
```

| field | meaning | default |
|---|---|---|
| `name` | used in the output folder name | `batch` |
| `seed` | master seed; the whole dataset is a pure function of the spec | `0` |
| `episodes` | number of episodes (up to 2,000,000) | `64` |
| `duration_s` | length of each episode in seconds | `4.0` |
| `params` | variables to fix, randomize or sweep ([below](#choosing-and-varying-variables)) | none: YAML defaults |
| `maneuver` | driver-input generator and its parameters ([below](#driver-inputs-maneuvers)) | `drift_schedule` |
| `export` | formats, signals, decimation, precision, shard size, output folder ([below](#export-options)) | NPZ, default signals |
| `run.workers` | worker processes; `0` = one per CPU core minus one (CPU only) | `0` |
| `run.device` | `cpu` (NumPy float64 reference), `mps` (Apple GPU), `cuda` (NVIDIA GPU), `auto` (best GPU, else CPU); see [GPU acceleration](#gpu-acceleration) | `cpu` |
| `run.precision` | GPU arithmetic: `float32` or `float64` (`float64` not on Apple GPUs) | `float32` |
| `run.compile` | fuse the GPU kernels with `torch.compile` (NVIDIA; ignored on Apple GPUs) | `false` |

`validate_spec(spec)` returns the normalized spec, a list of errors and a list of warnings; the CLI
and `run_batch` refuse to start on errors (unknown variable names, low > high, a mass that could be
negative, and so on).

## Choosing and varying variables

Every variable has a key. The full list with defaults, units and descriptions is in
[PARAMETERS.md](PARAMETERS.md); the main groups are:

| keys | what |
|---|---|
| `tire`, `surface` | tire compound and surface type, by name |
| `vehicle.*` | chassis: mass, wheelbase, CG position and height, wheel radius, steering geometry |
| `drivetrain.*` | layout, gearing, motor, battery, ESC, differentials |
| `actuators.*` | steering limits, servo speed, trim, control latency, steering gyro |
| `tire.*` | Magic Formula coefficients of the chosen compound (grip, stiffness, relaxation, temperature window) |
| `surface.*` | properties of the chosen surface (grip multiplier, looseness, plowing drag, rolling resistance) |
| `condition.*` | tire wear, wetness, dust and starting temperature (per wheel if you like) |
| `init.*` | starting speed, sideslip, yaw rate and pose |

Angles use degrees under `*_deg` names (`actuators.steer_max_deg`, `init.beta_deg`), like the YAML
files. Everything else is SI.

Each entry in `params` is a distribution:

```json
"vehicle.mass":        {"dist": "fixed", "value": 1.7}
"vehicle.cg_height":   {"dist": "uniform", "low": 0.03, "high": 0.04}
"actuators.steer_offset_deg": {"dist": "normal", "mean": 0.0, "std": 1.0, "low": -3, "high": 3}
"tire.relax_y":        {"dist": "loguniform", "low": 0.02, "high": 0.08}
"surface":             {"dist": "choice", "values": ["epoxy_ptile", "carpet"], "weights": [3, 1]}
"actuators.gyro_enabled": {"dist": "bernoulli", "p": 0.5}
"vehicle.cg_to_front": {"dist": "sweep", "values": [0.10, 0.133, 0.16]}
"init.speed":          {"dist": "linspace", "low": 0.0, "high": 3.0, "num": 7}
```

Two options:

- `"mode": "scale"` multiplies the default instead of replacing it. For example
  `"surface.mu_scale": {"dist": "normal", "mean": 1.0, "std": 0.05, "mode": "scale"}` gives every
  surface ±5 % grip around its own value, even when the surface itself is randomized.
- `"per_wheel": true` (only for `condition.*`) draws the four wheels independently, e.g. uneven wear.

**Sweeps.** Every `sweep` and `linspace` variable is one axis of a full-factorial grid. Episode `i`
takes grid point `i` modulo the grid size, with the first swept key in alphabetical order varying
fastest. Make `episodes` a multiple of the grid size to cover it evenly; random variables are still
drawn per episode on top of the grid.

**Structural settings.** `drivetrain.layout`, `drivetrain.reverse_enabled`, `tire.combined_mode`,
`sim.dt`, `sim.control_dt` and `sim.integrator` change the structure of the simulation. They can
vary too, but episodes are then split into separate vectorized groups, which is slower than varying
numeric variables (these can differ freely inside one batch).

## Driver inputs (maneuvers)

```json
"maneuver": {"type": "drift_schedule",
             "params": {"sustain_throttle": {"dist": "uniform", "low": 0.2, "high": 0.26},
                        "mirror_prob": {"dist": "fixed", "value": 0.5}}}
```

| type | inputs |
|---|---|
| `drift_schedule` | launch, full-throttle stab with a little steer, then counter-steer and a matched slide throttle |
| `constant` | one steering and throttle command for the whole episode |
| `step_steer` | straight, then a steering step |
| `sine_steer` | sinusoidal steering at constant throttle |
| `chirp` | steering sine sweeping from `f0_hz` to `f1_hz` (system identification) |
| `lane_change` | steer one way, then the other, then straight |
| `random` | band-limited random steering and throttle (broad state coverage) |

Every maneuver parameter accepts a distribution, and every maneuver has `mirror_prob`: the
probability of flipping the steering sign, for balanced left and right data. Parameters and defaults
are listed in [PARAMETERS.md](PARAMETERS.md#maneuvers).

## Export options

| option | meaning | default |
|---|---|---|
| `formats` | any of `npz`, `csv`, and `parquet` (when pyarrow is installed) | `["npz"]` |
| `signals` | signal groups to record, see [PARAMETERS.md](PARAMETERS.md#signals) | pose, velocity, derived, wheel_speeds, steering, actions, accel |
| `record_rate` | `"control"`: record at control steps (50 Hz); `"physics"`: record every integrator step (1 kHz), about 20x more rows | `"control"` |
| `decimation` | record every n-th step of the record rate | `1` (50 Hz) |
| `float32` | store single precision (half the size) | `true` |
| `compress` | compressed NPZ / zstd Parquet (smaller, slower) | `false` |
| `shard_episodes` | maximum episodes per shard file | `256` |
| `out_dir` | output root (relative paths are relative to the repository) | `exports` |

Signal groups include body velocities, wheel speeds, steering angle, commanded and applied
actions, IMU-like accelerations, tire forces and loads, slips, effective grip, tire temperatures,
load transfer, motor current and torque, motor voltage, motor RPM and battery voltage, and slip power.

## What gets written

```
exports/20260929-083740_my_dataset/
├── manifest.json         the full spec, versions, git commit, signal names and units, shards, timing, status
├── README.txt            how to load the files
├── all/                  every episode, spun out or not
│   ├── episodes.csv      one row per episode: every sampled variable and the outcome metrics
│   ├── shard_0000.npz    the time series of episodes 0..n
│   └── shard_0000.csv    (if requested) the same as a long table
└── not_spun/             the same layout with only the episodes that did not spin out (spun = False)
    ├── episodes.csv
    └── shard_0000.npz    the non-spun episodes of all/shard_0000.npz (no file if all of them spun)
```

`not_spun/` repeats those episodes' data, so it adds the non-spun fraction of the size on disk.

Reading an NPZ shard:

```python
import numpy as np
d = np.load("all/shard_0000.npz")   # or "not_spun/shard_0000.npz"
d["episode_id"]   # (n,)          rows of episodes.csv in the same folder
d["t"]            # (T,)          seconds
d["speed"]        # (n, T)        one array per scalar signal
d["omega"]        # (n, T, 4)     per-wheel signals, wheel order FL, FR, RL, RR
```

CSV and Parquet shards are long tables with one row per (episode, time) and per-wheel signals
expanded into `name_fl`, `name_fr`, `name_rl`, `name_rr` columns.

**Timing.** Row `k` is the state at `t = k * decimation * step`, where `step` is `control_dt`
(20 ms) at the default `record_rate` and the physics `dt` (1 ms) with `"record_rate": "physics"`.
`steer_cmd` / `throttle_cmd` are the commands in force at that time; `steer_applied` /
`throttle_applied` are the commands reaching the car after the control latency (zero until the first
command arrives). Commands change only at control steps, while the state (steering angle after the
servo, wheel and motor speed, motor current and voltage, ...) moves every physics step. Forces,
accelerations and voltages are evaluated at the recorded state with the applied command. The
physics rate gives the same states as the control rate at every 20th row; it is meant for comparing
with high-rate logs from a real car, and shards are made smaller so a shard's buffers stay under
about 256 MB.

**Outcome metrics** in `episodes.csv`, computed at the full control rate:

| column | meaning |
|---|---|
| `max_abs_beta_deg` | largest sideslip while moving |
| `drift_time_s` | time with 20° < abs(sideslip) < 80° and speed > 0.8 m/s |
| `spun` | abs(sideslip) reached 80° while moving |
| `max_speed`, `final_speed`, `distance` | in m/s and m |
| `max_abs_yaw_rate_deg_s` | largest yaw rate |
| `finite` | false if the simulation produced NaN (an extreme parameter combination) |
| `latency_steps`, `mirrored` | applied control delay in steps; whether the steering was mirrored |

## Reproducibility

Every (episode, variable) pair has its own random stream, seeded with `(seed, episode id, variable
name)`. So the same spec always gives the same data, and adding a new randomized variable does
not change the samples of the others. Each episode's data is also independent of the number of
worker processes and of how episodes are split into shards: it is bit-identical to simulating that
car on its own (only the grouping of episodes into shard files can differ). To regenerate a
dataset, pass its manifest back:

```bash
driftsim-datagen exports/20260929-083740_my_dataset/manifest.json
```

## Performance

Measured on the development laptop (Apple M4, 4 performance + 6 efficiency cores, NumPy CPU
backend, 1 kHz RK4 physics, 50 Hz control):

| batch | time | throughput |
|---|---|---|
| 1,000 episodes x 4 s, all tires x surfaces mixed | 6.6 s | 150 episodes/s |
| 5,000 episodes x 4 s, same | 19.6 s | 255 episodes/s, 51,000 control steps/s |

Tips: keep structural settings fixed; use `float32` and only the signal groups you need; use
`decimation` when 50 Hz is more than you need; for large batches use a GPU (below).

## GPU acceleration

With PyTorch installed (`pip install -e ".[gpu]"`; on an NVIDIA machine first install the CUDA
build that matches your driver from pytorch.org), `"run": {"device": "mps"}` (Apple Silicon) or
`"cuda"` (NVIDIA) simulates thousands of episodes at once on the GPU. The GUI has the same choice
under Export, Output options, Compute device. The same physics code runs on both backends:
`rc_drift_sim/sim/xp.py` maps the NumPy calls of the core to PyTorch.

What to expect:

* **Speed.** Apple M4 (10-core GPU), 20,000 episodes x 5 s of the drift dataset: 93 s on the CPU
  (9 NumPy workers, 54k control steps/s) vs 56 s on MPS (90k control steps/s, 1.7x). The GPU levels
  off at about 96k control steps/s from 16,000 cars; with a few hundred episodes the CPU is faster.
  NVIDIA GPUs have far more memory bandwidth and can also fuse the kernels (`run.compile`), so
  expect a much larger gain there; `python scripts/bench_devices.py` measures any machine.
* **Precision.** Apple GPUs only do float32, and float32 is the GPU default. Open-loop drifts are
  unstable, so a float32 episode drifts away from its float64 twin after a second or two; outcome
  statistics agree (both runs above spun 90.7 % of the time), single episodes do not. The CPU
  stays the bit-reproducible reference; on CUDA `"precision": "float64"` matches it to rounding.
* **Fusion.** `torch.compile` is not used on Apple GPUs: Metal allows 31 buffers per kernel and
  the fused tire kernels of this model need more, so MPS runs in eager mode. On CUDA it is opt-in
  and falls back to eager mode (with a warning) if compilation fails.

Episodes are simulated in GPU batches of up to 65,536 (and about 1 GB of recorded signals), then
written as the usual shards; `manifest.json` records the device and precision.

## How it works

1. The spec is validated and normalized.
2. Episodes are grouped by their structural settings. Inside a group, cars with different numeric
   parameters run together as one vectorized batch (`rc_drift_sim.sim.vehicle.VehicleBatch`); the
   result for every car is bit-identical to simulating it alone.
3. Groups are cut into shards and the shards run in parallel worker processes. Each worker samples
   its own episodes by id, simulates them with per-car control latency, records the signals and
   writes its shard.
4. Each shard goes into `all/`, and its non-spun episodes also into `not_spun/` under the same shard
   number. The two `episodes.csv` tables and `manifest.json` are written at the end. Cancelling or
   a failure keeps the shards already written and records the status in the manifest.

The code is in `rc_drift_sim/datagen/`: `catalog.py` (variables and their metadata), `spec.py`
(spec format and validation), `sampling.py`, `inputs.py` (maneuvers), `runner.py` and `export.py`.
