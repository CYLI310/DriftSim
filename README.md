# RC Drift Sim

Vectorized 2D vehicle dynamics for a 1/10-scale RC drift car, built to train an RL policy that
drifts across many surfaces and tire conditions and transfers to a real car.

Status: **Milestone 1 complete** (NumPy physics, visualizer, tests). Next: Milestone 2 (surface
maps, roughness, tire-condition sampling), then the JAX port.

## Layout

```
docs/DESIGN.md          conventions, state layout, module contracts, findings
scripts/                make_m1_outputs.py (figures + GIFs), tune_drift.py (re-tune the drift test)
outputs/m1/             generated Milestone 1 results
rc_drift_sim/
  configs/              vehicle, tire compounds, surfaces (YAML)
  sim/                  physics: vehicle, tire, surface, drivetrain, actuators, integrator,
                        equilibrium (trims, stability), stability (stiffness guard)
  control/              open-loop maneuvers, LQR drift controller
  viz/                  top-down renderer, time-series plots
  envs/ train/ sysid/   placeholders for later milestones
  tests/                pytest suite
```

## Commands

```bash
.venv/bin/pip install --no-cache-dir -e ".[dev]"
.venv/bin/pytest -q                             # ~1 min
.venv/bin/pytest -q -s -m benchmark             # throughput numbers
.venv/bin/python scripts/make_m1_outputs.py     # figures + GIFs into outputs/m1/
.venv/bin/python scripts/tune_drift.py          # re-tune the open-loop drift test
.venv/bin/python -m pyflakes rc_drift_sim scripts
```

## Quick start

```python
from rc_drift_sim.sim.vehicle import make_vehicle
v = make_vehicle("hard_plastic_drift", "epoxy_ptile")
traj = v.rollout(v.initial_state(), actions)            # actions: (T, 2) steer, throttle in [-1, 1]
states, _ = v.rollout_batch(s0_batch, actions_batch)     # (B, NS) envs in lock-step
```
