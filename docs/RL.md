# Reinforcement learning

`rc_drift_sim.rl` turns the simulator into reinforcement-learning environments (Milestone 4) and
includes a compact PPO trainer as the starting point for Milestone 5.

```python
import gymnasium as gym
import rc_drift_sim.rl as rl                      # registers the Gymnasium ids

env = gym.make("DriftSim/DriftHold-v0")           # one car, gymnasium.Env
envs = rl.DriftVectorEnv(1024, rl.EnvConfig(task="track"))           # 1024 cars, one simulation
envs = rl.DriftVectorEnv(8192, rl.EnvConfig(task="hold"), device="mps")  # physics on the GPU
```

```bash
python -m rc_drift_sim.rl.ppo --task hold --envs 1024 --steps 20000000 --out runs/hold
driftsim-train --task track --device mps --envs 8192 --out runs/track      # same, installed command
```

## Tasks

| id | task | reward per step (max) | ends early when |
|---|---|---|---|
| `DriftSim/DriftHold-v0` | sustain a drift: \|sideslip\| near `target_beta_deg` (30°), either direction, with the car rotating into the slide, at `target_speed` (1.5 m/s). Position does not matter (a steady donut). | 1.0 sideslip + 0.5 speed (1.5) | the car spins (\|sideslip\| > 80° while moving) |
| `DriftSim/DriftTrack-v0` | drive along a circle of radius `track_radius` (1.5 m; clockwise or counter-clockwise per episode) at `target_speed`, with a bonus for drifting into the turn | 1.0 line (× progress) + 0.5 speed along the line + 0.5 drift (2.0) | spin, or more than `max_track_error` (1 m) off the line |

Episodes last `episode_s` (8 s = 400 control steps at 50 Hz) and start with the car parked
(`init_speed` 0); `init_drift_prob` starts that share of episodes in the steady drift instead (a
curriculum: the policy can learn to hold a drift before it learns to enter one). Every step also
pays `-0.05 x` the squared change of the commands, and an early end costs 10. All weights are in
`RewardWeights` (`EnvConfig(reward=RewardWeights(beta=2.0, ...))`).

Reference scores (8 s episodes, 32 cars, `rl.evaluate`):

| policy | hold | track |
|---|---|---|
| model-based drift controller (`LQRDriftBaseline`, reads the true state) | 480 | - |
| circle-following grip driving (feed-forward steer + feedback) | - | 548 |
| open-loop drift schedule of the test suite | 130 | 63 (leaves the circle) |
| no input | 0 | 0 |
| uniform random | -15 | -5 |

A learned policy that drifts round the circle can reach about 800 on `track`.

## Observations and actions

Actions are `(steer, throttle)` in [-1, 1], the same commands as everywhere in the simulator; they
reach the car after the control latency (`actuators.latency`, per car) and go through the servo,
ESC and motor models.

`obs="sensors"` (default) is what the real car can measure, with Gaussian noise (`EnvConfig.noise`,
standard deviations): 

| name | signal | scale | noise |
|---|---|---|---|
| `gyro_z` | yaw rate (IMU gyro) | rad/s ÷ 5 | `gyro` 0.02 rad/s |
| `accel_x`, `accel_y` | body-frame specific force (IMU accelerometer) | ÷ 9.81 | `accel` 0.2 m/s² |
| `wheel_fl` ... `wheel_rr` | wheel surface speeds (front encoders, rear from the motor RPM) | m/s ÷ 3 | `wheel` 0.05 m/s |
| `vel_x`, `vel_y` | body velocity estimate (optical flow or an overhead tracker) | m/s ÷ 3 | `velocity` 0.05 m/s |
| `prev_steer`, `prev_throttle` | last commands | - | - |
| hold: `target_beta`, `target_speed` | the targets | rad, m/s ÷ 3 | - |
| track: `track_error`, `heading_err_sin/cos`, `direction` | distance to the line (+ = left), body heading vs the line, ±1 | m | - |

`obs="full"` gives the simulator state instead (velocities, wheel speeds, steering angle, motor
current, load transfer, relaxed slips, tire temperatures): useful for a privileged critic, for a
teacher policy, or to check how much the sensors lose. `env.obs_names` lists the columns.

## Domain randomization

`randomize` takes the same `params` dictionary as a batch-export spec ([DATA_GENERATION.md](DATA_GENERATION.md)),
and every episode draws its own car from it:

```python
rnd = {
    "tire": {"dist": "choice", "values": ["hard_plastic_drift", "rubber_onroad"]},
    "surface": {"dist": "choice", "values": ["epoxy_ptile", "polished_concrete"]},
    "vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8},
    "surface.mu_scale": {"dist": "normal", "mean": 1.0, "std": 0.1, "mode": "scale"},
    "condition.wear": {"dist": "uniform", "low": 0.0, "high": 0.6, "per_wheel": True},
    "actuators.latency": {"dist": "choice", "values": [0.02, 0.04, 0.06]},
    "init.speed": {"dist": "uniform", "low": 0.0, "high": 1.0},
}
env = rl.DriftVectorEnv(1024, rl.EnvConfig(task="hold", randomize=rnd))
```

Draws are reproducible: episode k of seed s is the same car in every run and on every device.
Structural settings (drivetrain layout, combined-slip mode, time steps, integrator) cannot vary
inside one environment; the latency can be up to 140 ms. `driftsim-train --randomize spec.json`
accepts a whole export spec (its `params` are used), so a spec saved from the GUI can be reused.

## Interfaces

* `DriftBatchEnv(num_envs, config, device="cpu"|"mps"|"cuda"|"auto")` is the core: `reset(seed)`
  returns the observations, `step(actions)` returns `(obs, reward, terminated, truncated, info)` as
  NumPy arrays, or torch tensors on the GPU (for GPU training loops). Finished cars are reset in the
  same step; `info["final_obs"]` and `info["final_idx"]` hold their last observations,
  `info["episode_return"]` / `["episode_length"]` their totals.
* `DriftVectorEnv` wraps it as a `gymnasium.vector.VectorEnv` (float32 NumPy, `AutoresetMode.SAME_STEP`,
  `infos["final_obs"]`, `infos["final_info"][i]["episode"]`), `DriftEnv` as a single `gymnasium.Env`.
  `gym.make_vec("DriftSim/DriftHold-v0", num_envs=N, vectorization_mode="vector_entry_point")` gives
  the vectorized one directly.
* `LQRDriftBaseline`, `zero_policy`, `random_policy` and `evaluate(env, policy)` are the reference
  policies and the scorer used above.

The NumPy physics runs about 25k car-steps/s in one process; on the Apple M4 GPU about 64k at 8,192
cars (see [DATA_GENERATION.md](DATA_GENERATION.md#gpu-acceleration) for the precision trade-off).

## PPO trainer

`rc_drift_sim.rl.ppo` (`driftsim-train`) is clipped PPO with GAE: Gaussian MLP policy (2 x 256, tanh),
MLP critic, running observation normalization, time-limit bootstrapping. Each iteration collects
`rollout` (32) steps from every car, then runs 5 epochs of 8 minibatches. It writes `<out>/log.jsonl`
(return, episode length, KL, action std per iteration) and `<out>/policy.pt`:

```python
from rc_drift_sim.rl.ppo import load_policy
model, cfg = load_policy("runs/hold/policy.pt")
env = rl.DriftBatchEnv(32, cfg)
print(rl.evaluate(env, lambda e: model.act(e.last_obs).numpy().astype(float)))
```

Next (Milestone 5 onwards): tune the trainer and rewards until the policy drifts as well as the LQR
baseline from sensor observations, then widen the randomization (Milestone 6) and export the
policy for the car (Milestone 8).
