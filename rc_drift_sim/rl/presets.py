"""Ready-made training setups, in the GUI's settings format (``rl.catalog.build_configs`` input).

``safe-adaptive``: safety-filtered, self-adapting controlled sliding on unknown low-friction surfaces.
The car must hold a commanded drift (sideslip and speed vary per episode) from a parked start on a
surface whose grip it is not told: three hard-floor surfaces with their grip scaled 0.3x to 1.3x
(peak friction coefficient about 0.09 to 0.46 with the plastic drift tires), dust, wear, stiffness,
mass, steering offset and latency all random, and in 30 % of the episodes the grip changes by up to
0.6x / 1.6x halfway through (a wet or dusty patch). The policy sees 16 readings (0.32 s) of the
gyro, accelerometer, body velocity and the rear (driven) wheel speed - no front encoders needed - a
grip estimator learns the friction from the same readings, the critic sees the true state, the
safety filter is on in training as on the car, and the best policy on a grip sweep is kept.

    from rc_drift_sim.rl.presets import preset_body
    body = preset_body("safe-adaptive")
"""
from __future__ import annotations

import copy

LOW_FRICTION_RANDOMIZE = {
    "surface": {"dist": "choice", "values": ["epoxy_ptile", "polished_concrete", "wet_asphalt"]},
    "surface.mu_scale": {"dist": "uniform", "low": 0.3, "high": 1.3, "mode": "scale"},
    "surface.stiffness_scale": {"dist": "uniform", "low": 0.7, "high": 1.2, "mode": "scale"},
    "condition.contamination": {"dist": "uniform", "low": 0.0, "high": 0.5},
    "condition.wear": {"dist": "uniform", "low": 0.0, "high": 0.5},
    "vehicle.mass": {"dist": "uniform", "low": 0.9, "high": 1.1, "mode": "scale"},
    "drivetrain.motor_kv": {"dist": "uniform", "low": 0.9, "high": 1.1, "mode": "scale"},
    "actuators.steer_offset_deg": {"dist": "normal", "mean": 0.0, "std": 1.0, "low": -2.5, "high": 2.5},
    "actuators.latency": {"dist": "choice", "values": [0.02, 0.03, 0.04, 0.06]},
}

PRESETS = {
    "safe-adaptive": dict(
        label="Safe adaptive sliding on unknown low-friction surfaces",
        desc="The recommended setup for the final model: safety filter on, 16-step sensor history, grip estimator, "
             "privileged critic, unknown low grip (with mid-episode changes), commanded sideslip and speed, "
             "best policy kept by a grip-sweep evaluation. 60 M steps.",
        body=dict(
            name="safe_adaptive",
            env=dict(task="hold", episode_s=10.0, target_beta_deg=30.0, target_speed=1.5, target_beta_jitter_deg=8.0,
                     target_speed_jitter=0.3, init_speed=0.0, init_drift_prob=0.3, obs="sensors", history=16,
                     front_wheel_speeds=False,
                     grip_change_prob=0.3, grip_change_min=0.6, grip_change_max=1.6),
            noise=dict(gyro=0.02, accel=0.3, wheel=0.05, velocity=0.06),
            reward=dict(action_rate=0.1, intervention=1.0, spin=10.0),
            safety=dict(enabled=True),
            ppo=dict(total_steps=60_000_000, num_envs=4096, rollout=32, lr=3e-4, lr_schedule="linear",
                     grip_estimator=True, privileged_critic=True, eval_every=25, eval_cars=8),
            run=dict(device="auto", precision="float32", snapshots=20),
            randomize_mode="custom",
            randomize=LOW_FRICTION_RANDOMIZE,
        )),
    "baseline": dict(
        label="Plain drift (no filter, no adaptation)",
        desc="The original Milestone 4 setup: one known car and surface, current sensor readings only. Useful as a "
             "comparison: it does not cope with unknown grip.",
        body=dict(name="baseline", randomize_mode="none", randomize={}),
    ),
}
ALIASES = {"safe_adaptive": "safe-adaptive", "safe-adaptive-low-friction": "safe-adaptive", "final": "safe-adaptive"}


def preset_body(name: str) -> dict:
    key = ALIASES.get(name, name)
    if key not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[key]["body"])


def presets_for_gui() -> list[dict]:
    return [dict(key=k, label=v["label"], desc=v["desc"], body=copy.deepcopy(v["body"])) for k, v in PRESETS.items()]


__all__ = ["PRESETS", "LOW_FRICTION_RANDOMIZE", "preset_body", "presets_for_gui"]
