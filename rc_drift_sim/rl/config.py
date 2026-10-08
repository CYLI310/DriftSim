"""Configuration of the drift RL environments (Milestone 4).

Two tasks:

* ``hold``  - sustain a drift: sideslip magnitude near ``target_beta_deg`` (either direction, the car
  rotating into the slide) at ``target_speed``. Position does not matter: a steady donut.
* ``track`` - drive along a circle of radius ``track_radius`` (clockwise or counter-clockwise per
  episode) at ``target_speed`` with a bonus for drifting, ending the episode if the car strays more
  than ``max_track_error`` from the line (in the spirit of the Wheeled Lab oval task).

Domain randomization uses the batch-export spec language: ``randomize`` is a datagen ``params``
dict (``{"vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8}, "tire": {...}, ...}``) and every
episode draws its own car, tire condition and (optionally, ``init.*`` keys) initial state from it.

Safe, self-adapting sliding (``rl.presets``): ``safety`` puts the safety filter of ``deploy.safety``
between the policy and the car, ``history`` gives the policy the last N sensor readings so it can
infer the grip, ``grip_change_prob`` changes the grip in the middle of an episode (a wet patch) and
the target jitters make the targets per-episode commands.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ..deploy.safety import SafetyConfig

TASKS = ("hold", "track")
OBS_MODES = ("sensors", "full")
MAX_LATENCY_STEPS = 7          # action history kept for the per-car control latency (140 ms at 50 Hz)
MAX_HISTORY = 64               # sensor history the policy can see (1.28 s at 50 Hz)


@dataclass
class RewardWeights:
    """Per-step reward terms (all bounded) and the one-off spin-out penalty."""
    beta: float = 1.0           # hold: |sideslip| near the target (Gaussian, beta_sigma_deg), car rotating into the slide
    speed: float = 0.5          # speed near the target (hold: |v|, track: speed along the path)
    track: float = 1.0          # track: distance from the line (Gaussian, track_sigma) x progress along it
    drift: float = 0.5          # track: bonus while drifting into the turn (|sideslip| above drift_min_deg)
    action_rate: float = 0.05   # penalty on sum of squared command changes between steps
    spin: float = 10.0          # penalty when the episode ends early (spin-out, off track, numerical failure)
    intervention: float = 1.0   # penalty on the squared change the safety filter made to the commands
    beta_sigma_deg: float = 10.0
    speed_sigma: float = 0.5    # m/s
    track_sigma: float = 0.3    # m
    drift_min_deg: float = 15.0


@dataclass
class EnvConfig:
    task: str = "hold"
    episode_s: float = 8.0                  # episode length (truncation), s
    seed: int = 0
    randomize: dict = field(default_factory=dict)     # datagen spec "params" (domain randomization)
    obs: str = "sensors"                    # "sensors": what the real car can measure; "full": simulator state
    noise: dict = field(default_factory=lambda: dict(gyro=0.02, accel=0.2, wheel=0.05, velocity=0.05))
    target_beta_deg: float = 30.0
    target_speed: float = 1.5               # m/s
    track_radius: float = 1.5               # m
    track_both_ways: bool = True            # track: half of the episodes run clockwise
    max_track_error: float = 1.0            # m
    init_speed: float = 0.0                 # m/s, straight-line start (unless randomize has init.* keys)
    init_drift_prob: float = 0.0            # share of episodes that start in a steady drift (curriculum)
    spin_beta_deg: float = 80.0             # |sideslip| that counts as a spin-out while moving
    history: int = 1                        # sensor readings per observation (newest last; 1 = current only)
    front_wheel_speeds: bool = True         # sensors: include the front wheel speeds (False: rear encoder only)
    target_beta_jitter_deg: float = 0.0     # hold: per-episode target sideslip = target +- U(jitter)
    target_speed_jitter: float = 0.0        # per-episode target speed = target +- U(jitter), m/s
    grip_change_prob: float = 0.0           # share of episodes whose grip changes once, mid-episode
    grip_change_min: float = 0.5            # the grip is multiplied by a log-uniform factor in [min, max]
    grip_change_max: float = 1.5
    reward: RewardWeights = field(default_factory=RewardWeights)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def __post_init__(self):
        if isinstance(self.reward, dict):
            self.reward = RewardWeights(**self.reward)
        if isinstance(self.safety, dict):
            self.safety = SafetyConfig(**self.safety)

    def validate(self) -> None:
        from ..datagen.catalog import STRUCTURAL
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.obs not in OBS_MODES:
            raise ValueError(f"obs must be one of {OBS_MODES}, got {self.obs!r}")
        if self.episode_s <= 0 or self.target_speed <= 0 or self.track_radius <= 0:
            raise ValueError("episode_s, target_speed and track_radius must be positive")
        if not 0.0 <= self.init_drift_prob <= 1.0:
            raise ValueError("init_drift_prob must be in [0, 1]")
        bad = sorted(set(self.randomize) & set(STRUCTURAL))
        if bad:
            raise ValueError(f"structural settings cannot vary inside one environment batch: {bad}")
        unknown = sorted(set(self.noise) - {"gyro", "accel", "wheel", "velocity"})
        if unknown:
            raise ValueError(f"unknown noise keys {unknown}")
        if not 1 <= int(self.history) <= MAX_HISTORY or int(self.history) != self.history:
            raise ValueError(f"history must be a whole number from 1 to {MAX_HISTORY}")
        if self.target_beta_jitter_deg < 0 or self.target_speed_jitter < 0:
            raise ValueError("target jitters must not be negative")
        if self.target_beta_deg - self.target_beta_jitter_deg <= 0 or self.target_speed - self.target_speed_jitter <= 0:
            raise ValueError("the jittered targets must stay positive")
        if not 0.0 <= self.grip_change_prob <= 1.0:
            raise ValueError("grip_change_prob must be in [0, 1]")
        if not 0.0 < self.grip_change_min <= self.grip_change_max:
            raise ValueError("need 0 < grip_change_min <= grip_change_max")
        self.safety.validate()

    def replace(self, **changes: Any) -> "EnvConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
