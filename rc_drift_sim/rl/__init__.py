"""Reinforcement-learning environments for drifting (Milestone 4).

    from rc_drift_sim.rl import DriftBatchEnv, DriftVectorEnv, DriftEnv, EnvConfig
    env = DriftVectorEnv(1024, EnvConfig(task="hold", randomize={...}), device="cpu")

Importing this package registers the Gymnasium ids ``DriftSim/DriftHold-v0`` and
``DriftSim/DriftTrack-v0``. See docs/RL.md for tasks, observations, rewards and randomization.
"""
from __future__ import annotations

import gymnasium as gym

from .baselines import LQRDriftBaseline, evaluate, random_policy, zero_policy
from .config import OBS_MODES, TASKS, EnvConfig, RewardWeights
from .env import DriftBatchEnv, mirror_state
from .gym_env import DriftEnv, DriftVectorEnv, make_vec

for _id, _task in (("DriftSim/DriftHold-v0", "hold"), ("DriftSim/DriftTrack-v0", "track")):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point="rc_drift_sim.rl.gym_env:DriftEnv",
                     vector_entry_point="rc_drift_sim.rl.gym_env:make_vec", kwargs={"task": _task})

__all__ = ["DriftBatchEnv", "DriftEnv", "DriftVectorEnv", "EnvConfig", "RewardWeights", "TASKS", "OBS_MODES",
           "LQRDriftBaseline", "evaluate", "random_policy", "zero_policy", "mirror_state", "make_vec"]
