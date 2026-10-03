"""Gymnasium interfaces of the drift environment.

    import gymnasium as gym
    import rc_drift_sim.rl                                     # registers the ids below

    env = gym.make("DriftSim/DriftHold-v0")                    # one car (gymnasium.Env)
    envs = gym.make_vec("DriftSim/DriftHold-v0", num_envs=256, vectorization_mode="vector_entry_point")
    envs = DriftVectorEnv(256, EnvConfig(task="track"), device="mps")   # same, GPU physics

Ids: ``DriftSim/DriftHold-v0`` (sustain a drift) and ``DriftSim/DriftTrack-v0`` (drift around a
circle). Keyword arguments are ``EnvConfig`` fields (``randomize=...``, ``obs="full"``, ...).
Observations and actions are float32 NumPy arrays; actions are ``(steer, throttle)`` in [-1, 1].
``DriftVectorEnv`` steps all cars in one vectorized simulation (not one Python env per car) and
resets finished cars in the same step (``AutoresetMode.SAME_STEP``: ``infos["final_obs"]``,
``infos["final_info"]``).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from ..sim.xp import to_numpy
from .config import EnvConfig
from .env import DriftBatchEnv


def _config(config: EnvConfig | None, kwargs: dict) -> EnvConfig:
    cfg = config if config is not None else EnvConfig()
    return cfg.replace(**kwargs) if kwargs else cfg


def _spaces(n_obs: int) -> tuple[spaces.Box, spaces.Box]:
    obs = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
    act = spaces.Box(-1.0, 1.0, (2,), np.float32)
    return obs, act


class DriftEnv(gym.Env):
    """One car (the NumPy float64 physics). ``info`` has the reward terms, sideslip and speed."""

    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None, **kwargs: Any):
        self.core = DriftBatchEnv(1, _config(config, kwargs), autoreset=False)
        self.observation_space, self.action_space = _spaces(self.core.n_obs)
        self.obs_names = self.core.obs_names
        self.render_mode = render_mode

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self.core.reset(seed=seed)
        return obs[0], {}

    def step(self, action):
        obs, reward, term, trunc, info = self.core.step(np.asarray(action, dtype=np.float64)[None])
        return (obs[0], float(reward[0]), bool(term[0]), bool(trunc[0]),
                {k: (v[0].item() if np.ndim(v) else v) for k, v in info.items()
                 if k not in ("done",) and np.ndim(v) <= 1})


class DriftVectorEnv(VectorEnv):
    """``num_envs`` cars stepped together (NumPy, or PyTorch on ``device`` "mps" / "cuda" / "auto")."""

    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, num_envs: int = 1, config: EnvConfig | None = None, device: str = "cpu",
                 precision: str = "float32", **kwargs: Any):
        self.core = DriftBatchEnv(num_envs, _config(config, kwargs), device=device, precision=precision,
                                  autoreset=True)
        self.num_envs = int(num_envs)
        self.single_observation_space, self.single_action_space = _spaces(self.core.n_obs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.obs_names = self.core.obs_names
        self.render_mode = None

    def reset(self, *, seed: int | list | None = None, options: dict | None = None):
        if isinstance(seed, (list, tuple)):
            seed = seed[0]
        obs = self.core.reset(seed=seed)
        return to_numpy(obs).astype(np.float32), {}

    def step(self, actions):
        obs, reward, term, trunc, info = self.core.step(actions)
        obs = to_numpy(obs).astype(np.float32)
        infos: dict[str, Any] = {}
        for k in ("beta_deg", "speed", "track_error"):
            if k in info:
                infos[k] = to_numpy(info[k])
        if "final_idx" in info:
            idx = info["final_idx"]
            final_obs = to_numpy(info["final_obs"]).astype(np.float32)
            infos["final_obs"] = np.empty(self.num_envs, dtype=object)
            infos["_final_obs"] = np.zeros(self.num_envs, dtype=bool)
            infos["final_info"] = np.empty(self.num_envs, dtype=object)
            infos["_final_info"] = np.zeros(self.num_envs, dtype=bool)
            ret, length = to_numpy(info["episode_return"]), to_numpy(info["episode_length"])
            for j, i in enumerate(idx):
                infos["final_obs"][i] = final_obs[j]
                infos["_final_obs"][i] = True
                infos["final_info"][i] = {"episode": {"r": float(ret[i]), "l": int(length[i])}}
                infos["_final_info"][i] = True
        return (obs, to_numpy(reward).astype(np.float32), to_numpy(term).astype(bool),
                to_numpy(trunc).astype(bool), infos)


def make_vec(num_envs: int = 1, **kwargs: Any) -> DriftVectorEnv:
    """Vector entry point for ``gymnasium.make_vec(..., vectorization_mode="vector_entry_point")``."""
    kwargs.pop("render_mode", None)
    return DriftVectorEnv(num_envs, **kwargs)


__all__ = ["DriftEnv", "DriftVectorEnv", "make_vec"]
