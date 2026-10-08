"""Grip-sweep evaluation: how well a policy slides on surfaces from very slippery to grippy.

Used by the trainer to keep the best policy (``best.pt``), by the GUI's final-model report and by
anyone comparing policies. Every grip level gets ``cars`` cars (each one episode, deterministic
actions, parked start, nominal targets, no mid-episode grip change); the rest of the run's domain
randomization (car, latency, sensor noise, ...) still applies, so the levels are judged on the same
kind of unknown cars the policy trained on.

    res = evaluate_grip_sweep(model.act_with_grip, cfg, levels=(0.3, 0.6, 0.9), cars=8)
    res["score"], res["levels"][0]["spin_share"], res["levels"][0]["grip"]
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ..sim.xp import to_numpy
from .config import EnvConfig
from .env import MOVING, DriftBatchEnv

DEFAULT_LEVELS = (0.25, 0.45, 0.65, 0.85, 1.05)    # surface grip multipliers (surface.mu_scale)


def sweep_config(cfg: EnvConfig, levels: tuple | list) -> EnvConfig:
    """The evaluation version of ``cfg``: one fixed grip level per car (car k gets level k % n)."""
    rnd = dict(cfg.randomize)
    rnd = {k: v for k, v in rnd.items() if not k.startswith("init.")}
    rnd["surface.mu_scale"] = {"dist": "sweep", "values": [float(x) for x in levels]}
    return cfg.replace(randomize=rnd, grip_change_prob=0.0, target_beta_jitter_deg=0.0, target_speed_jitter=0.0,
                       init_drift_prob=0.0, init_speed=0.0)


class GripSweep:
    """A reusable evaluation environment (building one compiles every car, so the trainer keeps it)."""

    def __init__(self, cfg: EnvConfig, levels: tuple | list = DEFAULT_LEVELS, cars: int = 8, device: str = "cpu",
                 precision: str = "float32", seed: int = 777):
        self.levels = [float(x) for x in levels]
        self.cars, self.seed = int(cars), int(seed)
        self.cfg = sweep_config(cfg, self.levels)
        n = len(self.levels) * self.cars
        self.env = DriftBatchEnv(n, self.cfg, device=device, precision=precision, autoreset=False)
        self.level_of_car = np.arange(n) % len(self.levels)

    def run(self, policy: Callable[[Any], Any]) -> dict:
        """``policy(obs) -> actions`` or ``(actions, grip estimate)`` (e.g. ``ActorCritic.act_with_grip``)."""
        env, cfg = self.env, self.cfg
        env.reset(seed=self.seed)
        B = env.num_envs
        alive = np.ones(B, dtype=bool)
        ret = np.zeros(B)
        spun = np.zeros(B, dtype=bool)
        length = np.zeros(B)
        n_moving = np.zeros(B)
        beta_err = np.zeros(B)
        speed_err = np.zeros(B)
        overrides = np.zeros(B)
        grip_true = np.zeros(B)
        grip_abs_err = np.zeros(B)
        n_grip = np.zeros(B)
        half = env.max_steps // 2
        tb = math.radians(cfg.target_beta_deg)
        for k in range(env.max_steps):
            out = policy(env.last_obs)
            act, g_hat = out if isinstance(out, tuple) else (out, None)
            g_true = to_numpy(env.last_grip).astype(np.float64)
            if g_hat is not None:
                gh = to_numpy(g_hat).astype(np.float64).reshape(-1)
                grip_abs_err += np.where(alive, np.abs(gh - g_true), 0.0)
                n_grip += alive
            grip_true += np.where(alive, g_true, 0.0)
            _, r, term, trunc, info = env.step(act if env.device is not None else to_numpy(act).astype(np.float64),
                                               grip=g_hat)
            r, term, trunc = to_numpy(r), to_numpy(term), to_numpy(trunc)
            ret += np.where(alive, r, 0.0)
            length += alive
            if "override" in info:
                overrides += np.where(alive, to_numpy(info["override"]) > 0.01, 0.0)
            if k >= half:                      # tracking error once the drift should be established
                speed = to_numpy(info["speed"])
                mov = alive & (speed > MOVING)
                b = np.radians(np.abs(to_numpy(info["beta_deg"])))
                beta_err += np.where(mov, np.abs(b - tb), 0.0)
                speed_err += np.where(mov, np.abs(speed - cfg.target_speed), 0.0)
                n_moving += mov
            spun |= alive & term
            alive &= ~(term | trunc)
            if not alive.any():
                break
        rows = []
        for j, lvl in enumerate(self.levels):
            m = self.level_of_car == j
            mv = max(float(n_moving[m].sum()), 1.0)
            rows.append(dict(
                level=lvl, grip=round(float(grip_true[m].sum() / length[m].sum()), 3),
                mean_return=round(float(ret[m].mean()), 1), spin_share=round(float(spun[m].mean()), 3),
                override_share=round(float(overrides[m].sum() / length[m].sum()), 4),
                beta_error_deg=round(math.degrees(float(beta_err[m].sum()) / mv), 1) if n_moving[m].sum() else None,
                speed_error=round(float(speed_err[m].sum()) / mv, 3) if n_moving[m].sum() else None,
                grip_error=round(float(grip_abs_err[m].sum() / n_grip[m].sum()), 3) if n_grip[m].sum() else None))
        return dict(score=round(float(ret.mean()), 2), spin_share=round(float(spun.mean()), 3),
                    override_share=round(float(overrides.sum() / length.sum()), 4), levels=rows,
                    cars_per_level=self.cars, episode_s=cfg.episode_s)


def evaluate_grip_sweep(policy: Callable[[Any], Any], cfg: EnvConfig, levels: tuple | list = DEFAULT_LEVELS,
                        cars: int = 8, device: str = "cpu", seed: int = 777) -> dict:
    return GripSweep(cfg, levels, cars, device=device, seed=seed).run(policy)


__all__ = ["GripSweep", "evaluate_grip_sweep", "sweep_config", "DEFAULT_LEVELS"]
