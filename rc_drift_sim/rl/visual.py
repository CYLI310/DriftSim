"""Record one episode of a policy as plain lists, for the GUI's rollout viewer (and anyone who wants
to plot an episode): pose, sideslip, speeds, commands, reward and its terms, per control step.

    data = record_rollout(cfg, policy=model.act)          # a trained ActorCritic
    data = record_rollout(cfg, policy="lqr")              # the model-based reference controller
    data = record_rollout(cfg, model.act_with_grip, grip=0.4)   # on a chosen surface grip

With the safety filter on, ``steer`` / ``throttle`` are the commands the car got and ``raw_steer`` /
``raw_throttle`` what the policy asked for; ``grip_true`` / ``grip_est`` compare the true friction
coefficient with the policy's estimate.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ..sim import state as S
from .baselines import LQRDriftBaseline
from .config import EnvConfig
from .env import DriftBatchEnv

_TERMS = {"hold": ("r_beta", "r_speed"), "track": ("r_track", "r_speed", "r_drift")}


def record_rollout(cfg: EnvConfig, policy: Any = "zero", seed: int = 0, init_drift: bool | None = None,
                   stochastic: bool = False, grip: float | None = None, grip_change: tuple | None = None) -> dict:
    """One car, one episode (NumPy physics). ``policy``: "zero", "lqr" (hold only) or a callable
    ``act(obs (1, n_obs), deterministic) -> actions``, e.g. ``ActorCritic.act``. ``init_drift``
    overrides ``cfg.init_drift_prob`` (True: start drifting, False: start as configured with 0)."""
    if init_drift is not None:
        cfg = cfg.replace(init_drift_prob=1.0 if init_drift else 0.0)
    if grip is not None:
        cfg = cfg.replace(randomize=dict(cfg.randomize, **{"surface.mu_scale": {"dist": "fixed", "value": float(grip)}}))
    if grip_change is not None:
        cfg = cfg.replace(grip_change_prob=0.0)
    env = DriftBatchEnv(1, cfg, autoreset=False)
    env.reset(seed=seed)
    if policy == "zero":
        act: Callable[[], np.ndarray] = lambda: np.zeros((1, 2))  # noqa: E731
    elif policy == "lqr":
        ctrl = LQRDriftBaseline(env)
        act = lambda: ctrl(env)  # noqa: E731
    else:
        def act():
            out = policy(env.last_obs, deterministic=not stochastic)
            a, g = out if isinstance(out, tuple) else (out, None)
            a = np.asarray(a.cpu().numpy() if hasattr(a, "cpu") else a, dtype=np.float64).reshape(1, 2)
            return a, (None if g is None else np.asarray(g.cpu().numpy() if hasattr(g, "cpu") else g, dtype=np.float64).reshape(1))
    ep = env._sampler.episode(int(env.episode_id[0]))
    p = ep.params
    keys = ("t", "x", "y", "yaw_deg", "beta_deg", "speed", "yaw_rate_deg_s", "delta_deg", "steer", "throttle",
            "reward", *_TERMS[cfg.task], *(("track_error",) if cfg.task == "track" else ()),
            "grip_true", "grip_est", "override", "raw_steer", "raw_throttle")
    out: dict[str, list] = {k: [] for k in keys}

    def push_state(s: np.ndarray) -> None:
        out["x"].append(s[S.X]); out["y"].append(s[S.Y]); out["yaw_deg"].append(math.degrees(s[S.YAW]))  # noqa: E702
        out["speed"].append(math.hypot(s[S.VX], s[S.VY]))
        out["beta_deg"].append(math.degrees(math.atan2(s[S.VY], s[S.VX])) if out["speed"][-1] > 0.05 else 0.0)
        out["yaw_rate_deg_s"].append(math.degrees(s[S.R])); out["delta_deg"].append(math.degrees(s[S.DELTA]))  # noqa: E702

    push_state(env.state[0])
    out["t"].append(0.0)
    ended, total = "time limit", 0.0
    for k in range(env.max_steps):
        res = act()
        a, g = res if isinstance(res, tuple) else (res, None)
        a = np.clip(a, -1.0, 1.0)
        out["grip_true"].append(float(np.asarray(env.last_grip)[0]))
        out["grip_est"].append(None if g is None else float(g[0]))
        if grip_change is not None and k == int(round(grip_change[1] / env.control_dt)):
            env.model.tm.mu_scale[:] = env.model.tm.mu_scale * float(grip_change[0])
        _, r, term, trunc, info = env.step(a, grip=g)
        applied = np.asarray(info["action"])
        out["t"].append((k + 1) * env.control_dt)
        out["steer"].append(float(applied[0, 0])); out["throttle"].append(float(applied[0, 1]))  # noqa: E702
        out["raw_steer"].append(float(a[0, 0])); out["raw_throttle"].append(float(a[0, 1]))  # noqa: E702
        out["override"].append(float(np.asarray(info["override"])[0]) if "override" in info else 0.0)
        out["reward"].append(float(r[0]))
        for key in _TERMS[cfg.task]:
            out[key].append(float(info[key][0]))
        if cfg.task == "track":
            out["track_error"].append(float(info["track_error"][0]))
        push_state(env.state[0])
        total += float(r[0])
        if term[0] or trunc[0]:
            if term[0]:
                ended = "off the line" if cfg.task == "track" and bool(info["off_track"][0]) else "spin-out"
            break
    vp = p.vehicle
    rounded = {k: [None if v is None else round(float(v), 4) for v in vals] for k, vals in out.items()}
    return dict(task=cfg.task, dt=env.control_dt, steps=len(rounded["reward"]), episode_return=round(total, 2),
                ended=ended, target_beta_deg=round(math.degrees(float(env.target_beta[0])), 2),
                target_speed=round(float(env.target_speed[0]), 3), safety=bool(cfg.safety.enabled),
                track=dict(center=[float(env.center[0, 0]), float(env.center[0, 1])], radius=cfg.track_radius,
                           direction=float(env.direction[0])) if cfg.task == "track" else None,
                car=dict(length=float(vp.body_length), width=float(vp.body_width), wheelbase=float(vp.wheelbase),
                         track_width=float(vp.track_width), wheel_radius=float(vp.wheel_radius)),
                tire=ep.values.get("tire"), surface=ep.values.get("surface"), series=rounded)


__all__ = ["record_rollout"]
