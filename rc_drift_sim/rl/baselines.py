"""Reference policies for the drift environments: a reward sanity check and a target for RL.

``LQRDriftBaseline`` is the model-based controller of Milestone 1 made batch-capable: launch,
throttle-stab entry, then the trim LQR (``control.lqr.TrimLQR`` gains, designed on the first car at
the target speed and sideslip, latency-aware). It reads the TRUE state (privileged), so it is an
upper reference for the ``hold`` task, not a deployable policy. ``random_policy`` and
``zero_policy`` are the lower references.

    env = DriftBatchEnv(64, EnvConfig(task="hold"))
    pol = LQRDriftBaseline(env)
    obs = env.reset(seed=0)
    for _ in range(env.max_steps):
        obs, r, term, trunc, info = env.step(pol(env))
"""
from __future__ import annotations

import math

import numpy as np

from ..control.lqr import TrimLQR
from ..control.maneuvers import LQR_ENTRY
from ..sim.equilibrium import CTRL, solve_trim
from ..sim.vehicle import Vehicle
from ..sim.xp import to_numpy
from .env import DriftBatchEnv, mirror_state


class LQRDriftBaseline:
    """Launch + throttle stab, then full-state LQR on the drift trim, for every car of a batch."""

    def __init__(self, env: DriftBatchEnv, entry: dict | None = None):
        cfg = env.cfg
        car = Vehicle(env._sampler.episode(0).params)
        self.trim = solve_trim(car, cfg.target_speed, beta=-math.radians(cfg.target_beta_deg))
        if not self.trim.success:
            raise RuntimeError(f"no drift equilibrium at the target: {self.trim.summary()}")
        self.delay = int(round(car.params.actuators.latency / car.control_dt))
        self.K = TrimLQR(car, self.trim, delay_steps=self.delay).K
        self.entry = dict(LQR_ENTRY if entry is None else entry)
        self.cdt = env.control_dt
        B = env.num_envs
        self.mode = np.zeros(B, dtype=np.int64)          # 0 launch, 1 throttle stab, 2 LQR
        self.pending = np.tile(self.trim.u, (B, max(self.delay, 1), 1))
        self.last_id = np.full(B, -1)
        self.side = np.ones(B)                          # -1: hold the mirror image (a right-hand drift)

    def __call__(self, env: DriftBatchEnv) -> np.ndarray:
        e, B = self.entry, env.num_envs
        s = to_numpy(env.state).astype(np.float64)
        t = to_numpy(env.t_step) * self.cdt
        new = env.episode_id != self.last_id               # a car started a new episode
        beta = np.degrees(np.arctan2(s[:, 4], s[:, 3]))          # sideslip atan2(vy, vx)
        drifting = new & (np.abs(beta) > e["switch_beta_deg"]) & (np.hypot(s[:, 3], s[:, 4]) > 0.5)
        self.mode[new] = np.where(drifting[new], 2, 0)    # an episode that starts in a drift: hold it at once
        self.side[new] = np.where(drifting[new] & (beta[new] > 0), -1.0, 1.0)
        self.pending[new] = self.trim.u
        self.last_id = env.episode_id.copy()
        s = np.where(self.side[:, None] < 0, mirror_state(s), s)
        beta = beta * self.side
        self.mode[(self.mode == 0) & (t >= e["launch_s"])] = 1
        switch = (self.mode == 1) & ((beta < -e["switch_beta_deg"]) | (t >= e["launch_s"] + e["kick_max_s"]))
        self.pending[switch] = self.trim.u
        self.mode[switch] = 2
        u = np.zeros((B, 2))
        u[self.mode == 0] = [0.0, e["launch_thr"]]
        u[self.mode == 1] = [e["kick_steer"], e["kick_thr"]]
        lqr = self.mode == 2
        if lqr.any():
            dx = s[lqr][:, CTRL] - self.trim.s[CTRL]
            z = np.concatenate([dx, (self.pending[lqr] - self.trim.u).reshape(lqr.sum(), -1)], axis=1) \
                if self.delay else dx
            ul = np.clip(self.trim.u - z @ self.K.T, [-1.0, 0.0], [1.0, 1.0])
            u[lqr] = ul
            if self.delay:
                self.pending[lqr] = np.concatenate([self.pending[lqr][:, 1:], ul[:, None]], axis=1)
        u[:, 0] *= self.side                              # steer back from the mirror image
        return u


def random_policy(env: DriftBatchEnv, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(-1.0, 1.0, (env.num_envs, 2))


def zero_policy(env: DriftBatchEnv) -> np.ndarray:
    return np.zeros((env.num_envs, 2))


def evaluate(env: DriftBatchEnv, policy, episodes_per_car: int = 1, seed: int = 0) -> dict:
    """Run ``policy(env) -> actions`` until every car finished ``episodes_per_car`` episodes; returns
    mean episode return and length, the early-termination share and the mean |sideslip| (deg)."""
    env.reset(seed=seed)
    B = env.num_envs
    done_count = np.zeros(B, dtype=np.int64)
    returns, lengths, early, betas = [], [], [], []
    while done_count.min() < episodes_per_car:
        _, _, term, trunc, info = env.step(policy(env))
        betas.append(np.abs(to_numpy(info["beta_deg"])))
        d = np.flatnonzero(to_numpy(term) | to_numpy(trunc))
        if len(d):
            ret, length, te = to_numpy(info["episode_return"]), to_numpy(info["episode_length"]), to_numpy(term)
            for i in d:
                if done_count[i] < episodes_per_car:
                    returns.append(ret[i]); lengths.append(length[i]); early.append(bool(te[i]))  # noqa: E702
                done_count[i] += 1
    return dict(mean_return=float(np.mean(returns)), mean_length=float(np.mean(lengths)),
                early_end_share=float(np.mean(early)), mean_abs_beta_deg=float(np.mean(betas)), episodes=len(returns))


__all__ = ["LQRDriftBaseline", "random_policy", "zero_policy", "evaluate"]
