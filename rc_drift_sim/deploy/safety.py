"""Safety filter between the drift policy and the car.

The same function runs inside the RL environment during training (NumPy or PyTorch arrays) and on
the car (NumPy, ``deploy.runtime``), so the policy learns with exactly the filter it will drive with.
It only uses what the car measures (body velocity, yaw rate) and the policy's own grip estimate,
never the simulator's true state. In order:

1. rate limits: each command moves at most ``steer_rate`` / ``throttle_rate`` per control step
   (protects the servo and the ESC, removes chatter);
2. sideslip envelope: from ``beta_soft_deg`` the command blends toward a recovery command, fully at
   ``beta_hard_deg``. Recovery = counter-steer (front wheels along the front-axle velocity, plus
   ``yaw_damping`` against the rotation) and the throttle down to ``throttle_recover``;
3. rotation envelope: a yaw rate above ``yaw_margin`` x grip x g / speed (the friction-limited turn
   rate of a steady slide) triggers the same recovery, so a spin is caught before the sideslip grows;
4. speed limit: the throttle fades out between ``v_max`` and ``v_max`` + 0.5 m/s.

The returned ``override`` (0..1) is how much of 2-4 acted; the environment penalizes the change the
filter made, so the policy learns to stay inside the envelope instead of leaning on it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

G = 9.81


@dataclass
class SafetyConfig:
    enabled: bool = False
    beta_soft_deg: float = 50.0      # |sideslip| where the recovery starts to blend in
    beta_hard_deg: float = 70.0      # |sideslip| of full recovery
    yaw_margin: float = 2.0          # allowed |yaw rate| = margin x grip x g / speed
    v_max: float = 3.0               # m/s; the throttle fades out above this speed
    steer_rate: float = 0.35         # max change of the steer command per control step (0 = off)
    throttle_rate: float = 0.2       # max change of the throttle command per control step (0 = off)
    throttle_recover: float = 0.0    # throttle ceiling during a full recovery
    yaw_damping: float = 0.1         # extra counter-steer during recovery, rad per rad/s of yaw rate
    min_speed: float = 0.5           # m/s; below this the sideslip and rotation checks are off
    nominal_grip: float = 0.3        # friction coefficient assumed when no grip estimate is given

    def validate(self) -> None:
        if not 0.0 < self.beta_soft_deg < self.beta_hard_deg <= 180.0:
            raise ValueError("safety: need 0 < beta_soft_deg < beta_hard_deg <= 180")
        if self.yaw_margin <= 0 or self.v_max <= 0 or self.min_speed <= 0 or self.nominal_grip <= 0:
            raise ValueError("safety: yaw_margin, v_max, min_speed and nominal_grip must be positive")
        if self.steer_rate < 0 or self.throttle_rate < 0 or self.yaw_damping < 0:
            raise ValueError("safety: rates and yaw_damping must not be negative")
        if not -1.0 <= self.throttle_recover <= 1.0:
            raise ValueError("safety: throttle_recover must be in [-1, 1]")


def safety_filter(xp: Any, a: Any, prev: Any, vx: Any, vy: Any, r: Any, grip: Any, cfg: SafetyConfig,
                  steer_max: Any, lf: Any) -> tuple[Any, Any]:
    """Filter commands ``a`` (B, 2) = (steer, throttle) in [-1, 1].

    xp: ``numpy`` or ``sim.xp.TORCH``; prev: (B, 2) the previous filtered commands; vx, vy, r: (B,)
    measured body velocity (m/s) and yaw rate (rad/s); grip: (B,) estimated friction coefficient or
    None (``cfg.nominal_grip``); steer_max: road-wheel angle of a full steer command (rad, scalar or
    (B,)); lf: CG to front axle (m). Returns ``(filtered (B, 2), override (B,) in [0, 1])``."""
    steer, thr = a[:, 0], a[:, 1]
    if cfg.steer_rate > 0:
        steer = prev[:, 0] + xp.clip(steer - prev[:, 0], -cfg.steer_rate, cfg.steer_rate)
    if cfg.throttle_rate > 0:
        thr = prev[:, 1] + xp.clip(thr - prev[:, 1], -cfg.throttle_rate, cfg.throttle_rate)
    v = xp.hypot(vx, vy)
    moving = xp.clip((v - cfg.min_speed) / 0.25, 0.0, 1.0)
    beta = xp.arctan2(vy, xp.abs(vx))                      # reversing counts like the mirror slide
    soft, hard = math.radians(cfg.beta_soft_deg), math.radians(cfg.beta_hard_deg)
    w_beta = xp.clip((xp.abs(beta) - soft) / (hard - soft), 0.0, 1.0)
    g = cfg.nominal_grip if grip is None else xp.clip(grip, 0.02, 2.0)
    r_lim = cfg.yaw_margin * g * G / xp.maximum(v, cfg.min_speed)
    w_yaw = xp.clip((xp.abs(r) - r_lim) / (0.5 * r_lim), 0.0, 1.0)
    w = xp.maximum(w_beta, w_yaw) * moving
    d_rec = xp.arctan2(vy + lf * r, xp.abs(vx)) - cfg.yaw_damping * r
    steer_rec = xp.clip(d_rec / steer_max, -1.0, 1.0)
    thr_rec = xp.minimum(thr, cfg.throttle_recover)
    steer = (1.0 - w) * steer + w * steer_rec
    thr = (1.0 - w) * thr + w * thr_rec
    w_v = xp.clip((v - cfg.v_max) / 0.5, 0.0, 1.0)
    thr = xp.where(thr > 0.0, thr * (1.0 - w_v), thr)
    out = xp.clip(xp.stack([steer, thr], axis=1), -1.0, 1.0)
    return out, xp.maximum(w, w_v)


__all__ = ["SafetyConfig", "safety_filter", "G"]
