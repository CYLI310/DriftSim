"""Actuators: radio-style command shaping, steering servo, Ackermann geometry, control latency.

Implements docs/DESIGN.md section 7 for the NumPy reference. All functions are pure and written
JAX-portably (``np.where``/``np.clip``, no value-dependent Python control flow; Python branches
only on configuration such as ``ap.gyro_enabled``). ``ActionDelay`` is the one stateful object
and lives outside the ODE (in the step wrapper), as the contract requires.

Conventions
-----------
* Commands ``steer_cmd``, ``thr_cmd`` are normalized to [-1, 1].
* Steering angle ``delta`` (rad) is positive = wheels turned LEFT; yaw rate ``r`` (rad/s) is
  positive = turning left. Per-wheel angles are returned in wheel order (FL, FR).
* Time in s, angles in rad, rates in rad/s.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from dataclasses import dataclass

from .params import ActuatorParams, VehicleParams
from .xp import is_torch, namespace


def _f64(x):
    return x if is_torch(x) else np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------------- precomputed model
@dataclass(frozen=True, slots=True)
class ActuatorModel:
    """Actuator constants. Floats for one car, or (B,) arrays for a batch of cars with different
    settings (every field is used against per-car scalars such as the steering command).

    ``gyro_gain_eff = gyro_gain * gyro_enabled`` so cars with and without a gyro can share a batch."""
    steer_max: object
    servo_rate: object
    servo_tau: object
    steer_deadband: object
    steer_offset: object
    steer_expo: object
    throttle_deadband: object
    throttle_expo: object
    gyro_gain_eff: object
    gyro_max_correction: object


def actuator_model(ap: ActuatorParams) -> ActuatorModel:
    return ActuatorModel(
        steer_max=float(ap.steer_max), servo_rate=float(ap.servo_rate), servo_tau=float(ap.servo_tau),
        steer_deadband=float(ap.steer_deadband), steer_offset=float(ap.steer_offset),
        steer_expo=float(ap.steer_expo), throttle_deadband=float(ap.throttle_deadband),
        throttle_expo=float(ap.throttle_expo),
        gyro_gain_eff=float(ap.gyro_gain) * (1.0 if ap.gyro_enabled else 0.0),
        gyro_max_correction=float(ap.gyro_max_correction))


def throttle_command_m(am: ActuatorModel, thr_cmd) -> np.ndarray:
    """ESC throttle from the raw command (deadband then expo) on a precomputed model."""
    return apply_expo(deadband(thr_cmd, am.throttle_deadband), am.throttle_expo)


def steering_target_m(am: ActuatorModel, steer_cmd, yaw_rate) -> np.ndarray:
    """Servo target angle (rad) on a precomputed model; see ``steering_target``."""
    xp = namespace(steer_cmd, yaw_rate)
    yaw_rate = _f64(yaw_rate)
    c = apply_expo(deadband(steer_cmd, am.steer_deadband), am.steer_expo)
    delta = c * am.steer_max + am.steer_offset
    correction = xp.clip(am.gyro_gain_eff * yaw_rate, -am.gyro_max_correction, am.gyro_max_correction)
    return xp.clip(delta - correction, -am.steer_max, am.steer_max)


def steering_rate_m(am: ActuatorModel, delta, delta_target) -> np.ndarray:
    """Servo slew ``clip((target - delta)/servo_tau, -servo_rate, servo_rate)`` (rad/s)."""
    return namespace(delta, delta_target).clip((_f64(delta_target) - _f64(delta)) / am.servo_tau,
                                               -am.servo_rate, am.servo_rate)


def ackermann_angles_m(L, half_track, ackermann, delta) -> tuple[np.ndarray, np.ndarray]:
    """``ackermann_angles`` on plain numbers or per-car arrays (wheelbase, half track, blend 0..1)."""
    xp = namespace(delta)
    delta = _f64(delta)
    s, c = xp.sin(delta), xp.cos(delta)
    full_fl = xp.arctan2(L * s, L * c - half_track * s)
    full_fr = xp.arctan2(L * s, L * c + half_track * s)
    return (1.0 - ackermann) * delta + ackermann * full_fl, (1.0 - ackermann) * delta + ackermann * full_fr


# ----------------------------------------------------------------------------- command shaping
def deadband(x: float | np.ndarray, db: float) -> np.ndarray:
    """Symmetric deadband with rescaling: 0 for ``|x| <= db``, then linear to +-1 at ``x = +-1``.

    ``y = sign(x) * max(|x| - db, 0) / (1 - db)``, input clipped to [-1, 1] first. ``db`` is a
    normalized command width in [0, 1). Odd-symmetric in ``x``.
    """
    xp = namespace(x, db)
    x = xp.clip(_f64(x), -1.0, 1.0)
    span = namespace(db).maximum(1.0 - db, 1e-12)      # config guard: db >= 1 would divide by zero
    return xp.sign(x) * xp.maximum(xp.abs(x) - db, 0.0) / span


def apply_expo(x: float | np.ndarray, expo: float) -> np.ndarray:
    """RC-style exponential: ``y = (1 - expo) * x + expo * x^3``.

    ``expo = 0`` is linear, ``expo = 1`` is purely cubic. Preserves ``y(+-1) = +-1`` and odd symmetry.
    """
    x = _f64(x)
    return (1.0 - expo) * x + expo * x ** 3


def throttle_command(ap: ActuatorParams, thr_cmd: float | np.ndarray) -> np.ndarray:
    """ESC throttle ``thr`` in [-1, 1] from the raw command: deadband then expo (dimensionless)."""
    return throttle_command_m(actuator_model(ap), thr_cmd)


# ----------------------------------------------------------------------------- steering
def steering_target(ap: ActuatorParams, steer_cmd: float | np.ndarray,
                    yaw_rate: float | np.ndarray) -> np.ndarray:
    """Servo target angle ``delta_target`` (rad) of the virtual center front wheel.

    ``c = expo(deadband(steer_cmd))``; ``delta = c * steer_max + steer_offset``;
    if the gyro is enabled: ``delta -= clip(gyro_gain * yaw_rate, -gyro_max_correction, +gyro_max_correction)``
    (counter-steer: a left yaw rate steers right); finally clipped to ``[-steer_max, steer_max]``.
    ``yaw_rate`` in rad/s. Odd-symmetric in ``steer_cmd`` when ``steer_offset = 0`` and the gyro
    term is zero.
    """
    return steering_target_m(actuator_model(ap), steer_cmd, yaw_rate)


def steering_rate(ap: ActuatorParams, delta: float | np.ndarray,
                  delta_target: float | np.ndarray) -> np.ndarray:
    """Servo dynamics ``d(delta)/dt`` (rad/s): first-order lag ``servo_tau`` with slew limit.

    ``clip((delta_target - delta) / servo_tau, -servo_rate, +servo_rate)``.
    """
    return steering_rate_m(actuator_model(ap), delta, delta_target)


def ackermann_angles(vp: VehicleParams, delta: float | np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-wheel front steer angles ``(delta_FL, delta_FR)`` (rad) from the center angle ``delta``.

    Parallel steering: both equal ``delta``. Full Ackermann (turn center on the rear-axle line,
    ``L = wheelbase``, ``w = track_width``): inner ``= atan(L / (L/tan(delta) - w/2))``,
    outer ``= atan(L / (L/tan(delta) + w/2))``. The two are blended by ``vp.ackermann`` (0..1).

    For both turn directions the contract's formulas reduce to
    ``delta_FL = atan2(L sin d, L cos d - (w/2) sin d)``, ``delta_FR = atan2(L sin d, L cos d + (w/2) sin d)``
    (left turn, d > 0: FL is inner and larger; right turn, d < 0: FR is inner and larger in
    magnitude). This form needs no ``tan(delta) ~ 0`` guard (it is exactly 0 at d = 0), has no
    singularity for any finite ``delta``, and is odd-symmetric: ``delta_FL(-d) = -delta_FR(d)``.
    """
    return ackermann_angles_m(float(vp.wheelbase), 0.5 * float(vp.track_width), float(vp.ackermann), delta)


# ----------------------------------------------------------------------------- latency buffer
class ActionDelay:
    """FIFO control-latency buffer of whole control steps (lives outside the ODE).

    ``n_delay = round(latency_s / control_dt)`` (half up); ``push(action)`` returns the action that was
    pushed ``n_delay`` calls earlier (``n_delay = 0`` returns ``action`` itself). ``reset(action0)``
    fills the whole buffer with ``action0`` so the first ``n_delay`` outputs equal ``action0``.

    Actions are ``(n_actions,)`` float64 arrays, ``(steer_cmd, throttle_cmd)`` by default.
    """

    def __init__(self, latency_s: float, control_dt: float, n_actions: int = 2) -> None:
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        self.latency_s = float(latency_s)
        self.control_dt = float(control_dt)
        self.n_actions = int(n_actions)
        # round half up (math.floor(x + 0.5)) so latency 0.05 / control_dt 0.02 gives 3, not
        # Python's banker's-rounded round(2.5) == 2
        self._n_delay = max(int(math.floor(self.latency_s / self.control_dt + 0.5)), 0)
        self._buf: deque[np.ndarray] = deque(maxlen=self._n_delay + 1)
        self.reset(np.zeros(self.n_actions, dtype=np.float64))

    @property
    def n_delay(self) -> int:
        """Delay in whole control steps."""
        return self._n_delay

    def _as_action(self, action) -> np.ndarray:
        a = np.array(action, dtype=np.float64).reshape(-1)     # copy: caller may reuse its array
        if a.shape[0] != self.n_actions:
            raise ValueError(f"action must have {self.n_actions} entries, got {a.shape[0]}")
        return a

    def reset(self, action0=None) -> None:
        """Fill the buffer with ``action0`` (zeros if None)."""
        a0 = (np.zeros(self.n_actions, dtype=np.float64) if action0 is None
              else self._as_action(action0))
        self._buf.clear()
        for _ in range(self._n_delay + 1):
            self._buf.append(a0.copy())

    def push(self, action) -> np.ndarray:
        """Enqueue ``action`` and return the action delayed by ``n_delay`` control steps."""
        self._buf.append(self._as_action(action))     # maxlen n_delay+1 evicts the oldest ...
        return self._buf[0].copy()                     # ... so index 0 is exactly n_delay pushes old

    def __len__(self) -> int:
        return len(self._buf)
