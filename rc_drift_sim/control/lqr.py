"""Linear-quadratic regulators around a trim, and the drift-capture policy used by tests and scripts.

``TrimLQR`` holds the car at a steady-state operating point from ``sim.equilibrium`` (e.g. an
open-loop-unstable drift equilibrium) with full-state feedback, designed on the one-control-period
discrete model of the real integrator and augmented for an integer control latency.
``LQRDriftPolicy`` wraps it into a complete manoeuvre: open-loop launch, throttle-stab entry, then
the LQR as soon as the sideslip passes a threshold, with the control latency applied inside so the
recorded actions are the ones that reached the car.

Both are callables ``policy(t, s, info) -> action`` for ``Vehicle.rollout`` (keep one instance per
rollout: they store the pending, not yet applied, inputs).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from ..sim import state as S
from ..sim.actuators import ActionDelay
from ..sim.equilibrium import CTRL, Trim, discrete_model
from ..sim.vehicle import Vehicle


def lqr_gain(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray, delay_steps: int = 0
             ) -> np.ndarray:
    """Discrete LQR gain. With ``delay_steps`` > 0 the model is augmented with the pending inputs
    (an input applied at step k acts at step k + delay), and the returned gain acts on
    ``[x, u_{k-d}, ..., u_{k-1}]``."""
    n, m = Bd.shape
    d = int(delay_steps)
    if d == 0:
        P = solve_discrete_are(Ad, Bd, Q, R)
        return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    N = n + d * m
    Aa = np.zeros((N, N))
    Ba = np.zeros((N, m))
    Aa[:n, :n] = Ad
    Aa[:n, n:n + m] = Bd                      # the oldest pending input acts now
    for i in range(d - 1):                    # shift register of pending inputs
        Aa[n + i * m:n + (i + 1) * m, n + (i + 1) * m:n + (i + 2) * m] = np.eye(m)
    Ba[n + (d - 1) * m:, :] = np.eye(m)       # newest input enters the register
    Qa = np.zeros((N, N))
    Qa[:n, :n] = Q
    Qa[n:, n:] = 1e-6 * np.eye(d * m)
    P = solve_discrete_are(Aa, Ba, Qa, R)
    return np.linalg.solve(R + Ba.T @ P @ Ba, Ba.T @ P @ Aa)


class TrimLQR:
    """Full-state LQR that holds the vehicle at a trim (e.g. a drift equilibrium).

    ``u = clip(u* - K [x - x*, pending inputs - u*])``. Accounts for a known integer control
    latency (in control steps). Callable as ``policy(t, s, info) -> action`` for
    ``Vehicle.rollout`` (keep one instance per rollout: it stores the pending inputs).
    """

    def __init__(self, vehicle: Vehicle, trim: Trim, Q: np.ndarray | None = None,
                 R: np.ndarray | None = None, delay_steps: int = 0):
        self.trim = trim
        self.delay = int(delay_steps)
        Ad, Bd = discrete_model(vehicle, trim)
        n = len(CTRL)
        if Q is None:
            q = np.full(n, 1e-3)
            q[:3] = [10.0, 10.0, 10.0]          # vx, vy, r: the drift itself
            Q = np.diag(q)
        if R is None:
            R = np.diag([1.0, 1.0])
        self.K = lqr_gain(Ad, Bd, Q, R, self.delay)
        self.Ad, self.Bd = Ad, Bd
        self.pending = [trim.u.copy() for _ in range(self.delay)]

    def __call__(self, t: float, s: np.ndarray, info=None) -> np.ndarray:
        dx = s[CTRL] - self.trim.s[CTRL]
        z = np.concatenate([dx] + [p - self.trim.u for p in self.pending]) if self.delay else dx
        u = np.clip(self.trim.u - self.K @ z, [-1.0, 0.0], [1.0, 1.0])
        if self.delay:
            self.pending = self.pending[1:] + [u.copy()]
        return u


class LQRDriftPolicy:
    """Launch + throttle-stab entry (open loop), then ``TrimLQR`` on the drift trim.

    entry : dict with ``launch_s, launch_thr, kick_steer, kick_thr, kick_max_s, switch_beta_deg``
        (see ``control.maneuvers.LQR_ENTRY``). The controller takes over when the sideslip drops
        below ``-switch_beta_deg`` (left-hand drift) or after ``kick_max_s`` of throttle stab.
    The control latency (``params.actuators.latency``) is applied inside, so pass the policy
    straight to ``Vehicle.rollout``; ``t_switch`` records the take-over time (s).
    """

    def __init__(self, vehicle: Vehicle, trim: Trim, entry: dict, design_vehicle: Vehicle | None = None):
        cdt = vehicle.control_dt
        design = vehicle if design_vehicle is None else design_vehicle
        d = int(round(design.params.actuators.latency / design.control_dt))
        self.trim, self.entry = trim, entry
        self.ctrl = TrimLQR(design, trim, delay_steps=d)
        self.delay = ActionDelay(vehicle.params.actuators.latency, cdt)
        self.delay.reset([0.0, 0.0])
        self.mode = 0
        self.t_switch: float | None = None

    def __call__(self, t: float, s: np.ndarray, info=None) -> np.ndarray:
        e = self.entry
        beta = np.degrees(np.arctan2(s[S.VY], s[S.VX]))
        if self.mode == 0 and t >= e["launch_s"]:
            self.mode = 1
        if self.mode == 1 and (beta < -e["switch_beta_deg"] or t >= e["launch_s"] + e["kick_max_s"]):
            self.mode, self.t_switch = 2, t
            self.ctrl.pending = [self.trim.u.copy() for _ in range(self.ctrl.delay)]
        if self.mode == 0:
            u = np.array([0.0, e["launch_thr"]])
        elif self.mode == 1:
            u = np.array([e["kick_steer"], e["kick_thr"]])
        else:
            u = self.ctrl(t, s)
        return self.delay.push(u)
