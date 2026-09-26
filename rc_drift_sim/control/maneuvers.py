"""Open-loop action schedules (drift entry, constant inputs, lane change) and the LQR drift entry.

Used by the tests, the output scripts and (later) as baselines for the RL policy.

An action is ``(steer_cmd, throttle_cmd)``, both in [-1, 1]. Steer > 0 turns the wheels LEFT and
a left-hand drift therefore has a NEGATIVE sideslip beta (nose left of the velocity), see
docs/DESIGN.md section 1.

A schedule is a plain list of ``(duration_s, steer_cmd, throttle_cmd)`` segments played back to
back with zero-order hold at the control period; the last segment is held to the end of the
rollout if the segments are shorter than the requested duration.
"""
from __future__ import annotations

import numpy as np

Segment = tuple[float, float, float]  # (duration_s, steer_cmd, throttle_cmd)

# Open-loop drift on the paved epoxy_ptile track (hard_plastic_drift tire), left-hand drift:
#   1. launch straight on moderate throttle,
#   2. "power-over" entry: a short full-throttle stab with a little left steer kicks the rear out,
#   3. full counter-steer right with the throttle matched to the drift speed.
# Found with a batched search (scripts/tune_drift.py). Why the numbers are so specific:
# a RWD drift equilibrium is open-loop UNSTABLE (one real eigenvalue of +3..+5 1/s, see
# sim/equilibrium.py), and the stiff motor makes throttle a rear-wheel-speed command, so a
# sustained slide at ~1.5 m/s needs rear slip in a narrow band. The sustain throttle 0.226 sits in
# the middle of the passing band [0.218, 0.232]. If the physics changes, run
# `python scripts/tune_drift.py` and paste its schedule here.
DRIFT_SCHEDULE: list[Segment] = [
    (1.40,  0.00, 0.370),   # straight-line launch
    (0.28,  0.25, 1.000),   # power-over entry: throttle stab with a little left steer
    (9.00, -1.00, 0.226),   # full counter-steer, throttle matched to the slide (held to the end)
]

# Entry used by the closed-loop (LQR) drift tests: launch, throttle stab, then hand over to the
# feedback controller as soon as the sideslip passes -5 deg.
LQR_ENTRY = dict(launch_s=1.0, launch_thr=0.37, kick_steer=0.25, kick_thr=1.0, kick_max_s=0.4,
                 switch_beta_deg=5.0)


def schedule_to_actions(schedule: list[Segment], control_dt: float, duration: float) -> np.ndarray:
    """Expand a segment schedule into a (T, 2) action array, T = round(duration / control_dt).

    Each segment lasts round(duration_s / control_dt) control steps. The array is truncated or
    padded (holding the last segment) to exactly T rows and clipped to [-1, 1].
    """
    n_total = int(round(duration / control_dt))
    rows = []
    for seg_duration, steer, throttle in schedule:
        n = int(round(float(seg_duration) / control_dt))
        if n > 0:
            rows.append(np.tile(np.array([[steer, throttle]], dtype=float), (n, 1)))
    actions = np.concatenate(rows, axis=0) if rows else np.zeros((0, 2), dtype=float)
    if actions.shape[0] < n_total:
        last = actions[-1:] if actions.shape[0] else np.zeros((1, 2), dtype=float)
        actions = np.concatenate([actions, np.repeat(last, n_total - actions.shape[0], axis=0)])
    return np.clip(actions[:n_total], -1.0, 1.0)


def open_loop_drift_actions(params, duration: float = 4.0,
                            schedule: list[Segment] | None = None) -> np.ndarray:
    """(T, 2) open-loop (steer, throttle) commands for a drift entry on the paved track.

    ``params.sim.control_dt`` sets the step; ``schedule`` defaults to DRIFT_SCHEDULE.
    """
    sched = DRIFT_SCHEDULE if schedule is None else schedule
    return schedule_to_actions(sched, float(params.sim.control_dt), duration)


def constant_actions(params, duration: float, steer: float, throttle: float) -> np.ndarray:
    """(T, 2) array holding one (steer, throttle) pair for ``duration`` seconds."""
    return schedule_to_actions([(duration, steer, throttle)], float(params.sim.control_dt), duration)


def lane_change_actions(params, duration: float, steer: float, throttle: float,
                        t_start: float = 0.5, half_period: float = 0.75) -> np.ndarray:
    """Straight, then steer +``steer`` for ``half_period`` s, then -``steer`` for ``half_period`` s,
    then straight again, all at constant ``throttle``. A smooth, non-drifting reference maneuver."""
    rest = max(duration - t_start - 2.0 * half_period, 0.0)
    sched = [(t_start, 0.0, throttle), (half_period, steer, throttle),
             (half_period, -steer, throttle), (rest, 0.0, throttle)]
    return schedule_to_actions(sched, float(params.sim.control_dt), duration)
