"""Vectorized open-loop input generators ("maneuvers") for batch data generation.

Each generator turns per-episode parameter arrays of shape (B,) into commanded actions of shape
(T, B, 2) = (steer_cmd, throttle_cmd) in [-1, 1] at the control rate. Every parameter can be given
a distribution in the batch spec, so e.g. the sustain throttle of a drift entry can be randomized
per episode. ``mirror_prob`` (all maneuvers) flips the steering sign of an episode with that
probability, giving balanced left/right data.

The ``MANEUVERS`` table is the catalog the GUI shows (names, defaults, units, descriptions).
"""
from __future__ import annotations

import numpy as np

_T = dict(unit="-", min=-1.0, max=1.0)          # a normalized command in [-1, 1]


def _p(name, default, desc, unit="-", **kw):
    return dict(name=name, default=default, desc=desc, unit=unit, **kw)


MIRROR = _p("mirror_prob", 0.0, "probability of mirroring the steering (left/right balance)", min=0.0, max=1.0)

MANEUVERS: dict[str, dict] = {
    "drift_schedule": dict(
        label="Drift entry (power-over)",
        desc="Straight launch, a full-throttle stab with a little steer to break the rear loose, then "
             "counter-steer with the throttle matched to the slide (the open-loop drift of the test suite).",
        params=[_p("launch_s", 1.40, "straight launch duration", "s", min=0.0),
                _p("launch_throttle", 0.37, "throttle during the launch", **_T),
                _p("kick_s", 0.28, "duration of the throttle stab", "s", min=0.0),
                _p("kick_steer", 0.25, "steering during the stab (+ = left)", **_T),
                _p("kick_throttle", 1.00, "throttle during the stab", **_T),
                _p("sustain_steer", -1.00, "counter-steer while sliding", **_T),
                _p("sustain_throttle", 0.226, "throttle while sliding (narrow band, see DESIGN.md)", **_T),
                MIRROR]),
    "constant": dict(
        label="Constant inputs",
        desc="Hold one steering and throttle command for the whole episode (donuts, coast-down, cruising).",
        params=[_p("steer", 0.3, "steering command (+ = left)", **_T),
                _p("throttle", 0.25, "throttle command", **_T), MIRROR]),
    "step_steer": dict(
        label="Step steer",
        desc="Drive straight, then step the steering at t_step (classic yaw-response test).",
        params=[_p("throttle", 0.20, "throttle command", **_T),
                _p("t_step_s", 1.0, "time of the steering step", "s", min=0.0),
                _p("steer", 0.4, "steering after the step", **_T), MIRROR]),
    "sine_steer": dict(
        label="Sine steer",
        desc="Sinusoidal steering at constant throttle (frequency-response data).",
        params=[_p("throttle", 0.20, "throttle command", **_T),
                _p("amplitude", 0.4, "steering amplitude", "-", min=0.0, max=1.0),
                _p("frequency_hz", 0.5, "steering frequency", "Hz", min=0.0),
                _p("phase_deg", 0.0, "phase at t = 0", "deg"), MIRROR]),
    "chirp": dict(
        label="Chirp steer (frequency sweep)",
        desc="Steering sine whose frequency sweeps linearly from f0 to f1 over the episode (system "
             "identification excitation).",
        params=[_p("throttle", 0.20, "throttle command", **_T),
                _p("amplitude", 0.4, "steering amplitude", "-", min=0.0, max=1.0),
                _p("f0_hz", 0.1, "start frequency", "Hz", min=0.0),
                _p("f1_hz", 3.0, "end frequency", "Hz", min=0.0), MIRROR]),
    "lane_change": dict(
        label="Lane change",
        desc="Straight, steer one way for half a period, the other way for half a period, straight again.",
        params=[_p("throttle", 0.20, "throttle command", **_T),
                _p("steer", 0.3, "steering amplitude", **_T),
                _p("t_start_s", 0.5, "start of the manoeuvre", "s", min=0.0),
                _p("half_period_s", 0.75, "duration of each half", "s", min=0.0), MIRROR]),
    "random": dict(
        label="Smoothed random inputs",
        desc="Band-limited random steering and throttle (first-order low-pass filtered white noise): broad "
             "coverage of the state space for learning dynamics models.",
        params=[_p("steer_std", 0.4, "steering standard deviation", "-", min=0.0),
                _p("throttle_mean", 0.25, "mean throttle", **_T),
                _p("throttle_std", 0.10, "throttle standard deviation", "-", min=0.0),
                _p("bandwidth_hz", 1.0, "low-pass cutoff of the inputs", "Hz", min=0.0),
                _p("throttle_min", 0.0, "lowest throttle allowed (use < 0 to allow braking)", **_T), MIRROR]),
}


def defaults(mtype: str) -> dict[str, float]:
    return {p["name"]: float(p["default"]) for p in MANEUVERS[mtype]["params"]}


def generate(mtype: str, prm: dict[str, np.ndarray], T: int, control_dt: float,
             episode_seeds: np.ndarray | None = None, return_flags: bool = False):
    """Commanded actions (T, B, 2) for ``mtype`` with per-episode parameters ``prm`` (each (B,)).

    ``episode_seeds`` (B,) int seeds make the random generator reproducible per episode (the
    runner passes a hash of (spec seed, episode id)); the mirror flag is drawn from the same seeds.
    With ``return_flags`` the result is ``(actions, mirrored (B,) bool)``.
    """
    if mtype not in MANEUVERS:
        raise ValueError(f"unknown maneuver {mtype!r}; expected one of {sorted(MANEUVERS)}")
    B = len(next(iter(prm.values())))
    t = (np.arange(T) * control_dt)[:, None]                        # (T, 1)
    g = lambda k: np.asarray(prm[k], dtype=np.float64)[None, :]    # noqa: E731  (1, B)
    seeds = np.arange(B) if episode_seeds is None else np.asarray(episode_seeds)
    rngs = [np.random.default_rng([int(s), 7]) for s in seeds]
    zeros = np.zeros((T, B))

    if mtype == "drift_schedule":
        e1, e2 = g("launch_s"), g("launch_s") + g("kick_s")
        steer = np.where(t < e1, 0.0, np.where(t < e2, g("kick_steer"), g("sustain_steer")))
        thr = np.where(t < e1, g("launch_throttle"), np.where(t < e2, g("kick_throttle"), g("sustain_throttle")))
    elif mtype == "constant":
        steer, thr = zeros + g("steer"), zeros + g("throttle")
    elif mtype == "step_steer":
        steer = np.where(t < g("t_step_s"), 0.0, g("steer"))
        thr = zeros + g("throttle")
    elif mtype == "sine_steer":
        steer = g("amplitude") * np.sin(2 * np.pi * g("frequency_hz") * t + np.radians(g("phase_deg")))
        thr = zeros + g("throttle")
    elif mtype == "chirp":
        dur = max(T * control_dt, control_dt)
        f0, f1 = g("f0_hz"), g("f1_hz")
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t ** 2)
        steer = g("amplitude") * np.sin(phase)
        thr = zeros + g("throttle")
    elif mtype == "lane_change":
        t0, hp = g("t_start_s"), g("half_period_s")
        steer = np.where((t >= t0) & (t < t0 + hp), g("steer"),
                         np.where((t >= t0 + hp) & (t < t0 + 2 * hp), -g("steer"), 0.0))
        thr = zeros + g("throttle")
    else:  # random: first-order low-pass filtered white noise with the requested output std
        a = 1.0 - np.exp(-2 * np.pi * np.maximum(g("bandwidth_hz"), 1e-6) * control_dt)   # (1, B)
        gain = np.sqrt((2.0 - a) / a)                    # makes the filtered std equal the white std
        w = np.stack([r.standard_normal((T, 2)) for r in rngs], axis=1)                   # (T, B, 2)
        a_, gain_ = a[0][:, None], gain[0][:, None]                                      # (B, 1)
        y = np.empty((T, B, 2))
        state = w[0]                                     # a stationary sample (unit std) to start
        y[0] = state
        for k in range(1, T):                            # y_k = y_{k-1} + a (gain w_k - y_{k-1})
            state = state + a_ * (w[k] * gain_ - state)
            y[k] = state
        steer = g("steer_std") * y[:, :, 0]
        thr = g("throttle_mean") + g("throttle_std") * y[:, :, 1]
        thr = np.maximum(thr, g("throttle_min"))

    mirror_p = np.asarray(prm.get("mirror_prob", np.zeros(B)), dtype=np.float64)
    flip = np.array([r.random() < p for r, p in zip(rngs, mirror_p)])
    steer = np.where(flip[None, :], -steer, steer)
    actions = np.clip(np.stack([steer, thr], axis=-1), -1.0, 1.0)
    return (actions, flip) if return_flags else actions
