"""Re-tune the open-loop drift schedule used by tests/test_drift.py (batched search).

Why this tool exists: a RWD drift is open-loop unstable (see sim/equilibrium.py), so the open-loop
schedule is sensitive to the physics. After changing the model, run

    python scripts/tune_drift.py [--tire hard_plastic_drift] [--surface epoxy_ptile]

It keeps the "power-over" entry structure (launch, throttle stab, full counter-steer) and scans
the sustain throttle band in one vectorized rollout, then prints the band and a schedule centred in
it to paste into rc_drift_sim/control/maneuvers.py. If no throttle passes, it widens the search to the launch
and entry parameters with a random batched search.
"""
from __future__ import annotations

import argparse

import numpy as np

from rc_drift_sim.sim import state as S
from rc_drift_sim.sim.vehicle import make_vehicle

BETA_DRIFT, BETA_SPIN, V_MIN, MIN_S = 20.0, 80.0, 0.8, 1.5


def score(states: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """(longest drift window in s, spun flag) per env for states (T+1, B, NS)."""
    vx, vy = states[..., S.VX], states[..., S.VY]
    sp = np.hypot(vx, vy)
    beta = np.degrees(np.arctan2(vy, vx))
    moving = sp > 0.05
    spin = np.any(moving & (np.abs(beta) >= BETA_SPIN), axis=0)
    ok = moving & (np.abs(beta) > BETA_DRIFT) & (np.abs(beta) < BETA_SPIN) & (sp > V_MIN)
    best = np.zeros(ok.shape[1], int)
    cur = np.zeros(ok.shape[1], int)
    for k in range(ok.shape[0]):
        cur = np.where(ok[k], cur + 1, 0)
        best = np.maximum(best, cur)
    return np.maximum(best - 1, 0) * dt, spin


def schedule_actions(P: np.ndarray, T: int, dt: float) -> np.ndarray:
    """P (B, 6): launch_s, launch_thr, entry_s, entry_steer, entry_thr, sustain_thr -> (T, B, 2);
    the sustain phase uses full counter-steer (-1)."""
    L, a, F, f, b, d = P.T
    t = (np.arange(T) * dt)[:, None]
    steer = np.where(t < L, 0.0, np.where(t < L + F, f, -1.0))
    thr = np.where(t < L, a, np.where(t < L + F, b, d))
    return np.stack([steer, thr], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tire", default="hard_plastic_drift")
    ap.add_argument("--surface", default="epoxy_ptile")
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()
    v = make_vehicle(args.tire, args.surface)
    dt = v.control_dt
    T = int(round(args.duration / dt))
    entry = np.array([1.40, 0.37, 0.28, 0.25, 1.00])
    thr = np.round(np.arange(0.10, 0.45, 0.002), 3)
    P = np.column_stack([np.tile(entry, (len(thr), 1)), thr])
    states, _ = v.rollout_batch(v.initial_state(), schedule_actions(P, T, dt))
    dur, spin = score(states, dt)
    good = (~spin) & (dur >= MIN_S)
    if good.any():
        band = thr[good]
        centre = float(thr[good][np.argmin(np.abs(thr[good] - 0.5 * (band.min() + band.max())))])
        print(f"passing sustain-throttle band: [{band.min():.3f}, {band.max():.3f}] "
              f"(best drift {dur[good].max():.2f} s)")
        print("DRIFT_SCHEDULE = [\n"
              f"    ({entry[0]:.2f},  0.00, {entry[1]:.3f}),\n"
              f"    ({entry[2]:.2f},  {entry[3]:.2f}, {entry[4]:.3f}),\n"
              f"    (9.00, -1.00, {centre:.3f}),\n]")
        return
    print("no sustain throttle passes with the default entry; running a random batched search ...")
    rng = np.random.default_rng(0)
    lo = np.array([0.4, 0.3, 0.08, 0.1, 0.3, 0.1])
    hi = np.array([1.6, 1.0, 0.5, 1.0, 1.0, 0.6])
    P = lo + (hi - lo) * rng.random((4096, 6))
    states, _ = v.rollout_batch(v.initial_state(), schedule_actions(P, T, dt))
    dur, spin = score(states, dt)
    order = np.argsort(-(dur * ~spin))[:5]
    for i in order:
        print(f"  drift {dur[i]:.2f} s spin={spin[i]} | params {np.round(P[i], 3).tolist()}")


if __name__ == "__main__":
    main()
