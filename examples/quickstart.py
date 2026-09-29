"""Quick start: simulate a drift, change a variable, run several cars at once, save a plot and a GIF.

    python examples/quickstart.py

Images are written to examples/output/ (git-ignored).
"""
from pathlib import Path

import numpy as np

from rc_drift_sim import TireCondition, VehicleBatch, default_params, make_vehicle
from rc_drift_sim.control.maneuvers import open_loop_drift_actions
from rc_drift_sim.viz import plots, render

OUT = Path(__file__).resolve().parent / "output"


def describe(label, traj):
    beta = np.degrees(np.abs(traj.beta[traj.speed > 0.05]))
    print(f"{label:34s} top speed {traj.speed.max():4.2f} m/s, max sideslip {beta.max() if beta.size else 0:5.1f} deg")


def main():
    OUT.mkdir(exist_ok=True)

    # 1. one car: tire compound and surface are names from rc_drift_sim/configs/*.yaml
    car = make_vehicle("hard_plastic_drift", "epoxy_ptile")
    actions = open_loop_drift_actions(car.params, duration=4.0)   # (T, 2) steer, throttle in [-1, 1], 50 Hz
    traj = car.rollout(car.initial_state(), actions)               # t, states, actions, info
    describe("plastic tires on P-tile", traj)

    # 2. change any variable by name (field names as in the YAML files / docs/PARAMETERS.md)
    heavy = make_vehicle("hard_plastic_drift", "epoxy_ptile", mass=1.9, latency=0.04)
    describe("same, 1.9 kg and 40 ms latency", heavy.rollout(heavy.initial_state(), actions))
    grippy = make_vehicle("rubber_onroad", "dry_asphalt")
    describe("rubber tires on dry asphalt", grippy.rollout(grippy.initial_state(), actions))

    # 3. several different cars in one vectorized batch (one array operation for all of them)
    params = [default_params("hard_plastic_drift", s) for s in ("epoxy_ptile", "polished_concrete", "carpet")]
    conds = [TireCondition(wear=np.full(4, w)) for w in (0.0, 0.5, 1.0)]
    batch = VehicleBatch(params, conds=conds)
    s = batch.initial_states()
    for k in range(len(actions)):
        s, _ = batch.step(s, np.tile(actions[k], (len(batch), 1)))
    print("batch of 3 cars, final speeds:", np.round(np.hypot(s[:, 3], s[:, 4]), 2), "m/s")

    # 4. pictures
    plots.plot_timeseries(traj, car.params, OUT / "drift_timeseries.png")
    render.animate(traj, car.params, OUT / "drift.gif", fps=25, stride=3, dpi=60)
    print(f"saved {OUT / 'drift_timeseries.png'} and {OUT / 'drift.gif'}")


if __name__ == "__main__":
    main()
