"""Batch data generation: spec validation, reproducible sampling, and exports that equal an
independent single-car simulation through the real latency buffer."""
from __future__ import annotations

import glob
import json

import numpy as np

from rc_drift_sim.datagen import default_spec, run_batch, validate_spec
from rc_drift_sim.datagen import inputs
from rc_drift_sim.datagen.runner import _episode_seed
from rc_drift_sim.datagen.sampling import Sampler
from rc_drift_sim.sim.actuators import ActionDelay
from rc_drift_sim.sim.vehicle import Vehicle


def _spec():
    s = default_spec()
    s.update(name="t", episodes=12, duration_s=1.0, seed=3)
    s["params"] = {"vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8},
                   "tire": {"dist": "choice", "values": ["hard_plastic_drift", "rubber_onroad"]},
                   "actuators.latency": {"dist": "choice", "values": [0.0, 0.04]},
                   "condition.wear": {"dist": "uniform", "low": 0, "high": 1, "per_wheel": True}}
    s["maneuver"] = {"type": "random", "params": {"mirror_prob": {"dist": "fixed", "value": 0.5}}}
    s["export"].update(float32=False, shard_episodes=6, formats=["npz", "csv"])
    return s


def test_validation_catches_bad_specs():
    s = _spec()
    s["params"]["vehicle.mass"] = {"dist": "uniform", "low": -1.0, "high": 1.0}
    s["params"]["vehicle.nope"] = {"dist": "fixed", "value": 1}
    _, errors, _ = validate_spec(s)
    assert any("vehicle.mass" in e for e in errors) and any("vehicle.nope" in e for e in errors)
    assert validate_spec(_spec())[1] == []


def test_sampling_is_reproducible_and_independent_per_parameter():
    a = Sampler(validate_spec(_spec())[0]).episode(5).values["vehicle.mass"]
    s = _spec()
    s["params"]["vehicle.cg_height"] = {"dist": "uniform", "low": 0.03, "high": 0.04}
    b = Sampler(validate_spec(s)[0]).episode(5).values["vehicle.mass"]
    assert a == b, "adding a parameter must not change the samples of the others"


def test_export_matches_single_car_simulation(tmp_path):
    spec = _spec()
    res = run_batch(spec, out_root=tmp_path, workers=1)
    assert res["status"] == "complete" and res["episodes_written"] == 12
    man = json.load(open(f"{res['out_dir']}/manifest.json"))
    assert man["status"] == "complete" and len(man["shards"]) == 2
    norm = validate_spec(spec)[0]
    sampler = Sampler(norm)
    for f in sorted(glob.glob(f"{res['out_dir']}/shard_*.npz")):
        d = np.load(f)
        for row, i in enumerate(d["episode_id"]):
            ep = sampler.episode(int(i))
            car = Vehicle(ep.params, cond=ep.cond)
            s = car.initial_state()
            T = int(round(norm["duration_s"] / car.control_dt))
            cmd = inputs.generate("random", {k: np.array([v]) for k, v in ep.maneuver.items()}, T, car.control_dt,
                                  np.array([_episode_seed(norm["seed"], int(i))], dtype=np.uint64))[:, 0]
            delay = ActionDelay(ep.params.actuators.latency, car.control_dt)
            xs = [s[0]]
            for k in range(T):
                s, _ = car.step(s, delay.push(cmd[k]), want_info=False)
                xs.append(s[0])
            np.testing.assert_array_equal(np.array(xs), d["x"][row])
