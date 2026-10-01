"""Batch data generation: spec validation, reproducible sampling, and exports that equal an
independent single-car simulation through the real latency buffer."""
from __future__ import annotations

import csv
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


def test_warns_when_every_episode_would_be_identical():
    identical = lambda s: any("identical" in w for w in validate_spec(s)[2])  # noqa: E731
    s = default_spec()
    s.update(episodes=100, params={"tire": {"dist": "choice", "values": ["rubber_onroad"]},
                                   "vehicle.mass": {"dist": "uniform", "low": 1.6, "high": 1.6}})
    assert identical(s), "fixed params, one-value choice, empty range and a deterministic maneuver"
    assert not identical(_spec())
    s["params"]["vehicle.mass"]["high"] = 1.7
    assert not identical(s)
    s["params"].pop("vehicle.mass")
    s["maneuver"] = {"type": "drift_schedule", "params": {"mirror_prob": {"dist": "fixed", "value": 0.5}}}
    assert not identical(s), "a random left/right mirror makes episodes differ"
    s["maneuver"] = {"type": "random", "params": {}}
    assert not identical(s), "the random maneuver draws new inputs per episode"


def test_physics_rate_records_every_integrator_step():
    from rc_drift_sim.datagen.runner import plan, simulate_episodes
    s = _spec()
    s.update(episodes=4, duration_s=0.6)
    s["export"]["signals"] = ["pose", "wheel_speeds", "steering", "actions", "accel", "motor", "motor_torque",
                              "electrical"]
    ctrl = validate_spec(s)[0]
    s["export"]["record_rate"] = "physics"
    phys = validate_spec(s)[0]
    t_c, d_c, _ = simulate_episodes(ctrl, list(range(4)))
    t_p, d_p, _ = simulate_episodes(phys, list(range(4)))
    assert len(t_p) == 20 * (len(t_c) - 1) + 1 and np.allclose(np.diff(t_p), 0.001)
    for name in d_c:
        np.testing.assert_array_equal(d_p[name][:, ::20], d_c[name], err_msg=name)
    assert np.any(np.diff(d_p["delta"][:, 1:20], axis=1) != 0), "rows between control steps must be new states"
    np.testing.assert_allclose(d_p["motor_rpm"] * np.pi / 30, d_p["omega_m"], rtol=1e-12)
    assert len(plan(phys)[1]) >= 1


def test_motor_voltage_satisfies_the_motor_circuit():
    from rc_drift_sim.sim import state as S
    from rc_drift_sim.sim.vehicle import derivatives_model
    car = Vehicle(Sampler(validate_spec(_spec())[0]).episode(0).params)
    s = car.initial_state()
    s[S.VX], s[S.OMEGA], s[S.I_MOTOR] = 1.2, 45.0, 6.0
    dm = car.model.dm
    for thr in (0.0, 0.15, 0.6, 1.0):
        ds, info = derivatives_model(s, np.array([0.1, thr]), car.model, want_info=True)
        rhs = dm.r_m * s[S.I_MOTOR] + ds[S.I_MOTOR] / dm.inv_L + dm.ke * info["omega_m"]
        assert np.isclose(info["v_motor"], rhs), thr
        assert info["v_batt"] <= dm.v0


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
    for f in sorted(glob.glob(f"{res['out_dir']}/all/shard_*.npz")):
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


def test_not_spun_table_keeps_only_unspun_episodes_and_the_header(tmp_path):
    from rc_drift_sim.datagen.export import write_episodes_table
    rows = [dict(episode_id=2, spun=True, drift_time_s=0.3), dict(episode_id=0, spun=False, drift_time_s=1.5),
            dict(episode_id=1, spun=False, drift_time_s=2.0)]
    write_episodes_table(tmp_path / "a.csv", rows, keep=lambda r: not r["spun"])
    got = list(csv.DictReader(open(tmp_path / "a.csv")))
    assert [r["episode_id"] for r in got] == ["0", "1"] and all(r["spun"] == "False" for r in got)
    write_episodes_table(tmp_path / "b.csv", rows[:1], keep=lambda r: not r["spun"])
    assert open(tmp_path / "b.csv").read().strip() == "episode_id,spun,drift_time_s", "header even when all spun"


def test_episodes_that_did_not_spin_get_their_own_folder(tmp_path):
    s = default_spec()
    s.update(name="mix", episodes=6, duration_s=2.0, seed=1)
    s["maneuver"] = {"type": "constant", "params": {"steer": {"dist": "fixed", "value": 1.0},
                                                    "throttle": {"dist": "sweep", "values": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]}}}
    s["export"].update(formats=["npz", "csv"], shard_episodes=3)
    res = run_batch(s, out_root=tmp_path, workers=1)
    out = res["out_dir"]
    table = lambda folder: list(csv.DictReader(open(f"{out}/{folder}/episodes.csv")))  # noqa: E731
    everything, not_spun = table("all"), table("not_spun")
    assert [r["spun"] for r in everything] == ["False", "False", "True", "True", "True", "True"]
    assert not_spun == everything[:2]
    a, b = np.load(f"{out}/all/shard_0000.npz"), np.load(f"{out}/not_spun/shard_0000.npz")
    assert list(b["episode_id"]) == [0, 1] and set(b.files) == set(a.files)
    for k in a.files:
        np.testing.assert_array_equal(b[k], a[k] if k == "t" else a[k][:2], err_msg=k)
    assert "not_spun/shard_0000.csv" in res["files"] and "all/shard_0001.npz" in res["files"]
    assert not any(f.startswith("not_spun/shard_0001") for f in res["files"]), "an all-spun shard has no file"
    man = json.load(open(f"{out}/manifest.json"))
    assert [sh["not_spun"] for sh in man["shards"]] == [2, 0] and man["stats"]["not_spun_episodes"] == 2
