"""PyTorch backend of the physics core: the same RHS and batches as the NumPy float64 reference,
on PyTorch CPU (float64, tight) and on the GPUs this machine has (float32, loose)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_drift_sim.datagen import default_spec, run_batch, validate_spec
from rc_drift_sim.datagen.runner import simulate_episodes
from rc_drift_sim.sim import integrator, xp
from rc_drift_sim.sim.params import default_params
from rc_drift_sim.sim.vehicle import VehicleBatch, derivatives_model, rhs_model

torch = pytest.importorskip("torch")
DEVICES = xp.available_devices()
GPUS = [d for d in ("mps", "cuda") if DEVICES[d]]


def _mixed_batch(B=12):
    """Cars that exercise every configuration branch: AWD with overdrive, Ackermann, drag brake,
    a loose surface (plowing), per-car masses and steering limits."""
    p = default_params(tire="rubber_onroad", surface="loose_dirt")
    p = dataclasses.replace(p, drivetrain=dataclasses.replace(p.drivetrain, layout="awd_overdrive", drag_brake=0.1),
                            vehicle=dataclasses.replace(p.vehicle, ackermann=0.5))
    cars = [dataclasses.replace(p, vehicle=dataclasses.replace(p.vehicle, mass=1.3 + 0.05 * i),
                                actuators=dataclasses.replace(p.actuators, steer_max=np.radians(25.0 + i)))
            for i in range(B)]
    batch = VehicleBatch(cars, check_stiffness=False)
    rng = np.random.default_rng(1)
    s = batch.initial_states(v=rng.uniform(0.0, 3.0, B), beta=rng.uniform(-0.6, 0.6, B), yaw_rate=rng.uniform(-2, 2, B))
    s[:, 11] = rng.uniform(-5, 20, B)                 # motor current
    u = np.clip(rng.normal(0.1, 0.6, (B, 2)), -1, 1)
    return batch, s, u


def test_torch_float64_rhs_matches_numpy_including_info():
    batch, s, u = _mixed_batch()
    ds, info = derivatives_model(s, u, batch.model, want_info=True)
    m = xp.to_device(batch.model, "cpu", torch.float64)
    ds_t, info_t = derivatives_model(torch.as_tensor(s), torch.as_tensor(u), m, want_info=True)
    np.testing.assert_allclose(ds_t.numpy(), ds, rtol=1e-12, atol=1e-12)
    assert set(info_t) == set(info)
    for k in info:
        np.testing.assert_allclose(np.asarray(info_t[k]), info[k], rtol=1e-11, atol=1e-11, err_msg=k)


@pytest.mark.parametrize("device", GPUS)
def test_gpu_float32_rollout_stays_close_to_the_reference(device):
    batch, s, u = _mixed_batch()
    ref = batch.step(s, u)[0]
    for _ in range(24):                              # 0.5 s
        ref = batch.step(ref, u)[0]
    m = xp.to_device(batch.model, device, torch.float32)
    st, ut = torch.as_tensor(s, dtype=torch.float32, device=device), torch.as_tensor(u, dtype=torch.float32, device=device)
    for _ in range(25):
        st = integrator.integrate(rhs_model, st, batch.dt, batch.n_substeps, "rk4", ut, m)
    got = st.cpu().double().numpy()
    assert st.dtype == torch.float32 and np.isfinite(got).all()
    assert np.abs(got[:, :2] - ref[:, :2]).max() < 1e-4, "position within 0.1 mm after 0.5 s"
    assert np.abs(got[:, 3:6] - ref[:, 3:6]).max() < 1e-3


def _spec():
    s = default_spec()
    s.update(name="xp", episodes=8, duration_s=1.0, seed=5)
    s["params"] = {"vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8},
                   "surface": {"dist": "choice", "values": ["epoxy_ptile", "loose_dirt"]},
                   "actuators.latency": {"dist": "choice", "values": [0.0, 0.04]}}
    s["maneuver"] = {"type": "random", "params": {"mirror_prob": {"dist": "fixed", "value": 0.5}}}
    s["export"]["signals"] = ["pose", "derived", "wheel_speeds", "steering", "actions", "accel", "motor",
                              "electrical", "tire_forces"]
    return s


@pytest.mark.parametrize("rate", ["control", "physics"])
def test_torch_export_matches_numpy_export(rate):
    s = _spec()
    s["export"]["record_rate"] = rate
    s["run"]["precision"] = "float64"
    norm = validate_spec(s)[0]
    t_n, d_n, r_n = simulate_episodes(norm, list(range(8)))
    t_t, d_t, r_t = simulate_episodes(norm, list(range(8)), torch_device="cpu")
    np.testing.assert_array_equal(t_t, t_n)
    for k in d_n:
        np.testing.assert_allclose(d_t[k], d_n[k], rtol=1e-9, atol=1e-9, err_msg=k)
    for a, b in zip(r_n, r_t):
        assert a.keys() == b.keys() and a["spun"] == b["spun"] and a["mirrored"] == b["mirrored"]
        assert np.isclose(a["max_abs_beta_deg"], b["max_abs_beta_deg"], atol=1e-6)


def test_device_validation():
    s = _spec()
    s["run"]["device"] = "tpu"
    assert any("run.device" in e for e in validate_spec(s)[1])
    for dev in ("mps", "cuda"):
        s["run"].update(device=dev, precision="float32")
        errors = validate_spec(s)[1]
        assert (errors == []) == DEVICES[dev], (dev, errors)
    if DEVICES["mps"]:
        s["run"].update(device="mps", precision="float64")
        assert any("float32" in e for e in validate_spec(s)[1])


@pytest.mark.skipif(not GPUS, reason="no GPU")
def test_gpu_batch_export_runs_end_to_end(tmp_path):
    s = _spec()
    s["run"]["device"] = GPUS[0]
    s["export"]["shard_episodes"] = 3
    res = run_batch(s, out_root=tmp_path)
    assert res["status"] == "complete" and res["episodes_written"] == 8
    assert {"all/shard_0000.npz", "all/shard_0002.npz", "all/episodes.csv"} <= set(res["files"])
    d = np.load(f"{res['out_dir']}/all/shard_0001.npz")
    assert list(d["episode_id"]) == [3, 4, 5] and np.isfinite(d["x"]).all()
    import json
    man = json.load(open(f"{res['out_dir']}/manifest.json"))
    assert man["device"] == GPUS[0] and man["precision"] == "float32"
