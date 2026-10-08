"""Safe, self-adapting sliding: the safety filter, the sensor history, mid-episode grip changes, the
grip estimator and privileged critic in training, the grip-sweep evaluation, the final-model export
and the on-car runtime (which must reproduce the training environment exactly), and the BeamNG
frame conversions."""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request

import numpy as np
import pytest

import rc_drift_sim.rl as rl
from rc_drift_sim.deploy.beamng import body_readings, froude, to_policy_readings
from rc_drift_sim.deploy.safety import SafetyConfig, safety_filter
from rc_drift_sim.sim import state as S
from rc_drift_sim.sim.vehicle import derivatives_model


def _filt(a, prev, vx, vy, r, cfg, grip=None):
    arr = lambda v: np.array([float(v)])  # noqa: E731
    out, ov = safety_filter(np, np.array([a], float), np.array([prev], float), arr(vx), arr(vy), arr(r),
                            None if grip is None else arr(grip), cfg, math.radians(35.0), 0.133)
    return out[0], float(ov[0])


def test_safety_filter_envelope_recovery_and_symmetry():
    cfg = SafetyConfig(enabled=True, steer_rate=0.0, throttle_rate=0.0)
    v, b = 1.5, math.radians(30)
    a, ov = _filt([-0.4, 0.25], [0, 0], v * math.cos(b), -v * math.sin(b), 2.0, cfg)     # steady left drift
    assert ov == 0.0 and np.allclose(a, [-0.4, 0.25])
    b = math.radians(75)                                                                  # spinning left
    a, ov = _filt([0.5, 0.6], [0, 0], v * math.cos(b), -v * math.sin(b), 4.0, cfg)
    assert ov == 1.0 and a[0] == -1.0 and a[1] <= 0.0                                     # full counter-steer, no power
    m, ov_m = _filt([-0.5, 0.6], [0, 0], v * math.cos(b), v * math.sin(b), -4.0, cfg)    # the mirror image
    assert ov_m == ov and np.allclose(m, [-a[0], a[1]])
    _, ov = _filt([0, 0.3], [0, 0], 1.0, 0.0, 9.0, cfg, grip=0.3)                         # limit 2 x 0.3 g / 1 m/s = 5.9
    assert ov > 0.5
    _, ov = _filt([0, 0.3], [0, 0], 0.2, -0.2, 3.0, cfg)                                  # parked-ish: checks off
    assert ov == 0.0
    a, ov = _filt([0, 1.0], [0, 1.0], 3.4, 0.0, 0.0, cfg)                                 # over the speed limit
    assert 0 < ov < 1 and a[1] < 1.0
    a, _ = _filt([1.0, 1.0], [-1.0, 0.0], 1.0, 0.0, 0.0, SafetyConfig(enabled=True))      # rate limits
    assert np.allclose(a, [-1.0 + 0.35, 0.2])


def test_safety_filter_numpy_and_torch_agree():
    torch = pytest.importorskip("torch")
    from rc_drift_sim.sim.xp import TORCH
    rng = np.random.default_rng(0)
    n = 256
    a, prev = rng.uniform(-1, 1, (n, 2)), rng.uniform(-1, 1, (n, 2))
    vx, vy, r, g = rng.uniform(-1, 3, n), rng.uniform(-2, 2, n), rng.uniform(-6, 6, n), rng.uniform(0.05, 0.6, n)
    cfg = SafetyConfig(enabled=True)
    out_n, ov_n = safety_filter(np, a, prev, vx, vy, r, g, cfg, math.radians(35), 0.133)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)  # noqa: E731
    out_t, ov_t = safety_filter(TORCH, t(a), t(prev), t(vx), t(vy), t(r), t(g), cfg, math.radians(35), 0.133)
    np.testing.assert_allclose(out_t.numpy(), out_n, atol=1e-12)
    np.testing.assert_allclose(ov_t.numpy(), ov_n, atol=1e-12)


def test_history_layout_and_reset_fill():
    env = rl.DriftBatchEnv(4, rl.EnvConfig(history=5, episode_s=0.1, front_wheel_speeds=False))
    n_dyn = len(env.dyn_names)
    assert "wheel_fl" not in env.dyn_names and env.n_obs == 5 * n_dyn + 2
    hist = env.last_obs[:, :5 * n_dyn].reshape(4, 5, n_dyn)
    assert np.allclose(hist, hist[:, -1:, :])                       # filled with the first reading
    rng = np.random.default_rng(1)
    for _ in range(3):
        env.step(rng.uniform(-1, 1, (4, 2)))
    hist = env.last_obs[:, :5 * n_dyn].reshape(4, 5, n_dyn)
    assert np.allclose(hist[:, -1, -2:], env.prev_action)          # newest reading is last (prev action)
    assert not np.allclose(hist[:, 0], hist[:, -1])
    for _ in range(2):                                              # the 5-step episodes end: history refilled
        obs, _, _, trunc, info = env.step(rng.uniform(-1, 1, (4, 2)))
    assert trunc.all() and len(info["final_idx"]) == 4
    hist = obs[:, :5 * n_dyn].reshape(4, 5, n_dyn)
    assert np.allclose(hist, hist[:, -1:, :])
    final = info["final_obs"][:, :5 * n_dyn].reshape(4, 5, n_dyn)
    assert not np.allclose(final, final[:, -1:, :])
    assert rl.DriftBatchEnv(1).obs_names == rl.DriftBatchEnv(1, rl.EnvConfig(history=1)).obs_names


def test_grip_change_targets_and_privileged_info():
    cfg = rl.EnvConfig(grip_change_prob=1.0, grip_change_min=0.5, grip_change_max=0.5, target_beta_jitter_deg=10,
                       target_speed_jitter=0.5, episode_s=1.0)
    env = rl.DriftBatchEnv(8, cfg, privileged=True)
    mu0 = env.model.tm.mu_scale.copy()
    assert env.last_priv.shape == (8, env.n_priv) and np.all(env.last_grip > 0)
    tb = np.degrees(env.target_beta)
    assert np.all(np.abs(tb - 30) <= 10) and tb.std() > 0 and np.all(np.abs(env.target_speed - 1.5) <= 0.5)
    steps = env._gc_step.copy()
    assert np.all((steps >= 0.25 * env.max_steps) & (steps <= 0.75 * env.max_steps))
    for _ in range(int(steps.max()) + 1):
        env.step(np.zeros((8, 2)))
    np.testing.assert_allclose(env.model.tm.mu_scale, mu0 * 0.5)
    _, info = derivatives_model(env.state, env._applied(), env.model, want_info=True)
    np.testing.assert_allclose(env.last_grip, info["mu_y"].mean(axis=1), rtol=1e-12)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("torch")
    from rc_drift_sim.rl.catalog import build_configs
    from rc_drift_sim.rl.ppo import PPOConfig, train
    from rc_drift_sim.rl.presets import preset_body
    body = preset_body("safe-adaptive")
    body["env"]["episode_s"] = 0.6
    body["run"]["device"] = "cpu"
    body["ppo"].update(total_steps=3 * 32 * 16, num_envs=32, rollout=16, epochs=2, minibatches=2, eval_every=2,
                       eval_cars=2, eval_grip_levels=[0.3, 0.9])
    env_cfg, ppo, run, errors = build_configs(body)
    assert errors == []
    out = tmp_path_factory.mktemp("run")
    _, hist = train(env_cfg, PPOConfig(**ppo), out=out, log=None)
    return out, hist


def test_training_with_estimator_privileged_critic_and_evaluation(trained):
    out, hist = trained
    assert (out / "best.pt").is_file() and len(hist) == 3
    assert all(r["grip_error"] is not None and r["override_share"] is not None for r in hist)
    evals = [r["eval"] for r in hist if "eval" in r]
    assert len(evals) == 2 and evals[0]["best"]                     # iterations 2 and 3 (the last)
    lv = evals[0]["levels"]
    assert [x["level"] for x in lv] == [0.3, 0.9] and lv[0]["grip"] < lv[1]["grip"]
    from rc_drift_sim.rl.ppo import load_checkpoint
    model, cfg, ppo, ck = load_checkpoint(out / "best.pt")
    assert model.est is not None and model.n_priv > 0 and ck["obs_spec"]["history"] == 16 and cfg.safety.enabled


def test_export_and_runtime_reproduce_the_training_environment(trained):
    from rc_drift_sim.deploy.runtime import PolicyRuntime
    from rc_drift_sim.rl.export import export_final
    from rc_drift_sim.rl.ppo import load_checkpoint
    out, _ = trained
    res = export_final(out)
    assert res["checks"]["ok"] and (out / "final_model.zip").is_file()
    model, cfg, _, _ = load_checkpoint(out / "best.pt")
    cfg = cfg.replace(noise={k: 0.0 for k in cfg.noise}, grip_change_prob=0.0, target_beta_jitter_deg=0.0,
                      target_speed_jitter=0.0, init_drift_prob=1.0, randomize={})
    env = rl.DriftBatchEnv(1, cfg, autoreset=False)
    rt = PolicyRuntime(res["folder"])
    for _ in range(40):
        s = env.state[0]
        _, info = derivatives_model(env.state, env._applied(), env.model, want_info=True)
        w = s[S.OMEGA] * float(np.asarray(env.model.Rw).reshape(-1)[0])
        sensors = dict(gyro_z=s[S.R], accel_x=info["ax"][0], accel_y=info["ay"][0], wheel_rear=w[2],
                       vel_x=s[S.VX], vel_y=s[S.VY])
        a_ref, g_ref = model.act_with_grip(env.last_obs)
        cmd = rt.step(sensors)
        np.testing.assert_allclose([cmd.raw_steer, cmd.raw_throttle], a_ref.numpy()[0], atol=1e-4)
        assert abs(cmd.grip - float(g_ref[0])) < 1e-4
        _, _, term, _, info = env.step(a_ref.numpy().astype(np.float64), grip=g_ref.numpy().astype(np.float64))
        np.testing.assert_allclose([cmd.steer, cmd.throttle], info["action"][0], atol=1e-4)
        if term[0]:
            break


def test_sim_car_loop_runs_the_exported_model(trained, tmp_path):
    from rc_drift_sim.deploy.car_loop import SimCar, drive
    from rc_drift_sim.deploy.runtime import PolicyRuntime
    out, _ = trained
    from rc_drift_sim.rl.export import export_final
    folder = export_final(out, out=tmp_path / "final")["folder"]
    rt = PolicyRuntime(folder)
    rt.set_targets(beta_deg=90, speed=0.1)                          # clipped to the trained range
    assert math.degrees(rt.target_beta) == pytest.approx(38) and rt.target_speed == pytest.approx(1.2)
    res = drive(rt, SimCar(rt.meta, grip=0.5, grip_change=(0.6, 0.5)), 1.0, tmp_path / "log.csv", verbose=False)
    assert res["steps"] == 50 and res["grip_true_2nd_half"] > 0 and (tmp_path / "log.csv").is_file()


def test_beamng_frame_conversion_and_froude_scaling():
    dt, yaw, r, vx, vy = 0.05, 0.3, 0.8, 4.0, -1.0
    def world(yaw):
        f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        lft = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return f, vx * f + vy * lft
    f0, v0 = world(yaw)
    b0 = body_readings(f0, [0, 0, 1], v0, None, dt)
    assert b0["vx"] == pytest.approx(vx) and b0["vy"] == pytest.approx(vy)
    f1, v1 = world(yaw + r * dt)
    b1 = body_readings(f1, [0, 0, 1], v1, b0, dt)
    assert b1["r"] == pytest.approx(r) and b1["vx"] == pytest.approx(vx)
    # constant body velocity in a rotating frame: the IMU feels the centripetal acceleration
    assert b1["ay"] == pytest.approx(vx * r, rel=0.05) and b1["ax"] == pytest.approx(-vy * r, rel=0.05)
    k = 16.0
    p = to_policy_readings(b1, 5.0, k)
    assert p["vel_x"] == pytest.approx(vx / 4) and p["gyro_z"] == pytest.approx(r * 4) and p["wheel_rear"] == pytest.approx(1.25)
    assert froude(k)["time"] == pytest.approx(4.0)


def test_gui_exports_and_replays_the_best_policy(tmp_path):
    pytest.importorskip("torch")
    from rc_drift_sim.app.server import make_server
    srv = make_server("127.0.0.1", 0, root=tmp_path / "exports", runs_root=tmp_path / "rl_runs")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(method, path, body=None):
        req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    try:
        from rc_drift_sim.rl.presets import preset_body
        body = preset_body("safe-adaptive")
        body["env"]["episode_s"] = 0.4
        body["run"].update(device="cpu", snapshots=1)
        body["ppo"].update(total_steps=2 * 64 * 8, num_envs=64, rollout=8, epochs=1, minibatches=2, eval_every=1,
                           eval_cars=1, eval_grip_levels=[0.5])
        run = call("POST", "/api/rl/runs", body)
        t_end = time.time() + 120
        while call("GET", f"/api/rl/runs/{run['id']}")["status"] in ("queued", "running") and time.time() < t_end:
            time.sleep(0.3)
        d = call("GET", f"/api/rl/runs/{run['id']}")
        assert d["status"] == "complete" and d["best_score"] is not None and d["export"] is None
        ep = call("POST", f"/api/rl/runs/{run['id']}/rollout", {"policy": "best", "grip": 0.4, "grip_change": [0.5, 0.2]})
        assert ep["safety"] and len(ep["series"]["grip_est"]) == ep["steps"]
        ex = call("POST", f"/api/rl/runs/{run['id']}/export", {})
        assert ex["checks"]["ok"] and len(ex["sil"]) == 3
        assert call("GET", f"/api/rl/runs/{run['id']}")["export"]["checks"]["ok"]
        with urllib.request.urlopen(base + f"/api/rl/runs/{run['id']}/files/final_model.zip") as r:
            assert r.read(2) == b"PK"
    finally:
        srv.rl.stop_all()
        srv.rl.wait_idle(20)
        srv.shutdown()
        srv.server_close()


def _fake_beamngpy(k: float, mirror: bool, steer_sign: float, grip: float):
    """A stand-in for beamngpy: DriftSim physics enlarged k times (Froude-scaled), in a world that may
    be mirrored (left-handed) and with a steering input that may be reversed."""
    import sys
    import types

    from rc_drift_sim.sim.vehicle import make_vehicle
    car = make_vehicle(mu_scale=grip)
    sk = math.sqrt(k)

    class Vehicle:
        def __init__(self, vid, model, **kw):
            self.s = car.initial_state()
            self.u = np.zeros(2)
            self.sensors = self
            self._el = {}

        # sensors API
        def attach(self, name, sensor):
            pass

        def poll(self):
            self._el = {"wheelspeed": float(self.s[S.OMEGA][2] * car.model.Rw) * sk, "airspeed": 0.0}

        def __getitem__(self, name):
            return self._el

        @property
        def state(self):
            yaw, sgn = self.s[S.YAW], (-1.0 if mirror else 1.0)
            f = np.array([math.cos(yaw), sgn * math.sin(yaw), 0.0])
            lft = np.array([-math.sin(yaw), sgn * math.cos(yaw), 0.0])
            v = (self.s[S.VX] * f + self.s[S.VY] * lft) * sk
            return dict(pos=[self.s[S.X] * k, sgn * self.s[S.Y] * k, 0.0], dir=list(f), up=[0, 0, 1], vel=list(v))

        def control(self, steering, throttle, brake, parkingbrake=0.0, gear=None):
            self.u = np.array([steer_sign * steering, throttle - brake])

        def get_bbox(self):
            L = car.params.vehicle.body_length * k
            return {"a": [0, 0, 0], "b": [L, 0.4 * L, 0.3 * L]}

    class Ctl:
        def __init__(self, bng):
            self.bng = bng

        def pause(self):
            pass

        def step(self, n, wait=True):
            v = self.bng.vehicle
            model_time = n / self.bng.sps / sk
            v.s, _ = car.step(v.s, v.u, n_sub=max(1, int(round(model_time / car.dt))), want_info=False)

    class Settings:
        def __init__(self, bng):
            self.bng = bng

        def set_deterministic(self, sps):
            self.bng.sps = sps

    class ScenarioApi:
        def load(self, scenario):
            pass

        def start(self):
            pass

    class BeamNGpy:
        def __init__(self, host, port, home=None, user=None):
            self.control, self.settings, self.scenario = Ctl(self), Settings(self), ScenarioApi()

        def open(self, launch=True):
            return self

        def close(self):
            pass

    class Scenario:
        def __init__(self, level, name):
            pass

        def add_vehicle(self, vehicle, pos, rot_quat):
            self.vehicle = vehicle

        def make(self, bng):
            bng.vehicle = self.vehicle

    mod = types.ModuleType("beamngpy")
    mod.BeamNGpy, mod.Scenario, mod.Vehicle = BeamNGpy, Scenario, Vehicle
    sens = types.ModuleType("beamngpy.sensors")
    sens.Electrics = lambda: None
    mod.sensors = sens
    return {"beamngpy": mod, "beamngpy.sensors": sens}, sys


@pytest.mark.parametrize("mirror,steer_sign", [(False, 1.0), (False, -1.0), (True, 1.0), (True, -1.0)])
def test_beamng_driver_against_a_scaled_stand_in(trained, tmp_path, monkeypatch, mirror, steer_sign):
    from rc_drift_sim.deploy.beamng import BeamNGCar
    from rc_drift_sim.deploy.car_loop import drive
    from rc_drift_sim.deploy.runtime import PolicyRuntime
    from rc_drift_sim.rl.export import export_final
    out, _ = trained
    folder = export_final(out, out=tmp_path / "final")["folder"]
    rt = PolicyRuntime(folder)
    mods, sys = _fake_beamngpy(16.0, mirror, steer_sign, grip=0.7)
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    car = BeamNGCar(rt.meta, home="x")
    res = drive(rt, car, 1.0, tmp_path / "beamng.csv", verbose=False)
    assert car.scale == pytest.approx(16 * 1.3 / 1.3, rel=0.2)          # bbox length / trained body length
    # calibration: a positive policy steer must show up as a positive measured yaw rate, whatever
    # BeamNG's steering sign and world handedness (a consistently mirrored world is fine)
    assert car.sign == steer_sign * (-1.0 if mirror else 1.0)
    assert res["steps"] == 50 and math.isfinite(res["mean_speed_2nd_half"])
    # the readings the policy got are in trained-car units: a speed of a few m/s, not tens
    import csv
    rows = list(csv.DictReader(open(tmp_path / "beamng.csv")))
    assert max(abs(float(r["s_vel_x"])) for r in rows) < 5.0


def test_find_beamng_home_and_zip_models(trained, tmp_path, monkeypatch):
    from rc_drift_sim.deploy.beamng import find_beamng_home
    from rc_drift_sim.deploy.runtime import PolicyRuntime
    from rc_drift_sim.rl.export import export_final
    home = tmp_path / "BeamNG.drive"
    (home / "Bin64").mkdir(parents=True)
    (home / "Bin64" / "BeamNG.drive.x64.exe").write_bytes(b"")
    monkeypatch.setenv("BNG_HOME", str(home))
    assert find_beamng_home() == str(home)
    out, _ = trained
    res = export_final(out, out=tmp_path / "final")
    rt = PolicyRuntime(tmp_path / "final")
    import shutil
    zcopy = tmp_path / "copied" / "final_model.zip"
    zcopy.parent.mkdir()
    shutil.copy(res["zip"], zcopy)
    rz = PolicyRuntime(zcopy)                                       # the zip copied to another PC
    assert rz.n_obs == rt.n_obs and (tmp_path / "copied" / "final_model" / "final" / "policy.npz").is_file()


def test_gui_beamng_test_drive(trained, tmp_path, monkeypatch):
    """The GUI's Test in BeamNG button, against the scaled stand-in for beamngpy."""
    import shutil
    import sys as _sys
    from rc_drift_sim.app.server import make_server
    out, _ = trained
    runs = tmp_path / "rl_runs"
    run_dir = runs / "20260101-000000_trained"
    shutil.copytree(out, run_dir)
    (run_dir / "status.json").write_text(json.dumps(dict(status="complete", name="trained", iteration=3, n_iter=3)))
    mods, _ = _fake_beamngpy(16.0, False, 1.0, grip=0.7)
    for name, m in mods.items():
        monkeypatch.setitem(_sys.modules, name, m)
    srv = make_server("127.0.0.1", 0, root=tmp_path / "exports", runs_root=runs)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(method, path, body=None):
        req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    try:
        rid = run_dir.name
        assert call("GET", "/api/rl/catalog")["beamng"]["beamngpy"] is True
        call("POST", f"/api/rl/runs/{rid}/export", {})
        call("POST", f"/api/rl/runs/{rid}/beamng", {"home": "x", "seconds": 1.0, "beta": 30})
        t_end = time.time() + 120
        while time.time() < t_end:
            b = call("GET", f"/api/rl/runs/{rid}")["beamng"]
            if b["status"] not in ("starting", "driving"):
                break
            time.sleep(0.2)
        assert b["status"] == "complete", b
        assert b["summary"]["steps"] == 50 and any("steering sign" in m for m in b["messages"])
    finally:
        srv.shutdown()
        srv.server_close()


def test_gui_lists_runs_trained_on_the_command_line(trained, tmp_path):
    import shutil
    from rc_drift_sim.app.rl_runs import RLManager
    out, _ = trained
    shutil.copytree(out, tmp_path / "from_jetson", ignore=shutil.ignore_patterns("final*", "status.json"))
    mgr = RLManager(tmp_path)
    runs = mgr.list()
    assert [r["id"] for r in runs] == ["from_jetson"] and runs[0]["best_score"] is not None
    d = mgr.get("from_jetson")
    assert d["status"] == "complete" and len(d["history"]) == 3 and d["export"] is None
