"""RL pages of the GUI: the parameter catalog covers every config field, the GUI JSON converts and
validates, and a real server trains, records snapshots, replays episodes, stops and serves files."""
from __future__ import annotations

import dataclasses
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from rc_drift_sim.rl import EnvConfig, RewardWeights
from rc_drift_sim.rl.catalog import PPO_DEFAULTS, build_configs, catalog

torch = pytest.importorskip("torch")


def test_catalog_covers_every_field_with_the_dataclass_defaults():
    from rc_drift_sim.rl.ppo import PPOConfig
    cat = catalog()
    from rc_drift_sim.deploy.safety import SafetyConfig
    env_keys = {f.name for f in dataclasses.fields(EnvConfig)} - {"noise", "reward", "randomize", "safety"}
    assert {f["key"] for f in cat["env"]} == env_keys
    assert {f["key"] for f in cat["safety"]} == {f.name for f in dataclasses.fields(SafetyConfig)}
    assert {f["key"] for f in cat["reward"]} == {f.name for f in dataclasses.fields(RewardWeights)}
    assert {f["key"] for f in cat["noise"]} == set(EnvConfig().noise)
    assert {f["key"] for f in cat["ppo"]} == {f.name for f in dataclasses.fields(PPOConfig)}
    assert dataclasses.asdict(PPOConfig()) == PPO_DEFAULTS, "catalog.PPO_DEFAULTS must match ppo.PPOConfig"
    assert [p["key"] for p in cat["presets"]][0] == "safe-adaptive"
    assert cat["defaults"]["env"]["target_beta_deg"] == EnvConfig().target_beta_deg
    for part in ("env", "noise", "reward", "safety", "ppo", "run"):
        for f in cat[part]:
            assert f["label"] and f["desc"], f"{part}.{f['key']} needs a label and a description"


def test_build_configs_converts_and_reports_problems():
    env, ppo, run, errors = build_configs({"env": {"task": "track", "track_radius": 2.0}, "reward": {"drift": 1.5},
                                           "ppo": {"hidden": "128, 64", "num_envs": 64}, "noise": {"gyro": 0.0},
                                           "randomize": {"vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8}}})
    assert errors == [] and env.task == "track" and env.track_radius == 2.0 and env.reward.drift == 1.5
    assert ppo["hidden"] == (128, 64) and env.noise["gyro"] == 0.0 and "vehicle.mass" in env.randomize
    _, _, _, errors = build_configs({"env": {"task": "donut", "episode_s": -1}, "ppo": {"hidden": "a, b"}, "bogus": 1,
                                     "reward": {"nope": 1}})
    assert any("env.task" in e for e in errors) and any("episode_s" in e for e in errors)
    assert any("hidden" in e for e in errors) and any("unknown fields" in e for e in errors)
    _, _, _, errors = build_configs({"randomize": {"drivetrain.layout": {"dist": "fixed", "value": "awd_spool"}}})
    assert any("structural" in e for e in errors)


@pytest.fixture()
def server(tmp_path):
    from rc_drift_sim.app.server import make_server
    srv = make_server("127.0.0.1", 0, root=tmp_path / "exports", runs_root=tmp_path / "rl_runs")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.rl.stop_all()
    srv.rl.wait_idle(20)
    srv.shutdown()
    srv.server_close()


def call(srv, method, path, body=None, raw=False):
    url = f"http://127.0.0.1:{srv.server_address[1]}{path}"
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data))
    except urllib.error.HTTPError as e:
        return e.code, (e.read() if raw else json.loads(e.read()))


def wait(srv, rid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = call(srv, "GET", f"/api/rl/runs/{rid}")[1]
        if s["status"] not in ("queued", "running"):
            return s
        time.sleep(0.3)
    raise TimeoutError(rid)


SMALL = {"name": "gui test", "env": {"task": "hold", "episode_s": 1.0},
         "ppo": {"total_steps": 3 * 32 * 16, "num_envs": 32, "rollout": 16, "minibatches": 2, "epochs": 2},
         "run": {"snapshots": 2}}


def test_server_trains_records_and_replays(server):
    assert call(server, "GET", "/api/rl/catalog")[1]["available"]
    code, v = call(server, "POST", "/api/rl/validate", SMALL)
    assert code == 200 and v["errors"] == [] and v["iterations"] == 3
    assert call(server, "POST", "/api/rl/runs", {"env": {"task": "nope"}})[0] == 400
    code, run = call(server, "POST", "/api/rl/runs", SMALL)
    assert code == 201
    s = wait(server, run["id"])
    assert s["status"] == "complete" and s["n_history"] == 3 and set(s["baselines"]) >= {"no input", "random"}
    assert [x["iteration"] for x in s["snapshots"]] == [1, 2, 3]
    snap = call(server, "GET", f"/api/rl/runs/{run['id']}/snapshots/{s['snapshots'][-1]['name']}")[1]
    assert snap["steps"] == len(snap["series"]["reward"]) and len(snap["series"]["x"]) == snap["steps"] + 1
    assert call(server, "GET", f"/api/rl/runs/{run['id']}?since=2")[1]["history"][0]["iteration"] == 3
    ro = call(server, "POST", f"/api/rl/runs/{run['id']}/rollout", {"policy": "lqr", "seed": 1})[1]
    assert ro["ended"] == "time limit" and ro["episode_return"] > 0          # 1 s: still launching
    code, blob = call(server, "GET", f"/api/rl/runs/{run['id']}/files/policy.pt", raw=True)
    assert code == 200 and len(blob) > 1000
    assert call(server, "GET", f"/api/rl/runs/{run['id']}/snapshots/..%2Fconfig.json")[0] == 404
    assert call(server, "GET", f"/api/rl/runs/{run['id']}/files/..%2F..%2Fx")[0] == 404
    listed = call(server, "GET", "/api/rl/runs")[1]["runs"]
    assert listed[0]["id"] == run["id"] and listed[0]["best_return"] is not None
    cfg = json.loads((server.rl.root / run["id"] / "config.json").read_text())
    assert cfg["ppo"]["num_envs"] == 32


def test_server_stops_a_run_and_keeps_its_policy(server):
    long = dict(SMALL, name="long", ppo=dict(SMALL["ppo"], total_steps=10_000_000), run={"snapshots": 0})
    code, run = call(server, "POST", "/api/rl/runs", long)
    assert code == 201
    t0 = time.time()
    while call(server, "GET", f"/api/rl/runs/{run['id']}")[1]["n_history"] < 1 and time.time() - t0 < 120:
        time.sleep(0.3)
    assert call(server, "POST", f"/api/rl/runs/{run['id']}/stop", {})[0] == 200
    s = wait(server, run["id"], timeout=60)
    assert s["status"] == "cancelled" and s["n_history"] >= 1
    assert (server.rl.root / run["id"] / "policy.pt").is_file()
