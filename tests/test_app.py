"""Web GUI server: API end to end on a real HTTP server (ephemeral port, temporary export root)."""
from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile

import pytest

from rc_drift_sim.app import make_server


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("exports")
    httpd = make_server("127.0.0.1", 0, root)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd
    httpd.jobs.cancel_all()
    httpd.shutdown()
    httpd.server_close()


def call(server, method, path, body=None, headers=None, raw=False):
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = r.read()
            return r.status, (payload if raw else json.loads(payload or b"null"))
    except urllib.error.HTTPError as e:
        payload = e.read()
        try:
            return e.code, json.loads(payload)
        except ValueError:
            return e.code, payload


def small_spec(**kw):
    spec = {"name": "t", "episodes": 4, "duration_s": 0.3,
            "params": {"vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8}},
            "maneuver": {"type": "random"}, "export": {"formats": ["npz", "csv"]}, "run": {"workers": 1}}
    spec.update(kw)
    return spec


def wait_job(server, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, j = call(server, "GET", f"/api/jobs/{jid}")
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_page_and_static_files(server):
    code, html = call(server, "GET", "/", raw=True)
    assert code == 200 and b"DriftSim" in html and b"/app.js" in html
    for f in ("/app.js", "/style.css"):
        assert call(server, "GET", f, raw=True)[0] == 200


def test_catalog(server):
    code, r = call(server, "GET", "/api/catalog")
    assert code == 200
    ids = [g["id"] for g in r["catalog"]["groups"]]
    assert {"vehicle", "tire", "surface", "condition", "init"} <= set(ids)
    assert "drift_schedule" in r["catalog"]["maneuvers"] and r["default_spec"]["episodes"] > 0
    assert all("src" not in s for g in r["catalog"]["signals"] for s in g["signals"])


def test_validate_reports_errors_and_estimates(server):
    code, r = call(server, "POST", "/api/validate", {"spec": small_spec()})
    assert code == 200 and r["errors"] == [] and r["estimate"]["total_steps"] == 4 * 15
    bad = small_spec(params={"vehicle.mass": {"dist": "uniform", "low": -1, "high": 1}})
    code, r = call(server, "POST", "/api/validate", {"spec": bad})
    assert code == 200 and any(e.startswith("vehicle.mass") for e in r["errors"]) and r["estimate"] is None


def test_preview_returns_json_safe_series(server):
    code, r = call(server, "POST", "/api/preview", {"spec": small_spec(), "n": 3})
    assert code == 200 and len(r["episodes"]) == 3
    e = r["episodes"][0]
    assert len(e["t"]) == len(e["x"]) == len(e["speed"]) == 16
    assert {"episode_id", "tire", "surface", "max_abs_beta_deg"} <= set(e["row"])


def test_job_runs_and_dataset_downloads(server):
    code, j = call(server, "POST", "/api/jobs", {"spec": small_spec(name="gui_test")})
    assert code == 201 and j["status"] in ("queued", "running")
    j = wait_job(server, j["id"])
    assert j["status"] == "complete" and j["summary"]["episodes_written"] == 4 and j["dataset"]
    code, lst = call(server, "GET", "/api/datasets")
    assert any(d["id"] == j["dataset"] and d["status"] == "complete" for d in lst["datasets"])
    code, d = call(server, "GET", f"/api/datasets/{j['dataset']}")
    names = [f["name"] for f in d["files"]]
    assert {"manifest.json", "all/episodes.csv", "all/shard_0000.npz", "all/shard_0000.csv",
            "not_spun/episodes.csv"} <= set(names)
    assert d["manifest"]["spec"]["name"] == "gui_test"
    code, blob = call(server, "GET", f"/api/datasets/{j['dataset']}/zip", raw=True)
    assert code == 200
    zf = zipfile.ZipFile(io.BytesIO(blob))
    assert {f"{j['dataset']}/{n}" for n in names} == set(zf.namelist()) and zf.testzip() is None
    for folder in ("all", "not_spun"):
        code, csv = call(server, "GET", f"/api/datasets/{j['dataset']}/files/{folder}%2Fepisodes.csv", raw=True)
        assert code == 200 and csv.startswith(b"episode_id,")
    assert call(server, "GET", f"/api/datasets/{j['dataset']}/files/all%2F..%2F..%2Fx", raw=True)[0] == 404


def test_invalid_job_is_rejected(server):
    code, r = call(server, "POST", "/api/jobs", {"spec": small_spec(episodes=0)})
    assert code == 400 and r["errors"]


def test_cancel_a_running_job(server):
    code, j = call(server, "POST", "/api/jobs", {"spec": small_spec(name="long", episodes=64, duration_s=60.0)})
    assert code == 201
    for _ in range(100):
        if call(server, "GET", f"/api/jobs/{j['id']}")[1]["status"] == "running":
            break
        time.sleep(0.05)
    call(server, "POST", f"/api/jobs/{j['id']}/cancel", {})
    j = wait_job(server, j["id"], timeout=60)
    assert j["status"] == "cancelled"


def test_security_checks(server):
    port = server.server_address[1]
    assert call(server, "GET", "/api/catalog", headers={"Host": f"evil.example:{port}"})[0] == 403
    url = f"http://127.0.0.1:{port}/api/jobs"
    req = urllib.request.Request(url, data=b"spec=x", method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=10)
    assert ei.value.code == 415
    assert call(server, "GET", "/../pyproject.toml", raw=True)[0] == 404
    assert call(server, "GET", "/%2e%2e/server.py", raw=True)[0] == 404
    assert call(server, "GET", "/api/datasets/..%2F..%2Fetc", raw=True)[0] == 404
    assert call(server, "GET", "/api/datasets/nope/files/..%2Fmanifest.json", raw=True)[0] == 404
