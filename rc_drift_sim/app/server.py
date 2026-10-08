"""Local web GUI for batch dataset generation. Standard library only (``http.server``).

    python -m rc_drift_sim.app            # or: driftsim-gui
    -> http://127.0.0.1:8765/

The page (``static/``) edits a batch spec (see ``rc_drift_sim.datagen.spec``); this module serves it
and a small JSON API:

    GET  /api/catalog                    variables, maneuvers, signals, defaults, CPU count
    GET  /api/examples                   example specs from examples/specs/ (when present)
    POST /api/validate      {spec}       errors, warnings, normalized spec, size and time estimate
    POST /api/preview       {spec, n}    simulate the first n (<= 8) episodes, return time series
    GET  /api/jobs                       queued / running / finished jobs of this session
    POST /api/jobs          {spec}       queue a batch (jobs run one after another)
    GET  /api/jobs/<id>                  one job
    POST /api/jobs/<id>/cancel           cancel a queued or running job (written shards are kept)
    GET  /api/datasets                   every dataset folder under the export root (from manifests)
    GET  /api/datasets/<name>            manifest (incl. spec) and file list of one dataset
    GET  /api/datasets/<name>/zip        the whole dataset as a streamed .zip
    GET  /api/datasets/<name>/files/<f>  one file (<f> may be in a folder: all%2Fshard_0000.npz)
    POST /api/datasets/<name>/reveal     open the folder in the file manager

Security: binds to 127.0.0.1 by default, rejects requests whose Host header is not this server
(DNS-rebinding protection), requires ``Content-Type: application/json`` on POST (so plain
cross-site form posts cannot start jobs), sends no CORS headers, and only serves files inside the
static folder and the export root.
"""
from __future__ import annotations

import json
import math
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from .. import __version__
from ..datagen import build_catalog, default_spec, estimate, preview, run_batch, validate_spec
from ..sim.xp import available_devices
from ..datagen.runner import SpecError, default_out_root
from .rl_runs import RLManager, available as rl_available


def beamng_info() -> dict:
    """Whether BeamNG tests can run on this machine: beamngpy installed, BeamNG found (for the GUI)."""
    from ..deploy.beamng import beamngpy_installed, find_beamng_home
    return dict(beamngpy=beamngpy_installed(), home=find_beamng_home())

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "specs"
MAX_BODY = 4 * 1024 * 1024
MAX_PREVIEW_EPISODES = 8
MAX_PREVIEW_DURATION = 20.0
DEFAULT_RATE = 40_000.0          # control steps per second, until a finished job measures it
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
STORED = {".npz", ".parquet", ".gif", ".png"}   # already compressed: zip them without deflate


# ----------------------------------------------------------------------------- JSON helpers
def clean(obj: Any, digits: int | None = None) -> Any:
    """JSON-safe copy: numpy -> lists/floats, NaN/inf -> None, optional rounding of floats."""
    if isinstance(obj, dict):
        return {str(k): clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist(), digits)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if not math.isfinite(f):
            return None
        return round(f, digits) if digits is not None else f
    if isinstance(obj, slice):
        return None
    return obj


# ----------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, spec: dict, warnings: list[str]):
        self.id = uuid.uuid4().hex[:10]
        self.spec = spec
        self.warnings = warnings
        self.status = "queued"
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.progress: dict = {}
        self.out_dir: str | None = None
        self.summary: dict | None = None
        self.error: str | None = None
        self.cancel = threading.Event()

    def to_json(self) -> dict:
        return clean(dict(id=self.id, name=self.spec["name"], status=self.status, created=self.created,
                          started=self.started, finished=self.finished, progress=self.progress,
                          episodes=self.spec["episodes"], duration_s=self.spec["duration_s"],
                          dataset=Path(self.out_dir).name if self.out_dir else None,
                          summary=self.summary, error=self.error, warnings=self.warnings))


class JobManager:
    """Runs jobs one at a time in a background thread (each job itself uses all CPU cores)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.q: queue.Queue[str] = queue.Queue()
        self.rate = DEFAULT_RATE
        self.thread = threading.Thread(target=self._worker, name="driftsim-jobs", daemon=True)
        self.thread.start()

    def submit(self, spec: dict) -> Job:
        norm, errors, warnings = validate_spec(spec)
        if errors:
            raise SpecError(errors)
        job = Job(norm, warnings)
        with self.lock:
            self.jobs[job.id] = job
        self.q.put(job.id)
        return job

    def get(self, jid: str) -> Job | None:
        with self.lock:
            return self.jobs.get(jid)

    def list(self) -> list[dict]:
        with self.lock:
            jobs = list(self.jobs.values())
        return [j.to_json() for j in sorted(jobs, key=lambda j: j.created, reverse=True)]

    def cancel(self, jid: str) -> Job | None:
        job = self.get(jid)
        if job is not None and job.status in ("queued", "running"):
            job.cancel.set()
            if job.status == "queued":
                job.status, job.finished = "cancelled", time.time()
        return job

    def cancel_all(self) -> None:
        with self.lock:
            jobs = list(self.jobs.values())
        for j in jobs:
            j.cancel.set()

    def wait_idle(self, timeout: float = 15.0) -> bool:
        """Wait until no job is running (e.g. after cancel_all). True if idle in time."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                busy = any(j.status == "running" for j in self.jobs.values())
            if not busy:
                return True
            time.sleep(0.1)
        return False

    def active_dirs(self) -> set[str]:
        with self.lock:
            return {Path(j.out_dir).name for j in self.jobs.values() if j.out_dir and j.status == "running"}

    def _worker(self) -> None:
        while True:
            jid = self.q.get()
            job = self.get(jid)
            if job is None or job.cancel.is_set():
                continue
            job.status, job.started = "running", time.time()

            def progress(p: dict, job=job) -> None:
                job.progress = p
                if p.get("out_dir"):
                    job.out_dir = p["out_dir"]

            try:
                res = run_batch(job.spec, out_root=self.root, progress=progress, cancel=job.cancel)
                job.summary = res
                job.out_dir = res["out_dir"]
                job.status = res["status"]
                job.error = res.get("error")
                steps = res["episodes_written"] * job.spec["duration_s"] / 0.02
                if res["status"] == "complete" and res["seconds"] > 5.0 and steps > 0:   # long runs only: start-up excluded
                    self.rate = 0.5 * self.rate + 0.5 * steps / res["seconds"]
            except Exception as exc:
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.finished = time.time()


# ----------------------------------------------------------------------------- datasets on disk
def _dataset_files(d: Path) -> list[tuple[str, Path]]:
    """(relative posix name, path) of every file in a dataset, including its all/ and not_spun/
    folders (older exports are flat), sorted by name; unfinished *.tmp files are skipped."""
    out = [(f.relative_to(d).as_posix(), f) for f in d.rglob("*")
           if f.is_file() and not f.is_symlink() and not f.name.endswith(".tmp")]
    return sorted(out, key=lambda x: (x[0].count("/"), x[0]))


def _dir_size(d: Path) -> int:
    return sum(f.stat().st_size for _, f in _dataset_files(d))


def dataset_info(d: Path, active: set[str]) -> dict | None:
    mf = d / "manifest.json"
    if not mf.is_file():
        return None
    try:
        m = json.loads(mf.read_text())
    except (OSError, ValueError):
        return None
    status = m.get("status", "unknown")
    if status == "running" and d.name not in active:
        status = "interrupted"
    spec = m.get("spec", {})
    return clean(dict(id=d.name, name=spec.get("name", d.name), created=m.get("created"), status=status,
                      episodes=m.get("episodes"), episodes_written=m.get("episodes_written"),
                      seconds=m.get("seconds"), bytes=_dir_size(d), stats=m.get("stats", {}),
                      formats=spec.get("export", {}).get("formats", []),
                      maneuver=spec.get("maneuver", {}).get("type"), duration_s=spec.get("duration_s"),
                      error=m.get("error")))


def list_datasets(root: Path, active: set[str]) -> list[dict]:
    if not root.is_dir():
        return []
    out = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        info = dataset_info(d, active)
        if info:
            out.append(info)
    return out


def _open_folder(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)])


# ----------------------------------------------------------------------------- HTTP
class _ZipSink:
    """Minimal unseekable file object over the response stream, for zipfile."""

    def __init__(self, w):
        self.w = w

    def write(self, b) -> int:
        self.w.write(b)
        return len(b)

    def flush(self) -> None:
        self.w.flush()


class Handler(BaseHTTPRequestHandler):
    server_version = f"DriftSimGUI/{__version__}"
    protocol_version = "HTTP/1.0"      # one request per connection: streamed zips can end with close

    # --------------------------------------------------------------- plumbing
    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    @property
    def app(self) -> "GuiServer":
        return self.server  # type: ignore[return-value]

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(clean(obj)).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str, **extra) -> None:
        self._json(dict(error=message, **extra), status)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in self.app.allowed_hosts

    def _body(self) -> Any:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise _HttpError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "POST bodies must be application/json")
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise _HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "invalid JSON") from None

    def _dataset_dir(self, name: str) -> Path:
        name = unquote(name)
        root = self.app.root.resolve()
        d = (root / name).resolve()
        if not NAME_RE.match(name) or d.parent != root or not (d / "manifest.json").is_file():
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such dataset")
        return d

    # --------------------------------------------------------------- dispatch
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        try:
            if not self._host_ok():
                raise _HttpError(HTTPStatus.FORBIDDEN, "unexpected Host header")
            path = urlparse(self.path).path
            for m, pattern, fn in ROUTES:
                if m == method:
                    match = pattern.fullmatch(path)
                    if match:
                        return fn(self, *match.groups())
            if method == "GET" and not path.startswith("/api/"):
                return self._static(path)
            raise _HttpError(HTTPStatus.NOT_FOUND, "not found")
        except _HttpError as e:
            self._error(e.status, e.message, **e.extra)
        except SpecError as e:
            self._error(HTTPStatus.BAD_REQUEST, "invalid spec", errors=e.errors)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pragma: no cover - last-resort guard, keeps the server alive
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    # --------------------------------------------------------------- static files
    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "/index.html") else unquote(path.lstrip("/"))
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            raise _HttpError(HTTPStatus.NOT_FOUND, "not found")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, f.read_bytes(), ctype)

    # --------------------------------------------------------------- API
    def catalog(self) -> None:
        cat = dict(build_catalog())
        cat["signals"] = [dict(g, signals=[{k: v for k, v in s.items() if k not in ("src", "source")}
                                           for s in g["signals"]]) for g in cat["signals"]]
        self._json(dict(catalog=cat, default_spec=default_spec(), cpu_count=os.cpu_count() or 1,
                        devices=available_devices(),
                        rate=self.app.jobs.rate, version=__version__, export_root=str(self.app.root)))

    def examples(self) -> None:
        out = []
        if EXAMPLES.is_dir():
            for f in sorted(EXAMPLES.glob("*.json")):
                try:
                    out.append(dict(file=f.name, spec=json.loads(f.read_text())))
                except (OSError, ValueError):
                    continue
        self._json(dict(examples=out))

    def validate(self) -> None:
        body = self._body()
        spec = body.get("spec", body)
        norm, errors, warnings = validate_spec(spec)
        est = None if errors else estimate(norm)
        self._json(dict(errors=errors, warnings=warnings, spec=norm, estimate=est, rate=self.app.jobs.rate))

    def preview(self) -> None:
        body = self._body()
        spec = dict(body.get("spec", {}))
        n = max(1, min(int(body.get("n", 6)), MAX_PREVIEW_EPISODES))
        if isinstance(spec.get("duration_s"), (int, float)):
            spec["duration_s"] = min(float(spec["duration_s"]), MAX_PREVIEW_DURATION)
        t0 = time.perf_counter()
        with np.errstate(all="ignore"):
            res = preview(spec, n=n)
        res["seconds"] = time.perf_counter() - t0
        self._json(clean(res, digits=5))

    def jobs_list(self) -> None:
        self._json(dict(jobs=self.app.jobs.list(), rate=self.app.jobs.rate))

    def jobs_create(self) -> None:
        body = self._body()
        job = self.app.jobs.submit(body.get("spec", body))
        self._json(job.to_json(), HTTPStatus.CREATED)

    def job_get(self, jid: str) -> None:
        job = self.app.jobs.get(jid)
        if job is None:
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such job")
        self._json(job.to_json())

    def job_cancel(self, jid: str) -> None:
        self._body()
        job = self.app.jobs.cancel(jid)
        if job is None:
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such job")
        self._json(job.to_json())

    def datasets_list(self) -> None:
        self._json(dict(datasets=list_datasets(self.app.root, self.app.jobs.active_dirs()),
                        root=str(self.app.root)))

    def dataset_get(self, name: str) -> None:
        d = self._dataset_dir(name)
        m = json.loads((d / "manifest.json").read_text())
        files = [dict(name=rel, bytes=f.stat().st_size) for rel, f in _dataset_files(d)]
        self._json(dict(id=d.name, path=str(d), manifest=m, files=files))

    def dataset_zip(self, name: str) -> None:
        d = self._dataset_dir(name)
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{d.name}.zip"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True
        with zipfile.ZipFile(_ZipSink(self.wfile), "w") as zf:
            for rel, f in _dataset_files(d):
                kind = zipfile.ZIP_STORED if f.suffix in STORED else zipfile.ZIP_DEFLATED
                zf.write(f, arcname=f"{d.name}/{rel}", compress_type=kind, compresslevel=1)

    def dataset_file(self, name: str, fname: str) -> None:
        d = self._dataset_dir(name)
        fname = unquote(fname)
        f = (d / fname).resolve()
        if d not in f.parents or not f.is_file():
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such file")
        size = f.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{f.relative_to(d).as_posix().replace("/", "_")}"')
        self.end_headers()
        with open(f, "rb") as fh:
            while chunk := fh.read(1 << 20):
                self.wfile.write(chunk)

    def dataset_reveal(self, name: str) -> None:
        self._body()
        d = self._dataset_dir(name)
        _open_folder(d)
        self._json(dict(ok=True, path=str(d)))

    # --------------------------------------------------------------- RL training
    def _rl_folder(self, name: str) -> str:
        name = unquote(name)
        if not NAME_RE.match(name):
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such run")
        return name

    def rl_catalog(self) -> None:
        ok, why = rl_available()
        out = dict(available=ok, reason=why, devices=available_devices(), runs_root=str(self.app.rl.root),
                   beamng=beamng_info())
        if ok:
            from ..rl.catalog import catalog
            out["catalog"] = catalog()
        self._json(out)

    def rl_validate(self) -> None:
        body = self._body()
        ok, why = rl_available()
        if not ok:
            return self._json(dict(errors=[why], iterations=None))
        from ..rl.catalog import build_configs
        env_cfg, ppo, run, errors = build_configs(body if isinstance(body, dict) else {})
        it = None
        if ppo:
            it = max(1, ppo["total_steps"] // (ppo["num_envs"] * ppo["rollout"]))
        self._json(dict(errors=errors, iterations=it,
                        samples_per_iteration=ppo["num_envs"] * ppo["rollout"] if ppo else None))

    def rl_list(self) -> None:
        self._json(dict(runs=self.app.rl.list(), root=str(self.app.rl.root)))

    def rl_start(self) -> None:
        body = self._body()
        try:
            run = self.app.rl.start(body if isinstance(body, dict) else {})
        except ValueError as exc:
            raise _HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from None
        self._json(run.to_json(), HTTPStatus.CREATED)

    def rl_get(self, name: str) -> None:
        q = parse_qs(urlparse(self.path).query)
        since = int((q.get("since") or ["0"])[0] or 0)
        try:
            self._json(self.app.rl.get(self._rl_folder(name), max(0, since)))
        except (FileNotFoundError, OSError, ValueError):
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such run") from None

    def rl_stop(self, name: str) -> None:
        self._body()
        run = self.app.rl.stop(self._rl_folder(name))
        if run is None:
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such run in this session")
        self._json(run.summary())

    def rl_snapshot(self, name: str, snap: str) -> None:
        try:
            self._json(self.app.rl.snapshot(self._rl_folder(name), unquote(snap)))
        except (FileNotFoundError, OSError, ValueError):
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such snapshot") from None

    def rl_rollout(self, name: str) -> None:
        body = self._body() or {}
        policy = body.get("policy", "policy")
        if policy not in ("policy", "best", "lqr", "zero"):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "policy must be policy, best, lqr or zero")
        init = body.get("init_drift")
        grip, change = body.get("grip"), body.get("grip_change")
        try:
            grip = None if grip in (None, "") else float(grip)
            if grip is not None and not 0.01 <= grip <= 5.0:
                raise ValueError
            change = None if not change else (float(change[0]), float(change[1]))
            if change is not None and not (0.05 <= change[0] <= 5.0 and 0.0 <= change[1] <= 600.0):
                raise ValueError
        except (TypeError, ValueError, IndexError):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "grip must be 0.01-5 and grip_change [factor 0.05-5, seconds]") from None
        try:
            data = self.app.rl.rollout(self._rl_folder(name), policy, int(body.get("seed", 0)),
                                       None if init is None else bool(init), grip=grip, grip_change=change)
        except FileNotFoundError as exc:
            raise _HttpError(HTTPStatus.NOT_FOUND, str(exc) or "no such run") from None
        except RuntimeError as exc:
            raise _HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from None
        self._json(data)

    def rl_export(self, name: str) -> None:
        body = self._body() or {}
        which = body.get("which", "auto")
        if which not in ("auto", "best", "last"):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "which must be auto, best or last")
        try:
            self._json(self.app.rl.export(self._rl_folder(name), which))
        except FileNotFoundError as exc:
            raise _HttpError(HTTPStatus.NOT_FOUND, str(exc) or "no such run") from None
        except (ValueError, RuntimeError) as exc:
            raise _HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from None

    def rl_beamng(self, name: str) -> None:
        body = self._body() or {}
        try:
            if body.get("action") == "stop":
                self._json(self.app.rl.beamng_stop() or {})
            else:
                self._json(self.app.rl.beamng_start(self._rl_folder(name), body))
        except FileNotFoundError as exc:
            raise _HttpError(HTTPStatus.NOT_FOUND, str(exc) or "no such run") from None
        except ValueError as exc:
            raise _HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from None

    def rl_file(self, name: str, fname: str) -> None:
        try:
            f = self.app.rl.file(self._rl_folder(name), unquote(fname))
        except (FileNotFoundError, OSError):
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such file") from None
        size = f.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{self._rl_folder(name)}_{f.name}"')
        self.end_headers()
        with open(f, "rb") as fh:
            while chunk := fh.read(1 << 20):
                self.wfile.write(chunk)

    def rl_reveal(self, name: str) -> None:
        self._body()
        try:
            d = self.app.rl._dir(self._rl_folder(name))
        except FileNotFoundError:
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such run") from None
        _open_folder(d)
        self._json(dict(ok=True, path=str(d)))


class _HttpError(Exception):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = int(status), message, extra


_N = r"([A-Za-z0-9][A-Za-z0-9_.\-%]*)"
ROUTES = [(m, re.compile(p), fn) for m, p, fn in [
    ("GET", r"/api/catalog", Handler.catalog),
    ("GET", r"/api/examples", Handler.examples),
    ("POST", r"/api/validate", Handler.validate),
    ("POST", r"/api/preview", Handler.preview),
    ("GET", r"/api/jobs", Handler.jobs_list),
    ("POST", r"/api/jobs", Handler.jobs_create),
    ("GET", r"/api/jobs/([0-9a-f]{10})", Handler.job_get),
    ("POST", r"/api/jobs/([0-9a-f]{10})/cancel", Handler.job_cancel),
    ("GET", r"/api/datasets", Handler.datasets_list),
    ("GET", rf"/api/datasets/{_N}", Handler.dataset_get),
    ("GET", rf"/api/datasets/{_N}/zip", Handler.dataset_zip),
    ("GET", rf"/api/datasets/{_N}/files/{_N}", Handler.dataset_file),
    ("POST", rf"/api/datasets/{_N}/reveal", Handler.dataset_reveal),
    ("GET", r"/api/rl/catalog", Handler.rl_catalog),
    ("POST", r"/api/rl/validate", Handler.rl_validate),
    ("GET", r"/api/rl/runs", Handler.rl_list),
    ("POST", r"/api/rl/runs", Handler.rl_start),
    ("GET", rf"/api/rl/runs/{_N}", Handler.rl_get),
    ("POST", rf"/api/rl/runs/{_N}/stop", Handler.rl_stop),
    ("GET", rf"/api/rl/runs/{_N}/snapshots/{_N}", Handler.rl_snapshot),
    ("POST", rf"/api/rl/runs/{_N}/rollout", Handler.rl_rollout),
    ("POST", rf"/api/rl/runs/{_N}/export", Handler.rl_export),
    ("POST", rf"/api/rl/runs/{_N}/beamng", Handler.rl_beamng),
    ("GET", rf"/api/rl/runs/{_N}/files/{_N}", Handler.rl_file),
    ("POST", rf"/api/rl/runs/{_N}/reveal", Handler.rl_reveal),
]]


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, root: Path, verbose: bool = False, runs_root: Path | None = None):
        super().__init__((host, port), Handler)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = JobManager(self.root)
        rl_root = Path(runs_root) if runs_root is not None else self.root.parent / "rl_runs"
        rl_root.mkdir(parents=True, exist_ok=True)
        self.rl = RLManager(rl_root)
        self.verbose = verbose
        p = self.server_address[1]
        names = {"127.0.0.1", "localhost", "[::1]"}
        if host not in ("", "0.0.0.0", "::", "127.0.0.1", "localhost"):
            names.add(host.lower())
        self.allowed_hosts = {f"{n}:{p}" for n in names}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


def make_server(host: str = "127.0.0.1", port: int = 8765, root: str | Path | None = None,
                verbose: bool = False, port_tries: int = 20, runs_root: str | Path | None = None) -> GuiServer:
    """Create (not start) the server; if ``port`` is taken, the next free port is used. RL training
    runs go to ``runs_root`` (default: ``rl_runs`` next to the export root)."""
    root = Path(root) if root is not None else default_out_root()
    last: OSError | None = None
    for p in ([port] if port == 0 else range(port, port + port_tries)):
        try:
            return GuiServer(host, p, root, verbose, None if runs_root is None else Path(runs_root))
        except OSError as exc:
            last = exc
    raise OSError(f"no free port in {port}..{port + port_tries - 1}: {last}")
