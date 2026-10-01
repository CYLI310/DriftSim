"""Run a batch spec: sample episodes, simulate them in vectorized shards (in parallel worker
processes), and export the data.

    from rc_drift_sim.datagen import run_batch
    summary = run_batch(spec, out_root="exports", progress=print)

Pipeline
--------
1. ``validate_spec`` normalizes the spec (errors abort before anything is written).
2. Episodes are grouped by their structural settings (layout, combined-slip mode, integrator, time
   steps, reverse enable): cars that share them are simulated together as ONE per-car-parameter
   batch (``sim.vehicle.VehicleBatch``); numeric parameters may differ freely inside a batch.
3. Each group is cut into shards of ``export.shard_episodes`` episodes; shards run in a process
   pool (``run.workers``, 0 = automatic) and each worker samples its own episodes by id, so the
   parent never holds all parameter sets in memory.
4. Every shard is written as soon as it is done (``export.write_shard``) into ``all/``, and its
   episodes that did not spin out also into ``not_spun/`` (same shard number); the two
   ``episodes.csv`` tables and ``manifest.json`` are written at the end (the manifest is also written at the start with
   status "running", and on cancellation or failure with the matching status).

Control latency: each car's commands reach the car ``round(latency / control_dt)`` control steps
later (zeros before the first command arrives), matching ``sim.actuators.ActionDelay``.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import subprocess
import time
import traceback
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import __version__
from ..sim.vehicle import VehicleBatch, derivatives_model
from . import export as X
from . import inputs
from .sampling import Sampler, structural_signature
from .spec import estimate, validate_spec

BETA_DRIFT_DEG, BETA_SPIN_DEG, DRIFT_MIN_SPEED, MOVING = 20.0, 80.0, 0.8, 0.05
PROGRESS_EVERY = 10          # control steps between progress / cancel checks inside a shard
MAX_SHARD_BYTES = 256e6      # cap on a shard's float64 signal buffers (matters at the physics record rate)


class SpecError(ValueError):
    """The spec did not validate; ``.errors`` lists the problems."""

    def __init__(self, errors: list[str]):
        super().__init__("invalid batch spec:\n  " + "\n  ".join(errors))
        self.errors = errors


class Cancelled(Exception):
    pass


def _episode_seed(seed: int, i: int) -> int:
    return zlib.crc32(f"{seed}:{i}".encode()) | (int(i) << 32)


# ----------------------------------------------------------------------------- core simulation
def simulate_episodes(spec: dict, ids: list[int], groups: list[str] | None = None,
                      tick: Callable[[float], None] | None = None,
                      is_cancelled: Callable[[], bool] | None = None) -> tuple[np.ndarray, dict, list[dict]]:
    """Simulate episodes ``ids`` of a normalized spec as one vectorized batch.

    Returns ``(t (T_rec,), data {signal: (n, T_rec[, 4])}, rows)`` where rows are the per-episode
    records (sampled values + outcome metrics). All episodes must share their structural settings.
    ``tick(fraction_done)`` is called every few steps; ``is_cancelled()`` aborts with ``Cancelled``.
    """
    groups = spec["export"]["signals"] if groups is None else groups
    sampler = Sampler(spec)
    eps = [sampler.episode(i) for i in ids]
    sigs = {e.signature for e in eps}
    if len(sigs) != 1:
        raise ValueError("simulate_episodes needs episodes with identical structural settings")
    batch = VehicleBatch([e.params for e in eps], conds=[e.cond for e in eps], check_stiffness=False)
    B = len(eps)
    cdt = batch.control_dt
    T = max(1, int(round(spec["duration_s"] / cdt)))
    physics = spec["export"].get("record_rate") == "physics"
    n_rec = batch.n_substeps if physics else 1        # recordable steps per control step
    dec = int(spec["export"]["decimation"])
    rec = list(range(0, T * n_rec + 1, dec))
    T_rec = len(rec)
    t = np.asarray(rec, dtype=np.float64) * (batch.dt if physics else cdt)

    init = {k: np.array([e.init[k] for e in eps]) for k in ("x", "y", "yaw", "v", "beta", "yaw_rate")}
    s = batch.initial_states(x=init["x"], y=init["y"], yaw=init["yaw"], v=init["v"], beta=init["beta"],
                             yaw_rate=init["yaw_rate"])
    mtype = spec["maneuver"]["type"]
    prm = {k: np.array([e.maneuver[k] for e in eps]) for k in eps[0].maneuver}
    seeds = np.array([_episode_seed(spec["seed"], e.id) for e in eps], dtype=np.uint64)
    cmd, mirrored = inputs.generate(mtype, prm, T, cdt, seeds, return_flags=True)  # (T, B, 2)
    delay = np.array([int(math.floor(e.params.actuators.latency / cdt + 0.5)) for e in eps])
    applied = np.zeros_like(cmd)
    for d in np.unique(delay):
        cols = delay == d
        if d < T:
            applied[d:, cols] = cmd[:T - d, cols]

    sigs_sel = X.selected_signals(groups)
    want_info = X.needs_info(groups)
    data = {sg["name"]: np.empty((B, T_rec, 4) if sg["per_wheel"] else (B, T_rec)) for sg in sigs_sel}
    # outcome metrics at the full control rate
    max_beta = np.zeros(B)
    drift_time = np.zeros(B)
    spun = np.zeros(B, dtype=bool)
    max_speed = np.zeros(B)
    max_r = np.zeros(B)
    dist = np.zeros(B)

    def record(j_rec: int, k: int, s: np.ndarray) -> None:
        a_idx = min(k, T - 1)
        info = derivatives_model(s, applied[a_idx], batch.model, want_info=True)[1] if want_info else None
        for sg in sigs_sel:
            src, key = sg["source"], sg["src"]
            if src == "state":
                val = s[:, key]
            elif src == "derived":
                val = np.hypot(s[:, 3], s[:, 4]) if key == "speed" else np.arctan2(s[:, 4], s[:, 3])
            elif src == "action":
                val = (cmd if key.startswith("cmd") else applied)[a_idx, :, int(key[-1])]
            else:
                val = np.asarray(info[key])
            data[sg["name"]][:, j_rec] = val * sg["scale"] if "scale" in sg else val

    def metrics(s: np.ndarray) -> None:
        nonlocal max_beta, drift_time, spun, max_speed, max_r, dist
        sp = np.hypot(s[:, 3], s[:, 4])
        beta = np.degrees(np.abs(np.arctan2(s[:, 4], s[:, 3])))
        moving = sp > MOVING
        max_beta = np.where(moving, np.maximum(max_beta, beta), max_beta)
        drifting = moving & (beta > BETA_DRIFT_DEG) & (beta < BETA_SPIN_DEG) & (sp > DRIFT_MIN_SPEED)
        drift_time = drift_time + cdt * drifting
        spun = spun | (moving & (beta >= BETA_SPIN_DEG))
        max_speed = np.fmax(max_speed, sp)
        max_r = np.fmax(max_r, np.degrees(np.abs(s[:, 5])))
        dist = dist + sp * cdt

    j = 0
    with np.errstate(all="ignore"):
        if rec[0] == 0:
            record(0, 0, s)
            j = 1
        for k in range(T):
            for m in range(n_rec):                    # physics rate: one integrator step at a time
                s, _ = batch.step(s, applied[k], n_sub=None if n_rec == 1 else 1)
                idx = k * n_rec + m + 1
                if j < T_rec and rec[j] == idx:
                    record(j, idx // n_rec, s)
                    j += 1
            metrics(s)
            if (k + 1) % PROGRESS_EVERY == 0:
                if tick is not None:
                    tick((k + 1) / T)
                if is_cancelled is not None and is_cancelled():
                    raise Cancelled()
    finite = np.all(np.isfinite(s), axis=1)
    sp_end = np.hypot(s[:, 3], s[:, 4])
    rows = []
    for b, e in enumerate(eps):
        rows.append(dict(episode_id=e.id, **e.values,
                         max_abs_beta_deg=float(max_beta[b]), drift_time_s=float(drift_time[b]),
                         spun=bool(spun[b]), max_speed=float(max_speed[b]), final_speed=float(sp_end[b]),
                         distance=float(dist[b]), max_abs_yaw_rate_deg_s=float(max_r[b]),
                         finite=bool(finite[b]), latency_steps=int(delay[b]), mirrored=bool(mirrored[b])))
    if tick is not None:
        tick(1.0)
    return t, data, rows


def _run_shard(task: dict) -> dict:
    """Worker entry point: simulate one shard and write it. Picklable arguments only."""
    progress, cancel, key = task.get("progress"), task.get("cancel"), task["key"]
    n = len(task["ids"])
    t0 = time.perf_counter()

    def tick(frac: float) -> None:
        if progress is not None:
            progress[key] = frac * n

    def is_cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    try:
        t, data, rows = simulate_episodes(task["spec"], task["ids"], tick=tick, is_cancelled=is_cancelled)
    except Cancelled:
        return dict(key=key, status="cancelled")
    exp = task["spec"]["export"]
    root, ids = Path(task["out_dir"]), np.asarray(task["ids"])

    def write(folder: str, sel: Any) -> list[str]:
        sub = {k: v[sel] for k, v in data.items()}
        return [f"{folder}/{f}" for f in X.write_shard(root / folder, task["shard"], ids[sel], t, sub,
                                                       exp["formats"], exp["float32"], exp["compress"])]

    not_spun = np.array([not r["spun"] for r in rows])
    files = write(X.ALL_DIR, slice(None))
    if not_spun.any():                           # the same shard number, only the episodes that did not spin
        files += write(X.NOT_SPUN_DIR, not_spun)
    for r in rows:
        r["shard"] = task["shard"]
        r["group"] = task["group"]
    return dict(key=key, status="ok", shard=task["shard"], files=files, rows=rows, n=n,
                not_spun=int(not_spun.sum()),
                first_id=int(task["ids"][0]), last_id=int(task["ids"][-1]),
                seconds=time.perf_counter() - t0, T_rec=len(t))


# ----------------------------------------------------------------------------- planning
MIN_SHARD = 64     # below this many cars per shard the per-step Python overhead dominates


def _auto_workers(n_tasks_hint: int | None = None) -> int:
    w = max(1, (os.cpu_count() or 2) - 1)
    return w if n_tasks_hint is None else max(1, min(w, n_tasks_hint))


def plan(spec: dict, workers: int | None = None) -> tuple[dict, list[dict], list[str]]:
    """Validate and cut the spec into shard tasks. Returns (normalized spec, tasks, warnings).

    ``export.shard_episodes`` is the MAXIMUM shard size; when a group would give fewer shards than
    there are workers, shards are made smaller (but not below ``MIN_SHARD`` episodes) so every
    worker gets work.
    """
    norm, errors, warnings = validate_spec(spec)
    if errors:
        raise SpecError(errors)
    sampler = Sampler(norm)
    n = norm["episodes"]
    struct_keys = [k for k in norm["params"] if k in {"tire.combined_mode", "drivetrain.layout", "drivetrain.reverse_enabled",
                                                        "sim.dt", "sim.control_dt", "sim.integrator"}]
    if struct_keys:
        groups: dict[tuple, list[int]] = {}
        for i in range(n):
            groups.setdefault(sampler.structural(i), []).append(i)
    else:
        groups = {structural_signature(sampler.episode(0).params): list(range(n))}
    probs = []
    for sig in groups:
        ep = sampler.episode(groups[sig][0])
        try:
            ep.params.sim.n_substeps
        except ValueError as exc:
            probs.append(f"structural group {sig}: {exc}")
    if probs:
        raise SpecError(probs)
    est = estimate(norm)
    per_episode = est["records_per_episode"] * est["columns"] * 8
    max_size = max(1, min(norm["export"]["shard_episodes"], int(MAX_SHARD_BYTES // max(per_episode, 1))))
    w = workers if workers not in (None, 0) else (norm["run"]["workers"] or _auto_workers())
    tasks, shard = [], 0
    for gi, (sig, ids) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        size = max(1, min(max_size, max(MIN_SHARD, math.ceil(len(ids) / max(w, 1)))))
        for a in range(0, len(ids), size):
            tasks.append(dict(key=f"s{shard}", shard=shard, group=gi, ids=ids[a:a + size]))
            shard += 1
    return norm, tasks, warnings


def _record_dt(norm: dict) -> float:
    sim = Sampler(norm).episode(0).params.sim
    step = sim.dt if norm["export"].get("record_rate") == "physics" else sim.control_dt
    return float(norm["export"]["decimation"]) * float(step)


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=2)
        return out.stdout.strip() or None
    except Exception:
        return None


def default_out_root() -> Path:
    return Path(__file__).resolve().parents[2] / "exports"


def run_batch(spec: dict, out_root: str | Path | None = None,
              progress: Callable[[dict], None] | None = None,
              cancel: Any = None, workers: int | None = None) -> dict:
    """Run a batch and write it under ``out_root`` (default: <repo>/exports, or ``export.out_dir``).

    progress : called with dicts ``{done, total, fraction, episodes_per_s, eta_s, stage}``.
    cancel   : any object with ``is_set()`` (e.g. ``threading.Event``); setting it stops the run
               after the current control step of every shard (written shards are kept).
    workers  : overrides ``run.workers`` (0 = automatic, 1 = run in this process).
    Returns a summary dict (status, out_dir, episodes written, files, timing).
    """
    norm, tasks, warnings = plan(spec, workers)
    n = norm["episodes"]
    root = Path(out_root) if out_root is not None else None
    if root is None:
        od = Path(norm["export"].get("out_dir") or "exports")
        root = od if od.is_absolute() else Path(__file__).resolve().parents[2] / od
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = root / f"{stamp}_{norm['name']}"
    k = 1
    while out_dir.exists():
        out_dir = root / f"{stamp}_{norm['name']}_{k}"
        k += 1
    for folder in (X.ALL_DIR, X.NOT_SPUN_DIR):
        (out_dir / folder).mkdir(parents=True)
    (out_dir / "README.txt").write_text(X.README)
    t_start = time.time()
    manifest = dict(format_version=1, created=_dt.datetime.now().isoformat(timespec="seconds"),
                    rc_drift_sim_version=__version__, git_commit=_git_commit(), status="running",
                    spec=norm, warnings=warnings, episodes=n, signals=X.schema(norm["export"]["signals"]),
                    wheel_order=list(X.WHEELS), shards=[], groups=len({t["group"] for t in tasks}))
    X.write_json(out_dir / "manifest.json", manifest)

    w = norm["run"]["workers"] if workers is None else int(workers)
    w = _auto_workers(len(tasks)) if w <= 0 else min(w, len(tasks))
    rows: list[dict] = []
    shards: list[dict] = []
    status, error = "complete", None
    failed_shards: list[str] = []
    done_full = 0.0

    def report(partial: float, stage: str = "simulating") -> None:
        if progress is None:
            return
        done = done_full + partial
        el = max(time.time() - t_start, 1e-6)
        rate = done / el
        progress(dict(done=done, total=n, fraction=done / n, episodes_per_s=rate,
                      eta_s=(n - done) / rate if rate > 0 else None, stage=stage, out_dir=str(out_dir)))

    def collect(res: dict) -> None:
        nonlocal done_full
        if res["status"] == "ok":
            rows.extend(res["rows"])
            shards.append(dict(file=res["files"], shard=res["shard"], episodes=res["n"], not_spun=res["not_spun"],
                               first_id=res["first_id"], last_id=res["last_id"], seconds=round(res["seconds"], 3)))
            done_full += res["n"]

    def run_inline(todo: list[dict]) -> str:
        for task in todo:
            if cancel is not None and cancel.is_set():
                return "cancelled"
            res = _run_shard(dict(task, spec=norm, out_dir=str(out_dir), progress={}, cancel=cancel))
            if res["status"] == "cancelled":
                return "cancelled"
            collect(res)
            report(0.0)
        return "complete"

    try:
        if w == 1:
            status = run_inline(tasks)
        else:
            ctx = get_context("spawn")
            with ctx.Manager() as mgr:
                prog, stop = mgr.dict(), mgr.Event()
                with ProcessPoolExecutor(max_workers=w, mp_context=ctx) as pool:
                    futs = {pool.submit(_run_shard, dict(t, spec=norm, out_dir=str(out_dir), progress=prog,
                                                         cancel=stop)): t["key"] for t in tasks}
                    pending = set(futs)
                    while pending:
                        done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                        for f in done:
                            try:
                                res = f.result()
                            except BrokenProcessPool:
                                raise
                            except Exception as exc:     # a shard failed: keep going, report it
                                failed_shards.append(f"{futs[f]}: {exc!r}")
                                continue
                            prog.pop(res["key"], None)
                            if res["status"] == "cancelled":
                                status = "cancelled"
                            collect(res)
                        if cancel is not None and cancel.is_set() and not stop.is_set():
                            stop.set()
                            status = "cancelled"
                            for f in pending:
                                f.cancel()
                        try:
                            partial = float(sum(prog.values()))
                        except Exception:
                            partial = 0.0
                        report(partial)
    except BrokenProcessPool:
        # worker processes could not start (e.g. run_batch called from a script piped on stdin,
        # which the 'spawn' start method cannot re-import): finish the remaining shards here
        finished = {s["shard"] for s in shards}
        warnings.append("worker processes could not start; ran the batch in this process instead")
        w = 1
        status = run_inline([t for t in tasks if t["shard"] not in finished])
    except Exception as exc:  # keep what was written and record the failure
        status, error = "failed", "".join(traceback.format_exception_only(type(exc), exc)).strip()
        manifest["traceback"] = traceback.format_exc()
    if failed_shards and status == "complete":
        status, error = "failed", f"{len(failed_shards)} shard(s) failed: " + "; ".join(failed_shards[:5])
    elapsed = time.time() - t_start
    X.write_episodes_table(out_dir / X.ALL_DIR / "episodes.csv", rows)
    X.write_episodes_table(out_dir / X.NOT_SPUN_DIR / "episodes.csv", rows, keep=lambda r: not r["spun"])
    shards.sort(key=lambda s: s["shard"])
    written = sum(s["episodes"] for s in shards)
    stats = {}
    if rows:
        stats = dict(drift_fraction=float(np.mean([r["drift_time_s"] >= 1.0 for r in rows])),
                     spun_fraction=float(np.mean([r["spun"] for r in rows])),
                     not_spun_episodes=int(sum(not r["spun"] for r in rows)),
                     non_finite=int(sum(not r["finite"] for r in rows)),
                     mean_max_abs_beta_deg=float(np.nanmean([r["max_abs_beta_deg"] for r in rows])))
    manifest.update(status=status, error=error, shards=shards, episodes_written=written,
                    seconds=round(elapsed, 3), episodes_per_s=round(written / max(elapsed, 1e-9), 2),
                    workers=w, stats=stats, finished=_dt.datetime.now().isoformat(timespec="seconds"),
                    record_dt=_record_dt(norm))
    X.write_json(out_dir / "manifest.json", manifest)
    report(0.0, stage=status)
    paths = [p for p in out_dir.rglob("*") if p.is_file()]
    files = sorted(p.relative_to(out_dir).as_posix() for p in paths)
    size = sum(p.stat().st_size for p in paths)
    return dict(status=status, error=error, out_dir=str(out_dir), episodes=n, episodes_written=written,
                seconds=elapsed, workers=w, files=files, bytes=size, warnings=warnings, stats=stats)


def preview(spec: dict, n: int = 6) -> dict:
    """Simulate the first ``n`` episodes in this process (nothing is written) for the GUI preview."""
    norm, errors, warnings = validate_spec(spec)
    if errors:
        raise SpecError(errors)
    norm = dict(norm, episodes=min(norm["episodes"], n))
    norm["export"] = dict(norm["export"], decimation=1, record_rate="control")
    groups = ["pose", "velocity", "derived", "actions", "steering"]
    sampler = Sampler(norm)
    by_sig: dict[tuple, list[int]] = {}
    for i in range(norm["episodes"]):
        by_sig.setdefault(sampler.structural(i), []).append(i)
    out_eps: list[dict] = []
    t_arr = None
    for ids in by_sig.values():
        t, data, rows = simulate_episodes(norm, ids, groups=groups)
        t_arr = t
        for b, r in enumerate(rows):
            out_eps.append(dict(row=r, t=t, x=data["x"][b], y=data["y"][b], speed=data["speed"][b],
                                beta_deg=np.degrees(data["beta"][b]), yaw_rate_deg_s=np.degrees(data["yaw_rate"][b]),
                                steer_cmd=data["steer_cmd"][b], throttle_cmd=data["throttle_cmd"][b],
                                delta_deg=np.degrees(data["delta"][b])))
    out_eps.sort(key=lambda e: e["row"]["episode_id"])
    return dict(t=t_arr, episodes=out_eps, warnings=warnings)
