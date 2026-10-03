"""Training runs for the GUI's RL pages: one PPO run at a time in a background thread.

Each run lives in ``<runs root>/<timestamp>_<name>/``:

    config.json        the GUI JSON the run was started with (load it back into the editor)
    status.json        status, progress, reference scores, snapshot index (rewritten every iteration)
    log.jsonl          one line per PPO iteration (rl.ppo)
    policy.pt          the latest policy (rl.ppo.load_policy)
    snapshots/*.json   recorded episodes of the policy during training (rl.visual.record_rollout)

The physics, rewards and trainer are the library's (``rc_drift_sim.rl``); this module only schedules,
records and serves them. PyTorch and Gymnasium are imported lazily, so the dataset GUI keeps working
in builds without them (``available()`` says why RL is off).
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

RUN_NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def available() -> tuple[bool, str]:
    missing = [m for m in ("torch", "gymnasium") if importlib.util.find_spec(m) is None]
    if missing:
        return False, (f"{' and '.join(missing)} not installed: RL training needs PyTorch "
                       "(pip install -e \".[gpu]\", or the -WithTorch build of DriftSim.exe)")
    return True, ""


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


class RLRun:
    def __init__(self, body: dict, root: Path, n_iter: int, ppo: dict):
        self.id = uuid.uuid4().hex[:10]
        raw = str(body.get("name") or "run").strip() or "run"
        self.name = "".join(c if c in RUN_NAME_OK else "_" for c in raw)[:60]
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        d = root / f"{stamp}_{self.name}"
        k = 1
        while d.exists():
            d = root / f"{stamp}_{self.name}_{k}"
            k += 1
        self.dir = d
        self.body = body
        self.ppo = ppo
        self.status = "queued"
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.n_iter = n_iter
        self.history: list[dict] = []
        self.snapshots: list[dict] = []
        self.baselines: dict[str, float] = {}
        self.stage = "waiting"
        self.stop = threading.Event()

    @property
    def folder(self) -> str:
        return self.dir.name

    def summary(self) -> dict:
        last = self.history[-1] if self.history else {}
        best = max((r["episode_return"] for r in self.history if r.get("episode_return") is not None), default=None)
        elapsed = (self.finished or time.time()) - self.started if self.started else 0.0
        done = last.get("env_steps", 0)
        total = self.n_iter * self.ppo["num_envs"] * self.ppo["rollout"]
        eta = (total - done) / (done / elapsed) if done and self.status == "running" else None
        env = self.body.get("env", {})
        return dict(id=self.folder, run_id=self.id, name=self.name, status=self.status, error=self.error, stage=self.stage,
                    created=_dt.datetime.fromtimestamp(self.created).isoformat(timespec="seconds"),
                    task=env.get("task", "hold"), device=self.body.get("run", {}).get("device", "cpu"),
                    iteration=last.get("iteration", 0), n_iter=self.n_iter, env_steps=done, total_steps=total,
                    seconds=round(elapsed, 1), eta_s=eta, last_return=last.get("episode_return"), best_return=best,
                    baselines=self.baselines, n_snapshots=len(self.snapshots))

    def to_json(self, since: int = 0) -> dict:
        return dict(self.summary(), config=self.body, history=self.history[since:], n_history=len(self.history),
                    snapshots=self.snapshots)

    def save_status(self) -> None:
        _write_json(self.dir / "status.json", dict(self.summary(), snapshots=self.snapshots))


class RLManager:
    """Queue of training runs; the worker thread runs them one after another."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.runs: dict[str, RLRun] = {}                 # by folder name
        self.queue: list[RLRun] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._worker, name="driftsim-rl", daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------- API used by the server
    def start(self, body: dict) -> RLRun:
        from ..rl.catalog import build_configs
        ok, why = available()
        if not ok:
            raise ValueError(why)
        env_cfg, ppo, run_opts, errors = build_configs(body)
        if errors:
            raise ValueError("; ".join(errors))
        n_iter = max(1, ppo["total_steps"] // (ppo["num_envs"] * ppo["rollout"]))
        run = RLRun(body, self.root, n_iter, ppo)
        run.dir.mkdir(parents=True)
        (run.dir / "snapshots").mkdir()
        _write_json(run.dir / "config.json", body)
        run.save_status()
        with self.lock:
            self.runs[run.folder] = run
            self.queue.append(run)
        self.wake.set()
        return run

    def stop(self, folder: str) -> RLRun | None:
        run = self.runs.get(folder)
        if run is not None:
            run.stop.set()
            with self.lock:
                if run in self.queue:
                    self.queue.remove(run)
                    run.status, run.finished = "cancelled", time.time()
                    run.save_status()
        return run

    def stop_all(self) -> None:
        for run in list(self.runs.values()):
            self.stop(run.folder)

    def wait_idle(self, timeout: float = 15.0) -> bool:
        t_end = time.time() + timeout
        while time.time() < t_end:
            if not any(r.status in ("running", "queued") for r in self.runs.values()):
                return True
            time.sleep(0.1)
        return False

    def list(self) -> list[dict]:
        """Runs of this session and every run folder on disk, newest first."""
        out = {r.folder: r.summary() for r in self.runs.values()}
        if self.root.is_dir():
            for d in self.root.iterdir():
                if d.name in out or not (d / "status.json").is_file():
                    continue
                try:
                    st = json.loads((d / "status.json").read_text())
                except (OSError, ValueError):
                    continue
                st.pop("snapshots", None)
                if st.get("status") in ("running", "queued"):
                    st["status"] = "interrupted"
                out[d.name] = dict(st, id=d.name)
        return sorted(out.values(), key=lambda r: r["id"], reverse=True)

    def get(self, folder: str, since: int = 0) -> dict:
        run = self.runs.get(folder)
        if run is not None:
            return run.to_json(since)
        d = self._dir(folder)
        st = json.loads((d / "status.json").read_text())
        if st.get("status") in ("running", "queued"):
            st["status"] = "interrupted"
        hist = []
        if (d / "log.jsonl").is_file():
            hist = [json.loads(line) for line in (d / "log.jsonl").read_text().splitlines() if line.strip()]
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else {}
        return dict(st, id=folder, config=cfg, history=hist[since:], n_history=len(hist))

    def snapshot(self, folder: str, name: str) -> dict:
        f = (self._dir(folder) / "snapshots" / name).resolve()
        if f.parent != (self._dir(folder) / "snapshots").resolve() or not f.is_file() or f.suffix != ".json":
            raise FileNotFoundError(name)
        return json.loads(f.read_text())

    def rollout(self, folder: str, policy: str = "policy", seed: int = 0, init_drift: bool | None = None) -> dict:
        """A fresh episode of the run's latest policy (or of a reference policy) on the run's task."""
        from ..rl.ppo import load_policy
        from ..rl.visual import record_rollout
        d = self._dir(folder)
        if not (d / "policy.pt").is_file():
            raise FileNotFoundError("this run has no policy yet")
        model, cfg = load_policy(d / "policy.pt")
        pol = model.act if policy == "policy" else policy
        return record_rollout(cfg, pol, seed=seed, init_drift=init_drift)

    def file(self, folder: str, name: str) -> Path:
        d = self._dir(folder)
        f = (d / name).resolve()
        if f.parent != d.resolve() or not f.is_file():
            raise FileNotFoundError(name)
        return f

    def _dir(self, folder: str) -> Path:
        d = (self.root / folder).resolve()
        if d.parent != self.root.resolve() or not (d / "status.json").is_file():
            raise FileNotFoundError(folder)
        return d

    # ---------------------------------------------------------------- worker
    def _worker(self) -> None:
        while True:
            self.wake.wait()
            with self.lock:
                run = self.queue.pop(0) if self.queue else None
                if not self.queue:
                    self.wake.clear()
            if run is not None:
                self._run(run)

    def _run(self, run: RLRun) -> None:
        from ..rl import DriftBatchEnv, LQRDriftBaseline, evaluate, random_policy, zero_policy
        from ..rl.catalog import build_configs
        from ..rl.ppo import PPOConfig, train
        from ..rl.visual import record_rollout
        import numpy as np
        run.status, run.started = "running", time.time()
        try:
            env_cfg, ppo_kw, run_opts, _ = build_configs(run.body)
            run.stage = "reference scores"
            run.save_status()
            ref = DriftBatchEnv(16, env_cfg)
            rng = np.random.default_rng(0)
            refs = [("no input", zero_policy), ("random", lambda e: random_policy(e, rng))]
            if env_cfg.task == "hold":
                try:
                    refs.append(("LQR (true state)", LQRDriftBaseline(ref)))
                except RuntimeError:
                    pass                                      # no drift trim for this car
            for label, pol in refs:
                if run.stop.is_set():
                    break
                run.baselines[label] = round(evaluate(ref, pol, seed=12345)["mean_return"], 1)
            run.save_status()
            n_snap = int(run_opts["snapshots"])
            every = max(1, run.n_iter // n_snap) if n_snap else 0

            def on_iteration(row: dict, model: Any) -> None:
                run.history.append(row)
                it = row["iteration"]
                if every and (it == 1 or it % every == 0 or it == run.n_iter):
                    run.stage = "recording a snapshot"
                    data = record_rollout(env_cfg, model.act, seed=1000 + it, init_drift=False)
                    name = f"it{it:05d}.json"
                    _write_json(run.dir / "snapshots" / name, dict(data, iteration=it, env_steps=row["env_steps"]))
                    run.snapshots.append(dict(name=name, iteration=it, env_steps=row["env_steps"],
                                              episode_return=data["episode_return"], steps=data["steps"], ended=data["ended"]))
                run.stage = "training"
                run.save_status()

            run.stage = "training"
            ppo = PPOConfig(**ppo_kw)
            train(env_cfg, ppo, device=run_opts["device"], precision=run_opts["precision"], out=run.dir, log=None,
                  stop=run.stop.is_set, on_iteration=on_iteration)
            run.status = "cancelled" if run.stop.is_set() else "complete"
        except Exception as exc:                              # report the failure in the GUI
            run.status = "failed"
            run.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            (run.dir / "error.txt").write_text(traceback.format_exc())
        finally:
            run.finished = time.time()
            run.stage = run.status
            run.save_status()


__all__ = ["RLManager", "RLRun", "available"]
