"""Export a trained policy as the final model for the car (Jetson Orin Nano), BeamNG and other tools.

    driftsim-export rl_runs/<run>                 # best.pt if the run has one, else policy.pt
    driftsim-export runs/final --which last

writes ``<run>/final/`` and ``<run>/final_model.zip``:

    policy.json     what the model expects and does: observation layout (signals, scales, history), task
                    targets and their trained range, control period, latency, safety-filter settings,
                    car geometry the filter uses, network layout, evaluation results, parity checks
    policy.npz      the weights incl. the observation normalization (NumPy; deploy.runtime runs them)
    policy.onnx     the same network as an ONNX graph (raw observation -> action, grip) for TensorRT /
                    onnxruntime; skipped with a note if the onnx package is missing
    policy_ts.pt    TorchScript of the same graph (torch.jit.load, no DriftSim code needed)
    README.txt      how to run it

All formats are checked against the PyTorch policy on recorded observations (max difference stored).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import EnvConfig
from .env import DriftBatchEnv, nominal_params
from .ppo import GRIP_MEAN, GRIP_STD, ActorCritic, load_checkpoint

FORMAT = "driftsim-policy/1"


class DeployPolicy(nn.Module):
    """Raw observation (N, n_obs) -> (action mean clipped to [-1, 1] (N, 2), grip estimate (N, 1));
    the grip is -1 when the model has no estimator."""

    def __init__(self, model: ActorCritic):
        super().__init__()
        self.mean = nn.Parameter(model.norm.mean.detach().clone(), requires_grad=False)
        self.inv_std = nn.Parameter(1.0 / torch.sqrt(model.norm.var.detach() + 1e-8), requires_grad=False)
        self.clip = float(model.norm.clip)
        self.actor = model.actor
        self.est = model.est
        self.has_est = model.est is not None

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        on = torch.clamp((obs - self.mean) * self.inv_std, -self.clip, self.clip)
        if self.est is not None:
            g = GRIP_MEAN + GRIP_STD * self.est(on)
            gi = (torch.clamp(g, 0.0, 1.5) - GRIP_MEAN) / GRIP_STD
            a = self.actor(torch.cat([on, gi], dim=1))
        else:
            g = torch.full((obs.shape[0], 1), -1.0, dtype=obs.dtype, device=obs.device)
            a = self.actor(on)
        return torch.clamp(a, -1.0, 1.0), g


def _linears(seq: nn.Sequential | None) -> list[tuple[np.ndarray, np.ndarray]]:
    if seq is None:
        return []
    return [(m.weight.detach().cpu().numpy().astype(np.float32), m.bias.detach().cpu().numpy().astype(np.float32))
            for m in seq if isinstance(m, nn.Linear)]


def nominal_car(cfg: EnvConfig) -> dict:
    """Geometry of the nominal training car (fixed randomize values applied, random ones at their default)."""
    p = nominal_params(cfg)
    return dict(steer_max_deg=math.degrees(float(p.actuators.steer_max)), cg_to_front=float(p.vehicle.cg_to_front),
                wheelbase=float(p.vehicle.wheelbase), track_width=float(p.vehicle.track_width),
                body_length=float(p.vehicle.body_length), mass=float(p.vehicle.mass),
                wheel_radius=float(p.vehicle.wheel_radius), latency=float(p.actuators.latency))


def _sample_obs(cfg: EnvConfig, model: ActorCritic, n_steps: int = 60, n_cars: int = 16) -> np.ndarray:
    """Realistic observations: a short rollout of the policy itself (plus the parked starts)."""
    env = DriftBatchEnv(n_cars, cfg)
    obs = [np.asarray(env.last_obs)]
    pol = model.policy_function()
    for _ in range(n_steps):
        out = pol(env.last_obs)
        a, g = out if isinstance(out, tuple) else (out, None)
        env.step(a.numpy().astype(np.float64), grip=None if g is None else g.numpy().astype(np.float64))
        obs.append(np.asarray(env.last_obs))
    return np.concatenate(obs).astype(np.float32)


def numpy_forward(weights: dict[str, np.ndarray], obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The exported network in plain NumPy (identical to ``deploy.runtime``): obs (N, n) -> (actions, grip)."""
    x = np.clip((obs - weights["obs_mean"]) * weights["obs_inv_std"], -weights["obs_clip"], weights["obs_clip"])

    def mlp(prefix: str, h: np.ndarray) -> np.ndarray:
        k = 0
        while f"{prefix}{k}_w" in weights:
            h = h @ weights[f"{prefix}{k}_w"].T + weights[f"{prefix}{k}_b"]
            if f"{prefix}{k + 1}_w" in weights:
                h = np.tanh(h)
            k += 1
        return h

    if "est0_w" in weights:
        g = GRIP_MEAN + GRIP_STD * mlp("est", x)
        gi = (np.clip(g, 0.0, 1.5) - GRIP_MEAN) / GRIP_STD
        a = mlp("actor", np.concatenate([x, gi], axis=1))
    else:
        g = np.full((obs.shape[0], 1), -1.0, dtype=obs.dtype)
        a = mlp("actor", x)
    return np.clip(a, -1.0, 1.0), g


def export_final(run: str | Path, which: str = "auto", out: str | Path | None = None, onnx: bool = True) -> dict:
    """Export ``run``/best.pt (``which="best"``), policy.pt (``"last"``) or best-if-present (``"auto"``);
    ``run`` may also be a .pt file. Returns a summary (folder, zip, files, checks)."""
    run = Path(run)
    if run.suffix == ".pt":
        ckpt, run_dir = run, run.parent
    else:
        run_dir = run
        best, last = run / "best.pt", run / "policy.pt"
        ckpt = best if which == "best" or (which == "auto" and best.is_file()) else last
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found (train first)")
    model, cfg, ppo, ck = load_checkpoint(ckpt)
    if cfg.obs != "sensors":
        raise ValueError("this policy reads the simulator state (obs='full'); only sensor policies can run on a car")
    folder = Path(out) if out is not None else run_dir / "final"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    env = DriftBatchEnv(1, cfg)
    spec = ck.get("obs_spec") or env.obs_spec()
    dep = DeployPolicy(model).eval()

    weights: dict[str, np.ndarray] = dict(obs_mean=model.norm.mean.numpy().astype(np.float32),
                                          obs_inv_std=dep.inv_std.detach().numpy().astype(np.float32),
                                          obs_clip=np.float32(model.norm.clip))
    for prefix, seq in (("actor", model.actor), ("est", model.est)):
        for k, (w, b) in enumerate(_linears(seq)):
            weights[f"{prefix}{k}_w"], weights[f"{prefix}{k}_b"] = w, b
    np.savez(folder / "policy.npz", **weights)

    obs = _sample_obs(cfg, model)
    with torch.no_grad():
        a_ref, g_ref = model.act_with_grip(obs)
        a_dep, g_dep = dep(torch.as_tensor(obs))
    a_ref = a_ref.numpy()
    a_np, g_np = numpy_forward(dict(np.load(folder / "policy.npz")), obs)
    checks: dict[str, Any] = dict(samples=int(len(obs)), torch_graph=float(np.abs(a_dep.numpy() - a_ref).max()),
                                  numpy=float(np.abs(a_np - a_ref).max()))
    if g_ref is not None:
        checks["numpy_grip"] = float(np.abs(g_np[:, 0] - g_ref.numpy()).max())

    with warnings.catch_warnings():               # TorchScript is deprecated in new torch, still the most portable
        warnings.simplefilter("ignore", FutureWarning)
        ts = torch.jit.trace(dep, torch.as_tensor(obs[:1]))
        ts.save(str(folder / "policy_ts.pt"))
        a_ts, _ = torch.jit.load(str(folder / "policy_ts.pt"))(torch.as_tensor(obs))
    checks["torchscript"] = float(np.abs(a_ts.detach().numpy() - a_ref).max())
    files = ["policy.json", "policy.npz", "policy_ts.pt"]
    if onnx:
        note = _export_onnx(dep, obs, folder / "policy.onnx", a_ref, checks)
        if note is None:
            files.append("policy.onnx")
        else:
            checks["onnx_note"] = note

    ok = all(v < 1e-4 for k, v in checks.items() if isinstance(v, float))
    checks["ok"] = bool(ok)
    car = nominal_car(cfg)
    meta = dict(
        format=FORMAT, created=_dt.datetime.now().isoformat(timespec="seconds"), source=str(ckpt),
        iteration=ck.get("iteration"), evaluation=ck.get("evaluation"), task=cfg.task,
        control_dt=float(ck.get("control_dt") or env.control_dt), observation=spec, obs_names=ck["obs_names"],
        n_obs=len(ck["obs_names"]),
        actions=dict(names=["steer", "throttle"], range=[-1.0, 1.0],
                     meaning="steer +1 = full left lock (steer_max_deg), throttle +1 = full forward, negative = brake/reverse"),
        grip=dict(estimator=model.est is not None, meaning="estimated peak friction coefficient; -1 = no estimator"),
        targets=dict(beta_deg=cfg.target_beta_deg, beta_range_deg=[cfg.target_beta_deg - cfg.target_beta_jitter_deg,
                                                                   cfg.target_beta_deg + cfg.target_beta_jitter_deg],
                     speed=cfg.target_speed, speed_range=[cfg.target_speed - cfg.target_speed_jitter,
                                                          cfg.target_speed + cfg.target_speed_jitter],
                     sign="a left-hand drift has negative sideslip; the policy holds either direction"),
        safety=dict(cfg.safety.__dict__), car=car, front_wheel_speeds=cfg.front_wheel_speeds,
        network=dict(actor=[list(w.shape) for w, _ in _linears(model.actor)],
                     estimator=[list(w.shape) for w, _ in _linears(model.est)], activation="tanh",
                     grip_mean=GRIP_MEAN, grip_std=GRIP_STD),
        training=dict(env=cfg.to_dict(), ppo={k: (list(v) if isinstance(v, tuple) else v) for k, v in ppo.__dict__.items()}),
        checks=checks, files=files,
    )
    (folder / "policy.json").write_text(json.dumps(meta, indent=1))
    (folder / "README.txt").write_text(_readme(meta))
    files.append("README.txt")
    zpath = run_dir / "final_model.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(folder / f, f"final/{f}")
    return dict(folder=str(folder), zip=str(zpath), files=files, checks=checks, iteration=ck.get("iteration"),
                evaluation=ck.get("evaluation"), source=ckpt.name)


def _export_onnx(dep: DeployPolicy, obs: np.ndarray, path: Path, a_ref: np.ndarray, checks: dict) -> str | None:
    """Write the ONNX graph; returns None on success or a note on why it was skipped."""
    import importlib.util
    if importlib.util.find_spec("onnx") is None:
        return "onnx not installed (pip install onnx onnxscript): ONNX skipped"
    x = torch.as_tensor(obs[:1])
    err = None
    for dynamo in (False, True):
        try:
            torch.onnx.export(dep, (x,), str(path), input_names=["obs"], output_names=["action", "grip"],
                              dynamic_axes={"obs": {0: "n"}, "action": {0: "n"}, "grip": {0: "n"}},
                              opset_version=17, dynamo=dynamo)
            break
        except Exception as exc:                       # exporter differences between torch versions
            err = f"{type(exc).__name__}: {exc}"
    else:
        return f"ONNX export failed ({err})"
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        a_ort = sess.run(["action"], {"obs": obs})[0]
        checks["onnxruntime"] = float(np.abs(a_ort - a_ref).max())
    return None


def _readme(meta: dict) -> str:
    obs = meta["observation"]
    dyn = ", ".join(f"{f['name']} x {f['scale']:.4g}" for f in obs["dynamic"])
    task = ", ".join(f"{f['name']} x {f['scale']:.4g}" for f in obs["task"])
    ev = meta.get("evaluation") or {}
    return f"""DriftSim final model ({meta['format']}), exported {meta['created']} from {meta['source']}
(iteration {meta['iteration']}; grip-sweep score {ev.get('score', 'not evaluated')}).

Run it on the car (Jetson) or against a simulator with rc_drift_sim.deploy:

    driftsim-drive --model final --car sim                 # software-in-the-loop check in DriftSim
    driftsim-drive --model final --car beamng --beamng-home "C:/BeamNG.tech.v0.39"
    driftsim-drive --model final --car mypkg.mycar:MyCar   # your hardware driver

Without DriftSim: policy_ts.pt (torch.jit.load) or policy.onnx map obs (1, {meta['n_obs']}) -> action (1, 2),
grip (1, 1). Build obs every {meta['control_dt'] * 1000:.0f} ms from the readings in SI units:
  per step (stacked {obs['history']} times, oldest first): {dyn}
  then: {task}
prev_steer / prev_throttle are the commands sent last step (after the safety filter). Apply the safety
filter in policy.json ("safety"; rc_drift_sim/deploy/safety.py) to the action before sending it.
Steer +1 = full left lock ({meta['car']['steer_max_deg']:.0f} deg in training), throttle +1 = full forward.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="driftsim-export", description="Export a trained drift policy as the final model.")
    ap.add_argument("run", help="run folder (with best.pt / policy.pt) or a .pt file")
    ap.add_argument("--which", default="auto", choices=["auto", "best", "last"])
    ap.add_argument("--out", default=None, help="output folder (default: <run>/final)")
    ap.add_argument("--no-onnx", action="store_true")
    a = ap.parse_args(argv)
    res = export_final(a.run, a.which, a.out, onnx=not a.no_onnx)
    print(json.dumps({k: res[k] for k in ("folder", "zip", "files", "checks", "iteration")}, indent=1))
    return 0 if res["checks"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export_final", "DeployPolicy", "numpy_forward", "nominal_car", "FORMAT"]
