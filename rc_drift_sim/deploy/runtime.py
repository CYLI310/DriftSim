"""The exported drift policy as a NumPy runtime: readings in, filtered commands out, every control step.

    rt = PolicyRuntime("rl_runs/<run>/final")
    rt.set_targets(beta_deg=30, speed=1.5)          # hold task: the commanded slide
    rt.reset()                                      # before each run (fills the sensor history)
    cmd = rt.step(dict(gyro_z=.., accel_x=.., accel_y=.., wheel_rear=.., vel_x=.., vel_y=..))
    send(cmd.steer, cmd.throttle)

It does exactly what the training environment does: the same feature scaling and history, the same
network (``policy.npz``), the grip estimate and the same safety filter (``deploy.safety``). Only
NumPy is needed. Readings are SI: gyro_z rad/s (counter-clockwise +), accel_x / accel_y m/s^2 body
specific force (forward / left +), vel_x / vel_y m/s body velocity (forward / left +), wheel speeds m/s
at the tire surface (``wheel_rear`` fills both rear wheels; ``wheel_front`` both fronts).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .safety import SafetyConfig, safety_filter

GRIP_FALLBACK = (0.3, 0.15)


def unpack_model(zip_path: str | Path) -> Path:
    """Extract an exported final_model.zip next to itself (``<name>/final``) and return that folder."""
    import zipfile
    zip_path = Path(zip_path)
    dest = zip_path.with_suffix("")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():                     # only the export's own flat files
            if not name.startswith("final/") or ".." in name or name.endswith("/"):
                raise ValueError(f"{zip_path} is not a DriftSim final_model.zip ({name})")
        z.extractall(dest)
    return dest / "final"


@dataclass
class Command:
    steer: float                  # filtered command sent to the car, [-1, 1]
    throttle: float
    raw_steer: float              # what the policy asked for
    raw_throttle: float
    grip: float | None            # estimated friction coefficient (None: no estimator)
    override: float               # 0..1, how much the safety envelope took over
    compute_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class PolicyRuntime:
    def __init__(self, folder: str | Path, car: dict | None = None, safety: dict | None = None):
        """``folder``: the exported ``final`` folder (or its policy.json). ``car``: overrides of the
        geometry the safety filter uses (``steer_max_deg``, ``cg_to_front``) for the real car;
        ``safety``: overrides of the safety-filter settings (e.g. ``{"v_max": 2.0}``)."""
        folder = Path(folder)
        if folder.suffix == ".json":
            folder = folder.parent
        if folder.suffix == ".zip":                   # final_model.zip copied from the training machine
            folder = unpack_model(folder)
        self.folder = folder
        self.meta = json.loads((folder / "policy.json").read_text())
        if not str(self.meta.get("format", "")).startswith("driftsim-policy/"):
            raise ValueError(f"{folder} is not a DriftSim policy export")
        with np.load(folder / "policy.npz") as z:
            self.w = {k: z[k] for k in z.files}
        obs = self.meta["observation"]
        self.H = int(obs["history"])
        self.dyn = [(f["name"], float(f["scale"])) for f in obs["dynamic"]]
        self.task = [(f["name"], float(f["scale"])) for f in obs["task"]]
        self.n_obs = int(self.meta["n_obs"])
        if self.H * len(self.dyn) + len(self.task) != self.n_obs:
            raise ValueError("policy.json observation layout does not match n_obs")
        self.control_dt = float(self.meta["control_dt"])
        car_cfg = dict(self.meta["car"], **(car or {}))
        self.steer_max = math.radians(float(car_cfg["steer_max_deg"]))
        self.lf = float(car_cfg["cg_to_front"])
        self.safety = SafetyConfig(**{**self.meta["safety"], **(safety or {})})
        net = self.meta.get("network", {})
        self.grip_mean, self.grip_std = net.get("grip_mean", GRIP_FALLBACK[0]), net.get("grip_std", GRIP_FALLBACK[1])
        self.has_grip = "est0_w" in self.w
        t = self.meta["targets"]
        self.beta_range = [math.radians(x) for x in t["beta_range_deg"]]
        self.speed_range = list(t["speed_range"])
        self.target_beta = math.radians(t["beta_deg"])
        self.target_speed = float(t["speed"])
        self.track: dict[str, float] = {}
        self.reset()

    # ---------------------------------------------------------------- set-up
    def reset(self) -> None:
        """Start a new run: empty history (filled with the first reading), commands at zero."""
        self.hist: np.ndarray | None = None
        self.prev = np.zeros(2)
        self.steps = 0

    def set_targets(self, beta_deg: float | None = None, speed: float | None = None) -> None:
        """Hold task: commanded |sideslip| (deg) and speed (m/s), clipped to the range the policy trained on."""
        if beta_deg is not None:
            b = math.radians(abs(float(beta_deg)))
            self.target_beta = min(max(b, self.beta_range[0]), self.beta_range[1])
        if speed is not None:
            self.target_speed = min(max(float(speed), self.speed_range[0]), self.speed_range[1])

    def set_track(self, track_error: float, heading_err: float, direction: float) -> None:
        """Track task: distance to the circle (m, + = left), body heading minus the tangent (rad), +1 ccw / -1 cw."""
        self.track = dict(track_error=track_error, heading_err=heading_err, direction=direction)

    # ---------------------------------------------------------------- one control step
    def _reading(self, name: str, sensors: dict) -> float:
        if name == "prev_steer":
            return float(self.prev[0])
        if name == "prev_throttle":
            return float(self.prev[1])
        if name in sensors:
            return float(sensors[name])
        alias = {"wheel_rl": "wheel_rear", "wheel_rr": "wheel_rear", "wheel_fl": "wheel_front", "wheel_fr": "wheel_front"}
        if name in alias and alias[name] in sensors:
            return float(sensors[alias[name]])
        raise KeyError(f"missing reading {name!r}; this policy needs {self.required_readings()}")

    def required_readings(self) -> list[str]:
        return [n for n, _ in self.dyn if not n.startswith("prev_")]

    def observation(self, sensors: dict) -> np.ndarray:
        """Advance the history with this reading and return the observation vector (n_obs,)."""
        dyn = np.array([self._reading(n, sensors) * s for n, s in self.dyn], dtype=np.float32)
        if self.hist is None:
            self.hist = np.tile(dyn, (self.H, 1))
        else:
            self.hist = np.concatenate([self.hist[1:], dyn[None]], axis=0)
        task = []
        for n, s in self.task:
            if n == "target_beta":
                v = self.target_beta
            elif n == "target_speed":
                v = self.target_speed
            elif n == "track_error":
                v = min(max(self.track["track_error"], -2.0), 2.0)
            elif n == "heading_err_sin":
                v = math.sin(self.track["heading_err"])
            elif n == "heading_err_cos":
                v = math.cos(self.track["heading_err"])
            elif n == "direction":
                v = self.track["direction"]
            else:
                raise KeyError(n)
            task.append(v * s)
        return np.concatenate([self.hist.reshape(-1), np.asarray(task, dtype=np.float32)])

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, float | None]:
        """The network: raw observation (n_obs,) -> (action (2,), grip estimate or None)."""
        w = self.w
        x = np.clip((obs - w["obs_mean"]) * w["obs_inv_std"], -w["obs_clip"], w["obs_clip"])[None]

        def mlp(prefix: str, h: np.ndarray) -> np.ndarray:
            k = 0
            while f"{prefix}{k}_w" in w:
                h = h @ w[f"{prefix}{k}_w"].T + w[f"{prefix}{k}_b"]
                if f"{prefix}{k + 1}_w" in w:
                    h = np.tanh(h)
                k += 1
            return h

        if self.has_grip:
            g = self.grip_mean + self.grip_std * mlp("est", x)
            gi = (np.clip(g, 0.0, 1.5) - self.grip_mean) / self.grip_std
            a = mlp("actor", np.concatenate([x, gi], axis=1))
            return np.clip(a[0], -1.0, 1.0), float(g[0, 0])
        return np.clip(mlp("actor", x)[0], -1.0, 1.0), None

    def step(self, sensors: dict) -> Command:
        t0 = time.perf_counter()
        obs = self.observation(sensors)
        a, grip = self.forward(obs)
        if self.safety.enabled:
            arr = lambda v: np.array([float(v)])  # noqa: E731
            out, ov = safety_filter(np, a[None].astype(np.float64), self.prev[None], arr(sensors["vel_x"]),
                                    arr(sensors["vel_y"]), arr(sensors["gyro_z"]), None if grip is None else arr(grip),
                                    self.safety, self.steer_max, self.lf)
            cmd, override = out[0], float(ov[0])
        else:
            cmd, override = np.asarray(a, dtype=np.float64), 0.0
        self.prev = np.asarray(cmd, dtype=np.float64)
        self.steps += 1
        return Command(steer=float(cmd[0]), throttle=float(cmd[1]), raw_steer=float(a[0]), raw_throttle=float(a[1]),
                       grip=grip, override=override, compute_ms=(time.perf_counter() - t0) * 1000.0)

    def describe(self) -> str:
        ev = self.meta.get("evaluation") or {}
        return (f"{self.folder}: task {self.meta['task']}, {self.H} x {len(self.dyn)} readings every "
                f"{self.control_dt * 1000:.0f} ms ({', '.join(self.required_readings())}), grip estimator "
                f"{'yes' if self.has_grip else 'no'}, safety filter {'on' if self.safety.enabled else 'off'}, "
                f"grip-sweep score {ev.get('score', '-')}")


__all__ = ["PolicyRuntime", "Command", "unpack_model"]
