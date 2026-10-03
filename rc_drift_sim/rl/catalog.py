"""Parameter catalog of the RL page of the GUI, and the conversion of its JSON into configs.

Every field of ``EnvConfig``, ``RewardWeights`` and ``PPOConfig`` (plus the run options) is listed
with a label, unit, limits and a plain description; defaults come from the dataclasses themselves.
``tests/test_rl_gui.py`` checks that no field is missing. The GUI sends::

    {"name": "...", "env": {EnvConfig fields}, "noise": {...}, "reward": {RewardWeights fields},
     "ppo": {PPOConfig fields}, "run": {"device", "precision", "snapshots"},
     "randomize": {datagen "params" dict} }

and ``build_configs`` returns ``(EnvConfig, PPOConfig, run options, errors)``.
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

from .config import OBS_MODES, TASKS, EnvConfig, RewardWeights

PPO_DEFAULTS = dict(total_steps=20_000_000, num_envs=1024, rollout=32, epochs=5, minibatches=8, gamma=0.99,
                    lam=0.95, clip=0.2, lr=3e-4, entropy=0.0, value_coef=0.5, max_grad_norm=1.0,
                    init_log_std=-0.5, hidden=(256, 256), seed=0)        # = ppo.PPOConfig() (torch-free import)


def _f(key, label, unit="", typ="float", lo=None, hi=None, desc="", tasks=None, choices=None, level="basic"):
    return dict(key=key, label=label, unit=unit, type=typ, min=lo, max=hi, desc=desc, tasks=tasks,
                choices=choices, level=level)


ENV_FIELDS = [
    _f("task", "Task", typ="choice", choices=list(TASKS),
       desc="hold: keep a steady drift anywhere (a donut). track: drive round a circle, with a bonus for drifting."),
    _f("episode_s", "Episode length", "s", lo=0.2, hi=120, desc="time limit of one episode (truncation)"),
    _f("target_beta_deg", "Target sideslip", "deg", lo=1, hi=85, tasks=["hold"],
       desc="|sideslip| the drift reward is centred on (either direction)"),
    _f("target_speed", "Target speed", "m/s", lo=0.1, hi=15,
       desc="hold: ground speed; track: speed along the line"),
    _f("track_radius", "Circle radius", "m", lo=0.2, hi=50, tasks=["track"], desc="radius of the line to follow"),
    _f("track_both_ways", "Both directions", typ="bool", tasks=["track"],
       desc="half of the episodes go clockwise, half counter-clockwise"),
    _f("max_track_error", "Max distance from line", "m", lo=0.05, hi=20, tasks=["track"],
       desc="the episode ends (penalty) if the car strays further"),
    _f("init_speed", "Start speed", "m/s", lo=0, hi=15, desc="straight-line start (0 = parked)"),
    _f("init_drift_prob", "Drift starts", "share", lo=0, hi=1,
       desc="share of episodes that start in a steady drift (curriculum)"),
    _f("spin_beta_deg", "Spin-out threshold", "deg", lo=10, hi=180, desc="|sideslip| above which a moving car has spun"),
    _f("obs", "Observations", typ="choice", choices=list(OBS_MODES),
       desc="sensors: what the real car measures (with noise); full: the simulator state"),
    _f("seed", "Seed", typ="int", lo=0, desc="the episode sequence (cars, starts) is a function of the seed"),
]
NOISE_FIELDS = [
    _f("gyro", "Gyro noise", "rad/s", lo=0, desc="standard deviation of the yaw-rate reading"),
    _f("accel", "Accelerometer noise", "m/s²", lo=0, desc="standard deviation of the IMU specific force"),
    _f("wheel", "Wheel-speed noise", "m/s", lo=0, desc="standard deviation of the wheel surface speeds"),
    _f("velocity", "Velocity noise", "m/s", lo=0, desc="standard deviation of the body-velocity estimate"),
]
REWARD_FIELDS = [
    _f("beta", "Sideslip weight", lo=0, tasks=["hold"], desc="reward for |sideslip| near the target while rotating into the slide"),
    _f("speed", "Speed weight", lo=0, desc="reward for speed near the target"),
    _f("track", "Line weight", lo=0, tasks=["track"], desc="reward for staying on the line (times progress along it)"),
    _f("drift", "Drift bonus", lo=0, tasks=["track"], desc="bonus while drifting into the turn"),
    _f("action_rate", "Smoothness penalty", lo=0,
       desc="penalty on the squared change of the commands per step (raise it against chattering)"),
    _f("spin", "Early-end penalty", lo=0, desc="one-off penalty for a spin-out, leaving the line or a numerical failure"),
    _f("beta_sigma_deg", "Sideslip tolerance", "deg", lo=0.1, tasks=["hold"], desc="width of the sideslip reward"),
    _f("speed_sigma", "Speed tolerance", "m/s", lo=0.01, desc="width of the speed reward"),
    _f("track_sigma", "Line tolerance", "m", lo=0.01, tasks=["track"], desc="width of the line reward"),
    _f("drift_min_deg", "Drift bonus from", "deg", lo=0, hi=80, tasks=["track"], desc="|sideslip| where the drift bonus starts"),
]
PPO_FIELDS = [
    _f("total_steps", "Training steps", "steps", "int", lo=1000, desc="environment steps in total (cars x control steps)"),
    _f("num_envs", "Cars in parallel", "cars", "int", lo=1, hi=262144, desc="cars simulated together"),
    _f("rollout", "Rollout length", "steps", "int", lo=4, hi=4096, desc="control steps per car per iteration"),
    _f("epochs", "Epochs", "", "int", lo=1, hi=100, desc="passes over each batch of experience", level="advanced"),
    _f("minibatches", "Minibatches", "", "int", lo=1, hi=1024, desc="minibatches per epoch", level="advanced"),
    _f("lr", "Learning rate", "", lo=1e-7, hi=1.0, desc="Adam step size"),
    _f("gamma", "Discount", "", lo=0.5, hi=0.99999, desc="how far ahead the agent looks (0.99 ~ 100 steps = 2 s)"),
    _f("lam", "GAE lambda", "", lo=0, hi=1, desc="bias / variance trade-off of the advantage estimate", level="advanced"),
    _f("clip", "Clip range", "", lo=0.01, hi=1, desc="PPO ratio clipping", level="advanced"),
    _f("entropy", "Entropy bonus", "", lo=0, desc="encourages exploration", level="advanced"),
    _f("value_coef", "Value-loss weight", "", lo=0, level="advanced", desc="weight of the critic loss"),
    _f("max_grad_norm", "Gradient clip", "", lo=0.01, level="advanced", desc="maximum gradient norm"),
    _f("init_log_std", "Initial log std", "", lo=-5, hi=2, desc="initial exploration noise of the actions (log)"),
    _f("hidden", "Hidden layers", "", "intlist", desc="neurons per hidden layer, e.g. 256, 256"),
    _f("seed", "Training seed", "", "int", lo=0, desc="network initialization and action sampling"),
]
RUN_FIELDS = [
    _f("device", "Compute device", typ="choice", choices=["cpu", "mps", "cuda", "auto"],
       desc="cpu: NumPy physics + PyTorch policy on the CPU; mps / cuda: everything on the GPU"),
    _f("precision", "GPU precision", typ="choice", choices=["float32", "float64"], desc="Apple GPUs only do float32"),
    _f("snapshots", "Snapshots", "", "int", lo=0, hi=500,
       desc="policy rollouts recorded during training for the viewer (spread evenly; 0 = none)"),
]
RUN_DEFAULTS = dict(device="cpu", precision="float32", snapshots=20)


def catalog() -> dict:
    env, rw = EnvConfig(), RewardWeights()
    d = dict(env=_with_defaults(ENV_FIELDS, dataclasses.asdict(env)), noise=_with_defaults(NOISE_FIELDS, env.noise),
             reward=_with_defaults(REWARD_FIELDS, dataclasses.asdict(rw)), ppo=_with_defaults(PPO_FIELDS, PPO_DEFAULTS),
             run=_with_defaults(RUN_FIELDS, RUN_DEFAULTS))
    d["defaults"] = {k: {f["key"]: f["default"] for f in v} for k, v in d.items()}
    return d


def _with_defaults(fields: list[dict], defaults: dict) -> list[dict]:
    out = []
    for f in fields:
        dflt = defaults[f["key"]]
        out.append(dict(f, default=list(dflt) if isinstance(dflt, tuple) else dflt))
    return out


def _check(f: dict, v: Any, where: str, errors: list[str]) -> Any:
    name = f"{where}.{f['key']}"
    t = f["type"]
    if t == "choice":
        if v not in f["choices"]:
            errors.append(f"{name}: must be one of {f['choices']}")
        return v
    if t == "bool":
        if not isinstance(v, bool):
            errors.append(f"{name}: must be true or false")
        return bool(v)
    if t == "intlist":
        if isinstance(v, str):
            v = [x for x in re.split(r"[,\s]+", v.strip()) if x]
        try:
            vals = [int(x) for x in v]
        except (TypeError, ValueError):
            errors.append(f"{name}: a list of whole numbers, e.g. 256, 256")
            return v
        if not vals or any(x < 1 or x > 8192 for x in vals):
            errors.append(f"{name}: 1 to 8192 neurons per layer, at least one layer")
        return vals
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        errors.append(f"{name}: must be a number")
        return v
    if t == "int":
        if float(v) != int(v):
            errors.append(f"{name}: must be a whole number")
        v = int(v)
    else:
        v = float(v)
    if f["min"] is not None and v < f["min"]:
        errors.append(f"{name}: must be >= {f['min']}")
    if f["max"] is not None and v > f["max"]:
        errors.append(f"{name}: must be <= {f['max']}")
    return v


def build_configs(body: dict) -> tuple[EnvConfig | None, dict | None, dict, list[str]]:
    """GUI JSON -> (EnvConfig, PPO keyword dict, run options, errors). Missing fields take defaults."""
    errors: list[str] = []
    cat = catalog()
    parts: dict[str, dict] = {}
    for part in ("env", "noise", "reward", "ppo", "run"):
        given = body.get(part) or {}
        if not isinstance(given, dict):
            errors.append(f"{part} must be an object")
            given = {}
        unknown = set(given) - {f["key"] for f in cat[part]}
        if unknown:
            errors.append(f"{part}: unknown fields {sorted(unknown)}")
        parts[part] = {f["key"]: _check(f, given.get(f["key"], f["default"]), part, errors) for f in cat[part]}
    rnd = body.get("randomize") or {}
    if not isinstance(rnd, dict):
        errors.append("randomize must be an object")
        rnd = {}
    ppo = parts["ppo"]
    if not errors and ppo["num_envs"] * ppo["rollout"] < ppo["minibatches"]:
        errors.append("ppo: cars x rollout length must be at least the number of minibatches")
    if not errors and ppo["total_steps"] < ppo["num_envs"] * ppo["rollout"]:
        errors.append("ppo: training steps must cover at least one iteration (cars x rollout length)")
    if errors:
        return None, None, parts["run"], errors
    env = EnvConfig(**parts["env"], noise=parts["noise"], reward=RewardWeights(**parts["reward"]), randomize=rnd)
    try:
        env.validate()
    except ValueError as exc:
        errors.append(str(exc))
    if rnd and not errors:
        from ..datagen.spec import default_spec, validate_spec
        spec = default_spec()
        spec["params"] = rnd
        errors += [f"randomize: {e}" for e in validate_spec(spec)[1]]
    run = parts["run"]
    if run["device"] != "cpu" and not errors:
        from ..sim.xp import resolve_device
        try:
            dev = resolve_device(run["device"])
            if dev == "mps" and run["precision"] == "float64":
                errors.append("run.precision: Apple GPUs (mps) only support float32")
        except ValueError as exc:
            errors.append(f"run.device: {exc}")
    ppo = dict(ppo, hidden=tuple(ppo["hidden"]))
    return (None if errors else env), (None if errors else ppo), run, errors


__all__ = ["catalog", "build_configs", "ENV_FIELDS", "NOISE_FIELDS", "REWARD_FIELDS", "PPO_FIELDS", "RUN_FIELDS",
           "PPO_DEFAULTS"]
