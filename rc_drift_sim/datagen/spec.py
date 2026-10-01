"""Batch specification: the JSON document the GUI edits and the CLI reads.

A spec is a plain dict (JSON-serializable)::

    {
      "name": "drift_randomized", "seed": 0, "episodes": 1000, "duration_s": 4.0,
      "params": {                                  # only what differs from the YAML defaults
        "tire": {"dist": "choice", "values": ["hard_plastic_drift", "rubber_onroad"]},
        "surface": {"dist": "fixed", "value": "epoxy_ptile"},
        "vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8},
        "surface.mu_scale": {"dist": "normal", "mean": 1.0, "std": 0.05, "mode": "scale"},
        "condition.wear": {"dist": "uniform", "low": 0.0, "high": 0.5, "per_wheel": true},
        "actuators.latency": {"dist": "choice", "values": [0.02, 0.04]},
        "init.speed": {"dist": "fixed", "value": 0.0}
      },
      "maneuver": {"type": "drift_schedule",
                   "params": {"sustain_throttle": {"dist": "uniform", "low": 0.21, "high": 0.24}}},
      "export": {"formats": ["npz"], "signals": ["pose", "velocity", "derived", "actions"],
                 "record_rate": "control", "decimation": 1, "float32": true, "compress": false,
                 "shard_episodes": 256, "out_dir": "exports"},
      "run": {"workers": 0}
    }

Distributions (``dist``):
    fixed {value}             uniform {low, high}           normal {mean, std[, low, high]}
    loguniform {low, high}    choice {values[, weights]}    bernoulli {p}   (bool fields)
    sweep {values}            linspace {low, high, num}     (deterministic grid, see below)
Options: ``mode`` "set" (default) or "scale" (multiply the base value; numbers only);
``per_wheel`` (condition.* only) draws the four wheels independently.

Sweeps: every sweep/linspace parameter forms one axis of a full-factorial grid; episode i takes
grid point i modulo the grid size (the first sweep key in sorted order varies fastest). Set
``episodes`` to a multiple of the grid size to cover it evenly.

Units follow the YAML files (SI, angles in degrees under *_deg keys); see ``datagen.catalog``.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from .catalog import build_catalog, field_index
from .inputs import MANEUVERS

DISTS = {
    "fixed": ("value",), "uniform": ("low", "high"), "normal": ("mean", "std"),
    "loguniform": ("low", "high"), "choice": ("values",), "bernoulli": ("p",),
    "sweep": ("values",), "linspace": ("low", "high", "num"),
}
SWEEPS = ("sweep", "linspace")
RECORD_RATES = ("control", "physics")    # record every control step, or every physics (integrator) step
MAX_EPISODES = 2_000_000
MAX_DURATION_S = 600.0
DEFAULT_SIGNALS = ["pose", "velocity", "derived", "wheel_speeds", "steering", "actions", "accel"]


def default_spec() -> dict:
    return {
        "name": "batch", "seed": 0, "episodes": 64, "duration_s": 4.0,
        "params": {},
        "maneuver": {"type": "drift_schedule", "params": {}},
        "export": {"formats": ["npz"], "signals": list(DEFAULT_SIGNALS), "record_rate": "control",
                   "decimation": 1, "float32": True, "compress": False, "shard_episodes": 256,
                   "out_dir": "exports"},
        "run": {"workers": 0},
    }


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sweep_values(d: dict) -> list:
    """The grid values of a sweep/linspace distribution."""
    if d["dist"] == "sweep":
        return list(d["values"])
    n = int(d["num"])
    lo, hi = float(d["low"]), float(d["high"])
    return [lo + (hi - lo) * k / (n - 1) for k in range(n)] if n > 1 else [lo]


def _range_of(d: dict) -> list[float]:
    """Values a numeric distribution can produce at its extremes (for limit checks)."""
    kind = d["dist"]
    if kind == "fixed":
        return [d["value"]]
    if kind in ("uniform", "loguniform"):
        return [d["low"], d["high"]]
    if kind == "normal":
        lo = d.get("low", d["mean"] - 4 * d["std"])
        hi = d.get("high", d["mean"] + 4 * d["std"])
        return [lo, hi]
    if kind in ("choice", "sweep"):
        return list(d["values"])
    if kind == "linspace":
        return [d["low"], d["high"]]
    return []


def _varies(d: dict) -> bool:
    """True if the distribution can give two episodes different values."""
    kind = d["dist"]
    if kind == "fixed":
        return False
    if kind == "bernoulli":
        return 0 < d["p"] < 1
    if kind == "normal":
        return d["std"] > 0 and d.get("low", -math.inf) < d.get("high", math.inf)
    if kind in ("choice", "sweep"):
        w = d.get("weights", [1.0] * len(d["values"]))
        vals = [v for v, wi in zip(d["values"], w) if wi > 0]
        return any(v != vals[0] for v in vals[1:])
    return d["low"] != d["high"] and (kind != "linspace" or d["num"] > 1)


def _check_dist(key: str, d: Any, field: dict, errors: list[str]) -> dict | None:
    if not isinstance(d, dict):
        errors.append(f"{key}: expected an object like {{'dist': 'fixed', 'value': ...}}")
        return None
    d = dict(d)
    kind = d.get("dist", "fixed")
    d["dist"] = kind
    if kind not in DISTS:
        errors.append(f"{key}: unknown dist {kind!r} (use one of {', '.join(DISTS)})")
        return None
    missing = [k for k in DISTS[kind] if k not in d]
    if missing:
        errors.append(f"{key}: dist {kind!r} needs {', '.join(missing)}")
        return None
    ftype = field.get("type", "float")
    mode = d.get("mode", "set")
    if mode not in ("set", "scale"):
        errors.append(f"{key}: mode must be 'set' or 'scale'")
        return None
    if d.get("per_wheel") and not field.get("per_wheel"):
        errors.append(f"{key}: per_wheel is only available for condition.* fields")
    if ftype in ("choice",):
        allowed = field.get("choices", [])
        if kind not in ("fixed", "choice", "sweep"):
            errors.append(f"{key}: a categorical setting takes fixed, choice or sweep")
            return None
        vals = [d["value"]] if kind == "fixed" else list(d["values"])
        bad = [v for v in vals if v not in allowed]
        if bad or not vals:
            errors.append(f"{key}: {bad or 'no values'} not in {allowed}")
            return None
    elif ftype == "bool":
        if kind == "bernoulli":
            if not (_is_num(d["p"]) and 0.0 <= d["p"] <= 1.0):
                errors.append(f"{key}: bernoulli p must be in [0, 1]")
        elif kind in ("fixed", "choice", "sweep"):
            vals = [d["value"]] if kind == "fixed" else list(d["values"])
            if not vals or not all(isinstance(v, bool) for v in vals):
                errors.append(f"{key}: values must be true/false")
        else:
            errors.append(f"{key}: a true/false setting takes fixed, choice, sweep or bernoulli")
    else:  # numbers (optional_float accepts null in fixed/choice/sweep = "auto")
        if kind == "bernoulli":
            errors.append(f"{key}: bernoulli is only for true/false settings")
            return None
        nums = []
        for name in DISTS[kind]:
            v = d[name]
            if name == "values":
                if not isinstance(v, list) or not v:
                    errors.append(f"{key}: values must be a non-empty list")
                    return None
                ok = [x for x in v if not (ftype == "optional_float" and x is None)]
                if not all(_is_num(x) for x in ok):
                    errors.append(f"{key}: values must be numbers")
                    return None
                nums += ok
            elif name == "value" and ftype == "optional_float" and v is None:
                pass
            elif not _is_num(v):
                errors.append(f"{key}: {name} must be a finite number")
                return None
        if kind in ("uniform", "loguniform", "linspace") and d["low"] > d["high"]:
            errors.append(f"{key}: low > high")
        if kind == "loguniform" and d["low"] <= 0:
            errors.append(f"{key}: loguniform needs low > 0")
        if kind == "normal" and d["std"] < 0:
            errors.append(f"{key}: std must be >= 0")
        if kind == "normal" and "low" in d and "high" in d and d["low"] > d["high"]:
            errors.append(f"{key}: low > high")
        if kind == "linspace" and (not float(d["num"]).is_integer() or d["num"] < 1):
            errors.append(f"{key}: num must be a positive integer")
        if "weights" in d:
            w = d["weights"]
            if (not isinstance(w, list) or len(w) != len(d.get("values", [])) or not all(_is_num(x) and x >= 0 for x in w)
                    or sum(w) <= 0):
                errors.append(f"{key}: weights must be non-negative numbers, one per value, not all zero")
        # physical limits on the reachable values
        if mode == "set" and ftype == "float":
            reach = [x for x in _range_of(d) if _is_num(x)]
            if "min_exclusive" in field and any(x <= field["min_exclusive"] for x in reach):
                errors.append(f"{key}: must stay > {field['min_exclusive']} (a value of {min(reach)} is reachable)")
            if "min" in field and any(x < field["min"] for x in reach):
                errors.append(f"{key}: must stay >= {field['min']} (a value of {min(reach)} is reachable)")
            if "max" in field and any(x > field["max"] for x in reach):
                errors.append(f"{key}: must stay <= {field['max']} (a value of {max(reach)} is reachable)")
        if mode == "scale" and ("min_exclusive" in field or "min" in field):
            reach = [x for x in _range_of(d) if _is_num(x)]
            if any(x < 0 for x in reach) or ("min_exclusive" in field and any(x <= 0 for x in reach)):
                errors.append(f"{key}: a scale factor for this setting must stay positive")
    return d


def validate_spec(spec: dict) -> tuple[dict, list[str], list[str]]:
    """Normalize and check a spec. Returns ``(normalized, errors, warnings)``; run only if no errors."""
    errors: list[str] = []
    warnings: list[str] = []
    base = default_spec()
    if not isinstance(spec, dict):
        return base, ["the spec must be a JSON object"], []
    out = copy.deepcopy(base)
    unknown_top = set(spec) - set(base)
    if unknown_top:
        errors.append(f"unknown top-level keys: {sorted(unknown_top)}")
    for k in ("name", "seed", "episodes", "duration_s"):
        if k in spec:
            out[k] = spec[k]
    name = str(out["name"]).strip() or "batch"
    out["name"] = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:80]
    if not isinstance(out["seed"], int) or isinstance(out["seed"], bool) or out["seed"] < 0:
        errors.append("seed must be a non-negative integer")
    if not isinstance(out["episodes"], int) or isinstance(out["episodes"], bool) or not 1 <= out["episodes"] <= MAX_EPISODES:
        errors.append(f"episodes must be an integer between 1 and {MAX_EPISODES}")
    if not _is_num(out["duration_s"]) or not 0 < out["duration_s"] <= MAX_DURATION_S:
        errors.append(f"duration_s must be in (0, {MAX_DURATION_S}]")

    idx = field_index()
    params = spec.get("params", {}) or {}
    if not isinstance(params, dict):
        errors.append("params must be an object")
        params = {}
    norm_params = {}
    for key in sorted(params):
        if key not in idx:
            close = [k for k in idx if k.split(".")[-1] == key.split(".")[-1]]
            errors.append(f"unknown parameter {key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
            continue
        d = _check_dist(key, params[key], idx[key], errors)
        if d is not None:
            norm_params[key] = d
    out["params"] = norm_params

    man = spec.get("maneuver", base["maneuver"]) or {}
    mtype = man.get("type", "drift_schedule")
    if mtype not in MANEUVERS:
        errors.append(f"unknown maneuver type {mtype!r}")
        mtype = "drift_schedule"
    mfields = {p["name"]: dict(p, key=f"maneuver.{p['name']}", type="float") for p in MANEUVERS[mtype]["params"]}
    mparams = {}
    for k, d in sorted((man.get("params") or {}).items()):
        if k not in mfields:
            errors.append(f"maneuver {mtype!r} has no parameter {k!r} (has {sorted(mfields)})")
            continue
        nd = _check_dist(f"maneuver.{k}", d, mfields[k], errors)
        if nd is not None:
            mparams[k] = nd
    out["maneuver"] = {"type": mtype, "params": mparams}

    exp = dict(base["export"])
    exp.update(spec.get("export", {}) or {})
    cat = build_catalog()
    fmts = exp.get("formats", ["npz"])
    if not isinstance(fmts, list) or not fmts:
        errors.append("export.formats must be a non-empty list")
    else:
        bad = [f for f in fmts if f not in cat["formats"]]
        if bad:
            errors.append(f"export formats {bad} are not available here (available: {cat['formats']})")
    sig_keys = {s["key"] for s in cat["signals"]}
    sigs = exp.get("signals", [])
    if not isinstance(sigs, list) or not sigs:
        errors.append("export.signals must be a non-empty list")
    elif any(s not in sig_keys for s in sigs):
        errors.append(f"unknown signal groups {[s for s in sigs if s not in sig_keys]}")
    if exp.get("record_rate") not in RECORD_RATES:
        errors.append(f"export.record_rate must be one of {list(RECORD_RATES)}")
    if not isinstance(exp.get("decimation"), int) or exp["decimation"] < 1:
        errors.append("export.decimation must be an integer >= 1")
    if not isinstance(exp.get("shard_episodes"), int) or not 1 <= exp["shard_episodes"] <= 65536:
        errors.append("export.shard_episodes must be an integer in [1, 65536]")
    for flag in ("float32", "compress"):
        if not isinstance(exp.get(flag), bool):
            errors.append(f"export.{flag} must be true or false")
    out["export"] = exp
    run = dict(base["run"])
    run.update(spec.get("run", {}) or {})
    if not isinstance(run.get("workers"), int) or run["workers"] < 0:
        errors.append("run.workers must be an integer >= 0 (0 = automatic)")
    out["run"] = run

    if errors:
        return out, errors, warnings

    # ---- warnings and derived facts
    mirror = mparams.get("mirror_prob")
    mixed_mirror = mirror is not None and mirror["dist"] == "fixed" and 0 < mirror["value"] < 1
    if out["episodes"] > 1 and not (mtype == "random" or mixed_mirror
                                    or any(_varies(d) for d in [*norm_params.values(), *mparams.values()])):
        warnings.append(f"nothing varies between episodes, so all {out['episodes']} episodes will be identical "
                        f"(the simulation is deterministic): randomize or sweep a parameter, use the random "
                        f"maneuver, or set mirror_prob between 0 and 1")
    grid = 1
    for k, d in list(norm_params.items()) + [(f"maneuver.{k}", d) for k, d in mparams.items()]:
        if d["dist"] in SWEEPS:
            grid *= len(sweep_values(d))
    if grid > 1 and out["episodes"] % grid:
        warnings.append(f"the sweep grid has {grid} points but episodes = {out['episodes']} is not a multiple "
                        f"of it, so some grid points get one episode more than others")
    if grid > out["episodes"]:
        warnings.append(f"the sweep grid has {grid} points but only {out['episodes']} episodes: part of the "
                        f"grid is not covered")
    varying_struct = [k for k, d in norm_params.items() if idx[k].get("structural") and d["dist"] != "fixed"]
    if varying_struct:
        warnings.append(f"structural settings vary ({', '.join(varying_struct)}): episodes are split into "
                        f"separate batches, which is slower than varying numeric parameters")
    est = estimate(out)
    if est["bytes"] > 5e9:
        warnings.append(f"the export will be about {est['bytes'] / 1e9:.1f} GB; consider decimation, float32 or "
                        f"fewer signals")
    return out, errors, warnings


def estimate(spec: dict) -> dict:
    """Steps, records and approximate output size of a (normalized) spec."""
    from .export import signal_columns
    idx = field_index()

    def fixed(key: str) -> float:
        v = spec["params"].get(key, {}).get("value", idx[key]["default"])
        return float(v) if _is_num(v) else float(idx[key]["default"])

    dt = fixed("sim.control_dt")
    T = max(1, int(round(spec["duration_s"] / dt)))
    steps = T * max(1, int(round(dt / fixed("sim.dt")))) if spec["export"].get("record_rate") == "physics" else T
    T_rec = steps // spec["export"]["decimation"] + 1
    cols = sum(len(c) for c in signal_columns(spec["export"]["signals"]).values())
    bytes_per = 4 if spec["export"]["float32"] else 8
    n = spec["episodes"]
    fmt_factor = sum({"npz": 1.0, "csv": 2.6, "parquet": 0.6}.get(f, 1.0) for f in spec["export"]["formats"])
    return dict(control_steps=T, records_per_episode=T_rec, columns=cols, total_steps=n * T,
                bytes=int(n * T_rec * cols * bytes_per * fmt_factor))
