"""Per-episode sampling of parameters, tire condition, initial state and maneuver inputs.

Reproducibility: every (episode, parameter) pair draws from its own generator seeded with
``(spec seed, episode id, crc32(parameter key))``. Adding or removing one randomized parameter
therefore does not change the samples of any other, and a single episode can be regenerated on
its own (``Sampler(spec).episode(i)``) - which is also how worker processes sample their shards
without the parent materializing every episode.
"""
from __future__ import annotations

import dataclasses
import math
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ..sim import params as P
from ..sim.params import Params, TireCondition
from .catalog import ANGLE_FIELDS, STRUCTURAL, field_index
from .inputs import defaults as maneuver_defaults
from .spec import SWEEPS, sweep_values

_DEG_KEYS = {f"{g}.{n}": f for (g, f), (n, _) in ANGLE_FIELDS.items()}   # 'actuators.steer_max_deg' -> 'steer_max'
_GROUPS = ("vehicle", "drivetrain", "actuators", "sim", "tire", "surface")


@lru_cache(maxsize=1)
def _configs():
    vp, dp, ap, sp = P.load_vehicle()
    return dict(vehicle=vp, drivetrain=dp, actuators=ap, sim=sp), P.load_tires(), P.load_surfaces()


@dataclass
class Episode:
    """Everything needed to simulate one episode."""
    id: int
    params: Params
    cond: TireCondition
    init: dict[str, float]                  # SI: x, y, yaw, v, beta, yaw_rate
    maneuver: dict[str, float]
    values: dict[str, Any] = field(default_factory=dict)   # sampled spec values (catalog units)

    @property
    def signature(self) -> tuple:
        return structural_signature(self.params)


def structural_signature(p: Params) -> tuple:
    """The settings that must be shared inside one vectorized batch."""
    return tuple(getattr(getattr(p, k.split(".")[0]), k.split(".")[1]) for k in sorted(STRUCTURAL))


class Sampler:
    """Draws episodes of a normalized spec (see ``spec.validate_spec``)."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.seed = int(spec["seed"])
        self.idx = field_index()
        self.mtype = spec["maneuver"]["type"]
        allkeys = [(k, d) for k, d in spec["params"].items()] + \
                  [(f"maneuver.{k}", d) for k, d in spec["maneuver"]["params"].items()]
        self.sweeps = sorted((k, sweep_values(d)) for k, d in allkeys if d["dist"] in SWEEPS)
        self.strides = {}
        stride = 1
        for k, vals in self.sweeps:
            self.strides[k] = (stride, len(vals), vals)
            stride *= len(vals)
        self.grid_size = stride

    # ---------------------------------------------------------------- drawing
    def _rng(self, i: int, key: str) -> np.random.Generator:
        return np.random.default_rng([self.seed, int(i), zlib.crc32(key.encode())])

    def _one(self, d: dict, rng: np.random.Generator) -> Any:
        kind = d["dist"]
        if kind == "fixed":
            return d["value"]
        if kind == "uniform":
            return float(rng.uniform(d["low"], d["high"]))
        if kind == "normal":
            x = float(rng.normal(d["mean"], d["std"]))
            return float(np.clip(x, d.get("low", -np.inf), d.get("high", np.inf)))
        if kind == "loguniform":
            return float(math.exp(rng.uniform(math.log(d["low"]), math.log(d["high"]))))
        if kind == "choice":
            w = np.asarray(d.get("weights", [1.0] * len(d["values"])), dtype=float)
            return d["values"][int(rng.choice(len(d["values"]), p=w / w.sum()))]
        if kind == "bernoulli":
            return bool(rng.random() < d["p"])
        raise ValueError(kind)

    def draw(self, key: str, d: dict, i: int, base: Any = None) -> Any:
        """Sampled value of ``key`` for episode ``i`` (a list of 4 for per-wheel draws)."""
        if d["dist"] in SWEEPS:
            stride, n, vals = self.strides[key]
            v = vals[(i // stride) % n]
        elif d.get("per_wheel"):
            rng = self._rng(i, key)
            v = [self._one(d, rng) for _ in range(4)]
        else:
            v = self._one(d, self._rng(i, key))
        if d.get("mode") == "scale" and base is not None:
            v = [b * x for b, x in zip(np.broadcast_to(base, (4,)), v)] if isinstance(v, list) else base * v
        return v

    # ---------------------------------------------------------------- episodes
    def _choice(self, key: str, i: int) -> str:
        d = self.spec["params"].get(key)
        return self.draw(key, d, i) if d else self.idx[key]["default"]

    def structural(self, i: int) -> tuple:
        """Structural signature of episode ``i`` without building the whole episode (cheap)."""
        groups, _, _ = _configs()
        vals = {}
        for k in sorted(STRUCTURAL):
            g, f = k.split(".")
            d = self.spec["params"].get(k)
            if g == "tire":
                base = getattr(_configs()[1][self._choice("tire", i)], f)
            else:
                base = getattr(groups[g], f)
            vals[k] = self.draw(k, d, i, base) if d else base
        return tuple(vals[k] for k in sorted(STRUCTURAL))

    def episode(self, i: int) -> Episode:
        groups, tires, surfaces = _configs()
        tire = self._choice("tire", i)
        surface = self._choice("surface", i)
        values: dict[str, Any] = {"tire": tire, "surface": surface}
        base = dict(groups, tire=tires[tire], surface=surfaces[surface])
        overrides: dict[str, dict[str, Any]] = {g: {} for g in _GROUPS}
        cond_vals: dict[str, Any] = {}
        init = dict(x=0.0, y=0.0, yaw=0.0, v=0.0, beta=0.0, yaw_rate=0.0)
        for key, d in self.spec["params"].items():
            if key in ("tire", "surface"):
                continue
            g, name = key.split(".", 1)
            if g in _GROUPS:
                fname = _DEG_KEYS.get(key, name)
                b = getattr(base[g], fname)
                b_disp = math.degrees(b) if key in _DEG_KEYS else b
                v = self.draw(key, d, i, b_disp)
                values[key] = v
                overrides[g][fname] = math.radians(v) if key in _DEG_KEYS else v
            elif g == "condition":
                v = self.draw(key, d, i, self.idx[key]["default"])
                values[key] = v
                cond_vals[name] = v
            elif g == "init":
                v = self.draw(key, d, i, self.idx[key]["default"])
                values[key] = v
                init_key = {"speed": "v", "beta_deg": "beta", "yaw_rate_deg_s": "yaw_rate", "yaw_deg": "yaw"}.get(name, name)
                init[init_key] = math.radians(v) if name.endswith(("_deg", "_deg_s")) else float(v)
        new = {g: (dataclasses.replace(base[g], **ov) if ov else base[g]) for g, ov in overrides.items()}
        params = Params(vehicle=new["vehicle"], tire=new["tire"], surface=new["surface"],
                        drivetrain=new["drivetrain"], actuators=new["actuators"], sim=new["sim"])
        amb = float(params.sim.ambient_temp)
        c = lambda n, dflt: np.clip(np.broadcast_to(np.asarray(cond_vals.get(n, dflt), dtype=float), (4,)).copy(),  # noqa: E731
                                    0.0 if n != "temp0" else -np.inf, 1.0 if n != "temp0" else np.inf)
        cond = TireCondition(wear=c("wear", 0.0), wetness=c("wetness", 0.0),
                             contamination=c("contamination", 0.0), temp0=c("temp0", amb))
        man = maneuver_defaults(self.mtype)
        for k, d in self.spec["maneuver"]["params"].items():
            v = self.draw(f"maneuver.{k}", d, i, man[k])
            man[k] = float(v)
            values[f"maneuver.{k}"] = v
        return Episode(id=int(i), params=params, cond=cond, init=init, maneuver=man, values=values)
