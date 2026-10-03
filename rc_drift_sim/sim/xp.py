"""Array namespace for the physics core: NumPy (the float64 reference) or PyTorch (CPU, Apple MPS,
NVIDIA CUDA) for GPU-accelerated batches.

The ``*_m`` core functions of ``vehicle``, ``tire``, ``drivetrain``, ``actuators`` and ``integrator``
call ``xp = namespace(a, b, ...)`` on their array arguments and then use NumPy-named functions
(``xp.maximum``, ``xp.arctan2``, ``xp.concatenate``, ...). For NumPy inputs ``namespace`` returns the
``numpy`` module itself, so the reference path is unchanged. For torch tensors it returns
``TORCH``, a thin adapter with the same names and NumPy semantics for the subset the core uses
(Python scalars mixed with tensors, ``axis=`` keywords). Model constants are moved to a device once
with :func:`to_device`; Python floats, bools and strings stay Python values (configuration).

PyTorch is optional: nothing here imports it until a torch tensor or device is requested.

Public API
----------
    namespace(*arrays) -> numpy | TORCH
    is_torch(x) -> bool
    available_devices() -> {"cpu": True, "mps": bool, "cuda": bool}
    resolve_device(name) -> "cpu" | "mps" | "cuda"          name in DEVICES ("auto" picks a GPU)
    torch_dtype(precision) -> torch.float32 | torch.float64
    to_device(obj, device, dtype) -> obj with every float ndarray as a tensor on ``device``
    to_numpy(x) -> np.ndarray
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

DEVICES = ("cpu", "auto", "mps", "cuda")          # "cpu" = NumPy float64 reference (all cores)
PRECISIONS = ("float32", "float64")


def _torch():
    import torch
    return torch


def is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def namespace(*arrays: Any):
    """``numpy`` unless one of ``arrays`` is a torch tensor, then the torch adapter."""
    for a in arrays:
        if is_torch(a):
            return TORCH
    return np


class _TorchNamespace:
    """NumPy-named functions on torch tensors (only what the physics core needs)."""

    pi = np.pi

    @staticmethod
    def _like(x, ref):
        """``x`` as a tensor with ``ref``'s dtype and device (Python scalars and 0-d arrays)."""
        t = _torch()
        return x if isinstance(x, t.Tensor) else t.as_tensor(x, dtype=ref.dtype, device=ref.device)

    @staticmethod
    def _ref(*xs):
        t = _torch()
        return next(x for x in xs if isinstance(x, t.Tensor))

    def asarray(self, x, dtype=None):
        return x

    def maximum(self, a, b):
        t = _torch()
        if not isinstance(a, t.Tensor):
            a, b = b, a
        return t.clamp(a, min=b) if not isinstance(b, t.Tensor) else t.maximum(a, b)

    def minimum(self, a, b):
        t = _torch()
        if not isinstance(a, t.Tensor):
            a, b = b, a
        return t.clamp(a, max=b) if not isinstance(b, t.Tensor) else t.minimum(a, b)

    def fmax(self, a, b):
        return _torch().fmax(a, self._like(b, a))

    def clip(self, x, lo, hi):
        t = _torch()
        ref = self._ref(x, lo, hi)
        x = self._like(x, ref)
        if isinstance(lo, t.Tensor) or isinstance(hi, t.Tensor):
            return t.clamp(x, min=self._like(lo, ref), max=self._like(hi, ref))
        return t.clamp(x, min=float(lo), max=float(hi))

    def where(self, cond, a, b):
        ref = a if isinstance(a, _torch().Tensor) else b          # at least one branch is a tensor
        return _torch().where(cond, self._like(a, ref), self._like(b, ref))

    def stack(self, xs, axis=0):
        return _torch().stack(list(xs), dim=axis)

    def concatenate(self, xs, axis=0):
        return _torch().cat(list(xs), dim=axis)

    def sum(self, x, axis=None):
        return _torch().sum(x) if axis is None else _torch().sum(x, dim=axis)

    def broadcast_to(self, x, shape):
        return _torch().broadcast_to(x, tuple(shape))

    def zeros_like(self, x):
        return _torch().zeros_like(x)

    def ones_like(self, x):
        return _torch().ones_like(x)

    def hypot(self, a, b):
        return _torch().sqrt(a * a + b * b)

    def arctan2(self, y, x):
        return _torch().atan2(y, x)

    def arctan(self, x):
        return _torch().atan(x)

    def degrees(self, x):
        return x * (180.0 / np.pi)

    def isfinite(self, x):
        return _torch().isfinite(x)

    def all(self, x, axis=None):
        return _torch().all(x) if axis is None else _torch().all(x, dim=axis)

    def __getattr__(self, name):      # abs, sign, sqrt, exp, tanh, cos, sin, tan, ...
        return getattr(_torch(), name)


TORCH = _TorchNamespace()


# ----------------------------------------------------------------------------- devices
def available_devices() -> dict[str, bool]:
    """Which compute devices this machine has (PyTorch not installed -> only "cpu")."""
    out = {"cpu": True, "mps": False, "cuda": False}
    try:
        t = _torch()
    except ImportError:
        return out
    out["cuda"] = bool(t.cuda.is_available())
    out["mps"] = bool(getattr(t.backends, "mps", None) is not None and t.backends.mps.is_available())
    return out


def resolve_device(name: str) -> str:
    """``"auto"`` -> "cuda", else "mps", else "cpu"; other names are checked for availability."""
    avail = available_devices()
    if name == "auto":
        return "cuda" if avail["cuda"] else ("mps" if avail["mps"] else "cpu")
    if name not in avail:
        raise ValueError(f"unknown device {name!r}; expected one of {DEVICES}")
    if not avail[name]:
        raise ValueError(f"device {name!r} is not available on this machine (PyTorch "
                         f"{'missing' if not any(avail[d] for d in ('mps', 'cuda')) else 'found no such GPU'})")
    return name


def torch_dtype(precision: str):
    t = _torch()
    return {"float32": t.float32, "float64": t.float64}[precision]


def to_device(obj: Any, device: str, dtype: Any) -> Any:
    """Copy of a compiled model (nested frozen dataclasses / NamedTuples) with every float NumPy
    array as a torch tensor on ``device``; Python scalars, bools, strings and other fields are kept."""
    t = _torch()
    if isinstance(obj, np.ndarray) and obj.dtype.kind == "f":
        return t.as_tensor(obj, dtype=dtype, device=device)
    if isinstance(obj, np.ndarray):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if type(obj).__module__.endswith(".params"):      # the raw Params bundle: not used by the core
            return obj
        return dataclasses.replace(obj, **{f.name: to_device(getattr(obj, f.name), device, dtype)
                                           for f in dataclasses.fields(obj)})
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(to_device(v, device, dtype) for v in obj))
    return obj


def to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if is_torch(x) else np.asarray(x)


__all__ = ["DEVICES", "PRECISIONS", "namespace", "is_torch", "TORCH", "available_devices",
           "resolve_device", "torch_dtype", "to_device", "to_numpy"]
