"""Per-wheel surface parameters (Milestone 1: uniform surface only).

Milestone 1 scope: the whole track is one surface type. ``uniform_surface`` broadcasts a
scalar :class:`SurfaceParams` to one value per wheel so that the tire and vehicle code are
already written per wheel (wheel order FL, FR, RL, RR; per-wheel arrays have shape (4,)).

Milestone 2 replaces the uniform surface with a 2D surface-map lookup at the four contact
points; ``blend_surfaces`` (gradual transitions between two surface types) and ``per_wheel``
(a different surface under each wheel) are the building blocks that lookup will use.

All functions are pure, allocate new arrays (no in-place mutation) and use NumPy float64.
Only the fields listed in ``SurfaceParams.NUMERIC_FIELDS`` are numeric; ``name`` and
``color`` are carried through unchanged (or combined into a descriptive label).
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from .params import SurfaceParams

N_WHEELS = 4


def surface_field(surf: SurfaceParams, name: str) -> np.ndarray:
    """Return one numeric surface field as a (4,) float64 array.

    Scalars are broadcast to all four wheels; (4,) arrays are copied. ``name`` must be one
    of ``SurfaceParams.NUMERIC_FIELDS`` (dimensionless quantities, see ``params.py``).
    """
    value = np.asarray(getattr(surf, name), dtype=np.float64)
    return np.array(np.broadcast_to(value, (N_WHEELS,)), dtype=np.float64)


def uniform_surface(sp: SurfaceParams) -> SurfaceParams:
    """Broadcast a (possibly scalar) surface to a per-wheel surface.

    Every field in ``SurfaceParams.NUMERIC_FIELDS`` becomes a (4,) float64 array (one value
    per wheel, FL FR RL RR). ``name`` and ``color`` are kept. Idempotent.
    """
    kw = {name: surface_field(sp, name) for name in SurfaceParams.NUMERIC_FIELDS}
    return dataclasses.replace(sp, **kw)


def blend_surfaces(a: SurfaceParams, b: SurfaceParams, w) -> SurfaceParams:
    """Per-wheel linear blend of two surfaces: ``(1 - w) * a + w * b`` on every numeric field.

    Parameters
    ----------
    a, b : SurfaceParams
        Surfaces to blend (scalar or per-wheel fields).
    w : float or array of shape (4,)
        Blend weight per wheel, dimensionless, 0 = pure ``a`` ... 1 = pure ``b``.

    The result carries ``name = "<a.name>~<b.name>"`` and the color of ``a``.
    Milestone 2 uses this for gradual transitions at surface boundaries.
    """
    w_arr = np.array(np.broadcast_to(np.asarray(w, dtype=np.float64), (N_WHEELS,)),
                     dtype=np.float64)
    kw = {name: (1.0 - w_arr) * surface_field(a, name) + w_arr * surface_field(b, name)
          for name in SurfaceParams.NUMERIC_FIELDS}
    return dataclasses.replace(a, name=f"{a.name}~{b.name}", color=a.color, **kw)


def per_wheel(surfaces: Sequence[SurfaceParams]) -> SurfaceParams:
    """Stack one surface per wheel (list of exactly 4, order FL FR RL RR) into one SurfaceParams.

    Wheel ``i`` takes its value from ``surfaces[i]``; if ``surfaces[i]`` is itself per-wheel,
    its ``i``-th entry is used. ``name`` becomes ``"per_wheel(<n0>,<n1>,<n2>,<n3>)"``; ``color``
    is the common color if all four agree, otherwise a tuple of the four colors.
    """
    if len(surfaces) != N_WHEELS:
        raise ValueError(f"per_wheel expects exactly {N_WHEELS} surfaces, got {len(surfaces)}")
    kw = {}
    for name in SurfaceParams.NUMERIC_FIELDS:
        kw[name] = np.stack([surface_field(s, name)[i] for i, s in enumerate(surfaces)]).astype(
            np.float64)
    names = ",".join(str(s.name) for s in surfaces)
    colors = tuple(s.color for s in surfaces)
    color = colors[0] if all(c == colors[0] for c in colors) else colors
    return dataclasses.replace(surfaces[0], name=f"per_wheel({names})", color=color, **kw)
