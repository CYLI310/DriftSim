"""Surface maps (Milestone 2): a different surface under each wheel, friction patches, bumps, slopes.

A :class:`SurfaceMap` is a 2D grid of surface labels over a world rectangle. The vehicle model
looks it up at the four tire contact points in EVERY right-hand-side evaluation, so a car that
drives from the epoxy track onto loose dirt feels the change one wheel at a time, at the physics
rate, and the GPU (PyTorch) and NumPy paths do exactly the same thing.

What the map adds to the Milestone 1 physics
--------------------------------------------
* Per-wheel surface. Every cell carries the numeric surface fields of its surface type. Values are
  bilinearly interpolated between cell centers, so boundaries are blended over one cell width
  (the ``blend_surfaces`` transition, made spatial). Beyond the map the ``outside`` surface applies.
* Friction patches. ``mu_patch_std`` (surfaces.yaml) becomes a smooth random field of correlation
  length ``patch_size`` that multiplies the surface mu: ``mu *= max(1 + mu_patch_std * z, 0.2)``,
  ``z`` ~ unit normal.
* Roughness. ``roughness`` becomes bumps under the tires: ``Fz_i *= max(1 + roughness_i * n_i, 0)``,
  with ``n`` a smooth unit-std spatial noise of correlation length ``bump_size`` sampled at the
  contact point (it repeats every ``BUMP_TILE * bump_size``). Bumps are fixed to the ground, so the
  load noise a tire feels scales with its speed, and a parked car feels none.
* Slopes. An optional height map ``height`` (m, at cell centers) adds the downhill gravity component
  ``-g * grad(h)`` (small-slope approximation, mean of the four contact points) to the body
  acceleration. It is not a specific force: an IMU and the load-transfer lag see it only through the
  tire forces that resist it, which is what a real accelerometer on a slope measures.

How the map reaches the tire model
----------------------------------
The surface fields enter ``tire.tire_model`` through a handful of expressions that are linear in
the per-surface quantities in ``CHANNELS`` once the compound and the static tire condition are
fixed (``MapTire``). The map stores those quantities per cell; :func:`tire_on_map` rebuilds the
surface-dependent ``TireModel`` fields from the interpolated values. On a one-surface map with
patches and bumps off this reproduces ``tire_model(tp, uniform_surface(s), cond)`` to rounding.

Public API
----------
    SurfaceMap(surfaces, labels, cell, origin=(0, 0), outside=0, height=None, seed=0,
               patch_size=0.5, bump_size=0.03, mu_patches=True, bumps=True)
        .uniform(surface, ...), .from_labels(...), .from_function(...), .from_ascii(...)
        .extent, .shape, .names, .surface_at(x, y), .sample(x, y), .image(), .draw(ax)
    resolve_surface(name_or_params, config_dir=None) -> SurfaceParams
    compile_map(smap) -> MapModel                       arrays the model core reads
    map_tire(tp, cond) -> MapTire                       per-car constants for tire_on_map
    lookup_m(mm, x, y) -> (channels (..., 4, C), bump (..., 4))
    tire_on_map(tm, mt, channels) -> TireModel          surface fields replaced per wheel
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np

from .params import SurfaceParams, TireCondition, TireParams, load_surfaces
from .tire import C_MAX, TireModel, condition_scales
from .xp import is_torch, namespace

# per-cell quantities, all linear in the surface fields (see module docstring)
CHANNELS = ("mu",          # mu_scale (x friction patch)
            "mu_loose",    # mu_scale * looseness (x friction patch); times the compound's loose_affinity
            "stiffness",   # stiffness_scale
            "blend",       # shape_blend
            "blend_c",     # shape_blend * shape_c_target
            "e_shift",     # curvature_e_shift
            "slip_shift",  # peak_slip_shift
            "crr",         # rolling_resistance
            "loose_drag",  # plowing coefficient
            "roughness",   # bump amplitude, fraction of Fz
            "grad_x",      # dh/dx of the height map
            "grad_y")      # dh/dy
(MU, MU_LOOSE, STIFF, BLEND, BLEND_C, E_SHIFT, SLIP_SHIFT, CRR, LOOSE_DRAG, ROUGH, GRAD_X,
 GRAD_Y) = range(len(CHANNELS))
NC = len(CHANNELS)

BUMP_TILE = 256            # bump noise tile size in cells of ``bump_size``
MU_PATCH_FLOOR = 0.2       # friction patches never take a surface below 20 % of its mean mu
_BILINEAR_STD = 1.5        # 1 / std of a bilinear interpolation of iid unit normals (= 1/(2/3))


def resolve_surface(s: SurfaceParams | str, config_dir: Any = None) -> SurfaceParams:
    """A ``SurfaceParams`` as is, or the surface of that name in ``surfaces.yaml``."""
    if isinstance(s, SurfaceParams):
        return s
    table = load_surfaces(None if config_dir is None else f"{config_dir}/surfaces.yaml")
    if s not in table:
        raise KeyError(f"unknown surface {s!r}; known: {', '.join(sorted(table))}")
    return table[s]


def _scalar_field(s: SurfaceParams, name: str) -> float:
    v = np.asarray(getattr(s, name), dtype=np.float64)
    if v.ndim != 0:
        raise ValueError(f"surface map entries must be scalar surfaces ({s.name}.{name} is per-wheel)")
    return float(v)


def surface_channels(s: SurfaceParams) -> np.ndarray:
    """(NC,) channel values of one surface on flat ground, without patches."""
    f = lambda name: _scalar_field(s, name)  # noqa: E731
    return np.array([f("mu_scale"), f("mu_scale") * f("looseness"), f("stiffness_scale"),
                     f("shape_blend"), f("shape_blend") * f("shape_c_target"), f("curvature_e_shift"),
                     f("peak_slip_shift"), f("rolling_resistance"), f("loose_drag"), f("roughness"),
                     0.0, 0.0], dtype=np.float64)


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], corr: float, cell: float
                  ) -> np.ndarray:
    """Unit-std smooth noise at the centers of a (ny, nx) grid of ``cell``: iid normals on a coarse
    lattice of spacing ``corr``, bilinearly interpolated."""
    ny, nx = shape
    step = float(corr) / float(cell)                          # grid cells per lattice spacing
    if step <= 1.0:                                           # patches no larger than a cell
        return rng.standard_normal((ny, nx))
    ly, lx = int(np.ceil((ny - 1) / step)) + 2, int(np.ceil((nx - 1) / step)) + 2
    lat = rng.standard_normal((ly, lx))
    gy, gx = np.arange(ny) / step, np.arange(nx) / step
    iy, ix = np.floor(gy).astype(int), np.floor(gx).astype(int)
    fy, fx = (gy - iy)[:, None], (gx - ix)[None, :]
    out = (lat[iy][:, ix] * (1 - fy) * (1 - fx) + lat[iy][:, ix + 1] * (1 - fy) * fx
           + lat[iy + 1][:, ix] * fy * (1 - fx) + lat[iy + 1][:, ix + 1] * fy * fx)
    return out * _BILINEAR_STD


# ----------------------------------------------------------------------------- the map
@dataclass(frozen=True, eq=False)
class SurfaceMap:
    """A grid of surface labels over a world rectangle.

    surfaces : surface types, indexed by ``labels`` (``SurfaceParams`` or names in surfaces.yaml).
    labels : (ny, nx) integer array; ``labels[j, i]`` is the surface of the cell whose lower-left
        corner is ``origin + (i, j) * cell`` (row 0 = smallest y, as drawn with ``origin="lower"``).
    cell : cell size in m. Surfaces blend over one cell at a boundary.
    outside : surface index used beyond the map.
    height : optional (ny, nx) ground height in m at the cell centers (slopes); beyond the map the
        ground is flat.
    seed : seed of the friction patches and bumps (same seed -> same ground).
    patch_size, bump_size : correlation lengths of the friction patches and the bumps, m.
    mu_patches, bumps : switch the random friction patches / the roughness bumps on or off.
    """
    surfaces: tuple
    labels: np.ndarray
    cell: float
    origin: tuple = (0.0, 0.0)
    outside: int = 0
    height: np.ndarray | None = None
    seed: int = 0
    patch_size: float = 0.5
    bump_size: float = 0.03
    mu_patches: bool = True
    bumps: bool = True
    _compiled: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        surfs = tuple(resolve_surface(s) for s in self.surfaces)
        labels = np.asarray(self.labels)
        if labels.ndim != 2 or labels.size == 0:
            raise ValueError("labels must be a non-empty (ny, nx) array")
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.all(labels == np.round(labels)):
                raise ValueError("labels must be integer surface indices")
            labels = labels.astype(np.int64)
        if labels.min() < 0 or labels.max() >= len(surfs):
            raise ValueError(f"labels must index the {len(surfs)} surfaces (0..{len(surfs) - 1})")
        if not 0 <= int(self.outside) < len(surfs):
            raise ValueError("outside must index a surface")
        if not float(self.cell) > 0:
            raise ValueError("cell must be > 0")
        height = None
        if self.height is not None:
            height = np.asarray(self.height, dtype=np.float64)
            if height.shape != labels.shape:
                raise ValueError(f"height must have the labels' shape {labels.shape}, got {height.shape}")
        for s in surfs:
            surface_channels(s)                                    # scalar check
        object.__setattr__(self, "surfaces", surfs)
        object.__setattr__(self, "labels", labels.astype(np.int64, copy=True))
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "cell", float(self.cell))
        object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))
        object.__setattr__(self, "outside", int(self.outside))

    # -- constructors
    @classmethod
    def uniform(cls, surface: SurfaceParams | str, size: float = 20.0, center: Sequence[float] = (0.0, 0.0),
                cell: float | None = None, **kw: Any) -> "SurfaceMap":
        """One surface everywhere (patches and bumps still apply unless switched off)."""
        n = 1 if cell is None else max(1, int(np.ceil(size / cell)))
        c = float(size) / n
        org = (float(center[0]) - 0.5 * n * c, float(center[1]) - 0.5 * n * c)
        return cls((surface,), np.zeros((n, n), dtype=np.int64), c, org, **kw)

    @classmethod
    def from_labels(cls, labels: Any, surfaces: Sequence[SurfaceParams | str], cell: float,
                    origin: Sequence[float] = (0.0, 0.0), **kw: Any) -> "SurfaceMap":
        return cls(tuple(surfaces), np.asarray(labels), cell, tuple(origin), **kw)

    @classmethod
    def from_function(cls, fn: Callable[[np.ndarray, np.ndarray], Any], surfaces: Sequence[SurfaceParams | str],
                      extent: Sequence[float], cell: float, height_fn: Callable | None = None,
                      **kw: Any) -> "SurfaceMap":
        """Labels from ``fn(x, y) -> surface index`` evaluated at the cell centers of
        ``extent = (xmin, xmax, ymin, ymax)``; optional ``height_fn(x, y) -> m``."""
        xmin, xmax, ymin, ymax = map(float, extent)
        nx, ny = max(1, int(round((xmax - xmin) / cell))), max(1, int(round((ymax - ymin) / cell)))
        xc = xmin + (np.arange(nx) + 0.5) * cell
        yc = ymin + (np.arange(ny) + 0.5) * cell
        X, Y = np.meshgrid(xc, yc)
        labels = np.broadcast_to(np.asarray(fn(X, Y)), X.shape).astype(np.int64)
        height = None if height_fn is None else np.broadcast_to(np.asarray(height_fn(X, Y), dtype=np.float64),
                                                                X.shape)
        return cls(tuple(surfaces), labels, cell, (xmin, ymin), height=height, **kw)

    @classmethod
    def from_ascii(cls, rows: str | Sequence[str], legend: Mapping[str, SurfaceParams | str], cell: float,
                   origin: Sequence[float] = (0.0, 0.0), outside: str | None = None, **kw: Any) -> "SurfaceMap":
        """Map drawn as text: one character per cell, the FIRST row is the TOP (largest y).

        ``legend`` maps characters to surfaces; ``outside`` is a legend character (default: the
        first one)."""
        lines = [r for r in (rows.strip("\n").splitlines() if isinstance(rows, str) else rows)]
        lines = [r.strip() for r in lines if r.strip()]
        if not lines or len({len(r) for r in lines}) != 1:
            raise ValueError("ascii map rows must be non-empty and all the same length")
        keys = list(legend)
        bad = sorted({ch for r in lines for ch in r} - set(keys))
        if bad:
            raise ValueError(f"characters {bad} are not in the legend")
        labels = np.array([[keys.index(ch) for ch in r] for r in reversed(lines)], dtype=np.int64)
        out = 0 if outside is None else keys.index(outside)
        return cls(tuple(legend[k] for k in keys), labels, cell, tuple(origin), outside=out, **kw)

    def replace(self, **kw: Any) -> "SurfaceMap":
        return dataclasses.replace(self, _compiled={}, **kw)

    # -- geometry
    @property
    def shape(self) -> tuple[int, int]:
        """(ny, nx) cells."""
        return tuple(self.labels.shape)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) in m."""
        ny, nx = self.shape
        x0, y0 = self.origin
        return (x0, x0 + nx * self.cell, y0, y0 + ny * self.cell)

    @property
    def names(self) -> list[str]:
        return [str(s.name) for s in self.surfaces]

    def surface_at(self, x: Any, y: Any) -> np.ndarray:
        """Surface index of the cell containing each point (``outside`` beyond the map)."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        ny, nx = self.shape
        i = np.floor((x - self.origin[0]) / self.cell).astype(np.int64)
        j = np.floor((y - self.origin[1]) / self.cell).astype(np.int64)
        inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        lab = self.labels[np.clip(j, 0, ny - 1), np.clip(i, 0, nx - 1)]
        return np.where(inside, lab, self.outside)

    def sample(self, x: Any, y: Any) -> dict[str, np.ndarray]:
        """The interpolated channels (``CHANNELS``) and the bump value ``"bump"`` at points (x, y)."""
        ch, bump = lookup_m(compile_map(self), np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        out = {name: ch[..., k] for k, name in enumerate(CHANNELS)}
        out["bump"] = bump
        return out

    # -- drawing (``viz.render.draw_background`` calls ``.draw(ax)``)
    def image(self, shade: bool = True) -> np.ndarray:
        """(ny, nx, 3) RGB picture of the map (row 0 = smallest y); ``shade`` darkens low-mu patches
        and shows slopes as hillshading."""
        from matplotlib import colors as mcolors
        pal = np.array([mcolors.to_rgb(s.color if isinstance(s.color, str) else s.color[0])
                        for s in self.surfaces])
        img = pal[self.labels]
        if shade:
            mm = compile_map(self)
            mu = mm.grid.reshape(mm.hp, mm.wp, NC)[1:-1, 1:-1, MU]
            base = np.array([surface_channels(s)[MU] for s in self.surfaces])[self.labels]
            ratio = np.where(base > 0, mu / np.maximum(base, 1e-12), 1.0)
            k = np.clip(0.85 + 0.15 * ratio, 0.6, 1.15)[..., None]
            if self.height is not None:
                gy, gx = np.gradient(self.height, self.cell)
                k = k * np.clip(1.0 - 3.0 * (gx + gy) / np.sqrt(2.0), 0.6, 1.3)[..., None]
            img = np.clip(img * k, 0.0, 1.0)
        return img

    def draw(self, ax: Any, shade: bool = True, zorder: int = 0) -> Any:
        return ax.imshow(self.image(shade), extent=self.extent, origin="lower", interpolation="nearest",
                         zorder=zorder)


# ----------------------------------------------------------------------------- compiled form
class MapModel(NamedTuple):
    """What the model core reads. ``grid`` holds the cell-center channels of the map padded with one
    ring of ``outside`` cells, flattened to (hp*wp, NC); ``bump`` is the (BUMP_TILE**2,) bump tile."""
    grid: np.ndarray
    bump: np.ndarray
    x0: float           # world x of the center of padded cell (0, 0)
    y0: float
    inv_cell: float
    wp: int
    hp: int
    inv_bump: float     # 1 / bump_size
    has_bumps: bool
    has_slope: bool


def compile_map(smap: SurfaceMap) -> MapModel:
    """The :class:`MapModel` of a map (cached on the map; NumPy float64)."""
    hit = smap._compiled.get("mm")
    if hit is not None:
        return hit
    ny, nx = smap.shape
    table = np.stack([surface_channels(s) for s in smap.surfaces])          # (K, NC)
    cells = table[smap.labels].copy()                                     # (ny, nx, NC)
    rng = np.random.default_rng(int(smap.seed))
    if smap.mu_patches:
        std = np.array([_scalar_field(s, "mu_patch_std") for s in smap.surfaces])[smap.labels]
        if np.any(std != 0):
            z = _smooth_noise(rng, (ny, nx), smap.patch_size, smap.cell)
            mult = np.maximum(1.0 + std * z, MU_PATCH_FLOOR)
            cells[..., MU] *= mult
            cells[..., MU_LOOSE] *= mult
    if smap.height is not None:
        if ny > 1:
            cells[..., GRAD_Y] = np.gradient(smap.height, smap.cell, axis=0)
        if nx > 1:
            cells[..., GRAD_X] = np.gradient(smap.height, smap.cell, axis=1)
    padded = np.broadcast_to(table[smap.outside], (ny + 2, nx + 2, NC)).copy()
    padded[1:-1, 1:-1] = cells
    bump_rng = np.random.default_rng([int(smap.seed), 1])
    bump = bump_rng.standard_normal(BUMP_TILE * BUMP_TILE) * _BILINEAR_STD
    has_bumps = bool(smap.bumps) and bool(np.any(table[:, ROUGH] != 0))
    mm = MapModel(grid=padded.reshape(-1, NC), bump=bump,
                  x0=smap.origin[0] - 0.5 * smap.cell, y0=smap.origin[1] - 0.5 * smap.cell,
                  inv_cell=1.0 / smap.cell, wp=nx + 2, hp=ny + 2, inv_bump=1.0 / float(smap.bump_size),
                  has_bumps=has_bumps, has_slope=smap.height is not None)
    smap._compiled["mm"] = mm
    return mm


def _index(a: Any) -> Any:
    return a.long() if is_torch(a) else a.astype(np.intp)


def _bilinear(table: Any, xp: Any, gx: Any, gy: Any, w: int, h: int, wrap: bool) -> Any:
    """Bilinear lookup of a flattened (h*w, ...) table at continuous cell coordinates."""
    if wrap:
        ix, iy = xp.floor(gx), xp.floor(gy)
        fx, fy = gx - ix, gy - iy
        ix0, iy0 = _index(ix) % w, _index(iy) % h
        ix1, iy1 = (ix0 + 1) % w, (iy0 + 1) % h
    else:
        gx = xp.clip(gx, 0.0, w - 1.0)
        gy = xp.clip(gy, 0.0, h - 1.0)
        ix = xp.minimum(xp.floor(gx), w - 2.0)
        iy = xp.minimum(xp.floor(gy), h - 2.0)
        fx, fy = gx - ix, gy - iy
        ix0, iy0 = _index(ix), _index(iy)
        ix1, iy1 = ix0 + 1, iy0 + 1
    r0, r1 = iy0 * w, iy1 * w
    if table.ndim == 2:
        fx, fy = fx[..., None], fy[..., None]
    return ((table[r0 + ix0] * (1.0 - fx) + table[r0 + ix1] * fx) * (1.0 - fy)
            + (table[r1 + ix0] * (1.0 - fx) + table[r1 + ix1] * fx) * fy)


def lookup_m(mm: MapModel, x: Any, y: Any) -> tuple[Any, Any]:
    """Channels (..., NC) and bump value (...) at world points (x, y) (NumPy or torch, any shape)."""
    xp = namespace(x, mm.grid)
    ch = _bilinear(mm.grid, xp, (x - mm.x0) * mm.inv_cell, (y - mm.y0) * mm.inv_cell, mm.wp, mm.hp, wrap=False)
    if mm.has_bumps:
        bump = _bilinear(mm.bump, xp, x * mm.inv_bump, y * mm.inv_bump, BUMP_TILE, BUMP_TILE, wrap=True)
    else:
        bump = xp.zeros_like(x)
    return ch, bump


# ----------------------------------------------------------------------------- tire coupling
class MapTire(NamedTuple):
    """Per-car constants that turn map channels into ``TireModel`` fields (floats, or (B, 1) / (B, 4)
    arrays in a batch)."""
    loose_affinity: Any
    pcx1: Any
    pcy1: Any
    lam_mu: Any          # (4,) static condition multipliers (wear, wetness)
    lam_k: Any
    dC: Any


def map_tire(tp: TireParams, cond: TireCondition | None) -> MapTire:
    stat = condition_scales(tp, cond, None, contam=np.zeros(()))
    b = lambda v: np.array(np.broadcast_to(np.asarray(v, dtype=np.float64), (4,)))  # noqa: E731
    return MapTire(loose_affinity=float(tp.loose_affinity), pcx1=float(tp.pcx1), pcy1=float(tp.pcy1),
                   lam_mu=b(stat["lam_mu_x"]), lam_k=b(stat["lam_k_x"]), dC=b(stat["dC"]))


def tire_on_map(tm: TireModel, mt: MapTire, ch: Any) -> TireModel:
    """``tm`` with its surface-dependent fields rebuilt from per-wheel map channels ``ch`` (..., 4, NC),
    using the expressions of ``tire.tire_model``."""
    xp = namespace(ch)
    c = lambda k: ch[..., k]  # noqa: E731
    blend, blend_c = c(BLEND), c(BLEND_C)
    slip_scale = 1.0 + c(SLIP_SHIFT)
    return dataclasses.replace(
        tm,
        Cx=xp.minimum(mt.pcx1 + blend_c - blend * mt.pcx1 + mt.dC, C_MAX),
        Cy=xp.minimum(mt.pcy1 + blend_c - blend * mt.pcy1 + mt.dC, C_MAX),
        mu_scale=(c(MU) + mt.loose_affinity * c(MU_LOOSE)) * mt.lam_mu,
        k_scale=c(STIFF) * mt.lam_k,
        e_shift=c(E_SHIFT),
        slip_scale=slip_scale, inv_slip_scale=1.0 / slip_scale,
        crr=c(CRR), loose_drag=c(LOOSE_DRAG))


__all__ = ["SurfaceMap", "MapModel", "MapTire", "CHANNELS", "NC", "BUMP_TILE", "resolve_surface",
           "surface_channels", "compile_map", "map_tire", "lookup_m", "tire_on_map"]
