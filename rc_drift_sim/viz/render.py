"""Top-down renderer for the RC drift simulator (docs/DESIGN.md section 9).

Everything is drawn in the world frame (x east, y north, yaw CCW from +x, metres). The
renderer consumes a ``Trajectory`` as produced by ``vehicle.Vehicle.rollout``:

    traj.t        (T+1,)      time in s
    traj.states   (T+1, NS)   state vectors, indices from ``rc_drift_sim.sim.state``
    traj.actions  (T, 2)      (steer_cmd, throttle_cmd) in [-1, 1]
    traj.info     dict        stacked per-step info, every array has leading dim T;
                              per-wheel arrays are (T, 4) in FL, FR, RL, RR order.

Info keys used here (all optional, with fall-backs to the state vector): ``delta_w`` (per-wheel
steer angle, rad), ``kappa`` (slip ratio), ``alpha`` (slip angle, rad), ``Fx``/``Fy`` (tire
forces in the wheel frame, N), ``thr`` and ``delta_target`` (HUD only). ``info[k]`` is taken
to describe control step k, i.e. the interval starting at ``t[k]``; the final state ``T`` is
drawn with the info of step ``T-1``.

Matplotlib uses the Agg backend (no display) unless ``MPLBACKEND`` is set in the environment.
``live_view`` is an optional pygame viewer; pygame is imported lazily so the module never
requires it.

Public API:
    car_geometry(state, info_step, params) -> dict of world-frame polygons and vectors
    draw_car(ax, state, info_step, params, artists=None, ...) -> artists dict (updatable)
    render_frame(ax, traj, k, params, surf_map=None, path=None, trail=True, world_lim=None,
                 artists=None, ...) -> artists dict (updatable)
    animate(traj, params, out_path, fps=25, stride=2, follow=False, world_lim=None, **kw) -> path
    snapshot(traj, params, out_path, ks=None, ...) -> Figure
    live_view(traj, params, fps=None, stride=1, ...)   (pygame, optional)
    info_at(info, k), trajectory_limits(traj, path=None), add_slip_colorbar(ax, ...)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from rc_drift_sim.sim import state as S  # noqa: E402
from rc_drift_sim.sim.params import Params, WHEEL_NAMES  # noqa: E402

# ----------------------------------------------------------------------------- defaults
DEFAULT_FORCE_SCALE: float = 0.02      # m of arrow per N of tire force
DEFAULT_VEL_SCALE: float = 0.10        # m of arrow per m/s of CG velocity
DEFAULT_SLIP_RANGE: tuple[float, float] = (0.0, 1.0)   # hypot(kappa, alpha) mapped to the colormap
DEFAULT_SLIP_CMAP: str = "plasma"
WHEEL_WIDTH_FRACTION: float = 0.4      # drawn wheel width as a fraction of the wheel diameter

COLORS: dict[str, str] = {
    "body_face": "#e9e9e9",
    "body_edge": "#1a1a1a",
    "wheel_edge": "#101010",
    "force": "#ff4d1a",
    "velocity": "#1fbf5f",
    "heading": "#111111",
    "cg": "#111111",
    "path": "#ffd84d",
}

__all__ = [
    "car_geometry", "draw_car", "render_frame", "animate", "snapshot", "live_view",
    "info_at", "trajectory_limits", "add_slip_colorbar", "draw_background",
    "DEFAULT_FORCE_SCALE", "DEFAULT_VEL_SCALE", "DEFAULT_SLIP_RANGE", "DEFAULT_SLIP_CMAP",
]


# ----------------------------------------------------------------------------- small helpers
def _rect_corners(cx: np.ndarray, cy: np.ndarray, length: float, width: float,
                  theta: np.ndarray) -> np.ndarray:
    """World corners of axis-aligned-in-local-frame rectangles.

    cx, cy, theta: scalars or arrays of shape (...,) (m, m, rad). length is along the local
    x axis, width along local y. Returns (..., 4, 2) corners in CCW order.
    """
    cx = np.asarray(cx, dtype=float)[..., None]
    cy = np.asarray(cy, dtype=float)[..., None]
    theta = np.asarray(theta, dtype=float)[..., None]
    hl, hw = 0.5 * float(length), 0.5 * float(width)
    lx = np.array([hl, -hl, -hl, hl])
    ly = np.array([hw, hw, -hw, -hw])
    c, s = np.cos(theta), np.sin(theta)
    x = cx + lx * c - ly * s
    y = cy + lx * s + ly * c
    return np.stack([x, y], axis=-1)


def _wheel_quantity(info: Mapping[str, Any] | None, key: str, default: np.ndarray) -> np.ndarray:
    """info[key] as a float (4,) array, or ``default`` when the key is absent."""
    if info is None or key not in info or info[key] is None:
        return np.asarray(default, dtype=float).reshape(4)
    return np.asarray(info[key], dtype=float).reshape(4)


def _colormap(cmap: str | mcolors.Colormap) -> mcolors.Colormap:
    if isinstance(cmap, mcolors.Colormap):
        return cmap
    return matplotlib.colormaps[cmap]


def _contrast_color(background: Any) -> str:
    """Black or white, whichever contrasts with a background color."""
    r, g, b = mcolors.to_rgb(background)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.55 else "#f8f8f8"


def _states(traj: Any) -> np.ndarray:
    return np.asarray(traj.states, dtype=float)


def _times(traj: Any) -> np.ndarray:
    return np.asarray(traj.t, dtype=float).reshape(-1)


def info_at(info: Mapping[str, Any] | None, k: int) -> dict[str, Any]:
    """Slice a stacked info dict (leading dim T) at step k, clamped to the last step.

    Entries without a leading time axis (0-d) are passed through unchanged.
    """
    out: dict[str, Any] = {}
    if info is None:
        return out
    for key, value in info.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] > 0:
            out[key] = arr[min(int(k), arr.shape[0] - 1)]
        else:
            out[key] = arr
    return out


def trajectory_limits(traj: Any, path: np.ndarray | None = None, margin: float = 0.5,
                      square: bool = True) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) in m covering the CG trace (and the path), padded by margin."""
    st = _states(traj)
    xs, ys = st[:, S.X], st[:, S.Y]
    if path is not None:
        p = np.asarray(path, dtype=float).reshape(-1, 2)
        xs, ys = np.concatenate([xs, p[:, 0]]), np.concatenate([ys, p[:, 1]])
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if square:
        half = 0.5 * max(xmax - xmin, ymax - ymin) + margin
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        return (cx - half, cx + half, cy - half, cy + half)
    return (xmin - margin, xmax + margin, ymin - margin, ymax + margin)


def _hud_text(traj: Any, k: int, params: Params) -> str:
    st = _states(traj)
    t = _times(traj)
    k = min(int(k), st.shape[0] - 1)
    s = st[k]
    v = float(np.hypot(s[S.VX], s[S.VY]))
    beta = float(np.degrees(np.arctan2(s[S.VY], s[S.VX])))
    r = float(np.degrees(s[S.R]))
    delta = float(np.degrees(s[S.DELTA]))
    lines = [f"t {t[min(k, t.shape[0] - 1)]:6.2f} s",
             f"v {v:6.2f} m/s",
             f"beta {beta:6.1f} deg",
             f"r {r:6.1f} deg/s",
             f"delta {delta:6.1f} deg"]
    actions = getattr(traj, "actions", None)
    if actions is not None:
        a = np.asarray(actions, dtype=float)
        if a.ndim == 2 and a.shape[0] > 0:
            ak = a[min(k, a.shape[0] - 1)]
            lines.append(f"cmd  {ak[0]:+5.2f} {ak[1]:+5.2f}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- geometry
def car_geometry(state: np.ndarray, info_step: Mapping[str, Any] | None,
                 params: Params) -> dict[str, Any]:
    """World-frame drawing geometry of the car in one state.

    Parameters
    ----------
    state : (NS,) state vector (m, rad, m/s, ...), indices from ``state.py``.
    info_step : per-step info dict; per-wheel entries are (4,) in FL FR RL RR order. Missing
        entries fall back to the state: ``delta_w`` -> (DELTA, DELTA, 0, 0), ``kappa``/``alpha``
        -> the lagged slip states, ``Fx``/``Fy`` -> zeros.
    params : Params (uses ``params.vehicle`` for dimensions).

    Returns a dict with ``cg`` (2,), ``yaw`` (rad), ``body`` (4, 2) corners, ``wheels`` (4, 4, 2)
    corners, ``wheel_centers`` (4, 2), ``wheel_heading`` (4,) rad, ``slip`` (4,) =
    hypot(kappa, alpha), ``F_world`` (4, 2) N, ``v_world`` (2,) m/s, ``nose`` (2,) m.
    """
    vp = params.vehicle
    s = np.asarray(state, dtype=float).reshape(-1)
    x, y, yaw = s[S.X], s[S.Y], s[S.YAW]
    c, sn = np.cos(yaw), np.sin(yaw)

    delta_w = _wheel_quantity(info_step, "delta_w", np.array([s[S.DELTA], s[S.DELTA], 0.0, 0.0]))
    kappa = _wheel_quantity(info_step, "kappa", s[S.KAPPA_LAG])
    alpha = _wheel_quantity(info_step, "alpha", s[S.ALPHA_LAG])
    Fx = _wheel_quantity(info_step, "Fx", np.zeros(4))
    Fy = _wheel_quantity(info_step, "Fy", np.zeros(4))

    pb = vp.wheel_positions()                             # (4, 2) body frame
    px = x + pb[:, 0] * c - pb[:, 1] * sn
    py = y + pb[:, 0] * sn + pb[:, 1] * c
    heading_w = yaw + delta_w
    wheel_len = 2.0 * vp.wheel_radius
    wheel_wid = WHEEL_WIDTH_FRACTION * wheel_len
    wheels = _rect_corners(px, py, wheel_len, wheel_wid, heading_w)       # (4, 4, 2)

    # body centred on the wheelbase midpoint so the axles sit at +a / -b inside it
    xc = 0.5 * (vp.cg_to_front - vp.cg_to_rear)
    body = _rect_corners(x + xc * c, y + xc * sn, vp.body_length, vp.body_width, yaw)  # (4, 2)

    cw, sw = np.cos(heading_w), np.sin(heading_w)
    F_world = np.stack([Fx * cw - Fy * sw, Fx * sw + Fy * cw], axis=-1)   # (4, 2)
    vx, vy = s[S.VX], s[S.VY]
    v_world = np.array([vx * c - vy * sn, vx * sn + vy * c])
    nose_d = xc + 0.5 * vp.body_length
    nose = np.array([x + nose_d * c, y + nose_d * sn])

    return {
        "cg": np.array([x, y]), "yaw": float(yaw), "body": body, "wheels": wheels,
        "wheel_centers": np.stack([px, py], axis=-1), "wheel_heading": heading_w,
        "slip": np.hypot(kappa, alpha), "F_world": F_world, "v_world": v_world, "nose": nose,
        "delta_w": delta_w, "kappa": kappa, "alpha": alpha, "Fx": Fx, "Fy": Fy,
    }


# ----------------------------------------------------------------------------- car artists
def draw_car(ax: plt.Axes, state: np.ndarray, info_step: Mapping[str, Any] | None, params: Params,
             artists: dict[str, Any] | None = None, *,
             force_scale: float = DEFAULT_FORCE_SCALE, vel_scale: float = DEFAULT_VEL_SCALE,
             slip_range: tuple[float, float] = DEFAULT_SLIP_RANGE,
             cmap: str | mcolors.Colormap = DEFAULT_SLIP_CMAP,
             alpha: float = 1.0, show_forces: bool = True, show_velocity: bool = True,
             zorder: float = 3.0) -> dict[str, Any]:
    """Draw (or update) the car in one state on ``ax``.

    Draws the body rectangle (``vehicle.body_length`` x ``vehicle.body_width``, centred on the
    wheelbase midpoint), four wheel rectangles at ``vehicle.wheel_positions()`` rotated by the
    per-wheel steer angle and face-colored by hypot(kappa, alpha) through ``cmap`` over the
    fixed ``slip_range``, one force arrow per wheel for (Fx, Fy) rotated into the world frame
    (``force_scale`` m per N), the CG velocity arrow (``vel_scale`` m per m/s) and a heading
    tick from the CG to the nose.

    Pass the returned ``artists`` dict back in to update the existing artists in place
    (for animation) instead of creating new ones. Keys: ``body`` (Polygon), ``wheels`` (list of
    4 Polygons), ``force`` (Quiver), ``vel`` (Quiver), ``heading`` (Line2D), ``cg`` (Line2D).
    """
    g = car_geometry(state, info_step, params)
    norm = mcolors.Normalize(vmin=slip_range[0], vmax=slip_range[1], clip=True)
    wheel_rgba = _colormap(cmap)(norm(g["slip"]))          # (4, 4)
    wc, F, cg, v = g["wheel_centers"], g["F_world"], g["cg"], g["v_world"]
    nose = g["nose"]

    if artists is None:
        body = Polygon(g["body"], closed=True, facecolor=COLORS["body_face"],
                       edgecolor=COLORS["body_edge"], linewidth=1.0, alpha=alpha, zorder=zorder)
        ax.add_patch(body)
        wheels = []
        for i in range(4):
            w = Polygon(g["wheels"][i], closed=True, facecolor=wheel_rgba[i],
                        edgecolor=COLORS["wheel_edge"], linewidth=0.8, alpha=alpha,
                        zorder=zorder + 1, label=WHEEL_NAMES[i])
            ax.add_patch(w)
            wheels.append(w)
        quiver_style = dict(angles="xy", scale_units="xy", units="xy", width=0.006,
                            headwidth=3.0, headlength=4.0, headaxislength=3.5, pivot="tail",
                            alpha=alpha)
        force = ax.quiver(wc[:, 0], wc[:, 1], F[:, 0], F[:, 1], scale=1.0 / force_scale,
                          color=COLORS["force"], zorder=zorder + 2, **quiver_style)
        force.set_visible(show_forces)
        vel = ax.quiver([cg[0]], [cg[1]], [v[0]], [v[1]], scale=1.0 / vel_scale,
                        color=COLORS["velocity"], zorder=zorder + 3, **quiver_style)
        vel.set_visible(show_velocity)
        heading, = ax.plot([cg[0], nose[0]], [cg[1], nose[1]], color=COLORS["heading"],
                           linewidth=1.2, alpha=alpha, zorder=zorder + 4)
        cg_dot, = ax.plot([cg[0]], [cg[1]], marker="o", markersize=3.0, linestyle="none",
                          color=COLORS["cg"], alpha=alpha, zorder=zorder + 4)
        return {"body": body, "wheels": wheels, "force": force, "vel": vel,
                "heading": heading, "cg": cg_dot}

    artists["body"].set_xy(g["body"])
    for i, w in enumerate(artists["wheels"]):
        w.set_xy(g["wheels"][i])
        w.set_facecolor(wheel_rgba[i])
    artists["force"].set_offsets(wc)
    artists["force"].set_UVC(F[:, 0], F[:, 1])
    artists["force"].set_visible(show_forces)
    artists["vel"].set_offsets(cg[None, :])
    artists["vel"].set_UVC([v[0]], [v[1]])
    artists["vel"].set_visible(show_velocity)
    artists["heading"].set_data([cg[0], nose[0]], [cg[1], nose[1]])
    artists["cg"].set_data([cg[0]], [cg[1]])
    return artists


def add_slip_colorbar(ax: plt.Axes, slip_range: tuple[float, float] = DEFAULT_SLIP_RANGE,
                      cmap: str | mcolors.Colormap = DEFAULT_SLIP_CMAP, **kw: Any):
    """Colorbar for the wheel face colors (combined slip magnitude hypot(kappa, alpha))."""
    sm = ScalarMappable(norm=mcolors.Normalize(*slip_range), cmap=_colormap(cmap))
    sm.set_array([])
    kw.setdefault("fraction", 0.04)
    kw.setdefault("pad", 0.02)
    kw.setdefault("label", "wheel slip  hypot(kappa, alpha)")
    return ax.figure.colorbar(sm, ax=ax, **kw)


# ----------------------------------------------------------------------------- background
def draw_background(ax: plt.Axes, surf_map: Any, params: Params,
                    world_lim: Sequence[float] | None = None) -> Any:
    """Draw the surface background.

    ``surf_map`` may be: None (plain ``params.surface.color`` fill), an object with a
    ``draw(ax)`` method, a tuple ``(image (H, W, 3), extent (xmin, xmax, ymin, ymax))``, or a bare
    (H, W, 3) image which is stretched over ``world_lim``. Returns the created artist or None.
    """
    if surf_map is None:
        ax.set_facecolor(params.surface.color)
        return None
    if hasattr(surf_map, "draw"):
        return surf_map.draw(ax)
    if isinstance(surf_map, tuple) and len(surf_map) == 2:
        img, extent = surf_map
    else:
        img, extent = surf_map, world_lim
    img = np.asarray(img)
    if img.ndim != 3 or extent is None:
        raise TypeError("surf_map must be None, an object with .draw(ax), an (H, W, 3) image "
                        "(with world_lim given) or an (image, extent) tuple")
    return ax.imshow(img, extent=tuple(float(e) for e in extent), origin="lower",
                     interpolation="nearest", zorder=0)


# ----------------------------------------------------------------------------- frames
def render_frame(ax: plt.Axes, traj: Any, k: int, params: Params, surf_map: Any = None,
                 path: np.ndarray | None = None, trail: bool = True,
                 world_lim: Sequence[float] | None = None,
                 artists: dict[str, Any] | None = None, *,
                 hud: bool = True, follow: bool = False, follow_halfwidth: float = 1.0,
                 force_scale: float = DEFAULT_FORCE_SCALE, vel_scale: float = DEFAULT_VEL_SCALE,
                 slip_range: tuple[float, float] = DEFAULT_SLIP_RANGE,
                 cmap: str | mcolors.Colormap = DEFAULT_SLIP_CMAP) -> dict[str, Any]:
    """Draw frame ``k`` of a trajectory on ``ax`` (or update a previous frame's artists).

    ``k`` indexes ``traj.states``; the per-step info is ``info_at(traj.info, k)`` (clamped for
    the final state). ``path`` is an optional (N, 2) reference path in m. ``world_lim`` is
    (xmin, xmax, ymin, ymax) in m; None fits the whole trajectory. With ``follow`` the view is
    a ``2 * follow_halfwidth`` square centred on the car.

    Returns a dict: ``car`` (the ``draw_car`` artists), ``trail``, ``path``, ``background``,
    ``hud`` and ``world_lim``. Pass it back as ``artists`` to update in place.
    """
    st = _states(traj)
    k = int(np.clip(k, 0, st.shape[0] - 1))
    state = st[k]
    info_k = info_at(getattr(traj, "info", None), k)
    car_kw = dict(force_scale=force_scale, vel_scale=vel_scale, slip_range=slip_range, cmap=cmap)

    if artists is None:
        lim = tuple(float(v) for v in world_lim) if world_lim is not None \
            else trajectory_limits(traj, path)
        artists = {"world_lim": lim, "path": None, "trail": None, "hud": None}
        artists["background"] = draw_background(ax, surf_map, params, lim)
        line_color = _contrast_color(params.surface.color)
        if path is not None:
            p = np.asarray(path, dtype=float).reshape(-1, 2)
            artists["path"], = ax.plot(p[:, 0], p[:, 1], linestyle="--", linewidth=1.0,
                                       color=COLORS["path"], alpha=0.9, zorder=1)
        if trail:
            artists["trail"], = ax.plot(st[:k + 1, S.X], st[:k + 1, S.Y], linewidth=1.3,
                                        color=line_color, alpha=0.8, zorder=2)
        artists["car"] = draw_car(ax, state, info_k, params, None, **car_kw)
        if hud:
            artists["hud"] = ax.text(0.02, 0.98, _hud_text(traj, k, params), transform=ax.transAxes,
                                     va="top", ha="left", family="monospace", fontsize=8,
                                     zorder=10, bbox=dict(boxstyle="round,pad=0.3",
                                                          facecolor="white", alpha=0.75,
                                                          edgecolor="none"))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    else:
        if trail and artists.get("trail") is not None:
            artists["trail"].set_data(st[:k + 1, S.X], st[:k + 1, S.Y])
        draw_car(ax, state, info_k, params, artists["car"], **car_kw)
        if hud and artists.get("hud") is not None:
            artists["hud"].set_text(_hud_text(traj, k, params))

    if follow:
        x, y = float(state[S.X]), float(state[S.Y])
        ax.set_xlim(x - follow_halfwidth, x + follow_halfwidth)
        ax.set_ylim(y - follow_halfwidth, y + follow_halfwidth)
    return artists


def _frame_indices(n_states: int, stride: int) -> list[int]:
    stride = max(1, int(stride))
    ks = list(range(0, n_states, stride))
    if ks[-1] != n_states - 1:
        ks.append(n_states - 1)
    return ks


def _make_writer(out: Path, fps: float) -> tuple[animation.AbstractMovieWriter, Path]:
    """Movie writer for the requested extension: ffmpeg for .mp4 when available, else Pillow GIF."""
    if out.suffix.lower() == ".mp4":
        if animation.FFMpegWriter.isAvailable():
            return animation.FFMpegWriter(fps=fps, bitrate=1800), out
        gif = out.with_suffix(".gif")
        print(f"[rc_drift_sim.viz] ffmpeg not available: writing {gif.name} instead of {out.name}")
        out = gif
    return animation.PillowWriter(fps=fps), out


def animate(traj: Any, params: Params, out_path: str | Path, fps: float = 25, stride: int = 2,
            follow: bool = False, world_lim: Sequence[float] | None = None, *,
            surf_map: Any = None, path: np.ndarray | None = None, trail: bool = True,
            figsize: tuple[float, float] = (6.0, 6.0), dpi: int = 72, colorbar: bool = True,
            title: str | None = None, **kw: Any) -> str:
    """Save an animation of the trajectory; returns the path actually written.

    Every ``stride``-th state is one frame (at the 50 Hz control rate, fps=25 with stride=2 is
    real time). ``.gif`` uses PillowWriter; ``.mp4`` uses FFMpegWriter when ffmpeg is installed
    and otherwise falls back to a GIF next to the requested name (a note is printed). Extra
    keyword arguments (``hud``, ``follow_halfwidth``, ``force_scale``, ``vel_scale``,
    ``slip_range``, ``cmap``) go to ``render_frame``. ``dpi`` is kept modest to keep files small.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, out = _make_writer(out, fps)
    ks = _frame_indices(_states(traj).shape[0], stride)

    fig, ax = plt.subplots(figsize=figsize, dpi=min(int(dpi), 100))
    artists = render_frame(ax, traj, ks[0], params, surf_map, path, trail, world_lim, None,
                           follow=follow, **kw)
    if colorbar:
        add_slip_colorbar(ax, kw.get("slip_range", DEFAULT_SLIP_RANGE),
                          kw.get("cmap", DEFAULT_SLIP_CMAP))
    ax.set_title(title if title is not None else f"{params.tire.name} on {params.surface.name}")

    def _update(k: int) -> list:
        render_frame(ax, traj, k, params, surf_map, path, trail, world_lim, artists,
                     follow=follow, **kw)
        return []

    anim = animation.FuncAnimation(fig, _update, frames=ks, blit=False,
                                   interval=1000.0 / fps, repeat=False)
    anim.save(str(out), writer=writer, dpi=min(int(dpi), 100))
    plt.close(fig)
    return str(out)


def snapshot(traj: Any, params: Params, out_path: str | Path | None, ks: Sequence[int] | None = None,
             *, n_poses: int = 8, surf_map: Any = None, path: np.ndarray | None = None,
             world_lim: Sequence[float] | None = None, figsize: tuple[float, float] = (7.0, 7.0),
             dpi: int = 100, colorbar: bool = True, fade: bool = True, labels: bool = True,
             title: str | None = None, **kw: Any) -> plt.Figure:
    """Static figure: the full CG trace (colored by speed) with several car poses overlaid.

    ``ks`` are the state indices to draw (default: ``n_poses`` evenly spaced). Earlier poses are
    drawn fainter when ``fade`` is set and labelled with their time when ``labels`` is set.
    Saves to ``out_path`` when given and returns the Figure (caller closes it).
    """
    st = _states(traj)
    t = _times(traj)
    n = st.shape[0]
    if ks is None:
        ks = np.unique(np.round(np.linspace(0, n - 1, max(1, int(n_poses)))).astype(int))
    ks = [int(np.clip(k, 0, n - 1)) for k in ks]
    lim = tuple(float(v) for v in world_lim) if world_lim is not None \
        else trajectory_limits(traj, path)

    fig, ax = plt.subplots(figsize=figsize, dpi=min(int(dpi), 100))
    draw_background(ax, surf_map, params, lim)
    if path is not None:
        p = np.asarray(path, dtype=float).reshape(-1, 2)
        ax.plot(p[:, 0], p[:, 1], linestyle="--", linewidth=1.0, color=COLORS["path"],
                alpha=0.9, zorder=1)

    speed = np.hypot(st[:, S.VX], st[:, S.VY])
    pts = st[:, [S.X, S.Y]][:, None, :]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=1.8, zorder=2,
                        norm=mcolors.Normalize(0.0, max(float(np.max(speed)), 1e-6)))
    lc.set_array(0.5 * (speed[:-1] + speed[1:]))
    ax.add_collection(lc)
    if colorbar:
        fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.08, location="bottom",
                     orientation="horizontal", label="speed [m/s]")
        add_slip_colorbar(ax, kw.get("slip_range", DEFAULT_SLIP_RANGE),
                          kw.get("cmap", DEFAULT_SLIP_CMAP))

    info = getattr(traj, "info", None)
    text_color = _contrast_color(params.surface.color)
    for j, k in enumerate(ks):
        a = 1.0 if (not fade or len(ks) == 1) else 0.35 + 0.65 * j / (len(ks) - 1)
        draw_car(ax, st[k], info_at(info, k), params, None, alpha=a, **kw)
        if labels:
            ax.annotate(f"{t[min(k, t.shape[0] - 1)]:.2f} s", xy=(st[k, S.X], st[k, S.Y]),
                        xytext=(5, 5), textcoords="offset points", fontsize=7,
                        color=text_color, zorder=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title if title is not None else
                 f"{params.tire.name} on {params.surface.name}  ({t[0]:.1f}-{t[-1]:.1f} s)")
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=min(int(dpi), 100))
    return fig


# ----------------------------------------------------------------------------- pygame (optional)
def _rgb255(color: Any) -> tuple[int, int, int]:
    r, g, b = mcolors.to_rgb(color)
    return int(round(255 * r)), int(round(255 * g)), int(round(255 * b))


def live_view(traj: Any, params: Params, fps: float | None = None, stride: int = 1, *,
              size: tuple[int, int] = (800, 800), follow: bool = True, view_halfwidth: float = 1.0,
              path: np.ndarray | None = None, loop: bool = False,
              force_scale: float = DEFAULT_FORCE_SCALE, vel_scale: float = DEFAULT_VEL_SCALE,
              slip_range: tuple[float, float] = DEFAULT_SLIP_RANGE,
              cmap: str | mcolors.Colormap = DEFAULT_SLIP_CMAP) -> None:
    """Replay a trajectory in a pygame window (optional dependency, imported lazily).

    ``fps=None`` plays in real time (``1 / (stride * control_dt)`` from ``traj.t``). Keys: space
    pauses, Esc/q quits; the window closes when the replay ends unless ``loop`` is set.
    Raises ImportError with a hint when pygame is not installed.
    """
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - depends on the optional dependency
        raise ImportError("live_view needs the optional dependency pygame "
                          "(pip install pygame)") from exc

    st = _states(traj)
    t = _times(traj)
    info = getattr(traj, "info", None)
    ks = _frame_indices(st.shape[0], stride)
    if fps is None:
        dt = float(t[1] - t[0]) if t.shape[0] > 1 else 0.02
        fps = 1.0 / max(dt * max(1, int(stride)), 1e-3)
    lim = trajectory_limits(traj, path)
    cx0, cy0, half0 = 0.5 * (lim[0] + lim[1]), 0.5 * (lim[2] + lim[3]), 0.5 * (lim[1] - lim[0])
    W, H = int(size[0]), int(size[1])

    def to_px(pts: np.ndarray, cx: float, cy: float, half: float) -> list:
        pts = np.asarray(pts, dtype=float)
        scale = min(W, H) / (2.0 * half)
        px = (pts[..., 0] - cx) * scale + 0.5 * W
        py = 0.5 * H - (pts[..., 1] - cy) * scale
        return np.stack([px, py], axis=-1).tolist()

    norm = mcolors.Normalize(vmin=slip_range[0], vmax=slip_range[1], clip=True)
    cm_ = _colormap(cmap)
    bg = _rgb255(params.surface.color)
    col = {key: _rgb255(val) for key, val in COLORS.items()}
    col["trail"] = _rgb255(_contrast_color(params.surface.color))
    path_pts = None if path is None else np.asarray(path, dtype=float).reshape(-1, 2)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"rc_drift_sim  {params.tire.name} on {params.surface.name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    running, paused, i = True, False, 0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
        k = ks[i]
        g = car_geometry(st[k], info_at(info, k), params)
        cx, cy, half = (float(g["cg"][0]), float(g["cg"][1]), float(view_halfwidth)) if follow \
            else (cx0, cy0, half0)

        screen.fill(bg)
        if path_pts is not None and path_pts.shape[0] >= 2:
            pygame.draw.lines(screen, col["path"], False, to_px(path_pts, cx, cy, half), 1)
        if k >= 1:
            pygame.draw.lines(screen, col["trail"], False,
                              to_px(st[:k + 1, [S.X, S.Y]], cx, cy, half), 2)
        body_px = to_px(g["body"], cx, cy, half)
        pygame.draw.polygon(screen, col["body_face"], body_px)
        pygame.draw.polygon(screen, col["body_edge"], body_px, 2)
        wheel_rgba = cm_(norm(g["slip"]))
        for w in range(4):
            wp = to_px(g["wheels"][w], cx, cy, half)
            pygame.draw.polygon(screen, _rgb255(wheel_rgba[w][:3]), wp)
            pygame.draw.polygon(screen, col["wheel_edge"], wp, 1)
        tips = g["wheel_centers"] + force_scale * g["F_world"]
        for a, b in zip(to_px(g["wheel_centers"], cx, cy, half), to_px(tips, cx, cy, half)):
            pygame.draw.line(screen, col["force"], a, b, 2)
            pygame.draw.circle(screen, col["force"], b, 3)
        cg_px = to_px(g["cg"], cx, cy, half)
        v_px = to_px(g["cg"] + vel_scale * g["v_world"], cx, cy, half)
        pygame.draw.line(screen, col["velocity"], cg_px, v_px, 2)
        pygame.draw.circle(screen, col["velocity"], v_px, 3)
        pygame.draw.line(screen, col["heading"], cg_px, to_px(g["nose"], cx, cy, half), 2)
        pygame.draw.circle(screen, col["cg"], cg_px, 3)
        for row, line in enumerate(_hud_text(traj, k, params).split("\n")):
            screen.blit(font.render(line, True, (20, 20, 20), (255, 255, 255)), (8, 8 + 16 * row))
        pygame.display.flip()
        clock.tick(fps)

        if not paused:
            i += 1
            if i >= len(ks):
                if loop:
                    i = 0
                else:
                    running = False
    pygame.quit()
