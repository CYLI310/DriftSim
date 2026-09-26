"""Time-series, tire-curve and g-g plots for the RC drift simulator (docs/DESIGN.md section 9).

Trajectory layout (see ``vehicle.Vehicle.rollout``): ``traj.t`` (T+1,) s, ``traj.states``
(T+1, NS), ``traj.actions`` (T, 2) = (steer_cmd, throttle_cmd) in [-1, 1], ``traj.info`` a dict
of stacked per-step arrays with leading dim T (per-wheel arrays (T, 4), FL FR RL RR). Info
entry ``k`` is plotted at ``t[k]`` (the start of control step k, the same index as
``actions[k]``). Missing info keys are drawn as empty (NaN) series so the figure still renders
while other modules are under construction.

Matplotlib uses the Agg backend (no display) unless ``MPLBACKEND`` is set in the environment.

Public API:
    plot_timeseries(traj, params, out_path=None, show=False) -> Figure   (4 x 2 panels)
    plot_tire_curves(tire_params, surface_params_list, out_path=None, Fz=4.0) -> Figure
    plot_gg(traj, out_path=None) -> Figure
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rc_drift_sim.sim import state as S  # noqa: E402
from rc_drift_sim.sim.params import (  # noqa: E402
    Params, SurfaceParams, TireCondition, TireParams, WHEEL_NAMES,
)

WHEEL_COLORS: tuple[str, ...] = ("tab:blue", "tab:orange", "tab:green", "tab:red")
WHEEL_LINESTYLES: tuple[str, ...] = ("-", "-", "--", "--")   # fronts solid, rears dashed

__all__ = ["plot_timeseries", "plot_tire_curves", "plot_gg", "WHEEL_COLORS", "WHEEL_LINESTYLES"]


# ----------------------------------------------------------------------------- helpers
def _states(traj: Any) -> np.ndarray:
    return np.asarray(traj.states, dtype=float)


def _times(traj: Any) -> np.ndarray:
    return np.asarray(traj.t, dtype=float).reshape(-1)


def _n_steps(traj: Any) -> int:
    """Number of control steps T (leading dim of the info arrays / actions)."""
    actions = getattr(traj, "actions", None)
    if actions is not None:
        a = np.asarray(actions)
        if a.ndim >= 1 and a.shape[0] > 0:
            return int(a.shape[0])
    return int(_states(traj).shape[0] - 1)


def _info_array(info: Mapping[str, Any] | None, key: str, T: int,
                width: int | None = None) -> np.ndarray:
    """info[key] as a float array; a NaN array of shape (T,) or (T, width) when absent."""
    if info is not None and key in info and info[key] is not None:
        return np.asarray(info[key], dtype=float)
    return np.full((T,) if width is None else (T, width), np.nan)


def _t_for(t: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Time stamps for an array with a leading time axis of length <= len(t)."""
    return t[:arr.shape[0]]


def _plot_wheels(ax: plt.Axes, t: np.ndarray, y: np.ndarray, ylabel: str, title: str) -> None:
    y = np.asarray(y, dtype=float)
    tt = _t_for(t, y)
    for i in range(4):
        ax.plot(tt, y[:, i], color=WHEEL_COLORS[i], linestyle=WHEEL_LINESTYLES[i],
                linewidth=1.2, label=WHEEL_NAMES[i])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=4)


def _twin_legend(ax: plt.Axes, tw: plt.Axes) -> None:
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _finish(fig: plt.Figure, out_path: str | Path | None, show: bool, dpi: int) -> plt.Figure:
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=min(int(dpi), 100))
    if show:  # no-op under the Agg backend
        plt.show()
    return fig


# ----------------------------------------------------------------------------- time series
def plot_timeseries(traj: Any, params: Params, out_path: str | Path | None = None,
                    show: bool = False, *, figsize: tuple[float, float] = (13.0, 12.0),
                    dpi: int = 100) -> plt.Figure:
    """4 x 2 time-series panels of a trajectory, sharing the time axis.

    Panels: speed [m/s] and sideslip [deg]; yaw rate [deg/s] and lateral accel [m/s^2];
    steering angle vs delta_target vs command (all deg, command scaled by steer_max);
    throttle command / ESC throttle and motor current [A]; per-wheel slip ratio; per-wheel slip
    angle [deg]; per-wheel Fz [N]; per-wheel tire temperature [degC]. Wheel legends FL FR RL RR.
    Saves to ``out_path`` when given, returns the Figure.
    """
    st = _states(traj)
    t = _times(traj)
    T = _n_steps(traj)
    info = getattr(traj, "info", None)
    actions = np.asarray(traj.actions, dtype=float).reshape(-1, 2) if getattr(traj, "actions", None) is not None \
        else np.full((T, 2), np.nan)
    ta = _t_for(t, actions)

    fig, axes = plt.subplots(4, 2, figsize=figsize, dpi=min(int(dpi), 100), sharex=True,
                             constrained_layout=True)

    # --- speed & sideslip -------------------------------------------------------------
    ax = axes[0, 0]
    speed = np.hypot(st[:, S.VX], st[:, S.VY])
    beta_deg = np.degrees(np.arctan2(st[:, S.VY], st[:, S.VX]))
    ax.plot(t, speed, color="tab:blue", linewidth=1.4, label="speed [m/s]")
    ax.set_ylabel("speed [m/s]", color="tab:blue")
    tw = ax.twinx()
    tw.plot(t, beta_deg, color="tab:red", linewidth=1.2, label="sideslip beta [deg]")
    tw.set_ylabel("beta [deg]", color="tab:red")
    tw.axhline(0.0, color="tab:red", linewidth=0.5, alpha=0.4)
    ax.set_title("speed and sideslip", fontsize=10)
    ax.grid(True, alpha=0.3)
    _twin_legend(ax, tw)

    # --- yaw rate & lateral accel -----------------------------------------------------
    ax = axes[0, 1]
    ax.plot(t, np.degrees(st[:, S.R]), color="tab:blue", linewidth=1.4, label="yaw rate [deg/s]")
    ax.set_ylabel("r [deg/s]", color="tab:blue")
    tw = ax.twinx()
    ay = _info_array(info, "ay", T)
    tw.plot(_t_for(t, ay), ay, color="tab:red", linewidth=1.2, label="a_y [m/s^2]")
    tw.set_ylabel("a_y [m/s^2]", color="tab:red")
    ax.set_title("yaw rate and lateral acceleration", fontsize=10)
    ax.grid(True, alpha=0.3)
    _twin_legend(ax, tw)

    # --- steering -------------------------------------------------------------------
    ax = axes[1, 0]
    steer_max_deg = np.degrees(params.actuators.steer_max)
    ax.step(ta, actions[:, 0] * steer_max_deg, where="post", color="tab:gray", linewidth=1.0,
            label="command * steer_max [deg]")
    dt_deg = np.degrees(_info_array(info, "delta_target", T))
    ax.plot(_t_for(t, dt_deg), dt_deg, color="tab:orange", linewidth=1.2, linestyle="--",
            label="delta_target [deg]")
    ax.plot(t, np.degrees(st[:, S.DELTA]), color="tab:blue", linewidth=1.4, label="delta [deg]")
    ax.set_ylabel("steer [deg]")
    ax.set_title("steering angle vs target vs command", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # --- throttle & current -----------------------------------------------------------
    ax = axes[1, 1]
    ax.step(ta, actions[:, 1], where="post", color="tab:gray", linewidth=1.0,
            label="throttle cmd [-]")
    thr = _info_array(info, "thr", T)
    ax.plot(_t_for(t, thr), thr, color="tab:blue", linewidth=1.2, label="ESC throttle [-]")
    ax.set_ylabel("throttle [-]")
    ax.set_ylim(-1.1, 1.1)
    tw = ax.twinx()
    tw.plot(t, st[:, S.I_MOTOR], color="tab:red", linewidth=1.2, label="motor current [A]")
    tw.set_ylabel("i [A]", color="tab:red")
    ax.set_title("throttle and motor current", fontsize=10)
    ax.grid(True, alpha=0.3)
    _twin_legend(ax, tw)

    # --- per-wheel panels -------------------------------------------------------------
    _plot_wheels(axes[2, 0], t, _info_array(info, "kappa", T, 4), "kappa [-]", "slip ratio")
    _plot_wheels(axes[2, 1], t, np.degrees(_info_array(info, "alpha", T, 4)), "alpha [deg]",
                 "slip angle")
    _plot_wheels(axes[3, 0], t, _info_array(info, "Fz", T, 4), "Fz [N]", "wheel load")
    _plot_wheels(axes[3, 1], t, st[:, S.T_TIRE], "T [degC]", "tire temperature")

    for a in axes[-1]:
        a.set_xlabel("t [s]")
    fig.suptitle(f"{params.tire.name} on {params.surface.name}  "
                 f"(drivetrain {params.drivetrain.layout}, {params.sim.integrator})", fontsize=11)
    return _finish(fig, out_path, show, dpi)


# ----------------------------------------------------------------------------- tire curves
def plot_tire_curves(tire_params: TireParams | Params,
                     surface_params_list: Iterable[SurfaceParams] | Mapping[str, SurfaceParams] | SurfaceParams,
                     out_path: str | Path | None = None, Fz: float = 4.0, *,
                     T_tire: float = 25.0, kappa_max: float = 1.0, alpha_max_deg: float = 45.0,
                     n: int = 401, figsize: tuple[float, float] = (11.0, 4.5),
                     dpi: int = 100) -> plt.Figure:
    """Pure-slip Fx(kappa) and Fy(alpha) of one compound on several surfaces at load ``Fz`` [N].

    Uses ``rc_drift_sim.sim.tire`` (imported lazily): ``tire_coefficients`` with a neutral
    ``TireCondition`` at ``T_tire`` [degC] and ``pure_slip_forces``. Fy follows the contract's
    ISO sign (Fy = -MF_y(alpha): positive slip angle gives a negative, rightward force).
    ``tire_params`` may be a TireParams or a Params bundle; ``surface_params_list`` an iterable
    or name->SurfaceParams mapping. Saves to ``out_path`` when given, returns the Figure.
    """
    from rc_drift_sim.sim import tire as tire_mod
    from rc_drift_sim.sim.surface import uniform_surface

    tp: TireParams = tire_params.tire if isinstance(tire_params, Params) else tire_params
    if isinstance(surface_params_list, SurfaceParams):
        surfaces = [surface_params_list]
    elif isinstance(surface_params_list, Mapping):
        surfaces = list(surface_params_list.values())
    else:
        surfaces = list(surface_params_list)

    kappa = np.linspace(-kappa_max, kappa_max, int(n))
    alpha = np.radians(np.linspace(-alpha_max_deg, alpha_max_deg, int(n)))
    zeros = np.zeros_like(kappa)
    Fz4 = np.full(4, float(Fz))
    T4 = np.full(4, float(T_tire))
    cond = TireCondition.neutral(float(T_tire))

    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=figsize, dpi=min(int(dpi), 100),
                                     constrained_layout=True)
    for sp in surfaces:
        coef = tire_mod.tire_coefficients(tp, uniform_surface(sp), cond, Fz4, T4)
        fx = tire_mod.pure_slip_forces(coef, kappa[:, None], zeros[:, None])[0][:, 0]
        fy = tire_mod.pure_slip_forces(coef, zeros[:, None], alpha[:, None])[1][:, 0]
        ax_x.plot(kappa, fx, linewidth=1.4, label=sp.name)
        ax_y.plot(np.degrees(alpha), fy, linewidth=1.4, label=sp.name)

    ax_x.set_xlabel("slip ratio kappa [-]")
    ax_x.set_ylabel("Fx [N]")
    ax_x.set_title(f"longitudinal, pure slip (alpha = 0), Fz = {Fz:g} N", fontsize=10)
    ax_y.set_xlabel("slip angle alpha [deg]")
    ax_y.set_ylabel("Fy [N]  (ISO: Fy = -MF_y(alpha))")
    ax_y.set_title(f"lateral, pure slip (kappa = 0), Fz = {Fz:g} N", fontsize=10)
    for a in (ax_x, ax_y):
        a.grid(True, alpha=0.3)
        a.axhline(0.0, color="k", linewidth=0.5)
        a.axvline(0.0, color="k", linewidth=0.5)
        a.legend(fontsize=8)
    fig.suptitle(f"tire '{tp.name}' ({tp.combined_mode})", fontsize=11)
    return _finish(fig, out_path, False, dpi)


# ----------------------------------------------------------------------------- g-g
def plot_gg(traj: Any, out_path: str | Path | None = None, *, g: float = 9.81,
            figsize: tuple[float, float] = (6.0, 6.0), dpi: int = 100) -> plt.Figure:
    """g-g diagram: body-frame lateral (x axis) vs longitudinal (y axis) acceleration in g.

    Uses ``info['ax']`` and ``info['ay']`` [m/s^2] (total specific force incl. aero), colored by
    time, with friction-circle reference rings every 0.25 g. Saves when ``out_path`` is given.
    """
    t = _times(traj)
    T = _n_steps(traj)
    info = getattr(traj, "info", None)
    ax_ = _info_array(info, "ax", T) / g
    ay_ = _info_array(info, "ay", T) / g
    tt = _t_for(t, ax_)

    fig, ax = plt.subplots(figsize=figsize, dpi=min(int(dpi), 100), constrained_layout=True)
    sc = ax.scatter(ay_, ax_, c=tt, s=6, cmap="viridis", alpha=0.8, linewidths=0)
    r = np.hypot(ax_, ay_)
    rmax = float(np.nanmax(np.concatenate([r, [0.5]])))
    ring_max = float(np.ceil(rmax / 0.25) * 0.25)
    th = np.linspace(0.0, 2.0 * np.pi, 361)
    for rr in np.arange(0.25, ring_max + 1e-9, 0.25):
        ax.plot(rr * np.cos(th), rr * np.sin(th), color="gray", linewidth=0.6, alpha=0.6)
        ax.annotate(f"{rr:.2f} g", xy=(rr * np.cos(np.pi / 4), rr * np.sin(np.pi / 4)),
                    fontsize=7, color="gray")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.axvline(0.0, color="k", linewidth=0.5)
    lim = ring_max + 0.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lateral a_y [g]  (+ = left)")
    ax.set_ylabel("longitudinal a_x [g]  (+ = forward)")
    ax.set_title("g-g diagram", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03, label="t [s]")
    return _finish(fig, out_path, False, dpi)
