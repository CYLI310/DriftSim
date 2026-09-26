"""Shared fixtures and helpers for the pytest suite (docs/DESIGN.md section 10).

Fixtures
--------
params   : Params                 ``default_params()`` -> hard_plastic_drift tire on epoxy_ptile
vehicle  : Vehicle                ``Vehicle(params)`` (fresh per test)
tires    : dict[str, TireParams]  ``load_tires()``
surfaces : dict[str, SurfaceParams] ``load_surfaces()``
run      : callable               ``run(vehicle, s0, action_fn_or_array, duration) -> Trajectory``

Plain helper functions (importable as ``from rc_drift_sim.tests.conftest import ...``):
``run_rollout``, ``traj_states``, ``traj_speed``, ``traj_beta``, ``traj_dt``,
``assert_all_finite``, ``longest_true_run``.

All units SI (m, s, rad); sideslip beta = atan2(vy, vx) as in state.sideslip.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from rc_drift_sim.sim import state as S
from rc_drift_sim.sim.params import Params, default_params, load_surfaces, load_tires


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def params() -> Params:
    """Default parameter bundle: hard-plastic drift tire on the epoxy/P-tile track."""
    return default_params()


@pytest.fixture
def vehicle(params):
    """A fresh Vehicle wrapper around the default params (contract section 5)."""
    from rc_drift_sim.sim.vehicle import Vehicle
    return Vehicle(params)


@pytest.fixture(scope="session")
def tires():
    """Every tire compound from configs/tires.yaml, keyed by name."""
    return load_tires()


@pytest.fixture(scope="session")
def surfaces():
    """Every surface from configs/surfaces.yaml, keyed by name."""
    return load_surfaces()


@pytest.fixture
def run():
    """``run(vehicle, s0, action_fn_or_array, duration) -> Trajectory`` (see run_rollout)."""
    return run_rollout


# ----------------------------------------------------------------------------- helpers
def run_rollout(vehicle, s0, actions, duration: float):
    """Roll the vehicle out for ``duration`` seconds at the vehicle's control period.

    ``actions`` may be a callable ``(t, s, info) -> (steer, throttle)``, a (2,) constant action,
    or a (T, 2) array; a too-short array is extended by holding its last row, a too-long one is
    truncated. Returns the Trajectory produced by ``Vehicle.rollout`` (t, states, actions, info).
    """
    control_dt = vehicle.control_dt
    n_steps = int(round(duration / control_dt))
    if callable(actions):
        act: Callable | np.ndarray = actions
    else:
        arr = np.asarray(actions, dtype=float)
        if arr.ndim == 1:
            arr = np.broadcast_to(arr.reshape(1, 2), (n_steps, 2)).copy()
        elif arr.shape[0] < n_steps:
            pad = np.repeat(arr[-1:], n_steps - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:n_steps]
        act = np.clip(arr, -1.0, 1.0)
    return vehicle.rollout(np.asarray(s0, dtype=float), act, control_dt)


def traj_states(traj) -> np.ndarray:
    """(T+1, NS) float array of the trajectory states."""
    return np.asarray(traj.states, dtype=float)


def traj_dt(traj) -> float:
    """Control period of a trajectory from its time stamps (s)."""
    t = np.asarray(traj.t, dtype=float)
    return float(t[1] - t[0])


def traj_speed(traj) -> np.ndarray:
    """(T+1,) planar speed |v| = hypot(vx, vy) in m/s."""
    return S.speed(traj_states(traj))


def traj_beta(traj) -> np.ndarray:
    """(T+1,) vehicle sideslip beta = atan2(vy, vx) in rad (positive = sliding left of the nose)."""
    return S.sideslip(traj_states(traj))


def assert_all_finite(traj, what: str = "trajectory") -> None:
    """Fail with the first offending sample if any state (or action) is NaN or inf."""
    st = traj_states(traj)
    bad = ~np.isfinite(st)
    if bad.any():
        k, i = np.argwhere(bad)[0]
        raise AssertionError(
            f"{what}: non-finite state at sample {k} (t={np.asarray(traj.t)[k]:.3f} s), "
            f"entry {S.STATE_NAMES[i]!r}: {st[k, i]!r}")
    act = np.asarray(traj.actions, dtype=float)
    assert np.all(np.isfinite(act)), f"{what}: non-finite action in the rollout"


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """(start_index, length) of the longest run of consecutive True values (length 0 if none)."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return 0, 0
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, ends = edges[0::2], edges[1::2]
    lengths = ends - starts
    j = int(np.argmax(lengths))
    return int(starts[j]), int(lengths[j])
