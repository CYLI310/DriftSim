"""Shared fixtures and helpers for the pytest suite (docs/DESIGN.md section 10).

Fixtures
--------
params   : Params                 ``default_params()`` -> hard_plastic_drift tire on epoxy_ptile
vehicle  : Vehicle                ``Vehicle(params)`` (fresh per test)
tires    : dict[str, TireParams]  ``load_tires()``
surfaces : dict[str, SurfaceParams] ``load_surfaces()``
run      : callable               ``run(vehicle, s0, action_fn_or_array, duration) -> Trajectory``

Plain helper functions live in ``tests/helpers.py``.
"""
from __future__ import annotations

import pytest

from rc_drift_sim.sim.params import Params, default_params, load_surfaces, load_tires
from tests.helpers import run_rollout


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
