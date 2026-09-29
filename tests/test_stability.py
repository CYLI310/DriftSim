"""Integrator stiffness guard: every shipped configuration must be stable for its integrator.

A too-stiff coupling (e.g. a viscous center coupling that acts x G^2 at the wheels) makes RK4 at
1 ms diverge to NaN within a fraction of a second. ``sim.stability`` estimates the fast rates from
the parameters; this test sweeps every compound x surface x drivetrain layout.
"""
from __future__ import annotations

import dataclasses
import itertools
import warnings

import numpy as np
import pytest

from rc_drift_sim.sim.params import default_params, load_surfaces, load_tires
from rc_drift_sim.sim.stability import check_stiffness, stiffness_report
from rc_drift_sim.sim.vehicle import Vehicle


def test_every_shipped_configuration_is_stable():
    worst = []
    for tn, sn, layout in itertools.product(load_tires(), load_surfaces(), ("rwd", "awd_spool", "awd_overdrive")):
        p = default_params(tn, sn)
        p = dataclasses.replace(p, drivetrain=dataclasses.replace(p.drivetrain, layout=layout, overdrive_front=1.2))
        rep = stiffness_report(p)
        worst.append((max(rep.margin(k) for k in rep.rates), tn, sn, layout, rep.worst[0]))
        assert rep.ok, f"{tn}/{sn}/{layout}:\n{rep.table()}"
    worst.sort(reverse=True)
    print("\nclosest to the stability limit (margin < 1 is stable):")
    for m, tn, sn, layout, mode in worst[:3]:
        print(f"  {m:.2f}  {tn}/{sn}/{layout} ({mode})")


def test_too_stiff_coupling_is_caught_before_it_blows_up():
    p = default_params()
    stiff = dataclasses.replace(p, drivetrain=dataclasses.replace(p.drivetrain, layout="awd_spool", center_visc=0.02))
    with pytest.raises(ValueError, match="center coupling"):
        check_stiffness(stiff, strict=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = Vehicle(stiff)
    assert any("too stiff" in str(x.message) for x in w), "Vehicle() should warn about a stiff config"
    with np.errstate(all="ignore"):
        traj = v.rollout(v.initial_state(), np.tile([[0.4, 1.0]], (20, 1)))
    assert not np.all(np.isfinite(traj.states)), "the flagged config really does diverge"
