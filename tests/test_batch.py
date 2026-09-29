"""Per-car parameter batches: B cars with different numeric parameters stepped together must give
exactly the results of stepping each car on its own (sim.vehicle.compile_model_batch, VehicleBatch)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_drift_sim.sim.params import TireCondition, default_params
from rc_drift_sim.sim.vehicle import Vehicle, VehicleBatch


def _cars():
    rng = np.random.default_rng(7)
    pairs = [("hard_plastic_drift", "epoxy_ptile"), ("rubber_onroad", "dry_asphalt"), ("foam", "carpet"),
             ("offroad_pin", "sand"), ("hard_plastic_drift", "ice"), ("rubber_onroad", "loose_dirt")]
    params, conds = [], []
    for k, (tn, sn) in enumerate(pairs):
        p = default_params(tn, sn)
        p = dataclasses.replace(
            p, vehicle=dataclasses.replace(p.vehicle, mass=1.4 + 0.1 * k, ackermann=0.2 * k, cg_height=0.03 + 0.002 * k),
            drivetrain=dataclasses.replace(p.drivetrain, gear_ratio=7.0 + k, drag_brake=0.1 * k,
                                           rear_diff_lock=1.0 - 0.15 * k),
            actuators=dataclasses.replace(p.actuators, gyro_enabled=bool(k % 2), gyro_gain=0.1 * k,
                                          servo_tau=0.01 + 0.005 * k))
        params.append(p)
        conds.append(TireCondition(wear=rng.uniform(0, 1, 4), wetness=rng.uniform(0, 1, 4),
                                   contamination=rng.uniform(0, 1, 4), temp0=rng.uniform(10, 70, 4)))
    return params, conds


def test_batch_of_different_cars_equals_individual_rollouts():
    params, conds = _cars()
    batch = VehicleBatch(params, conds=conds)
    rng = np.random.default_rng(3)
    B, T = len(params), 40
    v0 = rng.uniform(0.0, 3.0, B)
    beta0 = rng.uniform(-0.3, 0.3, B)
    s = batch.initial_states(v=v0, beta=beta0)
    acts = np.clip(rng.normal(0, 0.5, (T, B, 2)) + [0.0, 0.3], -1, 1)
    states = [s]
    for k in range(T):
        s, _ = batch.step(s, acts[k])
        states.append(s)
    states = np.stack(states)
    for b in range(B):
        car = Vehicle(params[b], cond=conds[b])
        s0 = car.initial_state(v=v0[b], beta=beta0[b])
        np.testing.assert_array_equal(states[0, b], s0, err_msg=f"initial state of car {b}")
        single = car.rollout(s0, acts[:, b]).states
        np.testing.assert_array_equal(states[:, b], single, err_msg=f"car {b} ({params[b].tire.name})")


def test_batch_info_matches_single_car_info():
    params, conds = _cars()
    batch = VehicleBatch(params, conds=conds)
    s = batch.initial_states(v=2.0, beta=-0.2, yaw_rate=1.0)
    u = np.tile([0.3, 0.4], (len(params), 1))
    _, info = batch.derivatives(s, u)
    for b in range(len(params)):
        _, info1 = Vehicle(params[b], cond=conds[b]).derivatives(s[b], u[b])
        for key in ("Fx", "Fy", "Fz", "ax", "ay", "mu_y", "delta_target", "T_motor"):
            np.testing.assert_array_equal(np.asarray(info[key])[b], info1[key], err_msg=f"{key} car {b}")


@pytest.mark.parametrize("group,field,value", [("drivetrain", "layout", "awd_spool"),
                                                ("tire", "combined_mode", "ellipse"),
                                                ("sim", "dt", 0.0005)])
def test_structural_settings_must_be_shared(group, field, value):
    p = default_params()
    q = dataclasses.replace(p, **{group: dataclasses.replace(getattr(p, group), **{field: value})})
    with pytest.raises(ValueError, match="shared"):
        VehicleBatch([p, q])


def test_large_batch_is_fast_and_finite():
    params, conds = _cars()
    reps = 200
    batch = VehicleBatch([params[k % len(params)] for k in range(reps)],
                         conds=[conds[k % len(conds)] for k in range(reps)], check_stiffness=False)
    s = batch.initial_states(v=1.5)
    u = np.tile([0.2, 0.3], (reps, 1))
    for _ in range(5):
        s, _ = batch.step(s, u)
    assert np.all(np.isfinite(s))
