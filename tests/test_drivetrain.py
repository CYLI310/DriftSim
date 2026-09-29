"""Drivetrain and actuator tests (DESIGN.md sections 6, 7; section 10 item 9).

The drivetrain is exercised on its own with a small forward-Euler loop over the wheel speeds and
the motor current (dt = 0.1 ms, far below every time constant), so the tests check the formulas
and not the vehicle integration.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from rc_drift_sim.sim import actuators, drivetrain as dtm
from rc_drift_sim.sim.params import ActuatorParams, DrivetrainParams, VehicleParams

VP = VehicleParams()


def _spin(dp: DrivetrainParams, thr: float, tau: np.ndarray, T: float, omega0=None, dt=1e-4):
    """Integrate current and wheel speeds with fixed resisting torques ``tau`` (4,)."""
    omega = np.zeros(4) if omega0 is None else np.asarray(omega0, dtype=float).copy()
    i = 0.0
    i_peak = 0.0
    for _ in range(int(round(T / dt))):
        om_m = dtm.shaft_speed(dp, omega)
        di = dtm.current_derivative(dp, i, om_m, thr)
        Tm = dtm.motor_torque(dp, i, om_m)
        omega = omega + dt * dtm.wheel_accelerations(dp, VP, omega, Tm, tau)
        i = i + dt * di
        i_peak = max(i_peak, abs(i))
    return omega, i, i_peak


# ----------------------------------------------------------------------------- gearing and diffs
def test_gear_ratios_per_layout():
    base = DrivetrainParams(gear_ratio=8.0, overdrive_front=1.25)
    G, drv = dtm.gear_ratios(dataclasses.replace(base, layout="rwd"))
    assert list(G) == [0, 0, 8, 8] and list(drv) == [0, 0, 1, 1]
    G, drv = dtm.gear_ratios(dataclasses.replace(base, layout="awd_spool"))
    assert list(G) == [8, 8, 8, 8] and list(drv) == [1, 1, 1, 1]
    G, _ = dtm.gear_ratios(dataclasses.replace(base, layout="awd_overdrive"))
    assert G[0] == pytest.approx(8.0 / 1.25) and G[2] == 8.0
    with pytest.raises(ValueError):
        dtm.gear_ratios(dataclasses.replace(base, layout="fwd"))


def test_reflected_motor_inertia_counted_once():
    for layout in ("rwd", "awd_spool", "awd_overdrive"):
        dp = DrivetrainParams(layout=layout, overdrive_front=1.2)
        G, drv = dtm.gear_ratios(dp)
        I = dtm.wheel_inertias(dp, VP)
        # kinetic energy check: all driven wheels at speed w/G_i give the motor speed w
        w = 100.0
        e_extra = 0.5 * np.sum((I - VP.wheel_inertia) * (drv * w / np.where(G > 0, G, 1.0)) ** 2)
        assert e_extra == pytest.approx(0.5 * dp.motor_inertia * w ** 2, rel=1e-12), layout


def test_locked_rear_diff_keeps_wheels_together_open_diff_does_not():
    """Item 9: locked (spool) vs open vs limited-slip under asymmetric resisting torque."""
    tau = np.array([0.0, 0.0, 0.01, 0.03])
    locked = DrivetrainParams(rear_diff_lock=1.0, rear_diff_visc=0.02)
    open_ = DrivetrainParams(rear_diff_lock=0.0, rear_diff_visc=0.0)
    lsd = DrivetrainParams(rear_diff_lock=0.5, rear_diff_max_torque=0.005, rear_diff_visc=0.0)
    w_l, _, _ = _spin(locked, 0.3, tau, 0.5)
    w_o, _, _ = _spin(open_, 0.3, tau, 0.5)
    w_s, _, _ = _spin(lsd, 0.3, tau, 0.5)
    d_l, d_o, d_s = abs(w_l[2] - w_l[3]), abs(w_o[2] - w_o[3]), abs(w_s[2] - w_s[3])
    print(f"\n|omega_RL - omega_RR| after 0.5 s: locked {d_l:.2e}, LSD {d_s:.2f}, open {d_o:.2f} rad/s")
    assert d_l < 1e-6
    assert d_o > 10.0
    assert d_l < d_s < d_o


def test_axle_torque_split_formulas():
    TL, TR = dtm.axle_torques(1.0, 0.2, 0.1, 5.0, 5.0, 0.0, 1e9, 0.0)
    assert TL == pytest.approx(0.5) and TR == pytest.approx(0.5), "open diff splits equally"
    TL, TR = dtm.axle_torques(1.0, 0.2, 0.1, 5.0, 5.0, 1.0, 1e9, 0.0)
    assert (TL - 0.2) == pytest.approx(TR - 0.1), "locked diff: equal net torque (equal accelerations)"
    TL, TR = dtm.axle_torques(1.0, 0.2, 0.1, 5.0, 5.0, 1.0, 0.01, 0.0)
    assert TL - TR == pytest.approx(0.02), "LSD transfers at most max_torque per side"


def test_awd_overdrive_speed_ratio():
    dp = DrivetrainParams(layout="awd_overdrive", overdrive_front=1.2)
    w, _, _ = _spin(dp, 0.4, np.zeros(4), 0.4)
    ratio = w[:2].mean() / w[2:].mean()
    print(f"\nfront/rear wheel speed ratio {ratio:.4f} (target 1.2)")
    assert ratio == pytest.approx(1.2, rel=0.01)


def test_rwd_front_wheels_are_free():
    dp = DrivetrainParams(layout="rwd")
    acc = dtm.wheel_accelerations(dp, VP, np.full(4, 50.0), 0.1, np.array([0.001, 0.002, 0.0, 0.0]))
    I = dtm.wheel_inertias(dp, VP)
    assert acc[0] == pytest.approx(-0.001 / I[0]) and acc[1] == pytest.approx(-0.002 / I[1])


# ----------------------------------------------------------------------------- motor + ESC
def test_motor_no_load_speed_and_current_limit():
    dp = DrivetrainParams()
    w, i, i_peak = _spin(dp, 1.0, np.zeros(4), 1.5)
    omega_m = dtm.shaft_speed(dp, w)
    v_nl = dtm.battery_voltage(dp, i, 1.0) / dp.ke
    print(f"\nno-load motor speed {omega_m:.0f} rad/s (V_batt/Ke = {v_nl:.0f}), peak current {i_peak:.1f} A")
    assert omega_m == pytest.approx(v_nl, rel=0.02)
    assert i_peak <= dp.current_limit * 1.05


def test_battery_sags_under_load():
    dp = DrivetrainParams()
    assert dtm.battery_voltage(dp, 30.0, 1.0) < dtm.battery_voltage(dp, 0.0, 1.0) == dp.battery_voltage


def test_neutral_coasts_and_drag_brake_decelerates():
    w0 = np.full(4, 100.0)
    coast, i_c, _ = _spin(DrivetrainParams(drag_brake=0.0), 0.0, np.zeros(4), 0.3, omega0=w0)
    brake, _, _ = _spin(DrivetrainParams(drag_brake=1.0), 0.0, np.zeros(4), 0.3, omega0=w0)
    print(f"\nrear wheel speed after 0.3 s from 100 rad/s: coast {coast[2]:.1f}, drag brake {brake[2]:.1f}")
    assert abs(i_c) < 1.0
    assert brake[2] < coast[2] - 20.0
    assert coast[0] == pytest.approx(100.0), "undriven fronts keep rolling with no resisting torque"


def test_brake_only_esc_never_reverses():
    dp = DrivetrainParams(reverse_enabled=False)
    w_brake, _, _ = _spin(dp, -0.5, np.zeros(4), 0.2, omega0=np.full(4, 100.0))
    assert w_brake[2] < 100.0, "negative throttle must brake"
    w_rest, i_rest, _ = _spin(dp, -1.0, np.zeros(4), 0.3)
    assert np.all(w_rest >= -1e-9) and abs(i_rest) < 1e-9, "a brake-only ESC must not drive backwards"
    w_rev, _, _ = _spin(DrivetrainParams(reverse_enabled=True), -0.5, np.zeros(4), 0.3)
    assert w_rev[2] < -10.0, "with reverse enabled, negative throttle drives backwards"


def test_motor_torque_zero_at_rest():
    assert dtm.motor_torque(DrivetrainParams(), 0.0, 0.0) == 0.0


def test_drivetrain_functions_are_batch_consistent():
    rng = np.random.default_rng(3)
    for layout in ("rwd", "awd_spool", "awd_overdrive"):
        dp = DrivetrainParams(layout=layout, overdrive_front=1.15, rear_diff_lock=0.4,
                              rear_diff_max_torque=0.03, front_diff_lock=0.0)
        om = rng.uniform(-50, 300, (64, 4))
        tau = rng.uniform(-0.1, 0.1, (64, 4))
        Tm = rng.uniform(-0.2, 0.2, 64)
        batched = dtm.wheel_accelerations(dp, VP, om, Tm, tau)
        single = np.stack([dtm.wheel_accelerations(dp, VP, om[k], Tm[k], tau[k]) for k in range(64)])
        np.testing.assert_allclose(batched, single, rtol=1e-13, atol=1e-10)


# ----------------------------------------------------------------------------- actuators
def test_action_delay_counts_whole_steps():
    d = actuators.ActionDelay(0.04, 0.02)
    assert d.n_delay == 2
    d.reset([0.0, 0.0])
    outs = [d.push([k, -k]) for k in range(1, 6)]
    assert [o[0] for o in outs] == [0.0, 0.0, 1.0, 2.0, 3.0]
    assert actuators.ActionDelay(0.05, 0.02).n_delay == 3, "rounds half up"
    z = actuators.ActionDelay(0.0, 0.02)
    assert z.push([0.3, 0.4])[0] == 0.3


def test_steering_rate_limit_and_saturation():
    ap = ActuatorParams()
    assert actuators.steering_rate(ap, 0.0, 1.0) == pytest.approx(ap.servo_rate)
    assert actuators.steering_rate(ap, 0.0, -1.0) == pytest.approx(-ap.servo_rate)
    small = actuators.steering_rate(ap, 0.0, 1e-4)
    assert small == pytest.approx(1e-4 / ap.servo_tau)
    assert actuators.steering_target(ap, 5.0, 0.0) == pytest.approx(ap.steer_max)


def test_steering_target_is_odd_and_has_deadband():
    ap = ActuatorParams(steer_deadband=0.05, steer_expo=0.3)
    x = np.linspace(-1, 1, 201)
    np.testing.assert_allclose(actuators.steering_target(ap, x, 0.0), -actuators.steering_target(ap, -x, 0.0), atol=1e-15)
    assert actuators.steering_target(ap, 0.04, 0.0) == 0.0


def test_gyro_counter_steers_against_yaw_rate():
    ap = ActuatorParams(gyro_enabled=True, gyro_gain=0.2, gyro_max_correction=math.radians(20))
    assert actuators.steering_target(ap, 0.0, 1.0) == pytest.approx(-0.2), "left yaw -> steer right"
    assert actuators.steering_target(ap, 0.0, -1.0) == pytest.approx(0.2)
    assert actuators.steering_target(ap, 0.0, 100.0) == pytest.approx(-math.radians(20)), "clipped"
    off = dataclasses.replace(ap, gyro_enabled=False)
    assert actuators.steering_target(off, 0.0, 1.0) == 0.0


def test_ackermann_geometry():
    vp_full = dataclasses.replace(VP, ackermann=1.0)
    for d in (0.2, -0.2):
        fl, fr = actuators.ackermann_angles(vp_full, d)
        inner, outer = (fl, fr) if d > 0 else (fr, fl)
        assert abs(inner) > abs(outer), f"inner wheel must steer more (delta = {d})"
        # full Ackermann: both wheel axes meet on the rear-axle line
        L, hw = VP.wheelbase, VP.track_width / 2
        R_center = L / math.tan(abs(d))
        assert L / math.tan(abs(inner)) == pytest.approx(R_center - hw, rel=1e-9)
        assert L / math.tan(abs(outer)) == pytest.approx(R_center + hw, rel=1e-9)
    fl, fr = actuators.ackermann_angles(dataclasses.replace(VP, ackermann=0.0), 0.3)
    assert fl == fr == 0.3
    assert actuators.ackermann_angles(vp_full, 0.0) == (0.0, 0.0)
