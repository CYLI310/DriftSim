"""Vehicle model tests (DESIGN.md section 5; section 10 items 1, 2, 3, 4, 7, 8) plus batching,
per-wheel surfaces, drivetrain layouts, the gyro, tire temperature and contamination states.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_drift_sim.sim import drivetrain as dtm
from rc_drift_sim.sim import equilibrium as EQ
from rc_drift_sim.sim import state as S
from rc_drift_sim.sim import tire
from rc_drift_sim.sim.params import TireCondition
from rc_drift_sim.sim.surface import per_wheel, uniform_surface
from rc_drift_sim.sim.vehicle import Vehicle, derivatives, make_vehicle
from tests.helpers import assert_all_finite
from rc_drift_sim.control.maneuvers import constant_actions, lane_change_actions, open_loop_drift_actions


def _with(p, **groups):
    """Params with some fields of some groups replaced: _with(p, vehicle=dict(mass=2.0))."""
    return dataclasses.replace(p, **{g: dataclasses.replace(getattr(p, g), **kw) for g, kw in groups.items()})


# ----------------------------------------------------------------------------- item 1
def test_rest_stays_at_rest(vehicle, run):
    traj = run(vehicle, vehicle.initial_state(), np.zeros(2), 2.0)
    assert_all_finite(traj)
    st = traj.states
    assert np.max(np.abs(st[:, [S.X, S.Y, S.YAW, S.VX, S.VY, S.R]])) < 1e-9
    assert np.max(np.abs(st[:, S.OMEGA])) < 1e-9
    assert np.max(np.abs(st[:, S.T_TIRE] - vehicle.params.sim.ambient_temp)) < 1e-9


# ----------------------------------------------------------------------------- item 2
def test_straight_line_coasting_decays_correctly(vehicle, run):
    """Coasting from 3 m/s: every modeled loss (rolling resistance, aero, motor friction reflected
    through the gearing) over the effective mass (incl. wheel and rotor inertia) predicts the
    deceleration; the car decelerates monotonically and keeps a straight line."""
    p = vehicle.params
    vp, dp = p.vehicle, p.drivetrain
    traj = run(vehicle, vehicle.initial_state(v=3.0), np.zeros(2), 1.0)
    assert_all_finite(traj)
    sp = traj.speed
    assert np.all(np.diff(sp) <= 1e-12), "coasting must never gain speed"
    assert np.max(np.abs(traj.states[:, S.Y])) < 1e-9 and np.max(np.abs(traj.states[:, S.YAW])) < 1e-12

    m_eff = vp.mass + dtm.wheel_inertias(dp, vp).sum() / vp.wheel_radius ** 2

    def decel(v):
        om_m = dp.gear_ratio * v / vp.wheel_radius
        F = (float(np.asarray(p.surface.rolling_resistance)) * vp.mass * p.sim.gravity
             + 0.5 * vp.air_density * vp.drag_coeff_area * v ** 2
             + (dp.motor_friction_visc * om_m + dp.motor_friction_coulomb * np.tanh(om_m / 5.0))
             * dp.gear_ratio / vp.wheel_radius)
        return F / m_eff

    k = int(round(0.5 / vehicle.control_dt))
    measured = (sp[0] - sp[k]) / (traj.t[k] - traj.t[0])
    predicted = float(np.mean([decel(v) for v in sp[:k + 1]]))
    print(f"\ncoasting decel: measured {measured:.4f} m/s^2, predicted {predicted:.4f} m/s^2 "
          f"(effective mass {m_eff:.3f} kg)")
    assert measured == pytest.approx(predicted, rel=0.05)


# ----------------------------------------------------------------------------- item 3
@pytest.mark.parametrize("tire_name,surface", [("hard_plastic_drift", "epoxy_ptile"), ("foam", "carpet"),
                                               ("rubber_onroad", "ice"), ("offroad_pin", "sand")])
def test_low_speed_is_stable(tire_name, surface):
    v = make_vehicle(tire_name, surface)
    traj = v.rollout(v.initial_state(v=0.05), constant_actions(v.params, 3.0, 1.0, 0.0))
    assert_all_finite(traj)
    assert traj.speed.max() <= 0.05 + 1e-6, "speed must not grow when coasting at walking pace"
    assert np.max(np.abs(traj.states[:, S.OMEGA])) < 10.0
    # launch, full brake/reverse, then coast to a stop with full steer: no NaN, no explosion
    acts = np.concatenate([np.tile([[0.0, 1.0]], (40, 1)), np.tile([[1.0, -1.0]], (40, 1)),
                           np.tile([[1.0, 0.0]], (120, 1))])
    traj = v.rollout(v.initial_state(), acts)
    assert_all_finite(traj)
    assert np.max(np.abs(traj.states[:, [S.VX, S.VY]])) < 15.0


def test_zero_crossing_through_reverse_is_smooth():
    v = make_vehicle()
    acts = np.concatenate([np.tile([[0.0, 0.5]], (50, 1)), np.tile([[0.0, -0.6]], (150, 1))])
    traj = v.rollout(v.initial_state(), acts)
    assert_all_finite(traj)
    assert traj.states[-1, S.VX] < -0.5, "reverse must eventually drive backwards"
    assert np.max(np.abs(np.diff(traj.states[:, S.VX]))) < 0.5, "no velocity jumps at the zero crossing"


# ----------------------------------------------------------------------------- item 4
@pytest.mark.parametrize("cg_to_front,expect", [(0.090, "understeer"), (0.170, "oversteer")])
def test_steady_state_understeer_gradient_matches_bicycle_theory(params, cg_to_front, expect):
    """Trim solutions at low lateral acceleration (open rear diff, so the spool does not add its
    own yaw moment) must reproduce the linear bicycle-model understeer gradient
        K_us = m/L * (b/C_af - a/C_ar)       [rad per m/s^2]
    with axle cornering stiffnesses C = sum of the two wheels' Ky at static load. Measured as
    d(delta - L/R)/d(a_y) at a_y -> 0 (quadratic fit; lateral load transfer adds curvature)."""
    p = _with(params, vehicle=dict(cg_to_front=cg_to_front),
              drivetrain=dict(rear_diff_lock=0.0, rear_diff_visc=0.0))
    v = Vehicle(p)
    vp = p.vehicle
    c = tire.tire_coefficients(p.tire, v.surf, None, vp.static_loads(), None)
    Ky = np.asarray(c["Ky_eff"]) * np.ones(4)
    K_theory = vp.mass / vp.wheelbase * (vp.cg_to_rear / (Ky[0] + Ky[1]) - vp.cg_to_front / (Ky[2] + Ky[3]))
    ays = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    speed = 1.5
    trims = EQ.continuation(v, speed, ays / speed, key="r")
    assert all(t.success for t in trims), "trim solver failed"
    under = np.array([t.s[S.DELTA] - vp.wheelbase / t.radius for t in trims])
    K_meas = np.polyfit(ays, under, 2)[1]
    print(f"\n{expect}: K_us measured {K_meas * 1e3:+.3f} mrad/(m/s^2), bicycle theory {K_theory * 1e3:+.3f}")
    assert np.sign(K_meas) == np.sign(K_theory) == (1 if expect == "understeer" else -1)
    assert K_meas == pytest.approx(K_theory, rel=0.15)


def test_spool_adds_understeer(params):
    """A locked rear axle resists turning (inner and outer wheel forced to the same speed)."""
    slopes = {}
    for lock in (0.0, 1.0):
        v = Vehicle(_with(params, drivetrain=dict(rear_diff_lock=lock, rear_diff_visc=0.02 * lock)))
        trims = EQ.continuation(v, 1.5, np.array([0.2, 0.6, 1.0]) / 1.5, key="r")
        und = [t.s[S.DELTA] - params.vehicle.wheelbase / t.radius for t in trims]
        slopes[lock] = np.polyfit([0.2, 0.6, 1.0], und, 1)[0]
    print(f"\nundersteer slope: open {slopes[0.0] * 1e3:+.2f}, spool {slopes[1.0] * 1e3:+.2f} mrad/(m/s^2)")
    assert slopes[1.0] > slopes[0.0] + 5e-3


# ----------------------------------------------------------------------------- item 7
def test_rk4_and_semi_implicit_euler_agree_and_converge():
    """On a gentle lane change the two integrators agree within a few cm, and the semi-implicit
    Euler error shrinks at least linearly with dt, so RK4 at 1 kHz is the converged reference. (At
    1 ms Euler is not yet in its asymptotic regime - the light free front wheel is a fast, poorly
    damped mode for explicit Euler - which is why RK4 is the default.)"""
    base = make_vehicle()
    acts = lane_change_actions(base.params, 2.0, 0.3, 0.25)
    ref = base.rollout(base.initial_state(v=1.5), acts).states[-1]
    errs = {}
    for dt in (0.001, 0.0005):
        v = Vehicle(_with(base.params, sim=dict(integrator="semi_implicit_euler", dt=dt)))
        s = v.rollout(v.initial_state(v=1.5), acts).states[-1]
        errs[dt] = float(np.hypot(*(s[:2] - ref[:2])))
    half = Vehicle(_with(base.params, sim=dict(dt=0.0005)))
    rk4_half = float(np.hypot(*(half.rollout(half.initial_state(v=1.5), acts).states[-1][:2] - ref[:2])))
    print(f"\nEuler position error vs RK4: dt 1 ms {errs[0.001] * 100:.2f} cm, dt 0.5 ms {errs[0.0005] * 100:.2f} cm; "
          f"RK4 1 ms vs 0.5 ms {rk4_half * 1e6:.1f} um")
    assert errs[0.001] < 0.05
    assert errs[0.001] / errs[0.0005] > 1.6, "semi-implicit Euler must converge (at least first order)"
    assert rk4_half < 1e-4, "RK4 at 1 kHz must already be converged"


# ----------------------------------------------------------------------------- item 8
def test_mirror_symmetry():
    v = make_vehicle()
    acts = open_loop_drift_actions(v.params, 3.0)
    a = v.rollout(v.initial_state(), acts)
    b = v.rollout(v.initial_state(), acts * np.array([-1.0, 1.0]))
    sa, sb = a.states, b.states
    for idx, sign in ((S.X, 1), (S.Y, -1), (S.YAW, -1), (S.VX, 1), (S.VY, -1), (S.R, -1), (S.DELTA, -1)):
        np.testing.assert_allclose(sb[:, idx], sign * sa[:, idx], atol=1e-9, err_msg=S.STATE_NAMES[idx])
    np.testing.assert_allclose(sb[:, S.OMEGA], sa[:, S.OMEGA][:, [1, 0, 3, 2]], atol=1e-9)


# ----------------------------------------------------------------------------- batching
def test_batched_rollout_equals_individual_rollouts():
    v = make_vehicle()
    rng = np.random.default_rng(5)
    B, T = 6, 60
    acts = np.clip(rng.normal(0, 0.5, (T, B, 2)) + [0.0, 0.4], -1, 1)
    s0 = np.stack([v.initial_state(v=float(x)) for x in rng.uniform(0, 2, B)])
    batched, _ = v.rollout_batch(s0, acts)
    for b in range(B):
        single = v.rollout(s0[b], acts[:, b]).states
        np.testing.assert_allclose(batched[:, b], single, rtol=0, atol=1e-12)


def test_functional_and_compiled_derivatives_agree(params):
    v = Vehicle(params)
    s = v.initial_state(v=2.0, beta=-0.3, yaw_rate=1.5)
    ds1, i1 = derivatives(s, np.array([0.3, 0.5]), params, v.surf, v.cond)
    ds2, i2 = v.derivatives(s, np.array([0.3, 0.5]))
    np.testing.assert_array_equal(ds1, ds2)
    assert set(i1) == set(i2)


# ----------------------------------------------------------------------------- per-wheel surface
def test_split_mu_braking_yaws_toward_high_grip_side(params, surfaces):
    """Left wheels on ice, right wheels on asphalt: braking pulls the car toward the grippy side
    (right, negative yaw), a direct check that each wheel sees its own surface."""
    split = per_wheel([surfaces["ice"], surfaces["dry_asphalt"], surfaces["ice"], surfaces["dry_asphalt"]])
    v = Vehicle(_with(params, drivetrain=dict(drag_brake=1.0)), surf=split)
    traj = v.rollout(v.initial_state(v=3.0), np.tile([[0.0, -1.0]], (25, 1)))
    assert_all_finite(traj)
    print(f"\nsplit-mu braking: yaw after 0.5 s {np.degrees(traj.states[-1, S.YAW]):+.2f} deg")
    assert traj.states[-1, S.YAW] < -np.radians(0.5)
    uni = Vehicle(_with(params, drivetrain=dict(drag_brake=1.0)), surf=uniform_surface(surfaces["dry_asphalt"]))
    traj_u = uni.rollout(uni.initial_state(v=3.0), np.tile([[0.0, -1.0]], (25, 1)))
    assert abs(traj_u.states[-1, S.YAW]) < 1e-9


def test_loose_surface_plowing_slows_a_sliding_car(params, surfaces):
    sand = dataclasses.replace(params, surface=surfaces["sand"])
    no_plow = dataclasses.replace(params, surface=dataclasses.replace(surfaces["sand"], loose_drag=0.0))
    s0 = Vehicle(sand).initial_state(v=2.0, beta=np.radians(-40))
    ends = {}
    for label, p in (("plow", sand), ("no plow", no_plow)):
        v = Vehicle(p)
        ends[label] = v.rollout(s0, np.zeros((15, 2))).speed[-1]
    print(f"\nspeed after 0.3 s sliding sideways on sand: {ends['plow']:.2f} (plowing) vs {ends['no plow']:.2f} m/s")
    assert ends["plow"] < ends["no plow"] - 0.05


# ----------------------------------------------------------------------------- layouts, gyro
@pytest.mark.parametrize("layout", ["rwd", "awd_spool", "awd_overdrive"])
def test_every_layout_runs_and_accelerates(params, layout):
    v = Vehicle(_with(params, drivetrain=dict(layout=layout, overdrive_front=1.2)))
    traj = v.rollout(v.initial_state(), np.tile([[0.4, 1.0]], (100, 1)))
    assert_all_finite(traj)
    assert traj.speed[25] > 0.3


def test_awd_launches_harder_than_rwd(params):
    out = {}
    for layout in ("rwd", "awd_spool"):
        v = Vehicle(_with(params, drivetrain=dict(layout=layout)))
        out[layout] = v.rollout(v.initial_state(), np.tile([[0.0, 1.0]], (25, 1))).speed[-1]
    print(f"\nspeed after 0.5 s full throttle: rwd {out['rwd']:.2f}, awd {out['awd_spool']:.2f} m/s")
    assert out["awd_spool"] > 1.3 * out["rwd"], "driving all four wheels doubles the usable traction"


def test_gyro_feeds_back_yaw_rate_and_saves_a_spin(params):
    """The steering gyro counter-steers against yaw rate: (1) in grip driving it lowers the yaw-rate
    gain of a steering step; (2) after a throttle-stab that spins the car with the gyro off, the
    same inputs with the gyro on keep the sideslip below the spin limit."""
    cruise = np.concatenate([np.tile([[0.0, 0.19]], (30, 1)), np.tile([[0.4, 0.19]], (60, 1))])
    stab = np.concatenate([np.tile([[0.0, 0.37]], (50, 1)), np.tile([[0.25, 1.0]], (14, 1)),
                           np.tile([[0.0, 0.2]], (86, 1))])
    out = {}
    for on in (False, True):
        v = Vehicle(_with(params, actuators=dict(gyro_enabled=on, gyro_gain=0.3)))
        r = v.rollout(v.initial_state(v=1.5), cruise).states[-20:, S.R].mean()
        spin = np.degrees(np.abs(v.rollout(v.initial_state(), stab).beta)).max()
        out[on] = (r, spin)
    print(f"\nsteady yaw rate after a steer step: gyro off {np.degrees(out[False][0]):.0f}, on "
          f"{np.degrees(out[True][0]):.0f} deg/s; max |beta| after a throttle stab: off "
          f"{out[False][1]:.0f}, on {out[True][1]:.0f} deg")
    assert 0 < out[True][0] < 0.9 * out[False][0]
    assert out[False][1] > 80.0 > out[True][1]


# ----------------------------------------------------------------------------- load transfer, heat, dirt
def test_load_transfer_signs_and_magnitude(params):
    v = Vehicle(params)
    traj = v.rollout(v.initial_state(), np.tile([[0.0, 1.0]], (20, 1)))
    Fz = traj.info["Fz"][-1]
    ax = traj.info["ax"][-1]
    assert Fz[2:].sum() > Fz[:2].sum(), "acceleration moves load to the rear"
    dfz_expected = params.vehicle.mass * ax * params.vehicle.cg_height / params.vehicle.wheelbase
    assert traj.states[-1, S.DFZ_LONG] == pytest.approx(dfz_expected, rel=0.05)
    trims = EQ.continuation(v, 1.5, [0.5, 1.0], key="r")
    Fz_turn = trims[-1].info["Fz"]
    assert Fz_turn[1] + Fz_turn[3] > Fz_turn[0] + Fz_turn[2], "a left turn loads the right-hand wheels"
    assert trims[-1].s[S.DFZ_LAT] == pytest.approx(params.vehicle.mass * trims[-1].info["ay"]
                                                   * params.vehicle.cg_height / params.vehicle.track_width, rel=1e-6)
    assert np.all(Fz_turn.sum() == pytest.approx(params.vehicle.mass * params.sim.gravity))


def test_tires_heat_while_drifting_and_cool_afterwards(params):
    v = Vehicle(params)
    traj = v.rollout(v.initial_state(), open_loop_drift_actions(params, 3.0))
    T = traj.states[:, S.T_TIRE]
    print(f"\ntire temperature rise over a 3 s drift: rear {T[-1, 2:].mean() - T[0, 2:].mean():+.2f} degC, "
          f"front {T[-1, :2].mean() - T[0, :2].mean():+.2f} degC")
    assert T[-1, 2:].mean() > T[0, 2:].mean() + 0.3, "spinning rear tires must heat up"
    assert T[-1, 2:].mean() > T[-1, :2].mean(), "the rears heat more than the fronts in a drift"
    # a hot tire on a parked car cools exponentially with time constant heat_capacity/cool_coeff
    s_hot = v.initial_state()
    s_hot[S.T_TIRE] = 60.0
    cool = v.rollout(s_hot, np.zeros((200, 2)))
    tp, amb = params.tire, params.sim.ambient_temp
    expected = amb + (60.0 - amb) * np.exp(-4.0 * tp.cool_coeff / tp.heat_capacity)
    np.testing.assert_allclose(cool.states[-1, S.T_TIRE], expected, rtol=1e-6)


def test_contamination_wears_off_with_distance(params):
    cond = TireCondition(contamination=np.full(4, 0.8), temp0=np.full(4, 25.0))
    v = Vehicle(params, cond=cond)
    s0 = v.initial_state(v=2.0)
    assert np.all(s0[S.CONTAM] == 0.8)
    traj = v.rollout(s0, np.tile([[0.0, 0.2]], (100, 1)))
    dist = np.sum(traj.speed[:-1] * v.control_dt)
    expected = 0.8 * np.exp(-dist / params.tire.contamination_decay_dist)
    print(f"\ncontamination after {dist:.2f} m: {traj.states[-1, S.CONTAM].mean():.3f} (expected ~{expected:.3f})")
    assert traj.states[-1, S.CONTAM].mean() == pytest.approx(expected, rel=0.1)
    s_clean = s0.copy()
    s_clean[S.CONTAM] = 0.0
    assert v.derivatives(s_clean, np.zeros(2))[1]["mu_y"][0] > v.derivatives(s0, np.zeros(2))[1]["mu_y"][0]


def test_initial_state_rolls_without_slip():
    v = make_vehicle()
    s = v.initial_state(v=2.0, yaw_rate=1.0)
    _, info = v.derivatives(s, np.zeros(2))
    np.testing.assert_allclose(info["kappa"], 0.0, atol=1e-12)
