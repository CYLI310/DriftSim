"""Milestone 2: surface maps (per-wheel surface lookup, friction patches, roughness bumps, slopes)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_drift_sim.control.maneuvers import open_loop_drift_actions
from rc_drift_sim.sim import state as S
from rc_drift_sim.sim import tire
from rc_drift_sim.sim.params import TireCondition, default_params
from rc_drift_sim.sim.surface import per_wheel
from rc_drift_sim.sim.surface_map import (CHANNELS, MU, MU_PATCH_FLOOR, SurfaceMap, compile_map, lookup_m,
                                          map_tire, surface_channels, tire_on_map)
from rc_drift_sim.sim.vehicle import Vehicle, VehicleBatch, derivatives_model, make_vehicle

FLAT = dict(mu_patches=False, bumps=False)


def _rollout(car, s0, acts):
    return car.rollout(s0, acts)


# ----------------------------------------------------------------------------- equivalence with M1
@pytest.mark.parametrize("surface", ["epoxy_ptile", "loose_dirt", "ice"])
def test_uniform_map_matches_uniform_surface(surface):
    """A one-surface map without patches or bumps is the Milestone 1 uniform surface (incl. plowing
    on loose dirt and a worn, wet tire)."""
    cond = TireCondition(wear=np.array([0.3, 0.3, 0.6, 0.6]), wetness=np.array([0.0, 0.0, 0.5, 0.5]),
                         temp0=30.0, contamination=0.0)
    p = default_params(surface=surface)
    plain = Vehicle(p, cond=cond)
    mapped = Vehicle(p, cond=cond, surface_map=SurfaceMap.uniform(surface, size=50.0, cell=1.0, **FLAT))
    acts = open_loop_drift_actions(p, duration=2.0)
    s0 = plain.initial_state(v=1.0)
    a, b = _rollout(plain, s0, acts), _rollout(mapped, s0, acts)
    np.testing.assert_allclose(b.states, a.states, rtol=1e-9, atol=1e-9)


def test_tire_on_map_reproduces_tire_model_per_wheel(surfaces):
    """One-hot channels under each wheel give the same TireModel as tire_model(per_wheel(...))."""
    p = default_params()
    cond = TireCondition(wear=np.array([0.1, 0.2, 0.3, 0.4]), wetness=np.array([0.0, 1.0, 0.0, 0.5]),
                         temp0=25.0, contamination=0.0)
    four = [surfaces[n] for n in ("epoxy_ptile", "loose_dirt", "sand", "wet_asphalt")]
    ref = tire.tire_model(p.tire, per_wheel(four), cond)
    ch = np.stack([surface_channels(s) for s in four])                     # (4, NC)
    got = tire_on_map(tire.tire_model(p.tire, four[0], cond), map_tire(p.tire, cond), ch)
    for name in ("Cx", "Cy", "mu_scale", "k_scale", "e_shift", "slip_scale", "inv_slip_scale", "crr",
                 "loose_drag"):
        np.testing.assert_allclose(np.broadcast_to(getattr(got, name), (4,)),
                                   np.broadcast_to(getattr(ref, name), (4,)), rtol=1e-12, err_msg=name)


def test_no_map_leaves_info_unchanged(vehicle):
    _, info = vehicle.derivatives(vehicle.initial_state(v=1.0), np.zeros(2))
    assert "mu_surface" not in info and vehicle.model.smap is None


# ----------------------------------------------------------------------------- lookup
def _two_cell_map(**kw):
    return SurfaceMap.from_labels([[0, 1]], ["epoxy_ptile", "ice"], cell=1.0, origin=(0.0, 0.0),
                                  outside=1, **FLAT, **kw)


def test_lookup_at_cell_centers_midpoint_and_outside(surfaces):
    m = _two_cell_map()
    mu_e, mu_i = surfaces["epoxy_ptile"].mu_scale, surfaces["ice"].mu_scale
    got = m.sample(np.array([0.5, 1.5, 1.0, -5.0, 0.5]), np.array([0.5, 0.5, 0.5, 0.5, 9.0]))["mu"]
    np.testing.assert_allclose(got, [mu_e, mu_i, 0.5 * (mu_e + mu_i), mu_i, mu_i])


def test_lookup_is_continuous_across_a_boundary():
    m = _two_cell_map()
    x = np.linspace(0.5, 1.5, 1001)                               # epoxy center -> ice center
    mu = m.sample(x, np.full_like(x, 0.5))["mu"]
    assert np.max(np.abs(np.diff(mu))) < 2e-3                     # no jumps, one-cell ramp
    assert np.all(np.diff(mu) <= 1e-15)                           # epoxy -> ice: monotone


def test_from_ascii_first_row_is_top():
    m = SurfaceMap.from_ascii("""
        AB
        CD""", {"A": "epoxy_ptile", "B": "ice", "C": "sand", "D": "carpet"}, cell=1.0)
    assert m.names == ["epoxy_ptile", "ice", "sand", "carpet"]
    assert m.surface_at(np.array([0.5, 1.5, 0.5, 1.5]), np.array([1.5, 1.5, 0.5, 0.5])).tolist() == [0, 1, 2, 3]
    assert m.extent == (0.0, 2.0, 0.0, 2.0)


def test_invalid_maps_are_rejected():
    with pytest.raises(ValueError):
        SurfaceMap.from_labels([[0, 2]], ["epoxy_ptile", "ice"], cell=1.0)
    with pytest.raises(ValueError):
        SurfaceMap.from_labels([[0, 1]], ["epoxy_ptile", "ice"], cell=1.0, height=np.zeros((2, 2)))
    with pytest.raises(KeyError):
        SurfaceMap.uniform("no_such_surface")
    with pytest.raises(ValueError):
        SurfaceMap.from_ascii("AX", {"A": "ice"}, cell=1.0)


# ----------------------------------------------------------------------------- patches and bumps
def test_friction_patches_have_the_configured_spread(surfaces):
    m = SurfaceMap.uniform("gravel", size=60.0, cell=0.1, bumps=False, patch_size=0.5, seed=3)
    mu = compile_map(m).grid[:, MU].reshape(602, 602)[1:-1, 1:-1] / surfaces["gravel"].mu_scale
    std = surfaces["gravel"].mu_patch_std
    assert abs(mu.mean() - 1.0) < 0.02
    assert abs(mu.std() / std - 1.0) < 0.15
    assert mu.min() >= MU_PATCH_FLOOR - 1e-12
    same = compile_map(SurfaceMap.uniform("gravel", size=60.0, cell=0.1, bumps=False, seed=3)).grid
    other = compile_map(SurfaceMap.uniform("gravel", size=60.0, cell=0.1, bumps=False, seed=4)).grid
    assert np.array_equal(same, compile_map(m).grid) and not np.array_equal(other, same)


def test_bumps_are_unit_noise_scaled_by_roughness(surfaces):
    m = SurfaceMap.uniform("gravel", size=60.0, cell=1.0, mu_patches=False, bump_size=0.03)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-20, 20, 20000), rng.uniform(-20, 20, 20000)
    s = m.sample(x, y)
    assert abs(s["bump"].mean()) < 0.05 and abs(s["bump"].std() - 1.0) < 0.1
    np.testing.assert_allclose(s["roughness"], surfaces["gravel"].roughness)
    # correlated over less than a few bump sizes
    near = m.sample(x + 0.003, y)["bump"]
    far = m.sample(x + 0.3, y)["bump"]
    assert np.corrcoef(s["bump"], near)[0, 1] > 0.8 and abs(np.corrcoef(s["bump"], far)[0, 1]) < 0.1


def test_bumps_modulate_wheel_loads_only_on_rough_ground():
    smooth = make_vehicle(surface_map=SurfaceMap.from_labels([[0]], [dataclasses.replace(
        default_params().surface, roughness=0.0)], cell=10.0, origin=(-5, -5), mu_patches=False))
    assert not smooth.model.smap.has_bumps
    rough = make_vehicle(surface_map=SurfaceMap.uniform("gravel", size=20.0, mu_patches=False))
    s = rough.initial_state(v=2.0)
    _, info = rough.derivatives(s, np.zeros(2))
    _, info0 = make_vehicle().derivatives(s, np.zeros(2))
    np.testing.assert_allclose(info["Fz"], info0["Fz"] * info["fz_factor"], rtol=1e-12)
    assert np.all(info["fz_factor"] != 1.0)
    # a rolling car on gravel sees load noise of about the configured amplitude
    tr = rough.rollout(s, np.tile([0.0, 0.2], (100, 1)))
    rel = tr.info["fz_factor"] - 1.0
    assert 0.1 < rel.std() < 0.35                                  # gravel roughness = 0.2


# ----------------------------------------------------------------------------- slopes
def _hill(grade=0.1, **kw):
    return SurfaceMap.from_function(lambda x, y: np.zeros(x.shape, int), ["epoxy_ptile"], (-10, 10, -10, 10),
                                    0.25, height_fn=lambda x, y: grade * x, **FLAT, **kw)


def test_slope_pulls_downhill_in_the_body_frame():
    car = make_vehicle(surface_map=_hill())
    g = car.params.sim.gravity
    for yaw, expect in ((0.0, (-0.1 * g, 0.0)), (np.pi / 2, (0.0, 0.1 * g))):
        ds, info = car.derivatives(car.initial_state(yaw=yaw), np.zeros(2))
        np.testing.assert_allclose(info["a_slope"], expect, atol=1e-12)
        np.testing.assert_allclose(info["ax"], 0.0, atol=1e-12)          # IMU: no tire force yet
        np.testing.assert_allclose(ds[[S.VX, S.VY]], expect, atol=1e-12)


def test_parked_car_rolls_down_a_hill_and_tires_resist():
    car = make_vehicle(surface_map=_hill(0.05))
    tr = car.rollout(car.initial_state(), np.zeros((50, 2)))
    assert tr.states[-1, S.X] < -0.05 and tr.states[-1, S.VX] < -0.1
    assert np.mean(tr.info["ax"][10:]) > 0.0                    # rolling resistance + drivetrain push uphill
    flat = make_vehicle(surface_map=_hill(0.0))
    assert abs(flat.rollout(flat.initial_state(), np.zeros((50, 2))).states[-1, S.X]) < 1e-9


# ----------------------------------------------------------------------------- vehicle behaviour
def test_split_mu_launch_yaws_away_from_the_slippery_side():
    """Left wheels on ice, right wheels on epoxy: under full throttle the spool pushes harder on
    the grippy right side, so the car yaws left (positive yaw rate)."""
    m = SurfaceMap.from_function(lambda x, y: (y > 0).astype(int), ["epoxy_ptile", "ice"], (-5, 20, -5, 5),
                                 0.02, **FLAT)
    car = make_vehicle(surface_map=m)
    _, info = car.derivatives(car.initial_state(), np.zeros(2))
    assert np.all(info["mu_surface"][[0, 2]] < 0.5 * info["mu_surface"][[1, 3]])     # FL, RL on ice
    tr = car.rollout(car.initial_state(), np.tile([0.0, 1.0], (25, 1)))
    assert tr.states[-1, S.R] > 0.1
    uni = make_vehicle(surface_map=SurfaceMap.uniform("epoxy_ptile", size=60, **FLAT))
    assert abs(uni.rollout(uni.initial_state(), np.tile([0.0, 1.0], (25, 1))).states[-1, S.R]) < 1e-9


def test_driving_onto_ice_loses_grip():
    """A steady left circle that crosses onto ice: lateral acceleration collapses on the ice."""
    m = SurfaceMap.from_function(lambda x, y: (x > -1.0).astype(int), ["epoxy_ptile", "ice"], (-5, 5, -5, 5),
                                 0.05, **FLAT)
    car = make_vehicle(surface_map=m)
    tr = car.rollout(car.initial_state(x=-2.0, v=1.2), np.tile([0.5, 0.25], (150, 1)))
    on_ice = car.surface_map.surface_at(tr.xy[1:, 0], tr.xy[1:, 1]) == 1
    assert on_ice.any() and (~on_ice).any()
    ay = np.abs(tr.info["ay"])
    assert ay[on_ice].mean() < 0.6 * ay[~on_ice][20:].mean()


def test_batch_and_torch_match_single_car():
    m = SurfaceMap.from_ascii("AAB\nCDB\nCDD", {"A": "epoxy_ptile", "B": "loose_dirt", "C": "gravel",
                                                  "D": "carpet"}, cell=0.7, origin=(-1, -1), height=None)
    car = make_vehicle(surface_map=m)
    vb = VehicleBatch([car.params] * 3, surface_map=m)
    acts = open_loop_drift_actions(car.params, duration=1.0)
    sb = np.stack([car.initial_state(v=1.0, x=x) for x in (-0.5, 0.0, 0.5)])
    ss = sb[2].copy()
    for k in range(len(acts)):
        sb, _ = vb.step(sb, np.tile(acts[k], (3, 1)))
        ss, _ = car.step(ss, acts[k], want_info=False)
    np.testing.assert_array_equal(sb[2], ss)
    torch = pytest.importorskip("torch")
    from rc_drift_sim.sim.xp import to_device
    mt = to_device(vb.model, "cpu", torch.float64)
    u = np.tile(acts[-1], (3, 1))
    dn, info_n = derivatives_model(sb, u, vb.model, want_info=True)
    dt, info_t = derivatives_model(torch.as_tensor(sb), torch.as_tensor(u), mt, want_info=True)
    np.testing.assert_allclose(dt.numpy(), dn, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(info_t["fz_factor"].numpy(), info_n["fz_factor"], rtol=1e-12)


def test_map_channels_list_matches_lookup_width():
    mm = compile_map(_two_cell_map())
    ch, bump = lookup_m(mm, np.zeros((2, 4)), np.zeros((2, 4)))
    assert ch.shape == (2, 4, len(CHANNELS)) and bump.shape == (2, 4)


def test_map_draws_as_a_background(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rc_drift_sim.viz.render import snapshot
    m = SurfaceMap.from_ascii("AB\nBA", {"A": "epoxy_ptile", "B": "gravel"}, cell=2.0, origin=(-2, -2),
                              height=np.array([[0.0, 0.1], [0.1, 0.2]]))
    img = m.image()
    assert img.shape == (2, 2, 3) and img.min() >= 0 and img.max() <= 1
    car = make_vehicle(surface_map=m)
    tr = car.rollout(car.initial_state(v=1.0), np.tile([0.3, 0.3], (20, 1)))
    snapshot(tr, car.params, tmp_path / "map.png", surf_map=m)
    assert (tmp_path / "map.png").stat().st_size > 1000
    plt.close("all")
