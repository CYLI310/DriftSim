"""Tire model tests (DESIGN.md section 4; section 10 items 5 and 6 plus the tire-condition model).

Conventions under test: alpha > 0 (contact velocity left of the wheel heading) gives Fy < 0;
kappa > 0 (driving) gives Fx > 0; Fz = 0 gives zero force; the friction force never exceeds
max(mu_x, mu_y) * Fz on any compound, surface, combined-slip mode or tire condition.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from rc_drift_sim.sim import tire
from rc_drift_sim.sim.params import SimParams, TireCondition
from rc_drift_sim.sim.surface import per_wheel, uniform_surface

PAVED = ("dry_asphalt", "wet_asphalt", "polished_concrete", "epoxy_ptile", "carpet", "ice")
LOOSE = ("loose_dirt", "gravel", "sand")          # looseness >= 0.7: grip keeps building
ALPHA_GRID = np.radians(np.linspace(0.01, 80.0, 4000))


def _lateral_curve(tp, sp, Fz=4.0, T=None, cond=None, wheel=0):
    """|Fy|(alpha) of one wheel over ALPHA_GRID (conditions are per-wheel (4,) arrays)."""
    if cond is None:
        c = tire.tire_coefficients(tp, sp, None, Fz, T)
        _, fy = tire.pure_slip_forces(c, 0.0, ALPHA_GRID)
        return np.abs(fy)
    c = tire.tire_coefficients(tp, uniform_surface(sp), cond, np.full(4, Fz), T)
    _, fy = tire.pure_slip_forces(c, 0.0, ALPHA_GRID[:, None])
    return np.abs(fy[:, wheel])


# ----------------------------------------------------------------------------- Magic Formula basics
def test_mf_matches_racer_nl_formula():
    """y = D sin(C atan(Bx - E(Bx - atan Bx))) + Sv with x -> x + Sh (hand-computed reference)."""
    B, C, D, E, Sh, Sv, x = 10.0, 1.3, 2.0, -0.5, 0.01, 0.05, 0.08
    bx = B * (x + Sh)                                      # 0.9
    expected = D * np.sin(C * np.arctan(bx - E * (bx - np.arctan(bx)))) + Sv
    assert tire.mf(x, B, C, D, E, Sh, Sv) == pytest.approx(expected, rel=1e-14)
    import math                                            # independent scalar re-implementation
    bxs = B * (x + Sh)
    ref = D * math.sin(C * math.atan(bxs - E * (bxs - math.atan(bxs)))) + Sv
    assert float(tire.mf(x, B, C, D, E, Sh, Sv)) == pytest.approx(ref, rel=1e-14)
    assert ref == pytest.approx(1.74395, abs=5e-5)         # by hand: bx = 0.9 -> 2 sin(1.01027) + 0.05


def test_mf_properties_peak_slope_asymptote():
    B, C, D, E = 8.0, 1.4, 3.0, 0.2
    x = np.linspace(0.0, 50.0, 200001)
    y = tire.mf(x, B, C, D, E)
    assert y.max() == pytest.approx(D, rel=1e-6), "D is the peak"
    h = 1e-7
    assert (tire.mf(h, B, C, D, E) - tire.mf(-h, B, C, D, E)) / (2 * h) == pytest.approx(B * C * D, rel=1e-6)
    assert tire.mf(1e7, B, C, D, E) == pytest.approx(D * np.sin(0.5 * np.pi * C), rel=1e-4), "asymptote"


def test_sign_conventions(tires, surfaces):
    c = tire.tire_coefficients(tires["rubber_onroad"], surfaces["dry_asphalt"], None, 4.0, None)
    fx, _ = tire.pure_slip_forces(c, 0.1, 0.0)
    _, fy = tire.pure_slip_forces(c, 0.0, np.radians(3.0))
    assert fx > 0.0, "positive slip ratio (driving) must push forward"
    assert fy < 0.0, "positive slip angle (velocity left of heading) must push right (ISO sign)"
    fx2, fy2 = tire.pure_slip_forces(c, -0.1, -np.radians(3.0))
    assert fx2 == pytest.approx(-fx) and fy2 == pytest.approx(-fy), "pure-slip curves are odd"


# ----------------------------------------------------------------------------- item 5: force bound
@pytest.mark.parametrize("mode", tire.MODES)
def test_force_never_exceeds_mu_fz(tires, surfaces, mode):
    """|F| <= max(mu_x, mu_y) * Fz on a dense (kappa, alpha, Fz) grid, every compound and surface."""
    kappa = np.linspace(-1.5, 1.5, 61)[:, None, None]
    alpha = np.radians(np.linspace(-80.0, 80.0, 81))[None, :, None]
    Fz = np.array([0.0, 1.0, 4.0, 8.0])[None, None, :]
    worst = 0.0
    for (tn, tp), (sn, sp) in itertools.product(tires.items(), surfaces.items()):
        c = tire.tire_coefficients(tp, sp, None, Fz, None)
        fx, fy = tire.combined_forces(c, kappa, alpha, mode)
        bound = np.maximum(c["mu_x"], c["mu_y"]) * c["Fz"]
        ratio = np.hypot(fx, fy) - bound * (1.0 + 1e-9)
        assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fy)), f"{tn}/{sn}: non-finite force"
        assert np.all(ratio <= 1e-12), f"{tn}/{sn}/{mode}: |F| exceeds mu*Fz by {ratio.max():.3e} N"
        worst = max(worst, float(np.max(np.hypot(fx, fy) / np.maximum(bound, 1e-12))))
    print(f"\n{mode}: max |F|/(mu Fz) over the grid = {worst:.6f}")


def test_force_bound_holds_with_tire_conditions_and_low_speed_blend(tires, surfaces):
    """Full tire_forces path (low-speed blend, temperature, wear, wetness, contamination)."""
    rng = np.random.default_rng(0)
    sim = SimParams()
    n = 20000
    for (tn, tp), (sn, sp) in itertools.product(tires.items(), surfaces.items()):
        cond = TireCondition(wear=rng.uniform(0, 1, 4), wetness=rng.uniform(0, 1, 4),
                             contamination=rng.uniform(0, 1, 4), temp0=np.full(4, 25.0))
        shape = (n // 4, 4)
        vxw = rng.uniform(-3, 8, shape) * rng.choice([1.0, 0.01], shape)
        vyw = rng.uniform(-4, 4, shape) * rng.choice([1.0, 0.01], shape)
        omega = rng.uniform(-50, 400, shape)
        k_lag, a_lag = rng.uniform(-3, 8, shape), rng.uniform(-1.4, 1.4, shape)
        Fz, T = rng.uniform(0, 9, shape), rng.uniform(-10, 120, shape)
        fx, fy, info = tire.tire_forces(tp, uniform_surface(sp), cond, k_lag, a_lag, vxw, vyw, omega,
                                        Fz, T, sim, R=0.032)
        bound = np.maximum(info["mu_x"], info["mu_y"]) * info["Fz"]
        assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fy))
        assert np.all(np.hypot(fx, fy) <= bound * (1 + 1e-9) + 1e-12), f"{tn}/{sn}: bound violated"


def test_zero_force_at_zero_load_and_at_rest(tires, surfaces):
    sim = SimParams()
    for tp in tires.values():
        c = tire.tire_coefficients(tp, surfaces["dry_asphalt"], None, 0.0, None)
        for mode in tire.MODES:
            fx, fy = tire.combined_forces(c, 0.7, 0.4, mode)
            assert fx == 0.0 and fy == 0.0, f"{tp.name}/{mode}: force at Fz = 0"
        z = np.zeros(4)
        fx, fy, _ = tire.tire_forces(tp, uniform_surface(surfaces["carpet"]), TireCondition.neutral(),
                                     z, z, z, z, z, np.full(4, 4.0), np.full(4, 25.0), sim, R=0.032)
        assert np.all(fx == 0.0) and np.all(fy == 0.0), f"{tp.name}: force at rest"


def test_extreme_inputs_are_finite(tires, surfaces):
    big = np.array([0.0, 1e-12, 1e6, -1e6, 1e12])
    for tp, mode in itertools.product(tires.values(), tire.MODES):
        c = tire.tire_coefficients(tp, surfaces["sand"], None, np.array([0.0, 1e-9, 4.0, 50.0, 1e4])[:, None], None)
        fx, fy = tire.combined_forces(c, big[None, :], np.clip(big, -1.57, 1.57)[None, :], mode)
        assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fy)), f"{tp.name}/{mode}"


def test_similarity_reproduces_pure_slip(tires, surfaces):
    k = np.linspace(-2, 2, 401)
    a = np.radians(np.linspace(-80, 80, 401))
    for tp, sp in itertools.product(tires.values(), surfaces.values()):
        c = tire.tire_coefficients(tp, sp, None, 4.0, None)
        fx0, _ = tire.pure_slip_forces(c, k, 0.0)
        _, fy0 = tire.pure_slip_forces(c, 0.0, a)
        fx, fy_zero = tire.combined_forces(c, k, 0.0, "similarity")
        fx_zero, fy = tire.combined_forces(c, 0.0, a, "similarity")
        np.testing.assert_allclose(fx, fx0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(fy, fy0, rtol=0, atol=1e-12)
        assert np.all(fy_zero == 0.0) and np.all(fx_zero == 0.0)


def test_combined_slip_rear_spin_collapses_lateral_grip(tires, surfaces):
    """The drift mechanism: a spinning tire loses most of its lateral force at the same slip angle."""
    c = tire.tire_coefficients(tires["hard_plastic_drift"], surfaces["epoxy_ptile"], None, 4.0, None)
    alpha = np.radians(20.0)
    _, fy_roll = tire.combined_forces(c, 0.0, alpha, "similarity")
    _, fy_spin = tire.combined_forces(c, 2.0, alpha, "similarity")
    print(f"\n|Fy| at 20 deg: rolling {abs(fy_roll):.3f} N, spinning (kappa=2) {abs(fy_spin):.3f} N")
    assert abs(fy_spin) < 0.35 * abs(fy_roll)


# ----------------------------------------------------------------------------- item 6: curve shape
def test_lateral_peak_angles_on_dry_asphalt(tires, surfaces):
    for tn, tp in tires.items():
        f = _lateral_curve(tp, surfaces["dry_asphalt"])
        a_pk = np.degrees(ALPHA_GRID[np.argmax(f)])
        print(f"\n{tn}: peak at {a_pk:.1f} deg, mu_y = {f.max() / 4.0:.2f}")
        assert 5.0 <= a_pk <= 20.0, f"{tn}: lateral peak at {a_pk:.1f} deg"


@pytest.mark.parametrize("surface", PAVED)
def test_paved_surfaces_have_a_sharp_peak(tires, surfaces, surface):
    for tn, tp in tires.items():
        f = _lateral_curve(tp, surfaces[surface])
        k = int(np.argmax(f))
        assert k < len(f) - 1, f"{tn} on {surface}: no interior peak"
        f3 = np.abs(tire.pure_slip_forces(tire.tire_coefficients(tp, surfaces[surface], None, 4.0, None),
                                          0.0, 3.0 * ALPHA_GRID[k])[1])
        assert f3 < 0.97 * f[k], f"{tn} on {surface}: force at 3x peak slip is {f3 / f[k]:.3f} of peak"


@pytest.mark.parametrize("surface", LOOSE)
def test_loose_surfaces_have_no_peak(tires, surfaces, surface):
    """Loose ground: no sharp peak, lateral force keeps building out to 80 deg (not just scaled mu)."""
    for tn, tp in tires.items():
        f = _lateral_curve(tp, surfaces[surface])
        assert np.all(np.diff(f) >= -1e-12), f"{tn} on {surface}: lateral force decreases somewhere"


def test_short_grass_has_at_most_a_negligible_bump(tires, surfaces):
    for tn, tp in tires.items():
        f = _lateral_curve(tp, surfaces["short_grass"])
        assert f[-1] >= 0.99 * f.max(), f"{tn} on short_grass: drops {1 - f[-1] / f.max():.3%} after peak"


def test_loose_differs_from_scaled_pavement(tires, surfaces):
    """Same peak mu, different shape: loose_dirt is not just dry_asphalt with mu scaled down."""
    tp = tires["rubber_onroad"]
    f_loose = _lateral_curve(tp, surfaces["loose_dirt"])
    f_paved = _lateral_curve(tp, surfaces["dry_asphalt"]) * (f_loose.max() / _lateral_curve(tp, surfaces["dry_asphalt"]).max())
    a_loose = ALPHA_GRID[np.argmax(f_loose)]
    a_paved = ALPHA_GRID[np.argmax(f_paved)]
    assert a_loose > 3 * a_paved, "loose ground must reach its peak at a much larger slip angle"
    k5 = np.searchsorted(ALPHA_GRID, np.radians(5.0))
    assert f_loose[k5] < 0.7 * f_paved[k5], "loose ground must be softer (lower initial stiffness)"


def test_drift_tire_is_flatter_after_peak_than_rubber(tires, surfaces):
    def drop(tp):
        f = _lateral_curve(tp, surfaces["epoxy_ptile"])
        k = int(np.argmax(f))
        c = tire.tire_coefficients(tp, surfaces["epoxy_ptile"], None, 4.0, None)
        return float(np.abs(tire.pure_slip_forces(c, 0.0, 3 * ALPHA_GRID[k])[1]) / f[k])
    plastic, rubber, foam = drop(tires["hard_plastic_drift"]), drop(tires["rubber_onroad"]), drop(tires["foam"])
    print(f"\nforce at 3x peak slip / peak: plastic {plastic:.3f}, rubber {rubber:.3f}, foam {foam:.3f}")
    assert plastic > rubber > foam


def test_pin_tire_prefers_loose_ground(tires, surfaces):
    """Off-road pin tire: worse than rubber on pavement, better on loose dirt (loose_affinity)."""
    def peak(tn, sn):
        return _lateral_curve(tires[tn], surfaces[sn]).max()
    assert peak("offroad_pin", "dry_asphalt") < peak("rubber_onroad", "dry_asphalt")
    assert peak("offroad_pin", "loose_dirt") > peak("rubber_onroad", "loose_dirt")
    assert peak("hard_plastic_drift", "loose_dirt") < peak("hard_plastic_drift", "packed_dirt")


def test_mu_scale_scales_the_peak(tires, surfaces):
    import dataclasses
    tp, sp = tires["rubber_onroad"], surfaces["dry_asphalt"]
    f1 = _lateral_curve(tp, sp).max()
    f2 = _lateral_curve(tp, dataclasses.replace(sp, mu_scale=0.5)).max()
    assert f2 == pytest.approx(0.5 * f1, rel=1e-3)


def test_per_wheel_surface_is_applied_per_wheel(tires, surfaces):
    tp = tires["rubber_onroad"]
    mixed = per_wheel([surfaces["ice"], surfaces["dry_asphalt"], surfaces["sand"], surfaces["carpet"]])
    c = tire.tire_coefficients(tp, mixed, None, np.full(4, 4.0), None)
    for i, name in enumerate(("ice", "dry_asphalt", "sand", "carpet")):
        ci = tire.tire_coefficients(tp, surfaces[name], None, 4.0, None)
        assert c["Dy"][i] == pytest.approx(float(ci["Dy"])), f"wheel {i} should be on {name}"


# ----------------------------------------------------------------------------- tire condition
def _peak_mu(tp, sp, cond=None, T=None):
    return _lateral_curve(tp, sp, cond=cond, T=T).max() / 4.0


def test_temperature_window(tires, surfaces):
    tp, sp = tires["rubber_onroad"], surfaces["dry_asphalt"]
    mu_opt = _peak_mu(tp, sp, T=tp.t_opt)
    mu_cold, mu_hot = _peak_mu(tp, sp, T=tp.t_opt - 40), _peak_mu(tp, sp, T=tp.t_opt + 40)
    mu_nominal = _peak_mu(tp, sp)
    print(f"\nrubber mu: optimal {mu_opt:.3f}, cold {mu_cold:.3f}, hot {mu_hot:.3f}")
    assert mu_opt == pytest.approx(mu_nominal, rel=1e-12), "YAML mu is the optimal-temperature mu"
    assert mu_cold < mu_opt and mu_hot < mu_opt
    assert mu_cold > (1 - tp.temp_mu_drop) * mu_opt * 0.999


def test_wear_lowers_peak_and_raises_stiffness(tires, surfaces):
    tp, sp = tires["rubber_onroad"], surfaces["dry_asphalt"]
    new = TireCondition(wear=np.zeros(4))
    worn = TireCondition(wear=np.ones(4))
    c_new = tire.tire_coefficients(tp, sp, new, np.full(4, 4.0), None)
    c_worn = tire.tire_coefficients(tp, sp, worn, np.full(4, 4.0), None)
    assert np.all(c_worn["Dy"] == pytest.approx((1 - tp.wear_mu_drop) * c_new["Dy"]))
    assert np.all(c_worn["Ky"] == pytest.approx((1 + tp.wear_k_gain) * c_new["Ky"]))


def test_asymmetric_wear_is_per_wheel(tires, surfaces):
    tp, sp = tires["hard_plastic_drift"], surfaces["epoxy_ptile"]
    cond = TireCondition(wear=np.array([0.0, 0.0, 0.9, 0.1]))
    c = tire.tire_coefficients(tp, sp, cond, np.full(4, 4.0), None)
    assert c["Dy"][2] < c["Dy"][3] < c["Dy"][0] == c["Dy"][1]


def test_wetness_lowers_grip_and_sharpens_the_drop(tires, surfaces):
    tp, sp = tires["rubber_onroad"], surfaces["dry_asphalt"]
    dry = _lateral_curve(tp, sp, cond=TireCondition(wetness=np.zeros(4)))
    wet = _lateral_curve(tp, sp, cond=TireCondition(wetness=np.ones(4)))
    assert wet.max() == pytest.approx((1 - tp.wet_mu_drop) * dry.max(), rel=1e-3)
    assert wet[-1] / wet.max() < dry[-1] / dry.max(), "a wet tire should fall off more after the peak"


def test_contamination_lowers_grip(tires, surfaces):
    tp, sp = tires["hard_plastic_drift"], surfaces["epoxy_ptile"]
    clean = _peak_mu(tp, sp, cond=TireCondition(contamination=np.zeros(4)))
    dirty = _peak_mu(tp, sp, cond=TireCondition(contamination=np.full(4, 0.5)))
    assert dirty == pytest.approx((1 - 0.5 * tp.contamination_mu_drop) * clean, rel=1e-3)


def test_neutral_condition_equals_no_condition(tires, surfaces):
    tp, sp = tires["foam"], surfaces["carpet"]
    a = tire.tire_coefficients(tp, uniform_surface(sp), None, np.full(4, 4.0), None)
    b = tire.tire_coefficients(tp, uniform_surface(sp), TireCondition.neutral(), np.full(4, 4.0), None)
    for key in ("Dx", "Dy", "Kx", "Ky", "Cx", "Cy", "Ex", "Ey"):
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0)


# ----------------------------------------------------------------------------- plowing and heat
def test_plowing_force(surfaces):
    vyw = np.array([-2.0, -0.5, 0.0, 0.5, 2.0, 50.0])
    f = tire.plowing_force(surfaces["sand"].loose_drag, vyw, 4.0)
    assert np.all(np.sign(f) == -np.sign(vyw)), "plowing opposes the lateral slide"
    assert f[2] == 0.0
    assert abs(f[4]) > abs(f[3]), "grows with lateral slide speed"
    assert abs(f[5]) <= tire.PLOW_CAP * surfaces["sand"].loose_drag * 4.0 * (1 + 1e-12), "saturates"
    assert np.all(tire.plowing_force(surfaces["dry_asphalt"].loose_drag, vyw, 4.0) == 0.0)


def test_thermal_rate_heats_under_slip_and_cools_to_ambient(tires):
    tp = tires["rubber_onroad"]
    heat = tire.thermal_rate(tp, 3.0, 0.0, 2.0, 0.0, 1.0, 25.0, 25.0)
    cool = tire.thermal_rate(tp, 0.0, 0.0, 0.0, 0.0, 3.0, 60.0, 25.0)
    assert heat == pytest.approx(3.0 * 2.0 / tp.heat_capacity)
    assert cool < 0.0
    assert tire.thermal_rate(tp, 0.0, 0.0, 0.0, 0.0, 3.0, 25.0, 25.0) == 0.0
