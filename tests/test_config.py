"""DESIGN.md section 10 item 12: YAML round trip and parameter plumbing.

Every compound and surface must load into its dataclass with no missing field, ``_deg`` keys
must arrive as radians, ``surface.uniform_surface`` must broadcast a scalar surface to (4,)
per-wheel arrays, and ``default_params()`` must carry every sub-bundle the contract names.
"""
from __future__ import annotations

import dataclasses
import math
import numbers

import numpy as np
import pytest

from rc_drift_sim.sim.params import (CONFIG_DIR, ActuatorParams, DrivetrainParams, Params, SimParams,
                                     SurfaceParams, TireParams, VehicleParams, default_params,
                                     load_vehicle, load_yaml)

EXPECTED_TIRES = {"hard_plastic_drift", "rubber_onroad", "foam", "offroad_pin"}
EXPECTED_SURFACES = {"dry_asphalt", "wet_asphalt", "polished_concrete", "epoxy_ptile", "carpet",
                     "packed_dirt", "loose_dirt", "gravel", "short_grass", "sand", "ice"}
COMBINED_MODES = {"similarity", "mf_weighting", "ellipse"}


def _is_finite_number(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x))


# ----------------------------------------------------------------------------- tires.yaml
def test_tires_yaml_round_trip(tires):
    raw = load_yaml(CONFIG_DIR / "tires.yaml")["tires"]
    assert set(tires) == set(raw), f"load_tires() keys {sorted(tires)} != YAML keys {sorted(raw)}"
    assert EXPECTED_TIRES <= set(tires), f"missing compounds: {sorted(EXPECTED_TIRES - set(tires))}"
    for name, tp in tires.items():
        assert isinstance(tp, TireParams), f"{name}: load_tires() returned {type(tp)}"
        assert tp.name == name, f"{name}: TireParams.name is {tp.name!r}"
        for f in dataclasses.fields(TireParams):
            val = getattr(tp, f.name)
            assert val is not None, f"{name}: field {f.name!r} missing (None)"
            if f.name not in ("name", "combined_mode"):
                assert _is_finite_number(val), f"{name}.{f.name} = {val!r} is not a finite number"
        for key, yaml_val in raw[name].items():
            assert getattr(tp, key) == yaml_val, (
                f"{name}.{key}: YAML {yaml_val!r} != loaded {getattr(tp, key)!r}")
        assert tp.combined_mode in COMBINED_MODES, f"{name}: combined_mode {tp.combined_mode!r}"
        assert tp.fz0 > 0 and tp.pdx1 > 0 and tp.pdy1 > 0 and tp.pkx1 > 0 and tp.pky1 > 0, (
            f"{name}: peak mu / stiffness coefficients must be positive")
        assert tp.pex1 <= 1.0 and tp.pey1 <= 1.0, f"{name}: curvature E must be <= 1 at nominal load"
    print("tire compounds loaded:", ", ".join(sorted(tires)))


# ----------------------------------------------------------------------------- surfaces.yaml
def test_surfaces_yaml_round_trip(surfaces):
    raw = load_yaml(CONFIG_DIR / "surfaces.yaml")["surfaces"]
    assert set(surfaces) == set(raw), f"load_surfaces() keys {sorted(surfaces)} != YAML {sorted(raw)}"
    assert EXPECTED_SURFACES <= set(surfaces), (
        f"missing surfaces: {sorted(EXPECTED_SURFACES - set(surfaces))}")
    for name, sp in surfaces.items():
        assert isinstance(sp, SurfaceParams), f"{name}: load_surfaces() returned {type(sp)}"
        assert sp.name == name
        for f in dataclasses.fields(SurfaceParams):
            assert getattr(sp, f.name) is not None, f"{name}: field {f.name!r} missing (None)"
        for key in SurfaceParams.NUMERIC_FIELDS:
            val = getattr(sp, key)
            assert _is_finite_number(val), f"{name}.{key} = {val!r} is not a finite scalar"
        for key, yaml_val in raw[name].items():
            assert getattr(sp, key) == yaml_val, (
                f"{name}.{key}: YAML {yaml_val!r} != loaded {getattr(sp, key)!r}")
        assert sp.mu_scale > 0, f"{name}: mu_scale must be positive"
        assert sp.stiffness_scale > 0, f"{name}: stiffness_scale must be positive"
        assert 0.0 <= sp.shape_blend <= 1.0, f"{name}: shape_blend {sp.shape_blend} not in [0, 1]"
        assert sp.peak_slip_shift >= 0.0, f"{name}: peak_slip_shift must be >= 0"
        assert 0.0 <= sp.looseness <= 1.0, f"{name}: looseness {sp.looseness} not in [0, 1]"
        assert isinstance(sp.color, str) and sp.color.startswith("#"), f"{name}: color {sp.color!r}"
    print("surfaces loaded:", ", ".join(sorted(surfaces)))


# ----------------------------------------------------------------------------- vehicle.yaml
def test_vehicle_yaml_round_trip():
    vp, dp, ap, sp = load_vehicle()
    assert isinstance(vp, VehicleParams) and isinstance(dp, DrivetrainParams)
    assert isinstance(ap, ActuatorParams) and isinstance(sp, SimParams)
    raw = load_yaml(CONFIG_DIR / "vehicle.yaml")
    # *_deg / *_deg_s keys arrive in radians with the suffix stripped
    assert ap.steer_max == pytest.approx(math.radians(raw["actuators"]["steer_max_deg"])), (
        f"steer_max {ap.steer_max} rad != radians({raw['actuators']['steer_max_deg']} deg)")
    assert ap.servo_rate == pytest.approx(math.radians(raw["actuators"]["servo_rate_deg_s"]))
    assert ap.gyro_max_correction == pytest.approx(
        math.radians(raw["actuators"]["gyro_max_correction_deg"]))
    assert ap.steer_offset == pytest.approx(math.radians(raw["actuators"]["steer_offset_deg"]))
    assert not hasattr(ap, "steer_max_deg"), "the _deg suffix must be stripped by the loader"
    assert dp.layout in ("rwd", "awd_spool", "awd_overdrive"), dp.layout
    assert sp.integrator in ("rk4", "semi_implicit_euler"), sp.integrator
    assert sp.n_substeps == int(round(sp.control_dt / sp.dt)) == 20, (
        f"n_substeps {sp.n_substeps} for dt={sp.dt}, control_dt={sp.control_dt}")
    assert vp.cg_to_rear == pytest.approx(vp.wheelbase - vp.cg_to_front)
    assert vp.iz > 0
    print(f"vehicle: m={vp.mass} kg L={vp.wheelbase} m a={vp.cg_to_front} m Iz={vp.iz:.5f} kg m^2; "
          f"steer_max={math.degrees(ap.steer_max):.1f} deg; n_substeps={sp.n_substeps}")


# ----------------------------------------------------------------------------- default_params
def test_default_params_fields_present(params):
    assert isinstance(params, Params)
    expected = {"vehicle": VehicleParams, "tire": TireParams, "surface": SurfaceParams,
                "drivetrain": DrivetrainParams, "actuators": ActuatorParams, "sim": SimParams}
    assert {f.name for f in dataclasses.fields(Params)} == set(expected)
    for name, cls in expected.items():
        sub = getattr(params, name, None)
        assert isinstance(sub, cls), f"default_params().{name} is {type(sub)}, expected {cls.__name__}"
        for f in dataclasses.fields(cls):
            val = getattr(sub, f.name)
            if (name, f.name) == ("vehicle", "yaw_inertia"):
                continue  # None is the documented 'estimate it' value
            assert val is not None, f"default_params().{name}.{f.name} is None"
    assert params.tire.name == "hard_plastic_drift", params.tire.name
    assert params.surface.name == "epoxy_ptile", params.surface.name
    assert params.sim.v_eps == pytest.approx(0.3), "contract section 1: v_eps = 0.3 m/s"
    # frozen: dataclasses.replace is the way to vary them
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.vehicle.mass = 2.0  # type: ignore[misc]
    replaced = dataclasses.replace(params, sim=dataclasses.replace(params.sim, integrator="semi_implicit_euler"))
    assert replaced.sim.integrator == "semi_implicit_euler" and params.sim.integrator == "rk4"


def test_default_params_every_compound_and_surface(tires, surfaces):
    for tname in tires:
        for sname in surfaces:
            p = default_params(tire=tname, surface=sname)
            assert p.tire.name == tname and p.surface.name == sname, (tname, sname)


def test_static_loads_consistent(params):
    vp = params.vehicle
    Fz = vp.static_loads(params.sim.gravity)
    assert Fz.shape == (4,)
    assert Fz.sum() == pytest.approx(vp.mass * params.sim.gravity, rel=1e-12), "static loads must sum to m g"
    assert Fz[0] == Fz[1] and Fz[2] == Fz[3], "left/right static loads must be equal"
    front_share = (Fz[0] + Fz[1]) / Fz.sum()
    assert front_share == pytest.approx(vp.cg_to_rear / vp.wheelbase), "front share = b/L"
    print(f"static loads FL FR RL RR = {np.round(Fz, 4)} N, front share {front_share:.3f}")


# ----------------------------------------------------------------------------- uniform_surface
def test_uniform_surface_broadcasts_to_four_wheels(surfaces):
    from rc_drift_sim.sim import surface as surface_mod
    for name, sp in surfaces.items():
        surf4 = surface_mod.uniform_surface(sp)
        assert isinstance(surf4, SurfaceParams), f"{name}: uniform_surface returned {type(surf4)}"
        assert surf4.name == sp.name, f"{name}: name not preserved"
        for key in SurfaceParams.NUMERIC_FIELDS:
            arr = np.asarray(getattr(surf4, key))
            assert arr.shape == (4,), f"{name}.{key}: shape {arr.shape}, expected (4,)"
            assert arr.dtype.kind == "f", f"{name}.{key}: dtype {arr.dtype}, expected float"
            assert np.all(arr == float(getattr(sp, key))), (
                f"{name}.{key}: {arr} != scalar {getattr(sp, key)}")
    print("uniform_surface: every numeric field broadcast to shape (4,) for", len(surfaces), "surfaces")


def test_uniform_surface_accepts_per_wheel_override(surfaces):
    """Milestone 2 hook: a SurfaceParams whose fields are already (4,) arrays must be usable."""
    from rc_drift_sim.sim import surface as surface_mod
    sp = surfaces["epoxy_ptile"]
    mixed = dataclasses.replace(sp, mu_scale=np.array([1.0, 0.9, 0.8, 0.7]))
    assert np.asarray(mixed.mu_scale).shape == (4,)
    surf4 = surface_mod.uniform_surface(sp)
    assert np.asarray(surf4.mu_scale).shape == (4,)
