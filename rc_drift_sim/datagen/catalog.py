"""Catalog of every variable a batch can set: defaults, units, descriptions, types and limits.

The catalog is built by introspecting the parameter dataclasses in ``sim.params`` (their field
comments carry the unit and description), so the GUI and the CLI can never drift out of sync with
the physics. Keys use the YAML conventions: ``group.field``, with angles exposed in degrees under
``*_deg`` / ``*_deg_s`` names exactly as in ``configs/vehicle.yaml``.

Key namespaces
--------------
    tire, surface              categorical: tire compound / surface type (names from the YAML files)
    vehicle.* drivetrain.* actuators.* sim.*     Params groups
    tire.*                     coefficients of the chosen compound (TireParams)
    surface.*                  numeric fields of the chosen surface (SurfaceParams)
    condition.*                per-episode tire condition: wear, wetness, contamination, temp0
    init.*                     initial state: speed, beta_deg, yaw_rate_deg_s, x, y, yaw_deg
    maneuver.<type>.*          input-generator parameters (see ``datagen.inputs``)
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import re
from functools import lru_cache
from typing import Any

from ..sim import params as P

# fields whose internal value is in rad (or rad/s) but that the YAML/GUI expose in degrees
ANGLE_FIELDS = {
    ("actuators", "steer_max"): ("steer_max_deg", "deg"),
    ("actuators", "servo_rate"): ("servo_rate_deg_s", "deg/s"),
    ("actuators", "steer_offset"): ("steer_offset_deg", "deg"),
    ("actuators", "gyro_max_correction"): ("gyro_max_correction_deg", "deg"),
}
# settings that change the structure of the simulation: cars that differ in these cannot share a
# vectorized batch, so the runner groups episodes by them
STRUCTURAL = {"tire.combined_mode", "drivetrain.layout", "drivetrain.reverse_enabled",
              "sim.dt", "sim.control_dt", "sim.integrator"}
CHOICES = {
    "drivetrain.layout": ["rwd", "awd_spool", "awd_overdrive"],
    "tire.combined_mode": ["similarity", "mf_weighting", "ellipse"],
    "sim.integrator": ["rk4", "semi_implicit_euler"],
}
# fields that must stay strictly positive (physical sizes, inertias, time constants, ...)
POSITIVE = {
    "vehicle.mass", "vehicle.wheelbase", "vehicle.track_width", "vehicle.cg_height", "vehicle.cg_to_front",
    "vehicle.body_length", "vehicle.body_width", "vehicle.wheel_radius", "vehicle.wheel_inertia",
    "vehicle.load_transfer_tau", "vehicle.air_density",
    "tire.fz0", "tire.pky2", "tire.relax_x", "tire.relax_y", "tire.heat_capacity", "tire.t_width",
    "tire.contamination_decay_dist",
    "drivetrain.gear_ratio", "drivetrain.overdrive_front", "drivetrain.motor_kv", "drivetrain.motor_resistance",
    "drivetrain.motor_tau_e", "drivetrain.motor_inertia", "drivetrain.current_limit",
    "drivetrain.battery_voltage", "drivetrain.throttle_min",
    "actuators.steer_max_deg", "actuators.servo_rate_deg_s", "actuators.servo_tau",
    "sim.dt", "sim.control_dt", "sim.v_eps", "sim.v_low", "sim.v_c_low", "sim.v_rr", "sim.gravity",
}
NONNEGATIVE = {
    "vehicle.drag_coeff_area", "tire.heat_capacity", "tire.cool_coeff", "tire.cool_speed_coeff",
    "drivetrain.motor_friction_visc", "drivetrain.motor_friction_coulomb", "drivetrain.battery_resistance",
    "drivetrain.rear_diff_max_torque", "drivetrain.front_diff_max_torque", "drivetrain.center_max_torque",
    "drivetrain.rear_diff_visc", "drivetrain.front_diff_visc", "drivetrain.center_visc",
    "actuators.latency", "actuators.steer_deadband", "actuators.throttle_deadband", "actuators.gyro_gain",
    "actuators.gyro_max_correction_deg",
    "surface.mu_scale", "surface.stiffness_scale", "surface.rolling_resistance", "surface.loose_drag",
    "surface.roughness", "surface.mu_patch_std", "sim.v_low_start",
}

# unit, description, level ("basic" fields are shown first; "advanced" ones sit behind a toggle)
_B, _A = "basic", "advanced"
META: dict[str, tuple[str, str, str]] = {
    # chassis
    "vehicle.mass": ("kg", "total mass of the car incl. battery", _B),
    "vehicle.wheelbase": ("m", "front-to-rear axle distance", _B),
    "vehicle.track_width": ("m", "left-to-right wheel distance", _B),
    "vehicle.cg_height": ("m", "centre-of-gravity height (drives load transfer)", _B),
    "vehicle.cg_to_front": ("m", "CG to front axle (smaller = more weight on the front)", _B),
    "vehicle.yaw_inertia": ("kg m^2", "yaw moment of inertia; empty = estimated from mass and size", _A),
    "vehicle.body_length": ("m", "body length (only used to estimate the yaw inertia)", _A),
    "vehicle.wheel_radius": ("m", "tire radius", _B),
    "vehicle.wheel_inertia": ("kg m^2", "rotational inertia of one wheel and tire", _A),
    "vehicle.roll_stiffness_front": ("-", "share of lateral load transfer on the front axle (0..1)", _A),
    "vehicle.load_transfer_tau": ("s", "suspension lag of the load transfer", _A),
    "vehicle.drag_coeff_area": ("m^2", "aerodynamic drag area Cd*A", _A),
    "vehicle.air_density": ("kg/m^3", "air density", _A),
    "vehicle.ackermann": ("-", "steering geometry: 0 = parallel, 1 = full Ackermann", _B),
    # motor, battery, drivetrain
    "drivetrain.layout": ("", "rear-wheel drive, AWD with a spool, or AWD with front overdrive", _B),
    "drivetrain.gear_ratio": ("-", "final drive: motor turns per rear-wheel turn", _B),
    "drivetrain.overdrive_front": ("-", "front / rear wheel speed ratio (awd_overdrive only)", _B),
    "drivetrain.motor_kv": ("rpm/V", "motor speed constant (higher = faster but weaker)", _B),
    "drivetrain.motor_resistance": ("ohm", "motor winding resistance", _A),
    "drivetrain.motor_tau_e": ("s", "motor electrical time constant (throttle lag)", _A),
    "drivetrain.motor_inertia": ("kg m^2", "rotor and pinion inertia", _A),
    "drivetrain.motor_friction_visc": ("N m s/rad", "viscous motor friction", _A),
    "drivetrain.motor_friction_coulomb": ("N m", "dry motor friction (dominates coasting)", _A),
    "drivetrain.current_limit": ("A", "ESC current limit", _B),
    "drivetrain.battery_voltage": ("V", "battery open-circuit voltage (2S LiPo: 7.4-8.4 V)", _B),
    "drivetrain.battery_resistance": ("ohm", "battery and wiring resistance (voltage sag)", _A),
    "drivetrain.drag_brake": ("-", "ESC drag brake at neutral throttle (0..1)", _B),
    "drivetrain.reverse_enabled": ("", "negative throttle drives backwards (off = brake only)", _B),
    "drivetrain.throttle_min": ("-", "throttle below which the ESC treats the command as neutral", _A),
    "drivetrain.rear_diff_lock": ("-", "rear differential: 0 = open ... 1 = locked (spool)", _B),
    "drivetrain.rear_diff_max_torque": ("N m", "limited-slip capacity at the wheels (lock < 1)", _A),
    "drivetrain.rear_diff_visc": ("N m s/rad", "rear viscous coupling at the wheels", _A),
    "drivetrain.front_diff_lock": ("-", "front differential lock (AWD)", _A),
    "drivetrain.front_diff_max_torque": ("N m", "front limited-slip capacity (AWD)", _A),
    "drivetrain.front_diff_visc": ("N m s/rad", "front viscous coupling at the wheels (AWD)", _A),
    "drivetrain.center_lock": ("-", "centre coupling lock (AWD)", _A),
    "drivetrain.center_max_torque": ("N m", "centre coupling capacity at the motor shaft (AWD)", _A),
    "drivetrain.center_visc": ("N m s/rad", "centre viscous coupling at the MOTOR SHAFT (x gear_ratio^2 at the wheels)", _A),
    # steering, throttle, latency
    "actuators.steer_max_deg": ("deg", "maximum steering angle", _B),
    "actuators.servo_rate_deg_s": ("deg/s", "servo slew-rate limit", _B),
    "actuators.servo_tau": ("s", "servo response time constant", _A),
    "actuators.steer_deadband": ("-", "steering command deadband", _A),
    "actuators.steer_offset_deg": ("deg", "steering trim offset", _B),
    "actuators.steer_expo": ("-", "steering expo (0 linear ... 1 cubic)", _A),
    "actuators.throttle_deadband": ("-", "throttle command deadband", _A),
    "actuators.throttle_expo": ("-", "throttle expo (0 linear ... 1 cubic)", _A),
    "actuators.latency": ("s", "control latency from command to actuator (whole control steps)", _B),
    "actuators.gyro_enabled": ("", "steering gyro (yaw-rate feedback, as on real RWD drift cars)", _B),
    "actuators.gyro_gain": ("rad per rad/s", "gyro counter-steer per unit of yaw rate", _B),
    "actuators.gyro_max_correction_deg": ("deg", "gyro authority limit", _A),
    # simulation
    "sim.dt": ("s", "physics time step", _A),
    "sim.control_dt": ("s", "control period (0.02 s = 50 Hz)", _B),
    "sim.integrator": ("", "ODE integrator (RK4 is the converged default)", _A),
    "sim.v_eps": ("m/s", "slip denominator regularization", _A),
    "sim.v_low_start": ("m/s", "below this speed the tire model is fully viscous", _A),
    "sim.v_low": ("m/s", "above this speed the full Magic Formula is used", _A),
    "sim.v_c_low": ("m/s", "slip-velocity scale of the low-speed tire model", _A),
    "sim.v_rr": ("m/s", "rolling-resistance sign smoothing", _A),
    "sim.ambient_temp": ("degC", "air and track temperature (tires cool toward it)", _B),
    "sim.gravity": ("m/s^2", "gravitational acceleration", _A),
    # tire compound (Pacejka Magic Formula, MF5.2-style)
    "tire.fz0": ("N", "nominal wheel load of the coefficients", _A),
    "tire.pcx1": ("-", "longitudinal shape factor C (> 1: grip falls off after the peak)", _A),
    "tire.pdx1": ("-", "peak longitudinal friction coefficient (dry asphalt, optimal temperature)", _B),
    "tire.pdx2": ("-", "load sensitivity of the longitudinal friction", _A),
    "tire.pex1": ("-", "longitudinal curvature factor E", _A),
    "tire.pex2": ("-", "load dependence of the longitudinal curvature", _A),
    "tire.pkx1": ("-", "longitudinal slip stiffness per unit load", _B),
    "tire.pkx2": ("-", "load dependence of the slip stiffness", _A),
    "tire.pkx3": ("-", "exponential load dependence of the slip stiffness", _A),
    "tire.pcy1": ("-", "lateral shape factor C", _A),
    "tire.pdy1": ("-", "peak lateral friction coefficient (dry asphalt, optimal temperature)", _B),
    "tire.pdy2": ("-", "load sensitivity of the lateral friction", _A),
    "tire.pey1": ("-", "lateral curvature factor E", _A),
    "tire.pey2": ("-", "load dependence of the lateral curvature", _A),
    "tire.pky1": ("-", "cornering stiffness factor", _B),
    "tire.pky2": ("-", "load (in units of fz0) where cornering stiffness saturates", _A),
    "tire.combined_mode": ("", "how braking/driving and cornering grip combine", _A),
    "tire.rbx1": ("-", "combined-slip weighting (mf_weighting mode only)", _A),
    "tire.rbx2": ("-", "combined-slip weighting (mf_weighting mode only)", _A),
    "tire.rcx1": ("-", "combined-slip weighting (mf_weighting mode only)", _A),
    "tire.rby1": ("-", "combined-slip weighting (mf_weighting mode only)", _A),
    "tire.rby2": ("-", "combined-slip weighting (mf_weighting mode only)", _A),
    "tire.rby3": ("-", "combined-slip weighting (mf_weighting mode only)", _A),
    "tire.rcy1": ("-", "combined-slip weighting (mf_weighting mode only)", _A),
    "tire.relax_x": ("m", "longitudinal relaxation length (tire response lag)", _B),
    "tire.relax_y": ("m", "lateral relaxation length (tire response lag)", _B),
    "tire.loose_affinity": ("-", "grip change on loose ground (+ pins and spikes, - slick plastic)", _B),
    "tire.heat_capacity": ("J/K", "tread heat capacity", _A),
    "tire.cool_coeff": ("W/K", "cooling at rest", _A),
    "tire.cool_speed_coeff": ("W/K per m/s", "extra cooling per m/s of rolling speed", _A),
    "tire.t_opt": ("degC", "optimal tire temperature (peak grip)", _B),
    "tire.t_width": ("degC", "width of the temperature grip window", _A),
    "tire.temp_mu_drop": ("-", "share of grip lost far outside the temperature window", _A),
    "tire.wear_mu_drop": ("-", "grip lost on a fully worn tire", _A),
    "tire.wear_k_gain": ("-", "stiffness gained on a fully worn tire", _A),
    "tire.wet_mu_drop": ("-", "grip lost on a fully wet tire", _A),
    "tire.wet_k_drop": ("-", "stiffness lost on a fully wet tire", _A),
    "tire.wet_c_gain": ("-", "sharper grip drop-off on a fully wet tire", _A),
    "tire.contamination_mu_drop": ("-", "grip lost with a fully dusty tread", _A),
    "tire.contamination_decay_dist": ("m", "rolling distance for dust to wear off (1/e)", _A),
    # surface
    "surface.mu_scale": ("-", "grip multiplier of this surface", _B),
    "surface.stiffness_scale": ("-", "tire stiffness multiplier on this surface", _B),
    "surface.peak_slip_shift": ("-", "moves the grip peak to larger slip (loose ground)", _A),
    "surface.shape_c_target": ("-", "shape factor the tire curve is blended toward", _A),
    "surface.shape_blend": ("-", "0 keeps the tire's own curve shape ... 1 uses shape_c_target", _A),
    "surface.curvature_e_shift": ("-", "added to the tire curvature factor E", _A),
    "surface.rolling_resistance": ("-", "rolling resistance coefficient Crr", _B),
    "surface.looseness": ("-", "0 paved ... 1 deep sand", _B),
    "surface.loose_drag": ("-", "plowing drag when sliding sideways through loose material", _B),
}
# settings that exist in the dataclasses but have no effect on the simulation yet
HIDDEN = {"vehicle.body_width", "surface.roughness", "surface.mu_patch_std"}

UNIT_OK = re.compile(r"^[A-Za-z0-9^/*. \-]{1,16}$")
GROUP_LABELS = {
    "vehicle": "Chassis", "drivetrain": "Motor, battery & drivetrain", "actuators": "Steering, throttle & latency",
    "sim": "Simulation", "tire": "Tire compound", "surface": "Surface", "condition": "Tire condition",
    "init": "Initial state",
}
CONDITION_FIELDS = [
    dict(key="condition.wear", name="wear", default=0.0, unit="0-1", min=0.0, max=1.0,
         desc="tread wear: lowers peak grip, stiffens the tire (per wheel)", per_wheel=True),
    dict(key="condition.wetness", name="wetness", default=0.0, unit="0-1", min=0.0, max=1.0,
         desc="water film on the tire: lowers grip, sharper drop after the peak (per wheel)", per_wheel=True),
    dict(key="condition.contamination", name="contamination", default=0.0, unit="0-1", min=0.0, max=1.0,
         desc="dust on the tread at the start: lowers grip, wears off with distance (per wheel)", per_wheel=True),
    dict(key="condition.temp0", name="temp0", default=25.0, unit="degC",
         desc="initial tire temperature (grip peaks at the compound's t_opt)", per_wheel=True),
]
INIT_FIELDS = [
    dict(key="init.speed", name="speed", default=0.0, unit="m/s", min=0.0, desc="initial speed"),
    dict(key="init.beta_deg", name="beta_deg", default=0.0, unit="deg", desc="initial sideslip angle"),
    dict(key="init.yaw_rate_deg_s", name="yaw_rate_deg_s", default=0.0, unit="deg/s", desc="initial yaw rate"),
    dict(key="init.x", name="x", default=0.0, unit="m", desc="initial x position"),
    dict(key="init.y", name="y", default=0.0, unit="m", desc="initial y position"),
    dict(key="init.yaw_deg", name="yaw_deg", default=0.0, unit="deg", desc="initial heading"),
]


def _field_comments(cls) -> dict[str, str]:
    """``name: type = default   # comment`` -> {name: comment} from the dataclass source."""
    out: dict[str, str] = {}
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):          # bytecode-only install (e.g. a frozen app): no descriptions
        return out
    for line in src.splitlines():
        m = re.match(r"^\s{4}(\w+):\s*[^=#]+=\s*[^#]*?(?:#\s*(.*))?$", line)
        if m and not m.group(1).isupper():
            out[m.group(1)] = (m.group(2) or "").strip()
    return out


def _split_unit(comment: str) -> tuple[str, str]:
    """Split 'kg m^2 per wheel' style comments into (unit, description) heuristically."""
    if not comment:
        return "", ""
    head, _, tail = comment.partition(",")
    head = head.strip()
    first = head.split(" ")[0] if head else ""
    if UNIT_OK.match(head) and not any(w in head for w in ("share", "fraction", "normalized", "0 ", "per ")):
        return head, tail.strip()
    if first in ("N", "m", "s", "kg", "ohm", "V", "A", "degC", "rad", "J/K", "W/K", "rpm/V"):
        return first, comment[len(first):].strip(" ,;")
    return "", comment


def _num(v: Any) -> Any:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _group_fields(group: str, cls, instance) -> list[dict]:
    comments = _field_comments(cls)
    rows = []
    skip = {"name", "color"} | ({"optical_texture", "wet"} if group == "surface" else set())
    for f in dataclasses.fields(cls):
        if f.name in skip or f.name.isupper():
            continue
        if group == "surface" and f.name not in P.SurfaceParams.NUMERIC_FIELDS:
            continue
        value = getattr(instance, f.name)
        unit, desc = _split_unit(comments.get(f.name, ""))
        name = f.name
        if (group, f.name) in ANGLE_FIELDS:
            name, unit = ANGLE_FIELDS[(group, f.name)]
            value = math.degrees(float(value))
        key = f"{group}.{name}"
        if key in HIDDEN:
            continue
        unit, desc, level = META.get(key, (unit, desc, "advanced"))
        if isinstance(value, bool):
            ftype = "bool"
        elif key in CHOICES:
            ftype = "choice"
        elif value is None:
            ftype = "optional_float"          # e.g. yaw_inertia: null = estimated
        elif isinstance(value, str):
            ftype = "choice"
        else:
            ftype = "float"
        row = dict(key=key, name=name, default=_num(value), unit=unit, desc=desc, type=ftype,
                   structural=key in STRUCTURAL, level=level)
        if key in CHOICES:
            row["choices"] = CHOICES[key]
        if key in POSITIVE:
            row["min_exclusive"] = 0.0
        elif key in NONNEGATIVE:
            row["min"] = 0.0
        rows.append(row)
    return rows


@lru_cache(maxsize=1)
def build_catalog() -> dict:
    """The full catalog (JSON-serializable). Cached: the configs are read once per process."""
    vp, dp, ap, sp = P.load_vehicle()
    tires = P.load_tires()
    surfaces = P.load_surfaces()
    default_tire, default_surface = "hard_plastic_drift", "epoxy_ptile"
    groups = []
    for gid, cls, inst in (("vehicle", P.VehicleParams, vp), ("drivetrain", P.DrivetrainParams, dp),
                           ("actuators", P.ActuatorParams, ap), ("sim", P.SimParams, sp)):
        groups.append(dict(id=gid, label=GROUP_LABELS[gid], fields=_group_fields(gid, cls, inst)))
    tire_fields = _group_fields("tire", P.TireParams, tires[default_tire])
    groups.append(dict(
        id="tire", label=GROUP_LABELS["tire"], choice_key="tire", choices=list(tires), default_choice=default_tire,
        fields=tire_fields,
        choice_defaults={n: {r["key"]: _num(getattr(t, r["name"])) for r in tire_fields} for n, t in tires.items()}))
    surf_fields = _group_fields("surface", P.SurfaceParams, surfaces[default_surface])
    groups.append(dict(
        id="surface", label=GROUP_LABELS["surface"], choice_key="surface", choices=list(surfaces),
        default_choice=default_surface, fields=surf_fields,
        choice_defaults={n: {r["key"]: _num(getattr(s, r["name"])) for r in surf_fields} for n, s in surfaces.items()},
        colors={n: s.color for n, s in surfaces.items()}))
    groups.append(dict(id="condition", label=GROUP_LABELS["condition"],
                       fields=[dict(r, type="float", structural=False, level="basic") for r in CONDITION_FIELDS]))
    groups.append(dict(id="init", label=GROUP_LABELS["init"],
                       fields=[dict(r, type="float", structural=False, level="basic") for r in INIT_FIELDS]))
    from .inputs import MANEUVERS
    from .export import FORMATS, SIGNAL_GROUPS, available_formats
    return dict(groups=groups, maneuvers=MANEUVERS, signals=SIGNAL_GROUPS, formats=available_formats(),
                all_formats=list(FORMATS), structural=sorted(STRUCTURAL))


def field_index() -> dict[str, dict]:
    """{key: field row} over every group (plus the 'tire' and 'surface' categorical keys)."""
    cat = build_catalog()
    idx: dict[str, dict] = {}
    for g in cat["groups"]:
        for r in g["fields"]:
            idx[r["key"]] = r
        if "choice_key" in g:
            idx[g["choice_key"]] = dict(key=g["choice_key"], name=g["choice_key"], type="choice",
                                        choices=g["choices"], default=g["default_choice"], structural=False,
                                        desc=f"{g['label']} (name from the YAML configs)", unit="")
    return idx
