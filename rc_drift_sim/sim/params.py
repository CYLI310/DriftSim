"""Parameter dataclasses and YAML loaders shared by the NumPy reference and the JAX port.

All dataclasses are frozen. Vary them with ``dataclasses.replace``. Angles are radians
internally; YAML keys ending in ``_deg`` are converted by the loaders and the suffix stripped.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

FL, FR, RL, RR = 0, 1, 2, 3
WHEEL_NAMES = ("FL", "FR", "RL", "RR")


# ----------------------------------------------------------------------------- vehicle
@dataclass(frozen=True)
class VehicleParams:
    mass: float = 1.6                 # kg
    wheelbase: float = 0.257          # m
    track_width: float = 0.19         # m
    cg_height: float = 0.035          # m
    cg_to_front: float = 0.133        # m, CG to front axle (a). b = wheelbase - a
    yaw_inertia: float | None = None  # kg m^2; None -> estimated from mass and dimensions
    body_length: float = 0.36         # m, used for the yaw inertia estimate and drawing
    body_width: float = 0.19          # m, drawing only
    wheel_radius: float = 0.032       # m
    wheel_inertia: float = 1.5e-5     # kg m^2 per wheel (tire + rim + axle)
    roll_stiffness_front: float = 0.5 # share of lateral load transfer carried by the front axle
    load_transfer_tau: float = 0.05   # s, first-order lag approximating suspension
    drag_coeff_area: float = 0.02     # Cd*A in m^2
    air_density: float = 1.2          # kg/m^3
    ackermann: float = 0.0            # 0 = parallel steering, 1 = full Ackermann

    @property
    def cg_to_rear(self) -> float:
        return self.wheelbase - self.cg_to_front

    @property
    def iz(self) -> float:
        if self.yaw_inertia is not None:
            return float(self.yaw_inertia)
        return self.mass * (self.body_length ** 2 + self.track_width ** 2) / 12.0

    def wheel_positions(self) -> np.ndarray:
        """(4, 2) body-frame contact positions, order FL FR RL RR."""
        a, b, hw = self.cg_to_front, self.cg_to_rear, 0.5 * self.track_width
        return np.array([[a, hw], [a, -hw], [-b, hw], [-b, -hw]], dtype=float)

    def static_loads(self, g: float = 9.81) -> np.ndarray:
        """(4,) static wheel loads in N."""
        w = self.mass * g
        f = w * self.cg_to_rear / (2.0 * self.wheelbase)
        r = w * self.cg_to_front / (2.0 * self.wheelbase)
        return np.array([f, f, r, r], dtype=float)


# ----------------------------------------------------------------------------- tire
@dataclass(frozen=True)
class TireParams:
    """One tire compound: MF5.2-style dimensionless Magic Formula coefficients.

    dfz = (Fz - fz0)/fz0. See docs/DESIGN.md section 4 for the formulas.
    """
    name: str = "hard_plastic_drift"
    fz0: float = 4.0                  # nominal wheel load (N)
    # longitudinal pure slip
    pcx1: float = 1.3
    pdx1: float = 0.45
    pdx2: float = -0.05
    pex1: float = -0.5
    pex2: float = 0.0
    pkx1: float = 12.0
    pkx2: float = 0.0
    pkx3: float = 0.0
    # lateral pure slip
    pcy1: float = 1.25
    pdy1: float = 0.42
    pdy2: float = -0.05
    pey1: float = -0.5
    pey2: float = 0.0
    pky1: float = 8.0
    pky2: float = 1.5
    # combined slip
    combined_mode: str = "similarity"   # "similarity" | "mf_weighting" | "ellipse"
    rbx1: float = 10.0
    rbx2: float = 8.0
    rcx1: float = 1.0
    rby1: float = 8.0
    rby2: float = 6.0
    rby3: float = 0.0
    rcy1: float = 1.0
    # relaxation lengths (m)
    relax_x: float = 0.03
    relax_y: float = 0.04
    # surface affinity: mu *= 1 + loose_affinity * surface.looseness
    loose_affinity: float = 0.0
    # thermal model
    heat_capacity: float = 15.0       # J/K
    cool_coeff: float = 0.15          # W/K at rest
    cool_speed_coeff: float = 0.05    # W/K per m/s
    t_opt: float = 45.0               # degC, optimal temperature: the YAML mu values hold here
    t_width: float = 30.0             # degC, 1/e half-width of the Gaussian grip window
    temp_mu_drop: float = 0.15        # peak-mu fraction lost far outside the window
    # tire condition sensitivities (see tire.condition_scales)
    wear_mu_drop: float = 0.2         # mu *= 1 - wear_mu_drop * wear
    wear_k_gain: float = 0.3          # K *= 1 + wear_k_gain * wear (worn tread is stiffer)
    wet_mu_drop: float = 0.4          # mu *= 1 - wet_mu_drop * wetness
    wet_k_drop: float = 0.15          # K *= 1 - wet_k_drop * wetness (water film softens the build-up)
    wet_c_gain: float = 0.15          # C += wet_c_gain * wetness (sharper drop after the peak when wet)
    contamination_mu_drop: float = 0.5  # mu *= 1 - contamination_mu_drop * contamination
    contamination_decay_dist: float = 3.0  # m of rolling to decay contamination by 1/e


@dataclass(frozen=True)
class TireCondition:
    """Per-wheel tire condition, sampled per episode. Arrays of shape (4,)."""
    wear: np.ndarray = field(default_factory=lambda: np.zeros(4))
    wetness: np.ndarray = field(default_factory=lambda: np.zeros(4))
    contamination: np.ndarray = field(default_factory=lambda: np.zeros(4))
    temp0: np.ndarray = field(default_factory=lambda: np.full(4, 25.0))

    @staticmethod
    def neutral(ambient: float = 25.0) -> "TireCondition":
        return TireCondition(temp0=np.full(4, float(ambient)))


# ----------------------------------------------------------------------------- surface
@dataclass(frozen=True)
class SurfaceParams:
    """One surface type. Numeric fields may be scalars or (4,) arrays (per wheel)."""
    name: str = "epoxy_ptile"
    mu_scale: Any = 1.0               # multiplies peak mu
    stiffness_scale: Any = 1.0        # multiplies Kx, Ky
    peak_slip_shift: Any = 0.0        # slip input divided by (1 + shift): peak moves to larger slip
    shape_c_target: Any = 1.0         # C blended toward this: C + shape_blend*(target - C)
    shape_blend: Any = 0.0            # 0 keeps the tire's own C; 1 forces C = shape_c_target
    curvature_e_shift: Any = 0.0      # additive on E (clamped <= 1)
    rolling_resistance: Any = 0.015   # Crr
    looseness: Any = 0.0              # 0 paved ... 1 deep sand; scales loose_affinity and plowing
    loose_drag: Any = 0.0             # plowing drag coefficient (fraction of Fz per plow_v_ref of lateral slide)
    roughness: Any = 0.0              # normal-load noise amplitude, fraction of Fz (Milestone 2)
    mu_patch_std: Any = 0.0           # per-patch random mu variation, std as fraction (Milestone 2)
    optical_texture: Any = 1.0        # 0..1, optical flow reliability (Milestone 6)
    wet: Any = 0.0                    # 0/1 flag for sensor dropouts (Milestone 6)
    color: Any = "#888888"            # drawing only

    NUMERIC_FIELDS = ("mu_scale", "stiffness_scale", "peak_slip_shift", "shape_c_target",
                      "shape_blend", "curvature_e_shift", "rolling_resistance", "looseness",
                      "loose_drag", "roughness", "mu_patch_std", "optical_texture", "wet")


# ----------------------------------------------------------------------------- drivetrain
@dataclass(frozen=True)
class DrivetrainParams:
    layout: str = "rwd"               # "rwd" | "awd_spool" | "awd_overdrive"
    gear_ratio: float = 8.0           # motor rev per rear wheel rev
    overdrive_front: float = 1.0      # front wheel speed / rear wheel speed for awd_overdrive
    motor_kv: float = 3000.0          # rpm/V
    motor_resistance: float = 0.03    # ohm
    motor_tau_e: float = 0.002        # s, electrical time constant (L = R*tau); also the throttle lag
    motor_inertia: float = 5e-6       # kg m^2 (rotor + pinion)
    motor_friction_visc: float = 1e-6 # N m s/rad
    motor_friction_coulomb: float = 0.002  # N m
    current_limit: float = 40.0       # A
    battery_voltage: float = 7.4      # V
    battery_resistance: float = 0.02  # ohm (pack + ESC + wiring)
    drag_brake: float = 0.0           # 0..1 ESC drag brake fraction at neutral throttle
    reverse_enabled: bool = True
    throttle_min: float = 0.02        # |thr| below which the ESC output is treated as neutral
    # rear differential: lock 0 = open, 1 = locked; max_torque = LSD clutch capacity (N m at wheel)
    rear_diff_lock: float = 1.0
    rear_diff_max_torque: float = 1e9
    rear_diff_visc: float = 0.02      # N m s/rad at the WHEELS, viscous coupling (also stabilizes spools)
    front_diff_lock: float = 1.0
    front_diff_max_torque: float = 1e9
    front_diff_visc: float = 0.02     # N m s/rad at the wheels
    center_lock: float = 1.0          # AWD layouts only
    center_max_torque: float = 1e9    # N m at the motor shaft
    center_visc: float = 5e-4         # N m s/rad at the MOTOR SHAFT: acts x G^2 at the wheels, keep small

    @property
    def ke(self) -> float:
        return 60.0 / (2.0 * math.pi * self.motor_kv)

    @property
    def kt(self) -> float:
        return self.ke

    @property
    def inductance(self) -> float:
        return self.motor_resistance * self.motor_tau_e


# ----------------------------------------------------------------------------- actuators
@dataclass(frozen=True)
class ActuatorParams:
    steer_max: float = math.radians(35.0)       # rad
    servo_rate: float = math.radians(600.0)     # rad/s
    servo_tau: float = 0.02                     # s
    steer_deadband: float = 0.02                # normalized command
    steer_offset: float = 0.0                   # rad (randomization / trim)
    steer_expo: float = 0.0                     # 0 linear ... 1 cubic-ish
    throttle_deadband: float = 0.02
    throttle_expo: float = 0.0
    latency: float = 0.02                       # s, control latency buffer
    gyro_enabled: bool = False
    gyro_gain: float = 0.0                      # rad of steering per rad/s of yaw rate
    gyro_max_correction: float = math.radians(20.0)


# ----------------------------------------------------------------------------- sim
@dataclass(frozen=True)
class SimParams:
    dt: float = 0.001                 # physics step
    control_dt: float = 0.02          # control period (50 Hz)
    integrator: str = "rk4"           # "rk4" | "semi_implicit_euler"
    v_eps: float = 0.3                # slip denominator regularization (m/s)
    v_low_start: float = 0.1          # low-speed blend fully viscous below this (m/s)
    v_low: float = 0.3                # low-speed blend fully Magic Formula above this (m/s)
    v_c_low: float = 0.15             # slip-velocity scale of the low-speed viscous model (m/s)
    v_rr: float = 0.05                # rolling-resistance sign smoothing (m/s)
    ambient_temp: float = 25.0        # degC
    gravity: float = 9.81

    @property
    def n_substeps(self) -> int:
        n = int(round(self.control_dt / self.dt))
        if abs(n * self.dt - self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of dt")
        return n


# ----------------------------------------------------------------------------- bundle
@dataclass(frozen=True)
class Params:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    tire: TireParams = field(default_factory=TireParams)
    surface: SurfaceParams = field(default_factory=SurfaceParams)
    drivetrain: DrivetrainParams = field(default_factory=DrivetrainParams)
    actuators: ActuatorParams = field(default_factory=ActuatorParams)
    sim: SimParams = field(default_factory=SimParams)


# ----------------------------------------------------------------------------- YAML
def _convert_deg_keys(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.endswith("_deg"):
            out[k[:-4]] = math.radians(float(v))
        elif k.endswith("_deg_s"):
            out[k[:-6]] = math.radians(float(v))
        else:
            out[k] = v
    return out


def from_dict(cls, d: dict[str, Any]):
    """Build a dataclass from a dict, converting *_deg keys; unknown keys raise."""
    d = _convert_deg_keys(dict(d))
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**d)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_vehicle(path: str | Path | None = None):
    """vehicle.yaml -> (VehicleParams, DrivetrainParams, ActuatorParams, SimParams)."""
    d = load_yaml(path or CONFIG_DIR / "vehicle.yaml")
    return (from_dict(VehicleParams, d.get("vehicle", {})),
            from_dict(DrivetrainParams, d.get("drivetrain", {})),
            from_dict(ActuatorParams, d.get("actuators", {})),
            from_dict(SimParams, d.get("sim", {})))


def load_tires(path: str | Path | None = None) -> dict[str, TireParams]:
    d = load_yaml(path or CONFIG_DIR / "tires.yaml")
    return {name: from_dict(TireParams, {"name": name, **spec}) for name, spec in d["tires"].items()}


def load_surfaces(path: str | Path | None = None) -> dict[str, SurfaceParams]:
    d = load_yaml(path or CONFIG_DIR / "surfaces.yaml")
    return {name: from_dict(SurfaceParams, {"name": name, **spec}) for name, spec in d["surfaces"].items()}


def default_params(tire: str = "hard_plastic_drift", surface: str = "epoxy_ptile",
                   config_dir: str | Path | None = None) -> Params:
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    vp, dp, ap, sp = load_vehicle(cdir / "vehicle.yaml")
    tires = load_tires(cdir / "tires.yaml")
    surfaces = load_surfaces(cdir / "surfaces.yaml")
    return Params(vehicle=vp, tire=tires[tire], surface=surfaces[surface],
                  drivetrain=dp, actuators=ap, sim=sp)


def replace(obj, **kw):
    return dataclasses.replace(obj, **kw)
