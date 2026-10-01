"""Planar 4-wheel vehicle model: the ODE right-hand side and the ``Vehicle`` step/rollout wrapper.

Implements docs/DESIGN.md section 5 for the NumPy reference. ``derivatives`` is a pure
function of the flat state ``s`` (layout in ``state.py``), the (already latency-delayed)
action ``u = (steer_cmd, throttle_cmd)`` in [-1, 1], the parameter bundle ``p``, a per-wheel
surface ``surf`` and a per-wheel tire condition ``cond``. It calls the contract modules
``tire``, ``drivetrain`` and ``actuators`` for the tire, motor/diff and servo math and never
re-implements them.

Conventions (DESIGN.md section 1)
---------------------------------
* SI units: m, s, kg, N, N m, rad, A, degC. Wheel order FL, FR, RL, RR; per-wheel arrays
  (..., 4) with an optional leading batch shape (states (..., NS)).
* World frame x east, y north, yaw CCW from +x. Body frame x forward, y left; positive yaw rate
  = turning left; positive steer = wheels turned left. Sideslip ``beta = atan2(vy, vx)``.
* Wheel-frame velocities ``(vxw, vyw)`` = body-frame contact velocity rotated by ``-delta_i``.
* Loads ``Fz >= 0`` (clamped) and built from the LAG states ``DFZ_LONG``/``DFZ_LAT``; the lag
  targets use the specific force computed in the same evaluation (no algebraic loop).
* The Magic Formula is fed the LAGGED slips ``KAPPA_LAG``/``ALPHA_LAG`` (relaxation-length ODE).

Style: NumPy float64, vectorized over the wheel axis, JAX-portable (no in-place mutation, no
Python branching on array values; Python control flow only on configuration).

Public API
----------
    compile_model(params, surf=None, cond=None) -> VehicleModel     precomputed constants
    derivatives_model(s, u, vm, want_info=False) -> (ds, info|None)   fast batch-capable core
    derivatives(s, u, p, surf, cond) -> (ds, info)                    contract API (cached model)
    rhs(s, u, p, surf, cond) -> ds, rhs_model(s, u, vm) -> ds          integrator adapters
    Trajectory                                                         t, states, actions, info
    Vehicle(params, surf=None, cond=None)
        .initial_state(x=0, y=0, yaw=0, v=0, beta=0, yaw_rate=0) -> s
        .step(s, action, n_sub=None, want_info=True) -> (s_new, info)   s (NS,) or (B, NS)
        .rollout(s0, actions, control_dt=None, n_steps=None, duration=None, t0=0.0) -> Trajectory
        .rollout_batch(s0 (B,NS), actions (T,B,2), record_info=False) -> (states (T+1,B,NS), info)
    compile_model_batch(params_list, surfs, conds) -> VehicleModel   B cars, different parameters
    VehicleBatch(params_list, surfs=None, conds=None)
        .initial_states(...) -> (B, NS), .step(s, u, want_info=False) -> (s_new, info)
    make_vehicle(tire="hard_plastic_drift", surface="epoxy_ptile", **overrides) -> Vehicle
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping

import numpy as np

from . import actuators, drivetrain, integrator, tire
from .params import Params, SurfaceParams, TireCondition, default_params
from .state import (ALPHA_LAG, CONTAM, DELTA, DFZ_LAT, DFZ_LONG, I_MOTOR, KAPPA_LAG, NS, OMEGA,
                    R, T_TIRE, VX, VY, X, Y, YAW)
from .surface import uniform_surface

# ``derivatives`` assembles ``ds`` by concatenation in the canonical order; make sure the
# layout in state.py is the one this module was written against.
assert (X, Y, YAW, VX, VY, R, DELTA, I_MOTOR, DFZ_LONG, DFZ_LAT) == (0, 1, 2, 3, 4, 5, 10, 11, 16, 17)
assert (OMEGA, T_TIRE, KAPPA_LAG, ALPHA_LAG, CONTAM) == (
    slice(6, 10), slice(12, 16), slice(18, 22), slice(22, 26), slice(26, 30))
assert NS == 30

# sign pattern of the longitudinal load-transfer state on the four wheels (FL, FR, RL, RR)
_LONG_SIGN = np.array([-1.0, -1.0, 1.0, 1.0])      # + DFZ_LONG = rear axle gains
_FRONT = np.array([1.0, 1.0, 0.0, 0.0])
_REAR = np.array([0.0, 0.0, 1.0, 1.0])

ActionFn = Callable[[float, np.ndarray, dict], Any]


# ----------------------------------------------------------------------------- compiled model
@dataclass(frozen=True, slots=True)
class VehicleModel:
    """Everything ``derivatives`` needs, precomputed once from (params, surface, condition).

    Build with :func:`compile_model`. Per-wheel arrays are (4,); scalars are Python floats.
    """
    params: Params
    tm: tire.TireModel
    ls: tire.LowSpeed
    dm: drivetrain.DrivetrainModel
    am: actuators.ActuatorModel
    inv_m: float
    m: float
    inv_iz: float
    Rw: float
    xi: np.ndarray            # (4,) contact x in the body frame (m)
    yi: np.ndarray            # (4,) contact y in the body frame (m)
    fz_static: np.ndarray     # (4,) static loads (N)
    long_pat: np.ndarray      # (4,) Fz change per N of DFZ_LONG
    lat_pat: np.ndarray       # (4,) Fz change per N of DFZ_LAT
    aero_k: float             # 0.5*rho*CdA (kg/m)
    lt_long: float            # m*h/L  (N per m/s^2)
    lt_lat: float             # m*h/w
    inv_tau_lt: float
    L: float
    half_track: float
    ackermann: float
    v_eps: float
    inv_v_rr: float
    t_amb: float


def compile_model(params: Params, surf: SurfaceParams | None = None,
                  cond: TireCondition | None = None) -> VehicleModel:
    """Precompute a :class:`VehicleModel`.

    ``surf`` defaults to ``uniform_surface(params.surface)``; ``cond`` to a neutral condition
    (new, dry, clean tires). Temperature and contamination are STATES and act dynamically;
    ``cond`` contributes the static wear and wetness factors (and the initial state values via
    ``Vehicle.initial_state``).
    """
    vp, sim = params.vehicle, params.sim
    surf = uniform_surface(params.surface) if surf is None else surf
    pos = vp.wheel_positions()
    rf = float(vp.roll_stiffness_front)
    return VehicleModel(
        params=params,
        tm=tire.tire_model(params.tire, surf, cond),
        ls=tire.low_speed_consts(sim),
        dm=drivetrain.drivetrain_model(params.drivetrain, vp),
        am=actuators.actuator_model(params.actuators),
        m=float(vp.mass), inv_m=1.0 / float(vp.mass), inv_iz=1.0 / float(vp.iz),
        Rw=float(vp.wheel_radius),
        xi=pos[:, 0].copy(), yi=pos[:, 1].copy(),
        fz_static=vp.static_loads(float(sim.gravity)),
        long_pat=0.5 * _LONG_SIGN,
        lat_pat=np.array([-rf, rf, -(1.0 - rf), 1.0 - rf]),
        aero_k=0.5 * float(vp.air_density) * float(vp.drag_coeff_area),
        lt_long=float(vp.mass) * float(vp.cg_height) / float(vp.wheelbase),
        lt_lat=float(vp.mass) * float(vp.cg_height) / float(vp.track_width),
        inv_tau_lt=1.0 / float(vp.load_transfer_tau),
        L=float(vp.wheelbase), half_track=0.5 * float(vp.track_width),
        ackermann=float(vp.ackermann),
        v_eps=float(sim.v_eps), inv_v_rr=1.0 / float(sim.v_rr), t_amb=float(sim.ambient_temp),
    )


# ----------------------------------------------------------------------------- per-car parameter batches
# Every numeric model field acts either on per-wheel quantities (shape (..., 4)) or on per-car
# scalars (shape (...)). Stacking B cars therefore gives (B, 1) / (B, 4) arrays for the first kind
# and (B,) arrays for the second, and ``derivatives_model`` runs unchanged on (B, NS) states.
_WHEEL_FLOATS = {  # float fields used against per-wheel arrays -> stacked to (B, 1)
    "VehicleModel": {"Rw", "v_eps", "inv_v_rr", "t_amb"},
    "TireModel": "all", "LowSpeed": "all", "DrivetrainModel": set(), "ActuatorModel": set(),
}
_SHARED = {  # structural fields that must be identical across the batch
    "TireModel": {"mode", "use_temp"}, "DrivetrainModel": {"front_driven", "reverse_enabled", "driven"},
}
STRUCTURAL = ("tire.combined_mode", "drivetrain.layout", "drivetrain.reverse_enabled",
              "sim.dt", "sim.control_dt", "sim.integrator")


def _stack_fields(objs: list, kind: str, skip: tuple = ()) -> Any:
    """Stack a list of identical-type model objects (dataclass or NamedTuple) field by field;
    ``skip`` fields keep the first object's value (the caller replaces them)."""
    first = objs[0]
    names = first._fields if isinstance(first, tuple) else [f.name for f in fields(first)]
    wheel = _WHEEL_FLOATS.get(kind, set())
    shared = _SHARED.get(kind, set())
    out = {}
    for name in names:
        vals = [getattr(o, name) for o in objs]
        v0 = vals[0]
        if name in skip:
            out[name] = v0
        elif name in shared or isinstance(v0, (str, bool)):
            if name == "has_plow":
                out[name] = any(vals)
                continue
            same = all(np.array_equal(v, v0) if isinstance(v0, np.ndarray) else v == v0 for v in vals)
            if not same:
                raise ValueError(f"{kind}.{name} differs across the batch; it must be shared "
                                 f"(structural settings: {', '.join(STRUCTURAL)})")
            out[name] = v0
        elif isinstance(v0, np.ndarray) and v0.ndim >= 1:
            out[name] = np.stack([np.asarray(v, dtype=np.float64) for v in vals])
        elif isinstance(v0, np.ndarray) and kind == "TireModel":        # 0-d surface field
            out[name] = np.stack([np.broadcast_to(np.asarray(v, dtype=np.float64), (4,)) for v in vals])
        else:
            arr = np.array([float(v) for v in vals], dtype=np.float64)
            out[name] = arr[:, None] if (wheel == "all" or name in wheel) else arr
    return type(first)(**out)


def compile_model_batch(params_list: list[Params], surfs: list[SurfaceParams] | None = None,
                        conds: list[TireCondition | None] | None = None) -> VehicleModel:
    """One ``VehicleModel`` for B cars with different numeric parameters (mass, tire compound,
    surface, wear, latency-free actuator settings, gearing, ...), stepped together on (B, NS) states.

    Structural settings (``STRUCTURAL``: combined-slip mode, drivetrain layout, reverse enable,
    time steps, integrator) must be identical across the batch; group cars by them first.
    """
    B = len(params_list)
    if B == 0:
        raise ValueError("empty batch")
    surfs = [None] * B if surfs is None else list(surfs)
    conds = [None] * B if conds is None else list(conds)
    ref = params_list[0]
    for key in STRUCTURAL:
        g, f = key.split(".")
        vals = {getattr(getattr(p, g), f) for p in params_list}
        if len(vals) > 1:
            raise ValueError(f"{key} must be shared across a batch, got {sorted(map(str, vals))}")
    models = [compile_model(p, s, c) for p, s, c in zip(params_list, surfs, conds)]
    vm = _stack_fields(models, "VehicleModel", skip=("params", "tm", "ls", "dm", "am"))
    return dataclasses.replace(
        vm, params=ref,
        tm=_stack_fields([m.tm for m in models], "TireModel"),
        ls=_stack_fields([m.ls for m in models], "LowSpeed"),
        dm=_stack_fields([m.dm for m in models], "DrivetrainModel"),
        am=_stack_fields([m.am for m in models], "ActuatorModel"))


# small identity-keyed cache so the functional API ``derivatives(s, u, p, surf, cond)`` does not
# recompile on every call (params are frozen; the cache holds strong refs so ids stay unique)
_MODEL_CACHE: dict[tuple[int, int, int], tuple[Any, Any, Any, VehicleModel]] = {}


def _cached_model(p: Params, surf: SurfaceParams, cond: TireCondition | None) -> VehicleModel:
    key = (id(p), id(surf), id(cond))
    hit = _MODEL_CACHE.get(key)
    if hit is not None and hit[0] is p and hit[1] is surf and hit[2] is cond:
        return hit[3]
    vm = compile_model(p, surf, cond)
    if len(_MODEL_CACHE) > 64:
        _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = (p, surf, cond, vm)
    return vm


# ----------------------------------------------------------------------------- ODE right-hand side
def derivatives_model(s: np.ndarray, u: np.ndarray, vm: VehicleModel, want_info: bool = False
                      ) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """Time derivative of the state (the fast, batch-capable core; DESIGN.md section 5).

    s : (..., NS) state(s), any leading batch shape; never mutated.
    u : (..., 2) ``(steer_cmd, throttle_cmd)`` in [-1, 1] (broadcast against the batch).
    vm : compiled model (``compile_model``).
    want_info : also return the diagnostics dict (costs extra allocations; the integrator's
        inner stages call with False).

    Returns ``(ds, info)`` with ds (..., NS) in SI units per second and info None unless
    ``want_info``. See :func:`derivatives` for the info keys and the physics.
    """
    s = np.asarray(s, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    tm, dm, am = vm.tm, vm.dm, vm.am
    steer_cmd, thr_cmd = u[..., 0], u[..., 1]
    psi = s[..., YAW]
    vx, vy, r = s[..., VX], s[..., VY], s[..., R]
    omega = s[..., OMEGA]
    delta = s[..., DELTA]
    i_m = s[..., I_MOTOR]
    T_tire = s[..., T_TIRE]
    dfz_long, dfz_lat = s[..., DFZ_LONG], s[..., DFZ_LAT]
    kappa_lag, alpha_lag = s[..., KAPPA_LAG], s[..., ALPHA_LAG]
    contam = s[..., CONTAM]
    vx_, vy_, r_ = vx[..., None], vy[..., None], r[..., None]

    # --- per-wheel steer angles (rear wheels do not steer: cos = 1, sin = 0 there, so the trig
    #     is only evaluated for the front axle)
    if np.all(vm.ackermann == 0.0):                       # parameter check (parallel steer)
        delta_w = delta[..., None] * _FRONT
        cd = np.cos(delta)[..., None] * _FRONT + _REAR
        sd = np.sin(delta)[..., None] * _FRONT
    else:
        d_fl, d_fr = actuators.ackermann_angles_m(vm.L, vm.half_track, vm.ackermann, delta)
        d_front = np.stack([d_fl, d_fr], axis=-1)
        pad = np.zeros(d_front.shape, dtype=np.float64)
        delta_w = np.concatenate([d_front, pad], axis=-1)
        cd = np.concatenate([np.cos(d_front), pad + 1.0], axis=-1)
        sd = np.concatenate([np.sin(d_front), pad], axis=-1)

    # --- contact velocities: body frame, then rotated by -delta_i into the wheel frame
    vcx = vx_ - r_ * vm.yi
    vcy = vy_ + r_ * vm.xi
    vxw = cd * vcx + sd * vcy
    vyw = cd * vcy - sd * vcx

    # --- instantaneous slips and relaxation ODEs
    omega_R = omega * vm.Rw
    v_reg = np.maximum(np.abs(vxw), vm.v_eps)
    kappa = (omega_R - vxw) / v_reg
    alpha = np.arctan2(vyw, v_reg)
    dkappa_lag = (v_reg * tm.inv_relax_x) * (kappa - kappa_lag)
    dalpha_lag = (v_reg * tm.inv_relax_y) * (alpha - alpha_lag)

    # --- wheel loads from the lag STATES
    Fz = np.maximum(vm.fz_static + vm.long_pat * dfz_long[..., None]
                    + vm.lat_pat * dfz_lat[..., None], 0.0)

    # --- tire friction forces (wheel frame) on the LAGGED slips
    (Fx, Fy, c, w_low, vsx, vsy, v_wheel,
     F_mf_x, F_mf_y, F_low_x, F_low_y) = tire.forces_model(
        tm, vm.ls, kappa_lag, alpha_lag, vxw, vyw, omega_R, Fz, T_tire, contam)
    if tm.has_plow:                                       # configuration branch
        F_plow = tire.plowing_force(tm.loose_drag, vyw, Fz)
        Fy_tot = Fy + F_plow
    else:
        F_plow = None
        Fy_tot = Fy

    # --- body-frame forces, aero, accelerations
    Fbx = Fx * cd - Fy_tot * sd
    Fby = Fx * sd + Fy_tot * cd
    v = np.sqrt(vx * vx + vy * vy)
    ka = vm.aero_k * v
    ax = (np.sum(Fbx, axis=-1) - ka * vx) * vm.inv_m
    ay = (np.sum(Fby, axis=-1) - ka * vy) * vm.inv_m
    Mz = np.sum(vm.xi * Fby - vm.yi * Fbx, axis=-1)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    # --- wheel torques and drivetrain
    T_rr = tm.crr * Fz * vm.Rw * np.tanh(omega_R * vm.inv_v_rr)
    tau = vm.Rw * Fx + T_rr
    thr = actuators.throttle_command_m(am, thr_cmd)
    omega_m = drivetrain.shaft_speed_m(dm, omega)
    di = drivetrain.current_derivative_m(dm, i_m, omega_m, thr)
    T_motor = drivetrain.motor_torque_m(dm, i_m, omega_m)
    domega = drivetrain.wheel_accelerations_m(dm, omega, T_motor, tau)

    # --- steering servo
    delta_target = actuators.steering_target_m(am, steer_cmd, r)
    ddelta = actuators.steering_rate_m(am, delta, delta_target)

    # --- load-transfer lag (targets from this evaluation's specific force, incl. aero)
    ddfz_long = (vm.lt_long * ax - dfz_long) * vm.inv_tau_lt
    ddfz_lat = (vm.lt_lat * ay - dfz_lat) * vm.inv_tau_lt

    # --- tire thermal state and contamination wear-off
    heating = np.abs(Fx * vsx) + np.abs(Fy * vsy)
    cooling = (tm.cool_coeff + tm.cool_speed_coeff * v_wheel) * (T_tire - vm.t_amb)
    dT = (heating - cooling) * tm.inv_heat_capacity
    dcontam = -(v_wheel * tm.inv_contam_dist) * contam

    ds = np.concatenate([
        np.stack([vx * cpsi - vy * spsi, vx * spsi + vy * cpsi, r,
                  ax + vy * r, ay - vx * r, Mz * vm.inv_iz], axis=-1),  # X Y YAW VX VY R
        domega,                                                          # OMEGA (4)
        np.stack([np.broadcast_to(ddelta, di.shape), di], axis=-1),      # DELTA I_MOTOR
        dT,                                                              # T_TIRE (4)
        np.stack([ddfz_long, ddfz_lat], axis=-1),                        # DFZ_LONG DFZ_LAT
        dkappa_lag,                                                      # KAPPA_LAG (4)
        dalpha_lag,                                                      # ALPHA_LAG (4)
        dcontam,                                                         # CONTAM (4)
    ], axis=-1)
    if not want_info:
        return ds, None

    v_batt, v_motor = drivetrain.voltages_m(dm, i_m, omega_m, thr)
    info: dict[str, np.ndarray] = dict(
        Fx=Fx, Fy=Fy, Fz=Fz, kappa=kappa, alpha=alpha, kappa_lag=kappa_lag, alpha_lag=alpha_lag,
        delta_w=delta_w, vxw=vxw, vyw=vyw, Fbx=Fbx, Fby=Fby,
        F_plow=(np.zeros_like(Fy) if F_plow is None else F_plow),
        mu_x=c.mu_x, mu_y=c.mu_y, w_low=w_low, vsx=vsx, vsy=vsy, v_wheel=v_wheel,
        F_mf_x=F_mf_x, F_mf_y=F_mf_y, tau=tau, T_rr=T_rr,
        ax=ax, ay=ay, beta=np.arctan2(vy, vx), v=v,
        T_motor=T_motor, omega_m=omega_m, i_motor=i_m, v_motor=v_motor, v_batt=v_batt, thr=thr,
        delta_target=delta_target, F_aero=np.stack([-ka * vx, -ka * vy], axis=-1), Mz=Mz,
        dfz_long_target=vm.lt_long * ax, dfz_lat_target=vm.lt_lat * ay,
        slip_power=np.sum(heating, axis=-1),
    )
    return ds, info


def derivatives(s: np.ndarray, u: np.ndarray, p: Params, surf: SurfaceParams,
                cond: TireCondition | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Time derivative of the flat state and a diagnostics dict (DESIGN.md section 5).

    Parameters
    ----------
    s : (..., NS) float64 state (see ``state.py``); never mutated. A leading batch shape is
        allowed (all envs share ``p``, ``surf`` and ``cond``).
    u : (..., 2) ``(steer_cmd, throttle_cmd)`` in [-1, 1], already latency-delayed by the caller.
    p : ``Params`` bundle (vehicle, tire, surface, drivetrain, actuators, sim).
    surf : per-wheel ``SurfaceParams`` (numeric fields scalar or (4,)); ``p.surface`` is NOT
        used here so a map lookup can pass a different surface under each wheel.
    cond : per-wheel ``TireCondition`` (static wear and wetness) or None.

    Returns
    -------
    ds : (..., NS) float64 time derivative (SI units per second).
    info : dict with per-wheel (..., 4) arrays ``Fx, Fy`` (wheel-frame tire friction forces, N),
        ``F_plow`` (lateral plowing drag, N), ``Fz`` (N), ``kappa, alpha`` (instantaneous slips),
        ``kappa_lag, alpha_lag`` (the slips fed to the MF), ``delta_w`` (per-wheel steer angle,
        rad), ``vxw, vyw`` (wheel-frame contact velocities, m/s), ``Fbx, Fby`` (body-frame tire
        forces incl. plowing, N), ``mu_x, mu_y`` (effective peak friction incl. temperature and
        condition), ``w_low`` (MF weight of the low-speed blend), ``vsx, vsy`` (slip velocities),
        ``v_wheel``, ``F_mf_x, F_mf_y``, ``tau, T_rr`` (wheel torques, N m); and scalars ``ax, ay``
        (body-frame specific force incl. aero, m/s^2), ``beta`` (rad), ``v`` (m/s), ``T_motor``
        (N m), ``omega_m`` (rad/s), ``i_motor`` (A), ``v_motor`` (mean motor terminal voltage, V),
        ``v_batt`` (battery terminal voltage, V), ``thr``, ``delta_target`` (rad), ``F_aero``
        (..., 2) (N), ``Mz`` (N m), ``dfz_long_target, dfz_lat_target`` (N), ``slip_power`` (W).

    Physics:
        v_i   = (vx - r*y_i, vy + r*x_i) rotated by -delta_i -> (vxw_i, vyw_i)
        kappa, alpha instantaneous;  d(lag)/dt = (max(|vxw|, v_eps)/relax) * (slip - lag)
        Fz    = Fz_static + [-1,-1,+1,+1]*DFZ_LONG/2 + [-rf,+rf,-(1-rf),+(1-rf)]*DFZ_LAT, >= 0
        (Fx, Fy) = Magic Formula (lagged slips, temperature, contamination) blended at low speed
        Fy_body_input = Fy + F_plow (loose surfaces);  Fb = R(delta_i) (Fx, Fy + F_plow)
        F_aero = -0.5*rho*CdA*|v|*(vx, vy);  a = (sum Fb + F_aero)/m
        dvx = ax + vy*r;  dvy = ay - vx*r;  dr = sum(x_i*Fby_i - y_i*Fbx_i)/Iz
        tau_i = R*Fx_i + Crr_i*Fz_i*R*tanh(omega_i*R/v_rr);  domega from the drivetrain
        motor: omega_m = shaft speed, di/dt from the ESC, T_motor = Kt*i - friction
        steering: d(DELTA)/dt = servo(DELTA, target(steer_cmd, r))
        load transfer targets m*ax*h/L, m*ay*h/w with a first-order lag
        dT = (slip power - cooling)/heat_capacity;  d(contam)/dt = -|v_wheel|/decay_dist * contam
    """
    ds, info = derivatives_model(s, u, _cached_model(p, surf, cond), want_info=True)
    return ds, info


def rhs(s: np.ndarray, u: np.ndarray, p: Params, surf: SurfaceParams,
        cond: TireCondition | None) -> np.ndarray:
    """``derivatives`` without the info dict: the ``f(s, *args) -> ds`` the integrator expects."""
    return derivatives_model(s, u, _cached_model(p, surf, cond), want_info=False)[0]


def rhs_model(s: np.ndarray, u: np.ndarray, vm: VehicleModel) -> np.ndarray:
    """Fast integrator right-hand side on a compiled model."""
    return derivatives_model(s, u, vm, want_info=False)[0]


# ----------------------------------------------------------------------------- trajectory container
@dataclass
class Trajectory:
    """Result of ``Vehicle.rollout``.

    t : (T+1,) time in s (``t[0]`` = start, ``t[k+1] - t[k]`` = control period).
    states : (T+1, NS) state vectors; ``states[0]`` is the initial state.
    actions : (T, 2) ``(steer_cmd, throttle_cmd)`` applied over control step k (from ``t[k]``).
    info : dict of stacked diagnostics with leading dim T. ``info[key][k]`` is
        ``derivatives(states[k+1], actions[k])`` evaluated at the END of control step k
        (per-wheel entries (T, 4) in FL FR RL RR order, ``F_aero`` (T, 2), scalars (T,)).
    """
    t: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    info: dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of control steps T."""
        return len(self)

    @property
    def speed(self) -> np.ndarray:
        """(T+1,) ground speed ``hypot(vx, vy)`` in m/s."""
        return np.hypot(self.states[:, VX], self.states[:, VY])

    @property
    def beta(self) -> np.ndarray:
        """(T+1,) vehicle sideslip ``atan2(vy, vx)`` in rad."""
        return np.arctan2(self.states[:, VY], self.states[:, VX])

    @property
    def yaw_rate(self) -> np.ndarray:
        """(T+1,) yaw rate in rad/s (positive = turning left)."""
        return self.states[:, R]

    @property
    def xy(self) -> np.ndarray:
        """(T+1, 2) world position in m."""
        return self.states[:, [X, Y]]

    def as_dict(self, flat: bool = True) -> dict[str, Any]:
        """Plain dict of arrays. ``flat=True``: ``t, states, actions`` plus ``info/<key>`` entries
        (ready for ``np.savez``); ``flat=False``: ``info`` kept as a nested dict."""
        d: dict[str, Any] = {"t": self.t, "states": self.states, "actions": self.actions}
        if flat:
            d.update({f"info/{k}": v for k, v in self.info.items()})
        else:
            d["info"] = dict(self.info)
        return d


def stack_info(infos: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack a list of per-step info dicts into one dict of arrays with leading dim T.

    Keys are taken from the first dict; an empty list gives an empty dict.
    """
    if not infos:
        return {}
    return {k: np.stack([np.asarray(d[k], dtype=np.float64) for d in infos]) for k in infos[0]}


# ----------------------------------------------------------------------------- wrapper
class Vehicle:
    """Convenience wrapper: parameters + surface + tire condition, control-rate stepping, rollouts.

    ``Vehicle(params, surf=None, cond=None)``: ``surf`` defaults to
    ``uniform_surface(params.surface)`` (one value per wheel), ``cond`` to
    ``TireCondition.neutral(params.sim.ambient_temp)``. The physics constants are compiled once
    (``self.model``); every step uses the fast core ``derivatives_model``.

    The control latency buffer (``actuators.ActionDelay``) is NOT applied here: ``step`` and
    ``rollout`` apply the action they are given immediately, with a zero-order hold over the
    ``params.sim.n_substeps`` physics substeps of ``params.sim.dt``, integrated with
    ``params.sim.integrator``. The caller / Gym env owns the delay buffer.

    Batched use: ``step`` and ``rollout_batch`` accept states of shape (B, NS) and actions of
    shape (B, 2); all B envs share the same parameters (per-env parameters arrive with domain
    randomization in Milestone 6 and the JAX port).
    """

    def __init__(self, params: Params, surf: SurfaceParams | None = None,
                 cond: TireCondition | None = None) -> None:
        self.params = params
        self.surf = uniform_surface(params.surface) if surf is None else surf
        self.cond = TireCondition.neutral(params.sim.ambient_temp) if cond is None else cond
        self.model = compile_model(params, self.surf, self.cond)
        from .stability import check_stiffness
        self.stiffness = check_stiffness(params, self.surf)      # warns if dt is too large

    # -- shorthands
    @property
    def dt(self) -> float:
        """Physics step in s."""
        return float(self.params.sim.dt)

    @property
    def control_dt(self) -> float:
        """Control period in s."""
        return float(self.params.sim.control_dt)

    @property
    def n_substeps(self) -> int:
        """Physics substeps per control step."""
        return int(self.params.sim.n_substeps)

    @property
    def method(self) -> str:
        """Integrator name from ``params.sim.integrator``."""
        return str(self.params.sim.integrator)

    # -- API
    def initial_state(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, v: float = 0.0,
                      beta: float = 0.0, yaw_rate: float = 0.0) -> np.ndarray:
        """State at pose ``(x, y, yaw)`` (m, m, rad) moving at speed ``v`` (m/s) with sideslip
        ``beta`` (rad) and yaw rate ``yaw_rate`` (rad/s): ``vx = v cos(beta)``, ``vy = v sin(beta)``,
        every wheel rolling at its own contact-point longitudinal speed (no slip), tire
        temperatures at ``cond.temp0``, contamination at ``cond.contamination``, other states 0.
        """
        vp = self.params.vehicle
        vx, vy = float(v) * np.cos(beta), float(v) * np.sin(beta)
        vxw = vx - float(yaw_rate) * vp.wheel_positions()[:, 1]       # rolling without slip
        temps = np.broadcast_to(np.asarray(self.cond.temp0, dtype=np.float64), (4,))
        contam = np.broadcast_to(np.clip(np.asarray(self.cond.contamination, dtype=np.float64),
                                         0.0, 1.0), (4,))
        return np.concatenate([
            np.array([x, y, yaw, vx, vy, yaw_rate], dtype=np.float64),  # X Y YAW VX VY R
            vxw / float(vp.wheel_radius),                               # OMEGA
            np.zeros(2),                                                # DELTA, I_MOTOR
            temps,                                                      # T_TIRE
            np.zeros(2),                                                # DFZ_LONG, DFZ_LAT
            np.zeros(8),                                                # KAPPA_LAG, ALPHA_LAG
            contam,                                                     # CONTAM
        ])

    def derivatives(self, s: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """``(ds, info)`` at state ``s`` (..., NS) and action ``u`` with this configuration."""
        ds, info = derivatives_model(s, u, self.model, want_info=True)
        return ds, info

    def step(self, s: np.ndarray, action: Any, n_sub: int | None = None,
             want_info: bool = True) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Advance one control period: ``n_sub`` physics substeps of ``sim.dt`` with the action held
        constant (zero-order hold). ``n_sub`` defaults to ``sim.n_substeps``.

        s : (NS,) or (B, NS); action : (2,) or (B, 2) ``(steer_cmd, throttle_cmd)`` in [-1, 1],
        already latency-delayed. Returns ``(s_new, info)``, ``info`` = diagnostics at the END of
        the step (None when ``want_info`` is False, which is cheaper).
        """
        s = np.asarray(s, dtype=np.float64)
        u = np.asarray(action, dtype=np.float64)
        n = self.n_substeps if n_sub is None else int(n_sub)
        s_new = integrator.integrate(rhs_model, s, self.dt, n, self.method, u, self.model)
        if not want_info:
            return s_new, None
        return s_new, derivatives_model(s_new, u, self.model, want_info=True)[1]

    def rollout(self, s0: np.ndarray, actions: np.ndarray | ActionFn,
                control_dt: float | None = None, n_steps: int | None = None,
                duration: float | None = None, t0: float = 0.0) -> Trajectory:
        """Open- or closed-loop rollout of ONE env from ``s0`` -> ``Trajectory``.

        actions : (T, 2) array of ``(steer_cmd, throttle_cmd)`` applied with zero-order hold over
            successive control steps, OR a callable ``actions(t, s, info_prev) -> action`` called
            at the start of each control step with the current time (s), state and the info dict
            of the previous step (for step 0: the info at ``s0`` with a zero action). A callable
            needs the horizon: ``n_steps`` (control steps) or ``duration`` (s).
        control_dt : control period in s; defaults to ``sim.control_dt``. Must be an integer
            multiple of ``sim.dt``.
        t0 : time of ``s0`` in s.

        No latency buffer is applied (the caller owns ``actuators.ActionDelay``).
        """
        cdt = self.control_dt if control_dt is None else float(control_dt)
        n_sub = int(round(cdt / self.dt))
        if n_sub < 1 or abs(n_sub * self.dt - cdt) > 1e-9:
            raise ValueError("control_dt must be a positive integer multiple of sim.dt")

        s = np.asarray(s0, dtype=np.float64).reshape(NS)
        if callable(actions):
            if n_steps is None:
                if duration is None:
                    raise ValueError("a callable action needs n_steps or duration")
                n_steps = int(round(float(duration) / cdt))
            T = int(n_steps)
            policy: ActionFn | None = actions
            act_arr = None
        else:
            act_arr = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
            T = act_arr.shape[0] if n_steps is None else min(int(n_steps), act_arr.shape[0])
            policy = None

        states = [s]
        acts = []
        infos = []
        info_prev = derivatives_model(s, np.zeros(2), self.model, want_info=True)[1]
        t = float(t0)
        for k in range(T):
            u = (np.asarray(policy(t, s, info_prev), dtype=np.float64).reshape(2)
                 if policy is not None else act_arr[k])
            s, info_prev = self.step(s, u, n_sub)
            states.append(s)
            acts.append(u)
            infos.append(info_prev)
            t += cdt

        t_arr = float(t0) + cdt * np.arange(T + 1, dtype=np.float64)
        return Trajectory(t=t_arr, states=np.stack(states),
                          actions=(np.stack(acts) if acts else np.zeros((0, 2))),
                          info=stack_info(infos))

    def rollout_batch(self, s0: np.ndarray, actions: np.ndarray, record_info: bool = False
                      ) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Open-loop rollout of B envs in lock-step (one vectorized integration for all of them).

        s0 : (B, NS) initial states (or (NS,) broadcast to every env).
        actions : (T, B, 2) or (T, 2) (same action for every env).
        record_info : also return the end-of-step diagnostics stacked to (T, B, ...).

        Returns ``(states, info)`` with states (T+1, B, NS).
        """
        acts = np.asarray(actions, dtype=np.float64)
        T = acts.shape[0]
        s = np.asarray(s0, dtype=np.float64)
        if acts.ndim == 3 and s.ndim == 1:
            s = np.broadcast_to(s, (acts.shape[1], NS)).copy()
        out = np.empty((T + 1,) + s.shape, dtype=np.float64)
        out[0] = s
        infos = []
        for k in range(T):
            s, info = self.step(s, acts[k], want_info=record_info)
            out[k + 1] = s
            if record_info:
                infos.append(info)
        return out, (stack_info(infos) if record_info else None)


class VehicleBatch:
    """B cars with (possibly) different numeric parameters, stepped together.

    ``VehicleBatch(params_list, surfs=None, conds=None)``: one entry per car; ``surfs`` default to
    ``uniform_surface(params.surface)`` and ``conds`` to a neutral condition, per car. Structural
    settings (``STRUCTURAL``) must be shared. Like ``Vehicle``, no control latency is applied here.
    """

    def __init__(self, params_list: list[Params], surfs: list[SurfaceParams] | None = None,
                 conds: list[TireCondition | None] | None = None, check_stiffness: bool = True) -> None:
        self.params_list = list(params_list)
        B = len(self.params_list)
        self.surfs = [uniform_surface(p.surface) if s is None else s
                      for p, s in zip(self.params_list, surfs or [None] * B)]
        self.conds = [TireCondition.neutral(p.sim.ambient_temp) if c is None else c
                      for p, c in zip(self.params_list, conds or [None] * B)]
        self.model = compile_model_batch(self.params_list, self.surfs, self.conds)
        self.params = self.params_list[0]
        if check_stiffness:
            from .stability import stiffness_report
            bad = [i for i, (p, s) in enumerate(zip(self.params_list, self.surfs))
                   if not stiffness_report(p, s).ok]
            if bad:
                import warnings
                warnings.warn(f"{len(bad)} of {B} cars are too stiff for {self.params.sim.integrator} at "
                              f"dt = {self.params.sim.dt} s (first: car {bad[0]}); see sim.stability",
                              RuntimeWarning, stacklevel=2)

    def __len__(self) -> int:
        return len(self.params_list)

    @property
    def dt(self) -> float:
        return float(self.params.sim.dt)

    @property
    def control_dt(self) -> float:
        return float(self.params.sim.control_dt)

    @property
    def n_substeps(self) -> int:
        return int(self.params.sim.n_substeps)

    def initial_states(self, x=0.0, y=0.0, yaw=0.0, v=0.0, beta=0.0, yaw_rate=0.0) -> np.ndarray:
        """(B, NS) initial states; every argument is a scalar or a (B,) array. Wheels roll without
        slip, tire temperatures and contamination come from each car's condition."""
        B = len(self)
        f = lambda a: np.broadcast_to(np.asarray(a, dtype=np.float64), (B,))  # noqa: E731
        x, y, yaw, v, beta, r = map(f, (x, y, yaw, v, beta, yaw_rate))
        vx, vy = v * np.cos(beta), v * np.sin(beta)
        yi = self.model.yi                                            # (B, 4)
        omega = (vx[:, None] - r[:, None] * yi) / self.model.Rw         # (B, 4)
        temps = np.stack([np.broadcast_to(np.asarray(c.temp0, dtype=np.float64), (4,)) for c in self.conds])
        contam = np.stack([np.broadcast_to(np.clip(np.asarray(c.contamination, dtype=np.float64), 0, 1), (4,))
                           for c in self.conds])
        z2, z8 = np.zeros((B, 2)), np.zeros((B, 8))
        return np.concatenate([np.stack([x, y, yaw, vx, vy, r], axis=1), omega, z2, temps, z2, z8, contam],
                              axis=1)

    def derivatives(self, s: np.ndarray, u: np.ndarray, want_info: bool = True):
        return derivatives_model(s, u, self.model, want_info=want_info)

    def step(self, s: np.ndarray, u: np.ndarray, want_info: bool = False, n_sub: int | None = None):
        """One control period for all cars: s (B, NS), u (B, 2) -> (s_new, info or None).

        ``n_sub`` overrides the number of integrator steps (default ``n_substeps``, one control
        period); ``n_sub=1`` advances a single physics step of ``dt``."""
        n = self.n_substeps if n_sub is None else int(n_sub)
        s_new = integrator.integrate(rhs_model, np.asarray(s, dtype=np.float64), self.dt, n,
                                     str(self.params.sim.integrator), np.asarray(u, dtype=np.float64),
                                     self.model)
        if not want_info:
            return s_new, None
        return s_new, derivatives_model(s_new, u, self.model, want_info=True)[1]


# ----------------------------------------------------------------------------- factory
def _apply_overrides(p: Params, overrides: Mapping[str, Any]) -> Params:
    """Return ``p`` with overrides applied.

    Accepted forms: a ``Params`` group name (``vehicle, tire, surface, drivetrain, actuators,
    sim``) mapped to a dict of field overrides or to a replacement dataclass instance; or a
    bare field name (``mass=1.8``, ``dt=0.002``) that is unique across the groups.
    """
    group_names = [f.name for f in fields(Params)]
    field_owner: dict[str, list[str]] = {}
    for gname in group_names:
        for f in fields(getattr(p, gname)):
            field_owner.setdefault(f.name, []).append(gname)

    per_group: dict[str, dict[str, Any]] = {g: {} for g in group_names}
    replaced: dict[str, Any] = {}
    for key, val in overrides.items():
        if key in group_names:
            if isinstance(val, Mapping):
                per_group[key].update(val)
            else:
                replaced[key] = val
        elif key in field_owner:
            owners = field_owner[key]
            if len(owners) != 1:
                raise KeyError(f"override {key!r} is ambiguous between groups {owners}; "
                               f"pass it as <group>=dict({key}=...)")
            per_group[owners[0]][key] = val
        else:
            raise KeyError(f"unknown parameter override {key!r}")

    new_groups = {}
    for gname in group_names:
        base = replaced.get(gname, getattr(p, gname))
        new_groups[gname] = dataclasses.replace(base, **per_group[gname]) if per_group[gname] else base
    return dataclasses.replace(p, **new_groups)


def make_vehicle(tire: str = "hard_plastic_drift", surface: str = "epoxy_ptile",
                 config_dir: Any = None, **overrides: Any) -> Vehicle:
    """``Vehicle`` from the YAML defaults (``params.default_params``) with optional overrides.

    ``tire`` / ``surface`` name a compound in ``tires.yaml`` and a surface in ``surfaces.yaml``.
    ``overrides`` are either group dicts (``sim=dict(dt=0.002)``, ``drivetrain=dict(layout="awd_spool")``),
    replacement group instances (``sim=SimParams(...)``) or unique bare field names
    (``mass=1.8``, ``integrator="semi_implicit_euler"``).
    """
    p = default_params(tire=tire, surface=surface, config_dir=config_dir)
    if overrides:
        p = _apply_overrides(p, overrides)
    return Vehicle(p)


__all__ = ["VehicleModel", "compile_model", "compile_model_batch", "derivatives_model", "derivatives",
           "rhs", "rhs_model", "Trajectory", "stack_info", "Vehicle", "VehicleBatch", "make_vehicle",
           "STRUCTURAL"]
