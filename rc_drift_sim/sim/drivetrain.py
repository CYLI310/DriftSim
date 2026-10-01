"""Drivetrain: DC-equivalent brushless motor + ESC, gearing, differentials, wheel dynamics.

Implements docs/DESIGN.md section 6 for the NumPy reference. Written JAX-portably: pure
functions, no in-place mutation of inputs, ``np.where``/``np.clip`` instead of value-dependent
Python control flow. Python branches occur only on configuration (layout string, bools).

Conventions
-----------
* Wheel order everywhere: index 0 = FL, 1 = FR, 2 = RL, 3 = RR. Per-wheel arrays have shape
  (..., 4) (the wheel axis is last, any leading batch shape); motor quantities have shape (...).
* SI units: rad/s, N m, kg m^2, A, V, ohm, H, s.
* ``omega``   wheel angular speeds (rad/s), positive = rolling forward.
* ``tau``     resisting torque per wheel (N m) = R*Fx + T_rr (computed by vehicle.py);
              positive resists forward rotation.
* ``omega_m`` motor shaft speed (rad/s) = mean over driven wheels of G_i * omega_i.
* ``T_motor`` net motor shaft torque (N m) after motor friction, positive = drives forward.
* ``thr``     ESC throttle in [-1, 1] AFTER ``actuators.throttle_command`` (deadband/expo).

Layouts are expressed through per-wheel gear ratios ``G`` and a float ``driven`` mask so that
one code path handles "rwd", "awd_spool" and "awd_overdrive". The constants are precomputed
once into a :class:`DrivetrainModel` (``drivetrain_model(dp, vp)``); the contract's public
functions build it on the fly and call the same array functions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import DrivetrainParams, VehicleParams

LAYOUTS = ("rwd", "awd_spool", "awd_overdrive")

# Speed scale (rad/s) of the tanh that smooths the motor Coulomb friction (contract: 5.0).
COULOMB_SMOOTH_SPEED = 5.0


# ----------------------------------------------------------------------------- gearing
def gear_ratios(dp: DrivetrainParams) -> tuple[np.ndarray, np.ndarray]:
    """Per-wheel gear ratios and driven mask for the configured layout.

    Returns
    -------
    G : (4,) float64
        Motor shaft speed / wheel speed for each wheel, ``[Gf, Gf, Gr, Gr]`` (dimensionless).
        ``Gr = gear_ratio``; ``Gf = 0`` (rwd), ``Gr`` (awd_spool) or ``Gr / overdrive_front``
        (awd_overdrive: the front wheels spin ``overdrive_front`` times faster than the rear).
    driven : (4,) float64
        1.0 where ``G > 0`` (wheel connected to the motor), else 0.0.
    """
    layout = dp.layout
    if layout not in LAYOUTS:
        raise ValueError(f"unknown drivetrain layout {layout!r}; expected one of {LAYOUTS}")
    gr = float(dp.gear_ratio)
    if layout == "rwd":
        gf = 0.0
    elif layout == "awd_spool":
        gf = gr
    else:  # awd_overdrive
        gf = gr / float(dp.overdrive_front)
    G = np.array([gf, gf, gr, gr], dtype=np.float64)
    driven = (G > 0.0).astype(np.float64)
    return G, driven


def wheel_inertias(dp: DrivetrainParams, vp: VehicleParams) -> np.ndarray:
    """Effective inertia per wheel (kg m^2), shape (4,).

    ``I_i = wheel_inertia + driven_i * motor_inertia * G_i^2 / n_driven`` - the rotor inertia is
    reflected to the wheel side and shared equally among the driven wheels, so the total inertia
    referred to the motor shaft counts ``motor_inertia`` exactly once.
    """
    G, driven = gear_ratios(dp)
    return float(vp.wheel_inertia) + driven * float(dp.motor_inertia) * G ** 2 / np.sum(driven)


# ----------------------------------------------------------------------------- precomputed model
@dataclass(frozen=True, slots=True)
class DrivetrainModel:
    """Drivetrain constants (see module docstring). Arrays are (4,) per wheel or (2,) per axle
    (front, rear); floats are per car. For a batch of cars with different parameters every numeric
    field gains a leading batch axis: (B, 4), (B, 2) and (B,) (see ``vehicle.compile_model_batch``).
    ``layout`` (through ``front_driven`` and ``driven``) and ``reverse_enabled`` must be shared."""
    G: np.ndarray            # (4,) gear ratio per wheel
    driven: np.ndarray       # (4,) 1.0 = connected to the motor
    G_driven: np.ndarray     # (4,) G * driven / n_driven  (shaft-speed weights)
    inv_inertia: np.ndarray  # (4,) 1 / effective inertia
    Gf: float                # front axle ratio (0 for rwd)
    Gr: float                # rear axle ratio
    inv_Gf: float            # 1/Gf (0 for rwd)
    inv_Gr: float
    front_driven: float      # 1.0 for AWD layouts
    center_lock: float
    center_max: float
    center_visc: float
    lock: np.ndarray         # (2,) [front, rear] diff lock
    max_torque: np.ndarray   # (2,)
    visc: np.ndarray         # (2,)
    ke: float
    kt: float
    r_m: float
    inv_L: float
    b_m: float
    tc_m: float
    i_max: float
    v_lim: float             # i_max * R_m
    v0: float                # battery open-circuit voltage
    r_batt: float
    inv_thr_min: float
    reverse_enabled: bool
    drag_brake: float


def drivetrain_model(dp: DrivetrainParams, vp: VehicleParams) -> DrivetrainModel:
    """Precompute the drivetrain constants for ``dp`` on vehicle ``vp``."""
    G, driven = gear_ratios(dp)
    n_driven = float(np.sum(driven))
    Gf, Gr = float(G[0]), float(G[2])
    return DrivetrainModel(
        G=G, driven=driven, G_driven=G * driven / n_driven,
        inv_inertia=1.0 / wheel_inertias(dp, vp),
        Gf=Gf, Gr=Gr, inv_Gf=(1.0 / Gf if Gf > 0.0 else 0.0), inv_Gr=1.0 / Gr,
        front_driven=float(driven[0]),
        center_lock=float(dp.center_lock), center_max=float(dp.center_max_torque),
        center_visc=float(dp.center_visc),
        lock=np.array([dp.front_diff_lock, dp.rear_diff_lock], dtype=np.float64),
        max_torque=np.array([dp.front_diff_max_torque, dp.rear_diff_max_torque], dtype=np.float64),
        visc=np.array([dp.front_diff_visc, dp.rear_diff_visc], dtype=np.float64),
        ke=float(dp.ke), kt=float(dp.kt), r_m=float(dp.motor_resistance),
        inv_L=1.0 / float(dp.inductance),
        b_m=float(dp.motor_friction_visc), tc_m=float(dp.motor_friction_coulomb),
        i_max=float(dp.current_limit), v_lim=float(dp.current_limit) * float(dp.motor_resistance),
        v0=float(dp.battery_voltage), r_batt=float(dp.battery_resistance),
        inv_thr_min=1.0 / max(float(dp.throttle_min), 1e-9),
        reverse_enabled=bool(dp.reverse_enabled), drag_brake=float(dp.drag_brake),
    )


# ----------------------------------------------------------------------------- array core
def shaft_speed_m(dm: DrivetrainModel, omega: np.ndarray) -> np.ndarray:
    """Motor shaft speed (rad/s) = mean over driven wheels of ``G_i * omega_i``; omega (..., 4)."""
    return np.sum(dm.G_driven * omega, axis=-1)


def esc_command_m(dm: DrivetrainModel, i, omega_m, thr) -> tuple[np.ndarray, np.ndarray]:
    """ESC bridge voltage ``V_cmd`` (V) and effective conductance ``g_eff`` (0..1); see ``esc_command``."""
    abs_thr = np.abs(thr)
    v_batt = dm.v0 - dm.r_batt * np.abs(i) * abs_thr
    v_cmd = thr * v_batt
    g = np.minimum(abs_thr * dm.inv_thr_min, 1.0)
    if not dm.reverse_enabled:                         # configuration branch
        brake = thr < 0.0
        v_cmd = np.where(brake, 0.0, v_cmd)
        g = np.where(brake, abs_thr, g)
    if np.any(dm.drag_brake != 0.0):                   # parameter check, not a traced value
        g = g + (1.0 - g) * dm.drag_brake
    emf = dm.ke * omega_m
    v_cmd = np.clip(v_cmd, emf - dm.v_lim, emf + dm.v_lim)
    return v_cmd, g


def voltages_m(dm: DrivetrainModel, i, omega_m, thr) -> tuple[np.ndarray, np.ndarray]:
    """Battery terminal voltage and mean motor terminal voltage ``Ke*omega_m + g_eff*(V_cmd - Ke*omega_m)``
    (V), i.e. ``R_m*i + L di/dt + Ke*omega_m`` of ``current_derivative_m``. Diagnostics only."""
    v_cmd, g = esc_command_m(dm, i, omega_m, thr)
    emf = dm.ke * omega_m
    return dm.v0 - dm.r_batt * np.abs(i) * np.abs(thr), emf + g * (v_cmd - emf)


def current_derivative_m(dm: DrivetrainModel, i, omega_m, thr) -> np.ndarray:
    """``di/dt = (g_eff*(V_cmd - Ke*omega_m) - R_m*i)/L`` (A/s)."""
    v_cmd, g = esc_command_m(dm, i, omega_m, thr)
    return (g * (v_cmd - dm.ke * omega_m) - dm.r_m * i) * dm.inv_L


def motor_torque_m(dm: DrivetrainModel, i, omega_m) -> np.ndarray:
    """Net motor shaft torque ``Kt*i - b_m*omega_m - Tc_m*tanh(omega_m/5)`` (N m)."""
    return dm.kt * i - dm.b_m * omega_m - dm.tc_m * np.tanh(omega_m * (1.0 / COULOMB_SMOOTH_SPEED))


def axle_torques(T_axle, tau_L, tau_R, omega_L, omega_R, lock, max_torque, visc
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Split an axle drive torque between its two sides (open / LSD / locked in one formula).

    All arguments broadcast against each other.

    ``T_c = clip(lock * (tau_L - tau_R)/2, -max_torque, +max_torque) + visc * (omega_R - omega_L)``
    ``T_L = T_axle/2 + T_c``, ``T_R = T_axle/2 - T_c``

    * ``lock = 1``, ``max_torque = inf``: with equal side inertias both sides get exactly the
      locked-axle acceleration ``(T_axle - tau_L - tau_R) / (2 I)``.
    * ``lock = 0``, ``visc = 0``: open differential, ``T_L = T_R = T_axle/2``.
    * intermediate ``lock`` with finite ``max_torque`` (N m at this axle's speed level): LSD.
    * ``visc`` (N m s/rad) is a viscous coupling / Baumgarte stabilizer; the side-to-side speed
      difference decays with time constant ``I / (2 visc)`` (equal inertias ``I``).
    """
    T_c = (np.clip(lock * (tau_L - tau_R) * 0.5, -max_torque, max_torque)
           + visc * (omega_R - omega_L))
    return 0.5 * T_axle + T_c, 0.5 * T_axle - T_c


def wheel_accelerations_m(dm: DrivetrainModel, omega: np.ndarray, T_motor, tau: np.ndarray
                          ) -> np.ndarray:
    """Wheel angular accelerations (rad/s^2), (..., 4). See ``wheel_accelerations``."""
    oFL, oFR, oRL, oRR = omega[..., 0], omega[..., 1], omega[..., 2], omega[..., 3]
    tFL, tFR, tRL, tRR = tau[..., 0], tau[..., 1], tau[..., 2], tau[..., 3]

    # --- center coupling, referred to the motor shaft (masked away for rwd)
    if dm.front_driven > 0.0:                         # configuration branch
        omega_f_ref = dm.Gf * 0.5 * (oFL + oFR)
        omega_r_ref = dm.Gr * 0.5 * (oRL + oRR)
        T_f_shaft, _ = axle_torques(T_motor, (tFL + tFR) * dm.inv_Gf, (tRL + tRR) * dm.inv_Gr,
                                    omega_f_ref, omega_r_ref,
                                    dm.center_lock, dm.center_max, dm.center_visc)
        T_r_shaft = T_motor - T_f_shaft                # torque conservation at the shaft
        TfL, TfR = axle_torques(dm.Gf * T_f_shaft, tFL, tFR, oFL, oFR,
                                dm.lock[..., 0], dm.max_torque[..., 0], dm.visc[..., 0])
    else:
        T_r_shaft = T_motor
        TfL = TfR = np.zeros_like(oFL)                # undriven front wheels: free rolling
    TrL, TrR = axle_torques(dm.Gr * T_r_shaft, tRL, tRR, oRL, oRR,
                            dm.lock[..., 1], dm.max_torque[..., 1], dm.visc[..., 1])
    T_wheel = np.stack([TfL, TfR, TrL, TrR], axis=-1)
    return (T_wheel - tau) * dm.inv_inertia


# ----------------------------------------------------------------------------- public API (contract)
def shaft_speed(dp: DrivetrainParams, omega: np.ndarray) -> np.ndarray:
    """Motor shaft speed (rad/s): mean over driven wheels of ``G_i * omega_i``; omega (..., 4)."""
    G, driven = gear_ratios(dp)
    return np.sum(driven * G * np.asarray(omega, dtype=np.float64), axis=-1) / np.sum(driven)


def battery_voltage(dp: DrivetrainParams, i, thr) -> np.ndarray:
    """Terminal voltage (V) with load sag: ``V_batt = V0 - R_batt * |i| * |thr|``."""
    return float(dp.battery_voltage) - float(dp.battery_resistance) * np.abs(
        np.asarray(i, dtype=np.float64)) * np.abs(np.asarray(thr, dtype=np.float64))


def esc_command(dp: DrivetrainParams, i, omega_m, thr) -> tuple[np.ndarray, np.ndarray]:
    """ESC output: commanded bridge voltage ``V_cmd`` (V) and effective conductance ``g_eff`` (0..1).

    * reverse enabled:                 ``V_cmd = thr * V_batt``, ``g = min(|thr| / throttle_min, 1)``
    * reverse disabled and ``thr < 0``: ``V_cmd = 0``, ``g = |thr|`` (proportional brake by shorting)
    * drag brake fills in at neutral:  ``g_eff = g + (1 - g) * drag_brake``
    * current limit (duty clamp):      ``V_cmd = clip(V_cmd, Ke*omega_m - i_max*R_m, Ke*omega_m + i_max*R_m)``
      so the quasi-steady current ``g_eff * (V_cmd - Ke*omega_m) / R_m`` never exceeds ``i_max``.
    """
    dm = drivetrain_model(dp, VehicleParams())
    f = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    return esc_command_m(dm, f(i), f(omega_m), f(thr))


def current_derivative(dp: DrivetrainParams, i, omega_m, thr) -> np.ndarray:
    """``di/dt`` (A/s) of the DC-equivalent motor driven by the ESC.

    ``di/dt = (g_eff * (V_cmd - Ke*omega_m) - R_m * i) / L`` with ``L = R_m * motor_tau_e``.
    With ``g_eff = 0`` (neutral, no drag brake) the current decays with time constant
    ``L/R_m = motor_tau_e``; ``g_eff = 1`` connects the winding to ``V_cmd``.
    """
    dm = drivetrain_model(dp, VehicleParams())
    f = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    return current_derivative_m(dm, f(i), f(omega_m), f(thr))


def motor_torque(dp: DrivetrainParams, i, omega_m) -> np.ndarray:
    """Net motor shaft torque (N m): ``Kt*i - b_m*omega_m - Tc_m*tanh(omega_m/5)``.

    Positive drives the car forward. Exactly zero at rest with zero current.
    """
    i = np.asarray(i, dtype=np.float64)
    omega_m = np.asarray(omega_m, dtype=np.float64)
    return (float(dp.kt) * i - float(dp.motor_friction_visc) * omega_m
            - float(dp.motor_friction_coulomb) * np.tanh(omega_m / COULOMB_SMOOTH_SPEED))


def wheel_accelerations(dp: DrivetrainParams, vp: VehicleParams, omega: np.ndarray,
                        T_motor, tau: np.ndarray) -> np.ndarray:
    """Wheel angular accelerations ``domega`` (rad/s^2), shape (..., 4), order FL FR RL RR.

    omega : (..., 4) wheel speeds (rad/s); T_motor : (...) net shaft torque (N m);
    tau : (..., 4) resisting torques at the wheels (N m), ``R*Fx + T_rr``.

    1. Center coupling (AWD): ``axle_torques`` between the front and rear axles with speeds and
       torques referred to the motor shaft, ``omega_ref = G_axle * mean(omega_axle)``,
       ``tau_ref = sum(tau_axle) / G_axle``: ``T_f_shaft = T_motor/2 + T_cc``,
       ``T_r_shaft = T_motor - T_f_shaft``, each multiplied by ``G_axle``. For rwd the rear shaft
       receives all of ``T_motor``.
    2. Front and rear axle differentials via ``axle_torques`` (per-axle lock/max_torque/visc).
    3. Undriven wheels (front wheels of an rwd car) get no drive torque: ``domega = -tau/I_w``.
    4. ``domega_i = (T_i - tau_i) / I_i`` with ``I`` from ``wheel_inertias``.
    """
    return wheel_accelerations_m(drivetrain_model(dp, vp), np.asarray(omega, dtype=np.float64),
                                 np.asarray(T_motor, dtype=np.float64),
                                 np.asarray(tau, dtype=np.float64))
