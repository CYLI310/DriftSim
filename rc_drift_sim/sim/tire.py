"""Pacejka Magic Formula tire model: combined slip, surface effects, tire condition, low-speed
blend, loose-surface plowing drag and a lumped thermal model.

Implements docs/DESIGN.md section 4. MF5.2-style normalized-load parameterization
(http://www.racer.nl/reference/pacejka.htm) with dimensionless coefficients at a nominal
wheel load ``fz0`` (about 4 N for a 1.6 kg car).

Conventions (DESIGN.md section 1)
---------------------------------
* Wheel order FL, FR, RL, RR; per-wheel arrays have shape (..., 4). Every function broadcasts
  over leading axes, so batched states ``(B, 4)`` and test grids
  ``kappa (Nk,1,1) x alpha (1,Na,1) x Fz (1,1,Nf)`` (with a scalar-field surface) both work.
* Slip ratio ``kappa = (omega*R - vxw)/max(|vxw|, v_eps)``, positive = driving, gives +Fx.
* Slip angle ``alpha = atan2(vyw, max(|vxw|, v_eps))`` (rad), positive = contact-patch velocity
  points LEFT of the wheel heading. ISO-style force sign: ``Fy = -MF_y(alpha)``, i.e. a
  positive slip angle produces a negative (rightward) lateral force. The lateral Magic
  Formula input is ``tan(alpha)`` (this makes the similarity combined-slip formula reduce
  exactly to pure slip).
* Load ``Fz >= 0`` (clamped); all tire forces are exactly 0 at ``Fz = 0``.
* Units: N, m, s, rad, degC.

Structure
---------
The math lives once, in array functions that take a precomputed :class:`TireModel` (compound x
surface x static tire condition, built by :func:`tire_model`). The dict-returning public API of
the contract (``tire_coefficients``, ``pure_slip_forces``, ``combined_forces``, ``tire_forces``)
wraps the same functions, so the fast path used by ``vehicle.py`` and the reference API cannot
drift apart.

Style: NumPy float64, pure functions, no in-place mutation, no Python branching on array
values (only on configuration: mode strings and bool flags), so it ports to JAX by swapping ``np``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Union

import numpy as np

from .params import SimParams, SurfaceParams, TireCondition, TireParams, VehicleParams

ArrayLike = Union[float, np.ndarray]

EPS_BCD = 1e-6          # guard in B = K / (C*D + eps), see DESIGN.md section 4
RHO_EPS = 1e-9          # regularization of the normalized combined slip magnitude
DIV_EPS = 1e-12         # generic guard for divisions by a non-negative quantity
ALPHA_MAX = 0.5 * np.pi - 1e-6   # |alpha| clamp before tan() (rad); atan2-based alpha never reaches it
C_MAX = 1.95            # shape factor cap: C >= 2 would flip the sign of the large-slip asymptote

PLOW_V_REF = 1.0        # m/s: lateral slide speed at which the plow force ~ 0.9 * loose_drag * Fz
PLOW_CAP = 2.0          # plow force saturates at PLOW_CAP * loose_drag * Fz

MODES = ("similarity", "mf_weighting", "ellipse")
_R_KEYS = ("rbx1", "rbx2", "rcx1", "rby1", "rby2", "rby3", "rcy1")


# ----------------------------------------------------------------------------- helpers
def _f64(x: Any) -> np.ndarray:
    """Float64 array view (0-d for scalars)."""
    return np.asarray(x, dtype=np.float64)


def _mf_core(bx: ArrayLike, C: ArrayLike, D: ArrayLike, E: ArrayLike) -> np.ndarray:
    """Magic Formula evaluated on the product ``bx = B * x`` (dimensionless).

    ``D * sin(C * atan(bx - E * (bx - atan(bx))))``. Written on the product so the similarity
    method never divides by ``B`` (which is 0 at ``Fz = 0``).
    """
    return D * np.sin(C * np.arctan(bx - E * (bx - np.arctan(bx))))


# ----------------------------------------------------------------------------- public: basics
def mf(x: ArrayLike, B: ArrayLike, C: ArrayLike, D: ArrayLike, E: ArrayLike,
       Sh: ArrayLike = 0.0, Sv: ArrayLike = 0.0) -> np.ndarray:
    """Pacejka Magic Formula (racer.nl "Pacejka96" form).

    ``xs = x + Sh; Bx = B*xs; return D*sin(C*atan(Bx - E*(Bx - atan(Bx)))) + Sv``

    x : slip input (slip ratio, or tan(slip angle)), dimensionless.
    B : stiffness factor, C : shape factor, D : peak value (N), E : curvature factor (<= 1),
    Sh, Sv : horizontal / vertical shifts. Returns the force (N). Properties: ``D`` is the
    peak, ``B*C*D`` the slope at the origin, ``D*sin(pi*C/2)`` the large-slip asymptote.
    """
    xs = _f64(x) + _f64(Sh)
    return _mf_core(_f64(B) * xs, _f64(C), _f64(D), _f64(E)) + _f64(Sv)


def smoothstep(x: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> np.ndarray:
    """Cubic smoothstep: 0 for ``x <= lo``, 1 for ``x >= hi``, ``3t^2 - 2t^3`` in between.

    All arguments share one unit (m/s in the low-speed blend). ``hi - lo`` is guarded.
    """
    span = np.maximum(_f64(hi) - _f64(lo), DIV_EPS)
    t = np.clip((_f64(x) - _f64(lo)) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def slips(vxw: ArrayLike, vyw: ArrayLike, omega: ArrayLike, R: ArrayLike,
          v_eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous (un-relaxed) slip ratio and slip angle from wheel-frame velocities.

    vxw, vyw : contact-point velocity in the wheel frame (m/s); omega : wheel angular speed
    (rad/s); R : wheel radius (m); v_eps : denominator regularization (m/s).

    ``kappa = (omega*R - vxw)/max(|vxw|, v_eps)`` (dimensionless, + = driving),
    ``alpha = atan2(vyw, max(|vxw|, v_eps))`` (rad, + = velocity points left of the heading).
    """
    vxw, vyw = _f64(vxw), _f64(vyw)
    v_reg = np.maximum(np.abs(vxw), float(v_eps))
    kappa = (_f64(omega) * _f64(R) - vxw) / v_reg
    alpha = np.arctan2(vyw, v_reg)
    return kappa, alpha


# ----------------------------------------------------------------------------- tire condition
def temperature_factor(tp: TireParams, T: ArrayLike) -> np.ndarray:
    """Peak-friction multiplier from tire temperature ``T`` (degC), dimensionless.

    Gaussian grip window centred on ``t_opt`` with 1/e half-width ``t_width``:
    ``1 - temp_mu_drop * (1 - exp(-((T - t_opt)/t_width)^2))``. Equals 1 at ``t_opt`` (the YAML
    mu values are the optimal-temperature values) and ``1 - temp_mu_drop`` far outside the window
    (cold or overheated tire).
    """
    z = (_f64(T) - float(tp.t_opt)) / float(tp.t_width)
    return 1.0 - float(tp.temp_mu_drop) * (1.0 - np.exp(-z * z))


def condition_scales(tp: TireParams, cond: TireCondition | None,
                     T_tire: ArrayLike | None, contam: ArrayLike | None = None
                     ) -> dict[str, np.ndarray]:
    """Tire-condition and temperature multipliers on friction, stiffness and shape.

    Parameters
    ----------
    tp : compound. cond : per-wheel TireCondition (arrays (4,)) or None (= new, dry, clean).
    T_tire : tire temperature (degC) or None (= no temperature effect).
    contam : per-wheel contamination level 0..1 overriding ``cond.contamination`` (the vehicle
        model passes its CONTAM state here; the condition value is only the initial level).

    Returns dict (dimensionless, broadcast to the common shape of the inputs):
        lam_mu_x, lam_mu_y : peak-friction multipliers
            = temperature_factor(T) * (1 - wear_mu_drop*wear) * (1 - wet_mu_drop*wetness)
              * (1 - contamination_mu_drop*contamination)
        lam_k_x, lam_k_y   : stiffness multipliers = (1 + wear_k_gain*wear) * (1 - wet_k_drop*wetness)
        dC                 : additive shape-factor change = wet_c_gain*wetness (sharper post-peak
                             drop on a wet tire)
    wear, wetness and contamination are clipped to [0, 1].
    """
    lam_mu = np.ones(())
    lam_k = np.ones(())
    dC = np.zeros(())
    if T_tire is not None:
        lam_mu = lam_mu * temperature_factor(tp, T_tire)
    if cond is not None:
        wear = np.clip(_f64(cond.wear), 0.0, 1.0)
        wet = np.clip(_f64(cond.wetness), 0.0, 1.0)
        lam_mu = lam_mu * (1.0 - tp.wear_mu_drop * wear) * (1.0 - tp.wet_mu_drop * wet)
        lam_k = lam_k * (1.0 + tp.wear_k_gain * wear) * (1.0 - tp.wet_k_drop * wet)
        dC = dC + tp.wet_c_gain * wet
        if contam is None:
            contam = cond.contamination
    if contam is not None:
        lam_mu = lam_mu * (1.0 - tp.contamination_mu_drop * np.clip(_f64(contam), 0.0, 1.0))
    return dict(lam_mu_x=lam_mu, lam_mu_y=lam_mu, lam_k_x=lam_k, lam_k_y=lam_k, dC=dC)


# ----------------------------------------------------------------------------- precomputed model
@dataclass(frozen=True, slots=True)
class TireModel:
    """Constants of one compound on one (per-wheel) surface with a static tire condition.

    Array fields keep the native shape of the surface / condition inputs (0-d for a scalar
    surface, (4,) per wheel). Temperature and contamination are dynamic and applied in
    :func:`coefficients`. For a batch of cars with different tires/surfaces every float becomes a
    (B, 1) array and every per-wheel array (B, 4) (all fields act on per-wheel quantities); ``mode``
    and ``use_temp`` must be shared across the batch.
    """
    mode: str
    fz0: float
    inv_fz0: float
    pdx1: float
    pdx2: float
    pex1: float
    pex2: float
    pkx1: float
    pkx2: float
    pkx3: float
    pdy1: float
    pdy2: float
    pey1: float
    pey2: float
    pky1_fz0: float        # pky1 * fz0
    inv_pky2_fz0: float    # 1 / (pky2 * fz0)
    Cx: np.ndarray         # shape factors after the surface blend and the wet gain
    Cy: np.ndarray
    mu_scale: np.ndarray   # surface mu (incl. loose affinity) x static condition (wear, wet)
    k_scale: np.ndarray    # surface stiffness x static condition (wear, wet)
    e_shift: np.ndarray    # surface curvature shift
    slip_scale: np.ndarray
    inv_slip_scale: np.ndarray
    use_temp: bool
    t_opt: float
    inv_t_width: float
    temp_mu_drop: float
    contam_mu_drop: float
    inv_contam_dist: float
    rbx1: float
    rbx2: float
    rcx1: float
    rby1: float
    rby2: float
    rby3: float
    rcy1: float
    crr: np.ndarray        # rolling-resistance coefficient per wheel
    loose_drag: np.ndarray # plowing coefficient per wheel
    has_plow: bool
    inv_relax_x: float
    inv_relax_y: float
    inv_heat_capacity: float
    cool_coeff: float
    cool_speed_coeff: float


def tire_model(tp: TireParams, surf: SurfaceParams, cond: TireCondition | None = None,
               use_temp: bool = True) -> TireModel:
    """Precompute a :class:`TireModel` from a compound, a (per-wheel) surface and a condition.

    Surface modifications (DESIGN.md section 4):
        mu   *= mu_scale * (1 + loose_affinity * looseness)
        K    *= stiffness_scale
        C     = C + shape_blend * (shape_c_target - C)        (then + wet_c_gain*wetness, <= C_MAX)
        E     = min(E + curvature_e_shift, 1)
        slip inputs are divided by slip_scale = 1 + peak_slip_shift
    Static condition factors (wear, wetness) from :func:`condition_scales` are folded in here;
    temperature (``use_temp``) and contamination are applied per evaluation.
    """
    f = lambda name: _f64(getattr(surf, name))  # noqa: E731
    shape_blend, shape_c = f("shape_blend"), f("shape_c_target")
    mu_surf = f("mu_scale") * (1.0 + float(tp.loose_affinity) * f("looseness"))
    stat = condition_scales(tp, cond, None, contam=np.zeros(()))   # wear + wetness only
    slip_scale = 1.0 + f("peak_slip_shift")
    Cx = np.minimum(tp.pcx1 + shape_blend * (shape_c - tp.pcx1) + stat["dC"], C_MAX)
    Cy = np.minimum(tp.pcy1 + shape_blend * (shape_c - tp.pcy1) + stat["dC"], C_MAX)
    loose_drag = f("loose_drag")
    return TireModel(
        mode=str(tp.combined_mode),
        fz0=float(tp.fz0), inv_fz0=1.0 / float(tp.fz0),
        pdx1=float(tp.pdx1), pdx2=float(tp.pdx2), pex1=float(tp.pex1), pex2=float(tp.pex2),
        pkx1=float(tp.pkx1), pkx2=float(tp.pkx2), pkx3=float(tp.pkx3),
        pdy1=float(tp.pdy1), pdy2=float(tp.pdy2), pey1=float(tp.pey1), pey2=float(tp.pey2),
        pky1_fz0=float(tp.pky1) * float(tp.fz0), inv_pky2_fz0=1.0 / (float(tp.pky2) * float(tp.fz0)),
        Cx=Cx, Cy=Cy,
        mu_scale=mu_surf * stat["lam_mu_x"],
        k_scale=f("stiffness_scale") * stat["lam_k_x"],
        e_shift=f("curvature_e_shift"),
        slip_scale=slip_scale, inv_slip_scale=1.0 / slip_scale,
        use_temp=bool(use_temp), t_opt=float(tp.t_opt), inv_t_width=1.0 / float(tp.t_width),
        temp_mu_drop=float(tp.temp_mu_drop),
        contam_mu_drop=float(tp.contamination_mu_drop),
        inv_contam_dist=1.0 / float(tp.contamination_decay_dist),
        **{k: float(getattr(tp, k)) for k in _R_KEYS},
        crr=f("rolling_resistance"),
        loose_drag=loose_drag, has_plow=bool(np.any(loose_drag != 0.0)),
        inv_relax_x=1.0 / float(tp.relax_x), inv_relax_y=1.0 / float(tp.relax_y),
        inv_heat_capacity=1.0 / float(tp.heat_capacity),
        cool_coeff=float(tp.cool_coeff), cool_speed_coeff=float(tp.cool_speed_coeff),
    )


class Coef(NamedTuple):
    """Magic Formula coefficients for one evaluation (arrays broadcast to the load's shape)."""
    Dx: np.ndarray
    Cx: np.ndarray
    Bx: np.ndarray
    Ex: np.ndarray
    Kx: np.ndarray
    mu_x: np.ndarray
    Dy: np.ndarray
    Cy: np.ndarray
    By: np.ndarray
    Ey: np.ndarray
    Ky: np.ndarray
    mu_y: np.ndarray
    slip_scale: np.ndarray
    inv_slip_scale: np.ndarray
    Fz: np.ndarray
    rbx1: float = 10.0
    rbx2: float = 8.0
    rcx1: float = 1.0
    rby1: float = 8.0
    rby2: float = 6.0
    rby3: float = 0.0
    rcy1: float = 1.0


def coefficients(tm: TireModel, Fz: ArrayLike, T: ArrayLike | None = None,
                 contam: ArrayLike | None = None) -> Coef:
    """Magic Formula coefficients (``dfz = (Fz - fz0)/fz0``):

        mu_x = max((pdx1 + pdx2*dfz) * mu_scale * lam, 0),   Dx = mu_x*Fz
        Ex   = min(min(pex1 + pex2*dfz, 1) + e_shift, 1)
        Kx   = Fz*(pkx1 + pkx2*dfz)*exp(pkx3*dfz) * k_scale,   Bx = Kx/(Cx*Dx + 1e-6)
        mu_y = max((pdy1 + pdy2*dfz) * mu_scale * lam, 0),   Dy = mu_y*Fz
        Ey   = min(min(pey1 + pey2*dfz, 1) + e_shift, 1)
        Ky   = pky1*fz0*sin(2*atan(Fz/(pky2*fz0))) * k_scale,  By = Ky/(Cy*Dy + 1e-6)
    ``lam`` = temperature factor (when ``T`` is given and ``tm.use_temp``) x contamination factor
    (when ``contam`` is given). ``mu`` is clamped >= 0 so an extrapolated negative friction
    coefficient can never flip the force. At ``Fz = 0``: ``D = K = B = 0``.
    """
    Fz = np.maximum(_f64(Fz), 0.0)
    dfz = (Fz - tm.fz0) * tm.inv_fz0
    mu_scale = tm.mu_scale
    if T is not None and tm.use_temp:
        z = (_f64(T) - tm.t_opt) * tm.inv_t_width
        mu_scale = mu_scale * (1.0 - tm.temp_mu_drop * (1.0 - np.exp(-z * z)))
    if contam is not None:
        mu_scale = mu_scale * (1.0 - tm.contam_mu_drop * np.clip(_f64(contam), 0.0, 1.0))

    mu_x = np.maximum((tm.pdx1 + tm.pdx2 * dfz) * mu_scale, 0.0)
    Dx = mu_x * Fz
    Ex = np.minimum(np.minimum(tm.pex1 + tm.pex2 * dfz, 1.0) + tm.e_shift, 1.0)
    Kx = Fz * (tm.pkx1 + tm.pkx2 * dfz) * tm.k_scale
    if np.any(tm.pkx3 != 0.0):                           # parameter check, not a traced value
        Kx = Kx * np.exp(tm.pkx3 * dfz)
    Bx = Kx / (tm.Cx * Dx + EPS_BCD)

    mu_y = np.maximum((tm.pdy1 + tm.pdy2 * dfz) * mu_scale, 0.0)
    Dy = mu_y * Fz
    Ey = np.minimum(np.minimum(tm.pey1 + tm.pey2 * dfz, 1.0) + tm.e_shift, 1.0)
    z = Fz * tm.inv_pky2_fz0                        # sin(2 atan z) == 2z/(1+z^2), exactly
    Ky = tm.pky1_fz0 * (2.0 * z / (1.0 + z * z)) * tm.k_scale
    By = Ky / (tm.Cy * Dy + EPS_BCD)
    return Coef(Dx, tm.Cx, Bx, Ex, Kx, mu_x, Dy, tm.Cy, By, Ey, Ky, mu_y,
                tm.slip_scale, tm.inv_slip_scale, Fz,
                tm.rbx1, tm.rbx2, tm.rcx1, tm.rby1, tm.rby2, tm.rby3, tm.rcy1)


# ----------------------------------------------------------------------------- slip -> force (array core)
def _lateral_input(alpha: ArrayLike) -> np.ndarray:
    """Lateral MF input ``tan(alpha)`` with |alpha| clamped just below pi/2 (dimensionless)."""
    return np.tan(np.clip(_f64(alpha), -ALPHA_MAX, ALPHA_MAX))


def _pure(c: Coef, kappa: ArrayLike, alpha: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    Fx0 = _mf_core(c.Bx * (_f64(kappa) * c.inv_slip_scale), c.Cx, c.Dx, c.Ex)
    Fy0 = -_mf_core(c.By * (_lateral_input(alpha) * c.inv_slip_scale), c.Cy, c.Dy, c.Ey)
    return Fx0, Fy0


def _similarity(c: Coef, kappa: ArrayLike, alpha: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Bakker/Nyborg/Pacejka similarity method on stiffness-normalized slips.

    ``sx = Bx*Cx*kappa/s``, ``sy = By*Cy*tan(alpha)/s``, ``rho = |(sx, sy)|``;
    ``Fx = MFx(rho) * sx/rho``, ``Fy = -MFy(rho) * sy/rho`` with ``MF(rho) = mf(rho/(B*C), B, C, D, E)``
    evaluated on the product ``B*x = rho/C``. Exact pure slip when the other slip is 0, and
    ``|F| <= max(Dx, Dy)`` by construction.
    """
    sx = c.Bx * c.Cx * (_f64(kappa) * c.inv_slip_scale)
    sy = c.By * c.Cy * (_lateral_input(alpha) * c.inv_slip_scale)
    rho = np.sqrt(sx * sx + sy * sy)
    inv_rho = 1.0 / np.maximum(rho, RHO_EPS)
    Fx = _mf_core(rho / c.Cx, c.Cx, c.Dx, c.Ex) * (sx * inv_rho)
    Fy = -_mf_core(rho / c.Cy, c.Cy, c.Dy, c.Ey) * (sy * inv_rho)
    return Fx, Fy


def _mf_weighting(c: Coef, kappa: ArrayLike, alpha: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """MF5.2 cosine weighting functions (Shxa = Shyk = 0, no kappa-induced side force), then the
    ellipse safety clamp ``(Fx/Dx)^2 + (Fy/Dy)^2 <= 1`` (racer.nl CalcMF52 "Combined slip")."""
    k = _f64(kappa) * c.inv_slip_scale
    a = _f64(alpha) * c.inv_slip_scale
    Fx0, Fy0 = _pure(c, kappa, alpha)
    Gxa = np.cos(c.rcx1 * np.arctan(c.rbx1 * np.cos(np.arctan(c.rbx2 * k)) * a))
    Gyk = np.cos(c.rcy1 * np.arctan(c.rby1 * np.cos(np.arctan(c.rby2 * (a - c.rby3))) * k))
    Fx = Gxa * Fx0
    Fy = Gyk * Fy0
    nx = Fx / np.maximum(c.Dx, DIV_EPS)
    ny = Fy / np.maximum(c.Dy, DIV_EPS)
    scale = 1.0 / np.maximum(np.hypot(nx, ny), 1.0)
    return Fx * scale, Fy * scale


def _ellipse(c: Coef, kappa: ArrayLike, alpha: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """racer.nl simple method: ``Fx = Fx0``, ``Fy = Fy0 * sqrt(max(0, 1 - (Fx/Dx)^2))``."""
    Fx0, Fy0 = _pure(c, kappa, alpha)
    nx = Fx0 / np.maximum(c.Dx, DIV_EPS)
    Fy = Fy0 * np.sqrt(np.maximum(0.0, 1.0 - nx * nx))
    return Fx0 * np.ones_like(Fy), Fy


_COMBINED = {"similarity": _similarity, "mf_weighting": _mf_weighting, "ellipse": _ellipse}


def _combined(mode: str, c: Coef, kappa: ArrayLike, alpha: ArrayLike):
    try:
        fn = _COMBINED[mode]
    except KeyError:
        raise ValueError(f"unknown combined_mode {mode!r}; expected one of {MODES}") from None
    return fn(c, kappa, alpha)


def plowing_force(loose_drag: ArrayLike, vyw: ArrayLike, Fz: ArrayLike) -> np.ndarray:
    """Lateral plowing drag on deformable surfaces (N, wheel frame).

    A tire sliding sideways through loose material (dirt, gravel, sand, grass) pushes a berm of
    it and feels a reaction that grows with the lateral slide speed. It is separate from the
    friction force (so it is NOT bounded by mu*Fz) and opposes the lateral contact velocity:
        q = vyw / PLOW_V_REF,   F_plow = -loose_drag * Fz * q / sqrt(1 + (q/PLOW_CAP)^2)
    Linear (viscous-like) for small slides, saturating at ``PLOW_CAP * loose_drag * Fz``.
    Zero on paved surfaces (``loose_drag = 0``) and at zero lateral velocity.
    """
    q = _f64(vyw) * (1.0 / PLOW_V_REF)
    return -_f64(loose_drag) * np.maximum(_f64(Fz), 0.0) * q / np.sqrt(1.0 + (q * (1.0 / PLOW_CAP)) ** 2)


class LowSpeed(NamedTuple):
    """Low-speed blend constants from SimParams (m/s)."""
    v_low_start: float
    inv_span: float
    v_c2: float


def low_speed_consts(sim: SimParams) -> LowSpeed:
    span = max(float(sim.v_low) - float(sim.v_low_start), DIV_EPS)
    return LowSpeed(float(sim.v_low_start), 1.0 / span, float(sim.v_c_low) ** 2)


def forces_model(tm: TireModel, ls: LowSpeed, kappa_lag: ArrayLike, alpha_lag: ArrayLike,
                 vxw: ArrayLike, vyw: ArrayLike, omega_R: ArrayLike, Fz: ArrayLike,
                 T: ArrayLike | None = None, contam: ArrayLike | None = None):
    """Fast-path wheel-frame tire forces from a precomputed model.

    omega_R : wheel surface speed ``omega*R`` (m/s). Returns
    ``(Fx, Fy, c, w, vsx, vsy, v_wheel, F_mf_x, F_mf_y, F_low_x, F_low_y)``; see
    :func:`tire_forces` for the model.
    """
    c = coefficients(tm, Fz, T, contam)
    F_mf_x, F_mf_y = _combined(tm.mode, c, kappa_lag, alpha_lag)
    vsx = omega_R - vxw
    vsy = -vyw
    inv_den = 1.0 / np.sqrt(vsx * vsx + vsy * vsy + ls.v_c2)
    F_low_x = c.Dx * vsx * inv_den
    F_low_y = c.Dy * vsy * inv_den
    v_wheel = np.sqrt(vxw * vxw + vyw * vyw)
    t = np.clip((v_wheel - ls.v_low_start) * ls.inv_span, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    Fx = F_low_x + w * (F_mf_x - F_low_x)
    Fy = F_low_y + w * (F_mf_y - F_low_y)
    return Fx, Fy, c, w, vsx, vsy, v_wheel, F_mf_x, F_mf_y, F_low_x, F_low_y


# ----------------------------------------------------------------------------- public API (dicts)
def _as_coef(coef: Coef | Mapping[str, Any]) -> Coef:
    """Accept a :class:`Coef` or a dict (as returned by ``tire_coefficients`` or built by hand)."""
    if isinstance(coef, Coef):
        return coef
    s = _f64(coef.get("slip_scale", 1.0))
    d = TireParams()
    return Coef(Dx=_f64(coef["Dx"]), Cx=_f64(coef["Cx"]), Bx=_f64(coef["Bx"]), Ex=_f64(coef["Ex"]),
                Kx=_f64(coef.get("Kx", 0.0)), mu_x=_f64(coef.get("mu_x", 0.0)),
                Dy=_f64(coef["Dy"]), Cy=_f64(coef["Cy"]), By=_f64(coef["By"]), Ey=_f64(coef["Ey"]),
                Ky=_f64(coef.get("Ky", 0.0)), mu_y=_f64(coef.get("mu_y", 0.0)),
                slip_scale=s, inv_slip_scale=1.0 / s, Fz=_f64(coef.get("Fz", 0.0)),
                **{k: float(coef.get(k, getattr(d, k))) for k in _R_KEYS})


def tire_coefficients(tp: TireParams, surf: SurfaceParams, cond: TireCondition | None,
                      Fz: ArrayLike, T_tire: ArrayLike | None,
                      contam: ArrayLike | None = None) -> dict[str, np.ndarray]:
    """Magic Formula coefficients per wheel, including surface and condition modifications.

    tp : compound; surf : SurfaceParams with scalar or (4,) numeric fields; cond : per-wheel
    TireCondition or None; Fz : wheel load (N, clamped >= 0); T_tire : degC or None;
    contam : contamination override (see ``condition_scales``).

    Returns a dict with ``Dx, Cx, Bx, Ex, Kx, mu_x, Dy, Cy, By, Ey, Ky, mu_y, slip_scale, Fz,
    Kx_eff, Ky_eff`` (``K/slip_scale``, the real origin slopes after the slip stretch) and the
    combined-slip coefficients ``rbx1 ... rcy1``. Formulas: see :func:`coefficients` and
    :func:`tire_model`.
    """
    tm = tire_model(tp, surf, cond)
    if contam is None and cond is not None:
        contam = cond.contamination
    c = coefficients(tm, Fz, T_tire, contam)
    d = c._asdict()
    d.pop("inv_slip_scale")
    d["Kx_eff"] = c.Kx * c.inv_slip_scale
    d["Ky_eff"] = c.Ky * c.inv_slip_scale
    return d


def pure_slip_forces(coef: Coef | Mapping[str, Any], kappa: ArrayLike,
                     alpha: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Pure-slip forces ``(Fx0, Fy0)`` in N.

    kappa : slip ratio (-); alpha : slip angle (rad). Both are divided by ``slip_scale`` before
    entering the Magic Formula (the lateral input is ``tan(alpha)``).
    ``Fy0 = -mf(tan(alpha)/slip_scale, By, Cy, Dy, Ey)``.
    """
    return _pure(_as_coef(coef), kappa, alpha)


def combined_forces(coef: Coef | Mapping[str, Any], kappa: ArrayLike, alpha: ArrayLike,
                    mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Combined-slip forces ``(Fx, Fy)`` in N; ``mode`` in ``MODES``.

    ``"similarity"`` (normalized-slip similarity, exact pure slip when the other slip is 0,
    resultant <= max(Dx, Dy) by construction), ``"mf_weighting"`` (MF5.2 cosine weighting
    followed by an ellipse safety clamp) or ``"ellipse"`` (racer.nl simple method). In every mode
    ``hypot(Fx, Fy) <= max(mu_x, mu_y) * Fz`` up to round-off.
    """
    return _combined(mode, _as_coef(coef), kappa, alpha)


def tire_forces(tp: TireParams, surf: SurfaceParams, cond: TireCondition | None,
                kappa_lag: ArrayLike, alpha_lag: ArrayLike,
                vxw: ArrayLike, vyw: ArrayLike, omega: ArrayLike,
                Fz: ArrayLike, T_tire: ArrayLike | None, sim: SimParams,
                R: ArrayLike | None = None, contam: ArrayLike | None = None
                ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Wheel-frame tire friction forces ``(Fx, Fy, info)`` in N.

    kappa_lag, alpha_lag : relaxed slip states (fed to the Magic Formula).
    vxw, vyw : wheel-frame contact velocity (m/s); omega : wheel speed (rad/s); Fz : load (N);
    T_tire : degC; sim : SimParams (low-speed blend); R : wheel radius (m, default
    ``VehicleParams().wheel_radius``); contam : contamination state (default
    ``cond.contamination``).

    Low-speed model (``v_slip = (omega*R - vxw, -vyw)`` opposed, saturating at ``mu*Fz``):
        F_low = (Dx*vsx, Dy*vsy) / sqrt(vsx^2 + vsy^2 + v_c_low^2)
        w = smoothstep(hypot(vxw, vyw), v_low_start, v_low);   F = w*F_MF + (1 - w)*F_low
    At rest (zero slip velocity, zero lagged slips) the force is exactly 0. The plowing drag
    (:func:`plowing_force`) is NOT included here; vehicle.py adds it to the lateral force.

    info keys: ``mu_x, mu_y, Dx, Dy, Kx, Ky, Bx, By, Cx, Cy, Ex, Ey, slip_scale, Fz, w_low,
    F_low_x, F_low_y, F_mf_x, F_mf_y, kappa_used, alpha_used, vsx, vsy, v_wheel``.
    """
    R_w = float(VehicleParams().wheel_radius) if R is None else R
    if contam is None and cond is not None:
        contam = cond.contamination
    tm = tire_model(tp, surf, cond)
    (Fx, Fy, c, w, vsx, vsy, v_wheel,
     F_mf_x, F_mf_y, F_low_x, F_low_y) = forces_model(
        tm, low_speed_consts(sim), _f64(kappa_lag), _f64(alpha_lag), _f64(vxw), _f64(vyw),
        _f64(omega) * _f64(R_w), Fz, T_tire, contam)
    info = dict(mu_x=c.mu_x, mu_y=c.mu_y, Dx=c.Dx, Dy=c.Dy, Kx=c.Kx, Ky=c.Ky, Bx=c.Bx, By=c.By,
                Cx=c.Cx, Cy=c.Cy, Ex=c.Ex, Ey=c.Ey, slip_scale=c.slip_scale, Fz=c.Fz,
                w_low=w, F_low_x=F_low_x, F_low_y=F_low_y, F_mf_x=F_mf_x, F_mf_y=F_mf_y,
                kappa_used=_f64(kappa_lag), alpha_used=_f64(alpha_lag),
                vsx=vsx, vsy=vsy, v_wheel=v_wheel)
    return Fx, Fy, info


# ----------------------------------------------------------------------------- thermal
def thermal_rate(tp: TireParams, Fx: ArrayLike, Fy: ArrayLike, vsx: ArrayLike, vsy: ArrayLike,
                 v_wheel: ArrayLike, T: ArrayLike, T_ambient: ArrayLike) -> np.ndarray:
    """Tire temperature rate ``dT/dt`` in degC/s (lumped tread model).

    ``dT/dt = (|Fx*vsx| + |Fy*vsy|)/heat_capacity
              - (cool_coeff + cool_speed_coeff*|v_wheel|) * (T - T_ambient)/heat_capacity``
    Heating is the frictional slip power; cooling is convective (stronger when rolling).
    Fx, Fy : wheel-frame friction forces (N); vsx, vsy : slip velocity (m/s); v_wheel :
    contact-point speed over ground (m/s); T, T_ambient : degC. heat_capacity J/K,
    cool_coeff W/K, cool_speed_coeff W/K per m/s.
    """
    heating = np.abs(_f64(Fx) * _f64(vsx)) + np.abs(_f64(Fy) * _f64(vsy))
    cooling = (tp.cool_coeff + tp.cool_speed_coeff * np.abs(_f64(v_wheel))) * (_f64(T) - _f64(T_ambient))
    return (heating - cooling) / float(tp.heat_capacity)
