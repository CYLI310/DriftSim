"""Steady-state (trim) analysis of the full vehicle model: grip cornering and drift equilibria,
their linearization (continuous and one-control-period discrete) and stability.

The LQR controller built on these linearizations lives in ``rc_drift_sim.control.lqr``.

Why this exists
---------------
* Steady-state cornering tests (understeer gradient) are cleaner as trim solutions than as long
  time simulations that have to "settle".
* RWD drift equilibria (rear axle saturated, front counter-steered) are open-loop UNSTABLE
  (Hindiyeh & Gerdes, "A controller framework for autonomous drifting", 2014). The trim
  solver finds them, the eigenvalues quantify the instability, and ``control.lqr.TrimLQR``
  shows the simulated car can be held in a drift with full-state feedback through the real
  actuator chain (servo, motor, latency). That is exactly the capability the RL policy must learn.

Trim definition
---------------
Circular motion at constant speed ``v`` and sideslip ``beta``: every state derivative is zero
except the pose (X, Y, YAW advance) and the contamination (fixed at 0, it only decays). The
unknowns are the 23 "dynamic" states (VX, VY, R, OMEGA, DELTA, I_MOTOR, T_TIRE, DFZ, KAPPA_LAG,
ALPHA_LAG) plus the two inputs (steer, throttle); the equations are their 23 derivatives plus
``speed = v`` and ``atan2(vy, vx) = beta``. Solved with ``scipy.optimize.least_squares``.

All angles in rad, speeds in m/s. Input steer/throttle are the normalized commands in [-1, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import state as S
from .vehicle import Vehicle, derivatives_model

# least-squares evaluation budget per solve: converged trims need <= 40 evaluations (measured over
# grip and drift families), so 200 leaves 5x headroom while making a failed start cheap
MAX_NFEV = 200

# states that are in equilibrium in a steady turn (pose advances, contamination decays)
DYN = np.array([S.VX, S.VY, S.R, *range(6, 10), S.DELTA, S.I_MOTOR, *range(12, 16),
                S.DFZ_LONG, S.DFZ_LAT, *range(18, 22), *range(22, 26)])
# states used for feedback (temperature is slow and nearly irrelevant to the fast dynamics)
CTRL = np.array([S.VX, S.VY, S.R, *range(6, 10), S.DELTA, S.I_MOTOR, S.DFZ_LONG, S.DFZ_LAT,
                 *range(18, 22), *range(22, 26)])


@dataclass
class Trim:
    """A steady-state operating point."""
    s: np.ndarray            # (NS,) full state (pose zero, contamination zero)
    u: np.ndarray            # (2,) (steer_cmd, throttle_cmd)
    v: float                 # speed (m/s)
    beta: float              # sideslip (rad)
    r: float                 # yaw rate (rad/s)
    radius: float            # path radius v / r (m)
    ay: float                # lateral acceleration v*r (m/s^2)
    residual: float          # max |ds| over the dynamic states at the solution
    success: bool
    info: dict

    def summary(self) -> str:
        return (f"v={self.v:.2f} m/s beta={np.degrees(self.beta):+.1f} deg r={np.degrees(self.r):+.0f} deg/s "
                f"R={self.radius:.2f} m ay={self.ay:.2f} m/s^2 ({self.ay / 9.81:.2f} g) "
                f"steer={self.u[0]:+.3f} thr={self.u[1]:.3f} residual={self.residual:.1e}")


def _pack(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.concatenate([s[DYN], u])


def _unpack(z: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = template.copy()
    s[DYN] = z[:-2]
    return s, z[-2:]


def _initial_guess(vehicle: Vehicle, v: float, beta0: float, r_guess: float, steer_guess: float,
                   thr_guess: float | None, kappa_rear_guess: float) -> tuple[np.ndarray, np.ndarray]:
    p, vm = vehicle.params, vehicle.model
    s0 = vehicle.initial_state(v=v, beta=beta0, yaw_rate=r_guess)
    om = s0[S.OMEGA].copy()
    om[2:] *= 1.0 + kappa_rear_guess
    s0[S.OMEGA] = om
    s0[S.T_TIRE] = p.sim.ambient_temp + 10.0
    s0[S.DELTA] = steer_guess * p.actuators.steer_max
    s0[S.CONTAM] = 0.0
    _, info0 = derivatives_model(s0, np.array([steer_guess, 0.0]), vm, want_info=True)
    s0[S.KAPPA_LAG] = info0["kappa"]            # consistent lagged slips at the guess
    s0[S.ALPHA_LAG] = info0["alpha"]
    if thr_guess is None:
        v_nl = (p.drivetrain.battery_voltage / p.drivetrain.ke / p.drivetrain.gear_ratio
                * p.vehicle.wheel_radius)
        thr_guess = float(np.clip((1.0 + kappa_rear_guess) * v * np.cos(beta0) / v_nl + 0.02, 0.0, 1.0))
    return s0, np.array([steer_guess, thr_guess])


def _solve(vehicle: Vehicle, v: float, beta: float | None, r: float | None, s0: np.ndarray,
           u0: np.ndarray, max_nfev: int) -> Trim:
    vm = vehicle.model
    template = s0.copy()
    template[S.CONTAM] = 0.0
    template[S.X] = template[S.Y] = template[S.YAW] = 0.0
    # scale each residual so the solver sees O(1) numbers
    scale = np.ones(len(DYN))
    scale[3:7] = 1.0 / 100.0          # wheel accelerations (rad/s^2)
    scale[8] = 1.0 / 1000.0           # current derivative (A/s)
    scale[9:13] = 1.0 / 10.0          # temperature rates (degC/s)
    scale[13:15] = 1.0 / 10.0         # load-transfer rates (N/s)

    def resid(z: np.ndarray) -> np.ndarray:
        s, u = _unpack(z, template)
        ds, _ = derivatives_model(s, u, vm, want_info=False)
        sp = np.hypot(s[S.VX], s[S.VY])
        c2 = (np.arctan2(s[S.VY], s[S.VX]) - beta) if beta is not None else (s[S.R] - r)
        return np.concatenate([ds[DYN] * scale, [(sp - v) * 10.0, c2 * 10.0]])

    lo = np.full(len(DYN) + 2, -np.inf)
    hi = np.full(len(DYN) + 2, np.inf)
    lo[-2:], hi[-2:] = [-1.0, -1.0], [1.0, 1.0]
    z0 = np.clip(_pack(template, u0), lo + 1e-9, hi - 1e-9)
    sol = least_squares(resid, z0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14,
                        max_nfev=max_nfev)
    s, u = _unpack(sol.x, template)
    ds, info = derivatives_model(s, u, vm, want_info=True)
    residual = float(np.max(np.abs(ds[DYN])))
    constraint = float(np.max(np.abs(resid(sol.x)[-2:]))) / 10.0
    rr = float(s[S.R])
    speed = float(np.hypot(s[S.VX], s[S.VY]))
    return Trim(s=s, u=u, v=speed, beta=float(np.arctan2(s[S.VY], s[S.VX])), r=rr,
                radius=float(speed / rr) if abs(rr) > 1e-9 else np.inf, ay=speed * rr,
                residual=max(residual, constraint),
                success=bool(residual < 1e-6 and constraint < 1e-6), info=info)


def solve_trim(vehicle: Vehicle, v: float, beta: float | None = None, r: float | None = None,
               guess: Trim | None = None, r_guess: float | None = None,
               steer_guess: float | None = None, thr_guess: float | None = None,
               kappa_rear_guess: float | None = None, max_nfev: int = MAX_NFEV,
               max_starts: int | None = None) -> Trim:
    """Solve for a steady turn at speed ``v`` (m/s) with EITHER sideslip ``beta`` (rad; natural for
    drift equilibria) OR yaw rate ``r`` (rad/s; natural for grip cornering, ``radius = v/r``).

    ``guess``: a nearby :class:`Trim` to warm-start from (continuation). Without it, the solver
    tries the given guesses (``r_guess`` rad/s, ``steer_guess``, ``thr_guess``, ``kappa_rear_guess``)
    and, if they are not all given, a small multi-start set (``max_starts`` caps it), keeping the
    best solution. Use ``kappa_rear_guess`` ~0.3-1 to aim for the saturated-rear drift branch.
    """
    if (beta is None) == (r is None):
        raise ValueError("give exactly one of beta or r")
    if guess is not None:
        return _solve(vehicle, v, beta, r, guess.s, guess.u, max_nfev)
    beta0 = 0.0 if beta is None else float(beta)
    L, smax = vehicle.params.vehicle.wheelbase, vehicle.params.actuators.steer_max
    R_ANCHOR = 0.25                     # rad/s: cold solves near straight are ill-conditioned
    if r is not None and abs(r) < R_ANCHOR:
        anchor = solve_trim(vehicle, v, r=float(np.copysign(R_ANCHOR, r if r != 0 else 1.0)),
                            max_nfev=max_nfev)
        tr = anchor
        for rr in np.linspace(anchor.r, float(r), 4)[1:]:
            tr = _solve(vehicle, v, None, float(rr), tr.s, tr.u, max_nfev)
        return tr
    if r is not None:
        r0 = float(r)
        kin = float(np.clip(np.arctan(L * r0 / max(v, 0.1)) / smax, -1, 1))   # kinematic steer
        starts = [(kin, 0.0), (1.3 * kin, 0.02), (kin, 0.1)]
    else:
        r0 = (r_guess if r_guess is not None
              else -np.sign(beta0 if beta0 != 0 else -1.0) * 0.3 * 9.81 / max(v, 0.3))
        starts = [(sg, kr) for kr in (0.5, 1.0, 0.3, 2.0) for sg in (-0.5, -0.2, -0.8)]
    if steer_guess is not None or kappa_rear_guess is not None:
        starts = [(steer_guess if steer_guess is not None else starts[0][0],
                   kappa_rear_guess if kappa_rear_guess is not None else starts[0][1])] + starts
    if max_starts is not None:
        starts = starts[:max(int(max_starts), 1)]
    best: Trim | None = None
    for sg, kr in starts:
        s0, u0 = _initial_guess(vehicle, v, beta0, r0, sg, thr_guess, kr)
        tr = _solve(vehicle, v, beta, r, s0, u0, max_nfev)
        if best is None or tr.residual < best.residual:
            best = tr
        if tr.success:
            break
    return best


def _warm_step(vehicle: Vehicle, v: float, key: str, prev: Trim, target: float,
               max_halvings: int = 3) -> Trim:
    """Warm-started step from ``prev`` to ``target`` with step-size control: on failure the step is
    split into 2, 4, 8 sub-steps, each warm-started from the last converged point."""
    x0 = prev.beta if key == "beta" else prev.r
    tr = solve_trim(vehicle, v, guess=prev, **{key: float(target)})
    for h in range(1, max_halvings + 1):
        if tr.success:
            return tr
        cur = prev
        for x in np.linspace(x0, target, 2 ** h + 1)[1:]:
            tr = solve_trim(vehicle, v, guess=cur, **{key: float(x)})
            if not tr.success:
                break
            cur = tr
    return tr


def continuation(vehicle: Vehicle, v: float, values, key: str = "beta", start: Trim | None = None,
                 **first_kwargs) -> list[Trim]:
    """Follow a family of trims in the given order: solve at ``values[0]`` (``key`` = "beta" or "r";
    cold multi-start unless ``start`` is given) and warm-start each next value from the last
    converged solution, with step-size control (see ``_warm_step``). If even the sub-stepped warm
    solve fails, a cold multi-start is tried (cheap: ``MAX_NFEV`` evaluations per start). Near the
    end of a branch the equilibrium can simply stop existing; such points come back unconverged.
    Returns the list of trims in the order of ``values`` (check ``.success``)."""
    out: list[Trim] = []
    prev = start
    for x in values:
        kw = {key: float(x)}
        if prev is None:
            tr = solve_trim(vehicle, v, **kw, **first_kwargs)
        else:
            tr = _warm_step(vehicle, v, key, prev, float(x))
            if not tr.success:                              # last resort: cold multi-start
                tr2 = solve_trim(vehicle, v, **kw, **first_kwargs)
                tr = tr2 if tr2.residual < tr.residual else tr
        out.append(tr)
        prev = tr if tr.success else prev
    return out


def trim_family(vehicle: Vehicle, v: float, values, key: str = "beta", anchor: int | None = None,
                **first_kwargs) -> list[Trim]:
    """A family of trims over ``values``, solved cold at ``values[anchor]`` (default: the middle,
    where the equilibrium is best established) and continued outward in both directions.
    Returns the trims in the order of ``values``."""
    vals = list(values)
    a = len(vals) // 2 if anchor is None else int(anchor)
    first = solve_trim(vehicle, v, **{key: float(vals[a])}, **first_kwargs)
    start = first if first.success else None
    up = continuation(vehicle, v, vals[a + 1:], key=key, start=start, **first_kwargs)
    down = continuation(vehicle, v, vals[:a][::-1], key=key, start=start, **first_kwargs)
    return down[::-1] + [first] + up


def linearize(vehicle: Vehicle, trim: Trim, eps: float = 1e-6
              ) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time Jacobians ``A = d(ds)/ds``, ``B = d(ds)/du`` at the trim (central differences),
    restricted to the dynamic states ``DYN`` (A is 23x23, B is 23x2)."""
    vm = vehicle.model
    n = len(DYN)
    A = np.zeros((n, n))
    Bm = np.zeros((n, 2))
    for j, idx in enumerate(DYN):
        h = eps * max(1.0, abs(trim.s[idx]))
        sp, sm = trim.s.copy(), trim.s.copy()
        sp[idx] += h
        sm[idx] -= h
        A[:, j] = (derivatives_model(sp, trim.u, vm)[0][DYN] - derivatives_model(sm, trim.u, vm)[0][DYN]) / (2 * h)
    for j in range(2):
        up, um = trim.u.copy(), trim.u.copy()
        up[j] += eps
        um[j] -= eps
        Bm[:, j] = (derivatives_model(trim.s, up, vm)[0][DYN] - derivatives_model(trim.s, um, vm)[0][DYN]) / (2 * eps)
    return A, Bm


def open_loop_eigenvalues(vehicle: Vehicle, trim: Trim) -> np.ndarray:
    """Eigenvalues (1/s) of the linearized dynamics at the trim, sorted by real part (largest first)."""
    A, _ = linearize(vehicle, trim)
    ev = np.linalg.eigvals(A)
    return ev[np.argsort(-ev.real)]


def discrete_model(vehicle: Vehicle, trim: Trim, eps: float = 1e-6
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time model over ONE control period of the real integrator (zero-order hold),
    on the feedback states ``CTRL``: ``x_{k+1} - x* = Ad (x_k - x*) + Bd (u_k - u*)``.
    Obtained by finite differences through ``Vehicle.step`` (so servo, motor and RK4 are exact)."""
    n = len(CTRL)
    base, _ = vehicle.step(trim.s, trim.u, want_info=False)
    Ad = np.zeros((n, n))
    Bd = np.zeros((n, 2))
    for j, idx in enumerate(CTRL):
        h = eps * max(1.0, abs(trim.s[idx])) * 100.0
        sp, sm = trim.s.copy(), trim.s.copy()
        sp[idx] += h
        sm[idx] -= h
        Ad[:, j] = (vehicle.step(sp, trim.u, want_info=False)[0][CTRL]
                    - vehicle.step(sm, trim.u, want_info=False)[0][CTRL]) / (2 * h)
    for j in range(2):
        h = 1e-4
        up, um = trim.u.copy(), trim.u.copy()
        up[j] += h
        um[j] -= h
        Bd[:, j] = (vehicle.step(trim.s, up, want_info=False)[0][CTRL]
                    - vehicle.step(trim.s, um, want_info=False)[0][CTRL]) / (2 * h)
    return Ad, Bd
