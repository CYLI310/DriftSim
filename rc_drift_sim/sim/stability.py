"""Stiffness / integrator-stability report for a parameter set.

The fastest modes of the model are set by a handful of parameter combinations. For explicit
fixed-step integrators the step must satisfy ``lambda * dt < limit`` on the (negative) real axis:
RK4 ~2.785, (semi-implicit) Euler 2.0. This module estimates each fast rate from the parameters so
a configuration that would blow up (e.g. a stiff viscous coupling) is caught at construction time,
long before it produces NaNs in the middle of a training run.

Rates (1/s):
    electrical        R_m / L = 1 / motor_tau_e
    rear/front diff   2 * visc / I_side                      (side-to-side speed difference)
    center coupling   visc_shaft * (1/I_f,shaft + 1/I_r,shaft), I_x,shaft = sum I_i / G_i^2
    low-speed tire    mu_max * Fz_static,max / v_c_low * R^2 / I_wheel,min   (viscous tire regime;
                      SATURATING: the viscous force saturates at mu*Fz, so an excursion past the
                      linear limit stays bounded at |slip velocity| ~ v_c_low; flagged only above
                      SATURATING_FACTOR x the limit)
    tire vs wheel     sqrt(Kx * R^2 / (I_wheel,min * relax_x))             (slip-relaxation mode)
    servo             1 / servo_tau
    load transfer     1 / load_transfer_tau
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import drivetrain as dtm
from .params import Params
from .surface import uniform_surface

LIMITS = {"rk4": 2.785, "semi_implicit_euler": 2.0}
SATURATING = ("low-speed tire",)
SATURATING_FACTOR = 2.0


@dataclass
class StiffnessReport:
    rates: dict[str, float]     # 1/s
    dt: float
    method: str

    @property
    def limit(self) -> float:
        return LIMITS[self.method]

    def margin(self, name: str) -> float:
        """lambda*dt of mode ``name`` relative to what is allowed for it (< 1 is fine)."""
        allowed = self.limit * (SATURATING_FACTOR if name in SATURATING else 1.0)
        return self.rates[name] * self.dt / allowed

    @property
    def worst(self) -> tuple[str, float]:
        """(mode, lambda*dt) of the mode closest to (or furthest past) its allowed limit."""
        k = max(self.rates, key=self.margin)
        return k, self.rates[k] * self.dt

    @property
    def ok(self) -> bool:
        return all(self.margin(k) < 1.0 for k in self.rates)

    def table(self) -> str:
        rows = [f"  {k:16s} {r:10.1f} 1/s   lambda*dt = {r * self.dt:6.3f}" for k, r in
                sorted(self.rates.items(), key=lambda kv: -kv[1])]
        return (f"stiffness at dt = {self.dt * 1e3:.3g} ms ({self.method}, limit {self.limit}):\n"
                + "\n".join(rows))


def stiffness_report(p: Params, surf=None) -> StiffnessReport:
    vp, dp, tp, ap, sim = p.vehicle, p.drivetrain, p.tire, p.actuators, p.sim
    surf = uniform_surface(p.surface) if surf is None else surf
    G, driven = dtm.gear_ratios(dp)
    I = dtm.wheel_inertias(dp, vp)
    R = vp.wheel_radius
    rates: dict[str, float] = {"electrical": 1.0 / dp.motor_tau_e,
                               "servo": 1.0 / ap.servo_tau,
                               "load transfer": 1.0 / vp.load_transfer_tau}
    rates["rear diff"] = 2.0 * dp.rear_diff_visc / I[2]
    if driven[0] > 0:
        rates["front diff"] = 2.0 * dp.front_diff_visc / I[0]
        I_f_shaft = (I[0] + I[1]) / G[0] ** 2
        I_r_shaft = (I[2] + I[3]) / G[2] ** 2
        rates["center coupling"] = dp.center_visc * (1.0 / I_f_shaft + 1.0 / I_r_shaft)
    mu = float(np.max(np.asarray(surf.mu_scale) * (1.0 + tp.loose_affinity * np.asarray(surf.looseness)))) \
        * max(tp.pdx1, tp.pdy1)
    fz = float(np.max(vp.static_loads(sim.gravity))) * 1.5          # allow for braking load transfer
    rates["low-speed tire"] = mu * fz / sim.v_c_low * R ** 2 / float(I.min())
    kx = tp.pkx1 * fz * float(np.max(np.asarray(surf.stiffness_scale)))
    rates["tire-wheel mode"] = float(np.sqrt(kx * R ** 2 / (float(I.min()) * tp.relax_x)))
    return StiffnessReport(rates=rates, dt=float(sim.dt), method=str(sim.integrator))


def check_stiffness(p: Params, surf=None, strict: bool = False) -> StiffnessReport:
    """Warn (or raise with ``strict``) if the configuration is too stiff for its integrator."""
    rep = stiffness_report(p, surf)
    if not rep.ok:
        name, ldt = rep.worst
        msg = (f"'{name}' mode is too stiff for {rep.method} at dt = {rep.dt * 1e3:.3g} ms "
               f"(lambda*dt = {ldt:.2f} >= {rep.limit}); reduce the coupling or dt.\n{rep.table()}")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return rep
