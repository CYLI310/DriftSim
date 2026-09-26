"""Fixed-step ODE integrators for the flat vehicle state (docs/DESIGN.md section 8).

Every integrator takes a right-hand side ``f(s, *args) -> ds`` (a float64 array of the same
shape as ``s``, i.e. ``vehicle.rhs`` / ``vehicle.rhs_model``) and returns a NEW state array; the
input state is never mutated. States may carry a leading batch shape (..., NS). Time is in s; the state layout (indices) comes from ``state.py`` and is only
needed by the semi-implicit scheme, which treats the pose ``X, Y, YAW`` specially.

Written JAX-portably: pure functions, no in-place writes, no Python branching on array
values (the only Python branch is on the ``method`` configuration string).

Public API
----------
    rk4_step(f, s, dt, *args) -> s_new
    semi_implicit_euler_step(f, s, dt, *args) -> s_new
    step(f, s, dt, method, *args) -> s_new          method in METHODS
    integrate(f, s, dt, n_steps, method, *args) -> s_new
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .state import POSE, R, VX, VY, X, Y, YAW

Rhs = Callable[..., np.ndarray]

METHODS = ("rk4", "semi_implicit_euler")


def rk4_step(f: Rhs, s: np.ndarray, dt: float, *args: Any) -> np.ndarray:
    """One classical 4th-order Runge-Kutta step of size ``dt`` (s).

    ``k1 = f(s)``, ``k2 = f(s + dt/2 k1)``, ``k3 = f(s + dt/2 k2)``, ``k4 = f(s + dt k3)``,
    ``s_new = s + dt/6 (k1 + 2 k2 + 2 k3 + k4)``. ``f(s, *args)`` must return ``ds`` only.
    Returns a new array; ``s`` is untouched.
    """
    s = np.asarray(s, dtype=np.float64)
    h = float(dt)
    k1 = f(s, *args)
    k2 = f(s + 0.5 * h * k1, *args)
    k3 = f(s + 0.5 * h * k2, *args)
    k4 = f(s + h * k3, *args)
    return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def semi_implicit_euler_step(f: Rhs, s: np.ndarray, dt: float, *args: Any) -> np.ndarray:
    """One semi-implicit (symplectic-style) Euler step of size ``dt`` (s).

    All states except the world pose are advanced with explicit Euler,
    ``s_e = s + dt * f(s)``. The pose is then advanced with the NEW body velocities and the
    NEW yaw:
        yaw_new = yaw + dt * r_new
        x_new   = x + dt * (vx_new cos(yaw_new) - vy_new sin(yaw_new))
        y_new   = y + dt * (vx_new sin(yaw_new) + vy_new cos(yaw_new))
    First-order accurate; cheaper than RK4 (one RHS evaluation) and the pose update is
    consistent with the velocity that will hold over the next interval. Works on (..., NS).
    """
    s = np.asarray(s, dtype=np.float64)
    h = float(dt)
    s_e = s + h * f(s, *args)                              # explicit Euler for everything
    vx_n, vy_n, r_n = s_e[..., VX], s_e[..., VY], s_e[..., R]   # NEW body velocities / yaw rate
    yaw_n = s[..., YAW] + h * r_n
    c, sn = np.cos(yaw_n), np.sin(yaw_n)
    x_n = s[..., X] + h * (vx_n * c - vy_n * sn)
    y_n = s[..., Y] + h * (vx_n * sn + vy_n * c)
    return np.concatenate([np.stack([x_n, y_n, yaw_n], axis=-1), s_e[..., POSE.stop:]], axis=-1)


def step(f: Rhs, s: np.ndarray, dt: float, method: str, *args: Any) -> np.ndarray:
    """One step of size ``dt`` (s) with ``method`` in ``("rk4", "semi_implicit_euler")``."""
    if method == "rk4":
        return rk4_step(f, s, dt, *args)
    if method == "semi_implicit_euler":
        return semi_implicit_euler_step(f, s, dt, *args)
    raise ValueError(f"unknown integrator {method!r}; expected one of {METHODS}")


def integrate(f: Rhs, s: np.ndarray, dt: float, n_steps: int, method: str,
              *args: Any) -> np.ndarray:
    """Advance ``s`` by ``n_steps`` fixed steps of ``dt`` (s) with ``method``; returns the final state.

    ``n_steps`` is configuration (a Python int), so the loop is an ordinary Python loop
    (``jax.lax.fori_loop`` in the JAX port).
    """
    if method not in METHODS:
        raise ValueError(f"unknown integrator {method!r}; expected one of {METHODS}")
    s_cur = np.asarray(s, dtype=np.float64)
    for _ in range(int(n_steps)):
        s_cur = step(f, s_cur, dt, method, *args)
    return s_cur
