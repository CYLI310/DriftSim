"""A stand-in for ``beamngpy``: DriftSim physics enlarged ``k`` times at the same Froude number, behind
the BeamNGpy calls the BeamNG driver uses. It checks the driver (``deploy.beamng``) without BeamNG and
reproduces, on purpose, the ways a full-size BeamNG car differs from the RC car:

* ``engine``: the throttle is a torque request (a soft, engine-like drivetrain) instead of the RC
  motor's stiff wheel-speed command;
* ``auto_gearbox``: with the default automatic shifting the drive cuts out for ``shift_s`` on every
  up/down shift; ``set_shift_mode('realistic_manual...')`` + ``gear`` keeps one gear;
* ``assists``: traction control cuts the throttle when the rear wheels spin and the stability control
  brakes against a slide, until ``set_esc_mode`` picks the ``off`` mode (``cycle_esc_mode`` cycles
  ``on`` -> ``sport`` -> ``off``);
* ``steer_ratio``: road-wheel angle of full BeamNG steering / the trained car's lock;
* ``wheelspeed``: the Electrics ``wheelspeed`` is the mean of all four wheels (as a speedometer); the
  per-wheel speeds come from the driver's Lua query (``queue_lua_command`` with a response);
* ``mirror`` / ``steer_sign``: a left-handed world and a reversed steering input.

    with standin(k=10, engine=True, assists=True):
        car = BeamNGCar(meta, home="x")              # imports the stand-in instead of beamngpy
        drive(runtime, car, 20.0)
"""
from __future__ import annotations

import contextlib
import math
import sys
import types
from typing import Any, Iterator

import numpy as np

from ..sim import state as S

ESC_MODES = ("on", "sport", "off")
LUA_MARKER = "driftsim_wheels"                 # the driver's per-wheel query contains this name


def fake_beamngpy(k: float = 10.0, mirror: bool = False, steer_sign: float = 1.0, grip: float = 0.7, *,
                  engine: bool = False, auto_gearbox: bool = False, assists: bool = False, steer_ratio: float = 1.0,
                  lua_wheels: bool = True, shift_s: float = 0.3, body_scale: float = 1.0) -> dict[str, types.ModuleType]:
    """The ``beamngpy`` and ``beamngpy.sensors`` modules of the stand-in (see the module docstring)."""
    from ..sim.vehicle import make_vehicle
    over: dict[str, Any] = dict(mu_scale=grip)
    if engine:                                  # high winding resistance + high voltage: torque ~ throttle
        over.update(drivetrain=dict(motor_resistance=0.6, battery_voltage=30.0, current_limit=40.0))
    car = make_vehicle(**over)
    sk = math.sqrt(k)
    Rw = float(car.params.vehicle.wheel_radius)
    sgn = -1.0 if mirror else 1.0
    v_shift = 1.2                               # RC-equivalent rear-wheel speed per gear of the automatic

    class Vehicle:
        def __init__(self, vid: str, model: str, **kw: Any):
            self.vid, self.model, self.kw = vid, model, kw
            self.s = car.initial_state()
            self.cmd = dict(steering=0.0, throttle=0.0, brake=0.0, parkingbrake=0.0)
            self.sensors = self
            self._el: dict = {}
            self.esc = 0 if assists else 2          # index into ESC_MODES
            self.manual = False
            self.gear = 1
            self.cut = 0.0                          # seconds of shift torque cut left (RC time)
            self.esc_on = self.tcs_on = False
            self.calls: dict[str, int] = {}

        def _count(self, name: str) -> None:
            self.calls[name] = self.calls.get(name, 0) + 1

        # ---------------------------------------------------------------- sensors API
        def attach(self, name: str, sensor: Any) -> None:
            pass

        def _wheels(self) -> np.ndarray:
            return np.asarray(self.s[S.OMEGA], dtype=float) * Rw * sk     # FL FR RL RR surface speeds (BeamNG m/s)

        def poll(self) -> None:
            w = self._wheels()
            self._el = {"wheelspeed": float(np.mean(np.abs(w))), "airspeed": float(math.hypot(self.s[S.VX], self.s[S.VY]) * sk),
                        "gear": self.gear, "esc": 1 if ESC_MODES[self.esc] == "off" else (2 if self.esc_on else 0),
                        "esc_active": self.esc_on, "tcs_active": self.tcs_on, "steering_input": self.cmd["steering"]}

        def __getitem__(self, name: str) -> dict:
            return self._el

        @property
        def state(self) -> dict:
            yaw = self.s[S.YAW]
            f = np.array([math.cos(yaw), sgn * math.sin(yaw), 0.0])
            lft = np.array([-math.sin(yaw), sgn * math.cos(yaw), 0.0])
            v = (self.s[S.VX] * f + self.s[S.VY] * lft) * sk
            return dict(pos=[self.s[S.X] * k, sgn * self.s[S.Y] * k, 0.0], dir=list(f), up=[0, 0, 1], vel=list(v))

        # ---------------------------------------------------------------- vehicle API
        def control(self, steering: float | None = None, throttle: float | None = None, brake: float | None = None,
                    parkingbrake: float | None = None, clutch: float | None = None, gear: int | None = None) -> None:
            for key, val in (("steering", steering), ("throttle", throttle), ("brake", brake), ("parkingbrake", parkingbrake)):
                if val is not None:
                    self.cmd[key] = float(val)
            if gear is not None and self.manual:
                self.gear = int(gear)
            self._count("control")

        def set_shift_mode(self, mode: str) -> None:
            self.manual = "manual" in mode
            self._count("set_shift_mode")

        def get_esc_mode(self) -> str:
            return ESC_MODES[self.esc]

        def set_esc_mode(self, mode: str) -> None:
            if mode in ESC_MODES:
                self.esc = ESC_MODES.index(mode)

        def cycle_esc_mode(self) -> None:
            self.esc = (self.esc + 1) % len(ESC_MODES)

        def queue_lua_command(self, chunk: str, response: bool = False) -> Any:
            self._count("lua")
            if LUA_MARKER in chunk and lua_wheels and response:
                w = self._wheels() / Rw / sk                                  # rad/s at RC scale -> BeamNG rad/s
                r_b = Rw * k
                return {n: [float(w[i]) / sk, r_b, 0] for i, n in enumerate(("FL", "FR", "RL", "RR"))}
            return None

        def get_bbox(self) -> dict:
            L = car.params.vehicle.body_length * k * body_scale
            return {"a": [0, 0, 0], "b": [L, 0.4 * L, 0.3 * L]}

        # ---------------------------------------------------------------- physics (one BeamNG step chunk)
        def advance(self, model_time: float) -> None:
            c = self.cmd
            steer = float(np.clip(steer_sign * c["steering"] * steer_ratio, -1.0, 1.0))
            thr, brake = c["throttle"], c["brake"]
            w = self._wheels() / sk
            rear = 0.5 * (w[2] + w[3])
            vx = float(self.s[S.VX])
            self.esc_on = self.tcs_on = False
            if ESC_MODES[self.esc] != "off":
                slip = (rear - vx) / max(abs(vx), 0.5)
                if slip > (0.4 if ESC_MODES[self.esc] == "sport" else 0.15):
                    thr, self.tcs_on = thr * 0.15, True
                beta = math.atan2(self.s[S.VY], max(abs(vx), 1e-3))
                if ESC_MODES[self.esc] == "on" and abs(beta) > math.radians(12) and abs(vx) > 0.5:
                    brake, self.esc_on = max(brake, 0.35), True
            if auto_gearbox and not self.manual:
                g = int(np.clip(1 + math.floor(max(w[2], w[3]) / v_shift), 1, 5))
                if g != self.gear:
                    self.gear, self.cut = g, shift_s / sk
            gear_gain = 1.0 / max(self.gear, 1) ** 0.5 if (auto_gearbox and not self.manual) else 1.0
            if self.cut > 0:
                thr, self.cut = 0.0, self.cut - model_time
            # brakes slow the wheels down but never drive them (DriftSim's negative throttle is the motor's
            # brake / reverse): the torque opposes the rear wheels' rotation and fades out near standstill.
            # The parking brake is rear-only, like the motor's braking; the foot brake is approximated the same way.
            against = float(np.clip(rear / 0.3, -1.0, 1.0))           # a brake torque opposes the rotation
            u = np.array([steer, thr * gear_gain - (brake + c["parkingbrake"]) * against])
            n = max(1, int(round(model_time / car.dt)))
            self.s, _ = car.step(self.s, u, n_sub=n, want_info=False)

    class Ctl:
        def __init__(self, bng: "BeamNGpy"):
            self.bng = bng

        def pause(self) -> None:
            pass

        def step(self, n: int, wait: bool = True) -> None:
            self.bng.vehicle.advance(n / self.bng.sps / sk)
            self.bng.steps += n

    class Settings:
        def __init__(self, bng: "BeamNGpy"):
            self.bng = bng

        def set_deterministic(self, sps: int) -> None:
            self.bng.sps = sps

    class ScenarioApi:
        def load(self, scenario: Any) -> None:
            pass

        def start(self) -> None:
            pass

    class BeamNGpy:
        def __init__(self, host: str, port: int, home: str | None = None, user: str | None = None):
            self.control, self.settings, self.scenario = Ctl(self), Settings(self), ScenarioApi()
            self.sps, self.steps = 60, 0

        def open(self, launch: bool = True) -> "BeamNGpy":
            return self

        def close(self) -> None:
            pass

    class Scenario:
        def __init__(self, level: str, name: str):
            self.level = level

        def add_vehicle(self, vehicle: Vehicle, pos: Any, rot_quat: Any) -> None:
            self.vehicle = vehicle

        def make(self, bng: BeamNGpy) -> None:
            bng.vehicle = self.vehicle

    mod = types.ModuleType("beamngpy")
    mod.BeamNGpy, mod.Scenario, mod.Vehicle = BeamNGpy, Scenario, Vehicle
    mod.STANDIN = dict(k=k, car=car)
    sens = types.ModuleType("beamngpy.sensors")
    sens.Electrics = lambda: None
    mod.sensors = sens
    return {"beamngpy": mod, "beamngpy.sensors": sens}


@contextlib.contextmanager
def standin(**kw: Any) -> Iterator[types.ModuleType]:
    """Put the stand-in in ``sys.modules`` as ``beamngpy`` for the duration of the ``with`` block."""
    mods = fake_beamngpy(**kw)
    saved = {name: sys.modules.get(name) for name in mods}
    sys.modules.update(mods)
    try:
        yield mods["beamngpy"]
    finally:
        for name, m in saved.items():
            if m is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = m


__all__ = ["fake_beamngpy", "standin", "ESC_MODES", "LUA_MARKER"]
