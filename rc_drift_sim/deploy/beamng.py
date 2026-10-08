"""BeamNG as the car: test the exported model in an independent physics engine before the real car.

    pip install beamngpy                       (on the Windows PC that runs BeamNG)
    driftsim-drive --model final --car beamng --beamng-home "C:/BeamNG.tech.v0.39" --vehicle etk800 \\
                   --part-config vehicles/etk800/<a rear-drive config>.pc --seconds 20

BeamNGpy is made for BeamNG.tech (free for academic use on request); the Steam BeamNG.drive works
with it only in part, so this driver uses only the basic API: the vehicle state (position, direction,
up, velocity), the ``Electrics`` sensor (wheel speed) and ``vehicle.control``, in deterministic
stepping (BeamNG waits for the policy every control period, like the training simulator).

Readings are computed from the state the way the car's sensors measure them: body velocity (optical
flow), yaw rate (gyro, from the heading change), body acceleration (IMU, from the velocity change)
and the driven-wheel speed (``Electrics`` "wheelspeed" by default; ``--wheel-key`` picks another).

Scale: BeamNG has no 1/10 RC car, so a full-size car is driven at the same Froude number. With
``k`` = BeamNG car length / trained car length, speeds are divided by sqrt(k), yaw rates multiplied by
sqrt(k), accelerations and angles are unchanged and the control period is 20 ms x sqrt(k). Choose a
rear-drive car on a low-grip surface (or slippery tires): a plastic-tire RC drift car behaves like a
full-size car at about mu 0.3. ``scale=1`` drives an RC-size mod as is. The steering sign is
calibrated at the start (a short left-steer test), so it does not depend on BeamNG's conventions.
This tests the model, runtime and safety filter end to end on unfamiliar dynamics; it is not a
replacement for identifying the real car's parameters.
"""
from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .car_loop import CarInterface


# ----------------------------------------------------------------------------- finding BeamNG
BINARY_NAMES = ("BeamNG.drive.exe", "BeamNG.tech.exe", "Bin64/BeamNG.drive.x64.exe", "Bin64/BeamNG.tech.x64.exe",
                "Bin64/BeamNG.x64.exe", "BinLinux/BeamNG.drive.x64", "BinLinux/BeamNG.tech.x64")


def beamngpy_installed() -> bool:
    import importlib.util
    import sys
    return "beamngpy" in sys.modules or importlib.util.find_spec("beamngpy") is not None


def is_beamng_home(folder: str | Path) -> bool:
    return any((Path(folder) / b).is_file() for b in BINARY_NAMES)


def steam_libraries() -> list[Path]:
    """Steam library folders on this PC (from libraryfolders.vdf), the default install first."""
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam",
             Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Steam",
             Path.home() / ".steam" / "steam", Path.home() / ".local" / "share" / "Steam"]
    libs: list[Path] = []
    for root in roots:
        if root.is_dir():
            libs.append(root)
            vdf = root / "steamapps" / "libraryfolders.vdf"
            if vdf.is_file():
                for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(errors="ignore")):
                    libs.append(Path(m.group(1).replace("\\\\", "\\")))
    out: list[Path] = []
    for lib in libs:
        if lib not in out:
            out.append(lib)
    return out


def find_beamng_home() -> str | None:
    """The BeamNG install folder: $BNG_HOME, then BeamNG.drive / BeamNG.tech in the Steam libraries and
    the usual BeamNG.tech locations. None if not found."""
    cands = [Path(os.environ["BNG_HOME"])] if os.environ.get("BNG_HOME") else []
    for lib in steam_libraries():
        cands += [lib / "steamapps" / "common" / "BeamNG.drive", lib / "steamapps" / "common" / "BeamNG.tech"]
    for base in (Path("C:/"), Path.home(), Path.home() / "Documents", Path.home() / "Downloads"):
        if base.is_dir():
            try:
                cands += sorted(p for p in base.iterdir() if p.name.lower().startswith("beamng"))
            except OSError:
                pass
    for c in cands:
        if c.is_dir() and is_beamng_home(c):
            return str(c)
    return None


# ----------------------------------------------------------------------------- pure helpers (tested)
def body_frame(direction: Any, up: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (forward, left, up) from BeamNG's vehicle direction and up vectors."""
    f = np.asarray(direction, dtype=float)
    u = np.asarray(up, dtype=float)
    u = u / np.linalg.norm(u)
    left = np.cross(u, f)
    left /= np.linalg.norm(left)
    f = np.cross(left, u)
    return f, left, u


def body_readings(direction: Any, up: Any, vel: Any, prev: dict | None, dt: float) -> dict:
    """Body-frame velocity (vx, vy), yaw rate r and acceleration (ax, ay) from two consecutive
    vehicle states ``dt`` apart (``prev`` = the previous call's result, None on the first call)."""
    f, left, u = body_frame(direction, up)
    v = np.asarray(vel, dtype=float)
    out = dict(vx=float(v @ f), vy=float(v @ left), r=0.0, ax=0.0, ay=0.0, _f=f, _v=v)
    if prev is not None:
        fp = prev["_f"]
        out["r"] = math.atan2(float(np.cross(fp, f) @ u), float(fp @ f)) / dt
        a = (v - prev["_v"]) / dt                      # inertial acceleration over the step, felt in the
        fm = (fp + f) / np.linalg.norm(fp + f)          # mid-step body frame (second-order accurate)
        lm = np.cross(u, fm)
        out["ax"], out["ay"] = float(a @ fm), float(a @ lm)
    return out


def froude(scale: float) -> dict:
    """Factors from the BeamNG car to the trained car at the same Froude number."""
    s = math.sqrt(scale)
    return dict(speed=1.0 / s, rate=s, accel=1.0, time=s)


def to_policy_readings(b: dict, wheel_speed: float, scale: float, sign: float = 1.0) -> dict:
    """BeamNG body readings -> the policy's readings (trained-car units). ``sign`` = -1 mirrors the
    lateral quantities (used when BeamNG's frame turns out mirrored)."""
    fr = froude(scale)
    return dict(gyro_z=sign * b["r"] * fr["rate"], accel_x=b["ax"], accel_y=sign * b["ay"],
                vel_x=b["vx"] * fr["speed"], vel_y=sign * b["vy"] * fr["speed"],
                wheel_rear=wheel_speed * fr["speed"], wheel_front=b["vx"] * fr["speed"])


# ----------------------------------------------------------------------------- the driver
class BeamNGCar(CarInterface):
    realtime = False

    def __init__(self, meta: dict, home: str | None = None, host: str = "localhost", port: int = 25252,
                 user: str | None = None, level: str = "smallgrid", model: str = "etk800", part_config: str | None = None,
                 scale: Any = "auto", steer_lock_deg: float | None = None, throttle_gain: float = 1.0,
                 launch: bool = True, wheel_key: str = "wheelspeed", gear: int | None = None,
                 steps_per_second: int = 200, spawn: tuple = (0.0, 0.0, 0.3)):
        try:
            import beamngpy
            del beamngpy
        except ImportError as exc:
            raise RuntimeError("BeamNG needs the beamngpy package: pip install beamngpy (matching your BeamNG version)") from exc
        self.meta = meta
        if launch and not home:
            home = find_beamng_home()
            if home is None:
                raise RuntimeError("BeamNG was not found: pass its install folder (--beamng-home, e.g. "
                                   "C:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive) or set BNG_HOME")
        self.home, self.host, self.port, self.user, self.launch = home, host, int(port), user, bool(launch)
        self.level, self.model, self.part_config = level, model, part_config
        self.scale_opt = scale
        self.car_lock = math.radians(steer_lock_deg) if steer_lock_deg else None
        self.rc_lock = math.radians(meta["car"]["steer_max_deg"])
        self.throttle_gain, self.wheel_key, self.gear = float(throttle_gain), wheel_key, gear
        self.sps, self.spawn = int(steps_per_second), spawn
        self.sign = 1.0
        self.log = print                              # the GUI replaces this to show the messages
        self.prev: dict | None = None
        self.cmd = (0.0, 0.0)
        self.t = 0.0

    # ---------------------------------------------------------------- BeamNGpy (version-tolerant calls)
    @staticmethod
    def _call(obj: Any, names: list[str], *args: Any, **kw: Any) -> Any:
        for name in names:
            target = obj
            for part in name.split("."):
                target = getattr(target, part, None)
                if target is None:
                    break
            if target is not None:
                return target(*args, **kw)
        raise AttributeError(f"beamngpy has none of {names}")

    def _poll(self) -> tuple[dict, dict]:
        v = self.vehicle
        self._call(v, ["sensors.poll", "poll_sensors"])
        el = v.sensors["electrics"] if hasattr(v, "sensors") else v.sensor_cache["electrics"]
        el = getattr(el, "data", el)                  # sensor object (newer beamngpy) or plain dict
        el = dict(el.get("values", el)) if hasattr(el, "get") else dict(el)
        return v.state, el

    def _step(self, n: int) -> None:
        self._call(self.bng, ["control.step", "step"], n)

    def _control(self, steering: float, throttle: float, brake: float) -> None:
        kw = dict(steering=float(steering), throttle=float(throttle), brake=float(brake), parkingbrake=0.0)
        if self.gear is not None:
            kw["gear"] = int(self.gear)
        self.vehicle.control(**kw)

    def start(self, runtime: Any) -> None:
        from beamngpy import BeamNGpy, Scenario, Vehicle
        from beamngpy.sensors import Electrics
        self.log(f"BeamNG: {'starting ' + str(self.home) if self.launch else 'connecting to a running BeamNG'} "
              f"(port {self.port}); the first start takes a minute")
        self.bng = BeamNGpy(self.host, self.port, home=self.home, user=self.user)
        self.bng.open(launch=self.launch)
        scenario = Scenario(self.level, "driftsim_test")
        kw = dict(part_config=self.part_config) if self.part_config else {}
        self.vehicle = Vehicle("driftsim_ego", model=self.model, **kw)
        scenario.add_vehicle(self.vehicle, pos=self.spawn, rot_quat=(0, 0, 0, 1))
        scenario.make(self.bng)
        try:
            self.bng.settings.set_deterministic(self.sps)
        except (AttributeError, TypeError):
            self._call(self.bng, ["set_deterministic"])
            self._call(self.bng, ["set_steps_per_second"], self.sps)
        self._call(self.bng, ["scenario.load", "load_scenario"], scenario)
        self._call(self.bng, ["scenario.start", "start_scenario"])
        self._call(self.bng, ["control.pause", "pause"])
        try:
            self.vehicle.sensors.attach("electrics", Electrics())
        except AttributeError:
            self.vehicle.attach_sensor("electrics", Electrics())
        self.scale = self._scale()
        fr = froude(self.scale)
        self.dt_car = runtime.control_dt * fr["time"]
        self.n_steps = max(1, int(round(self.dt_car * self.sps)))
        self.dt_car = self.n_steps / self.sps
        if self.car_lock is None:
            self.car_lock = self.rc_lock
        _, el = self._poll()
        wheel_keys = sorted(k for k in el if "wheel" in k.lower() or "speed" in k.lower())
        self.log(f"BeamNG: {self.model} on {self.level}, scale {self.scale:.2f} (speeds / {math.sqrt(self.scale):.2f}), "
              f"control every {self.dt_car * 1000:.0f} ms ({self.n_steps} steps at {self.sps}/s); electrics with "
              f"wheel/speed values: {wheel_keys}")
        if self.wheel_key not in el:
            raise KeyError(f"electrics has no {self.wheel_key!r}; pick one of {wheel_keys} with --wheel-key")
        self._calibrate_steering()
        self.prev = None
        self.read()                                   # first state for the finite differences

    def _scale(self) -> float:
        if self.scale_opt not in (None, "auto"):
            return float(self.scale_opt)
        try:
            bbox = self._call(self.vehicle, ["get_bbox"])
            pts = np.array([list(p) for p in bbox.values()], dtype=float)
            ext = np.sort(pts.max(axis=0) - pts.min(axis=0))
            length = float(ext[-1])
        except Exception:                             # older / newer API: assume a typical car
            length = 4.5
            self.log("BeamNG: could not read the bounding box; assuming a 4.5 m car (set --scale)")
        return max(length / float(self.meta["car"]["body_length"]), 1e-3)

    def _calibrate_steering(self) -> None:
        """Drive off slowly, steer with a positive BeamNG input and see which way the car turns."""
        state0 = None
        for k in range(int(3.0 * self.sps / self.n_steps)):
            st, _ = self._poll()
            b = body_readings(st["dir"], st["up"], st["vel"], state0, self.dt_car)
            state0 = b
            steer = 0.3 if b["vx"] > 1.0 else 0.0
            self._control(steer, 0.25 * self.throttle_gain, 0.0)
            self._step(self.n_steps)
            if steer and k > 5 and abs(b["r"]) > 0.05:
                self.sign = 1.0 if b["r"] > 0 else -1.0
                break
        for _ in range(int(3.0 * self.sps / self.n_steps)):     # stop again
            self._control(0.0, 0.0, 1.0)
            self._step(self.n_steps)
        self.log(f"BeamNG: steering sign {self.sign:+.0f} (a positive policy steer turns the car left)")

    # ---------------------------------------------------------------- CarInterface
    def read(self) -> dict:
        st, el = self._poll()
        b = body_readings(st["dir"], st["up"], st["vel"], self.prev, self.dt_car)
        self.prev = b
        self.last = (st, b)
        return to_policy_readings(b, float(el[self.wheel_key]), self.scale)

    def write(self, steer: float, throttle: float) -> None:
        self.cmd = (steer, throttle)
        angle = steer * self.rc_lock
        s_in = float(np.clip(self.sign * angle / self.car_lock, -1.0, 1.0))
        thr = max(throttle, 0.0) * self.throttle_gain
        brake = max(-throttle, 0.0)
        self._control(s_in, min(thr, 1.0), min(brake, 1.0))

    def advance(self) -> None:
        self._step(self.n_steps)
        self.t += self.dt_car

    def truth(self) -> dict:
        st, b = self.last
        v = math.hypot(b["vx"], b["vy"])
        fr = froude(self.scale)
        return dict(speed=v * fr["speed"], beta_deg=math.degrees(math.atan2(b["vy"], b["vx"])) if v > 0.2 else 0.0,
                    speed_beamng=v, x=float(st["pos"][0]), y=float(st["pos"][1]), t_beamng=round(self.t, 3))

    def stop(self) -> None:
        try:
            self._control(0.0, 0.0, 1.0)
            self._step(self.n_steps)
        except Exception:
            pass
        if self.launch:
            try:
                self.bng.close()
            except Exception:
                pass
        time.sleep(0.1)


__all__ = ["beamngpy_installed", "BeamNGCar", "body_frame", "body_readings", "froude", "to_policy_readings", "find_beamng_home",
           "is_beamng_home"]
