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
and the rear (driven) wheel speed, read per wheel in the vehicle's Lua (``Electrics`` "wheelspeed"
as a fallback; ``--wheel-key`` picks another).

What makes a full-size BeamNG car drift unlike the RC car, and what the driver does about it (each
can be switched back; see ``BeamNGCar``): the engine is a torque source while the RC motor is a stiff
rear-wheel speed command (``MotorEmulator`` turns the engine into one), the automatic gearbox cuts
the drive at every shift (a fixed gear), ESC / traction control brake and cut the throttle in a slide
(switched off), and the steering ratio differs (the gain is measured in a slow turn against DriftSim's
trained car). ``deploy.beamng_standin`` reproduces all of these without BeamNG.

Scale: BeamNG has no 1/10 RC car, so a full-size car is driven at the same Froude number. With
``k`` = BeamNG car length / trained car length, speeds are divided by sqrt(k), yaw rates multiplied by
sqrt(k), accelerations and angles are unchanged and the control period is 20 ms x sqrt(k). Choose a
rear-drive car on a low-grip surface (or slippery tires): a plastic-tire RC drift car behaves like a
full-size car at about mu 0.3. ``scale=1`` drives an RC-size mod as is. The steering sign and gain
are calibrated at the start, so they do not depend on BeamNG's conventions or steering ratio.
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


# ----------------------------------------------------------------------------- RC motor emulation, wheels, steering
# Per-wheel angular speed (rad/s), radius (m) and broken flag, read in the vehicle's Lua VM. The name
# "driftsim_wheels" marks the query (the stand-in answers it). Unknown fields fall back to nil -> 0.
WHEEL_LUA = """-- driftsim_wheels
local out = {}
local rot = wheels and (wheels.wheelRotators or wheels.wheels)
if rot then
  for _, wd in pairs(rot) do
    if type(wd) == 'table' and wd.name then
      local av = wd.angularVelocity
      if av == nil and obj.getWheelAngularVelocity and wd.wheelID then av = obj:getWheelAngularVelocity(wd.wheelID) end
      out[wd.name] = {av or 0, wd.radius or 0, wd.isBroken and 1 or 0}
    end
  end
end
return out"""


def rc_motor(meta: dict) -> dict:
    """The trained car's motor line (``motor_noload_speed`` m/s, ``throttle_min``) from policy.json;
    exports made before it was recorded get it from the training config (or the default car)."""
    car = meta.get("car", {})
    if "motor_noload_speed" in car:
        return dict(v0=float(car["motor_noload_speed"]), thr_min=float(car.get("throttle_min", 0.02)))
    from ..rl.export import motor_line
    try:
        from ..rl.config import EnvConfig
        from ..rl.env import nominal_params
        line = motor_line(nominal_params(EnvConfig(**meta["training"]["env"])))
    except Exception:
        from ..sim.params import default_params
        line = motor_line(default_params())
    return dict(v0=line["motor_noload_speed"], thr_min=line["throttle_min"])


class MotorEmulator:
    """The RC car's motor + ESC on a BeamNG engine.

    The RC motor is stiff (about 1.3 g of drive force per m/s of wheel-speed error, up to 2 g), so its
    throttle is in effect a rear-wheel speed command: the rear wheels run at about throttle x no-load
    speed whatever the grip. An engine with a gearbox is a torque source instead, so raw throttle
    gives the policy a plant it never saw. Here a PI loop on the BeamNG throttle drives the rear wheels
    to ``throttle x v0`` and the parking brake (rear wheels only) brakes them when they run faster, as
    the motor's braking current does; negative throttle (the ESC's brake) brakes the rear wheels too.
    All speeds in trained-car units (m/s), ``dt`` in trained-car seconds."""

    def __init__(self, v0: float, thr_min: float = 0.02, kp: float = 1.5, ki: float = 6.0, kb: float = 0.5,
                 brake_max: float = 0.35):
        self.v0, self.thr_min = float(v0), float(thr_min)
        self.kp, self.ki, self.kb, self.brake_max = float(kp), float(ki), float(kb), float(brake_max)
        self.reset()

    def reset(self) -> None:
        self.i = 0.0

    def __call__(self, u: float, v_w: float, dt: float) -> tuple[float, float, float]:
        """(throttle, brake, parking brake) for BeamNG from the policy throttle ``u`` and the rear-wheel speed."""
        if abs(u) < self.thr_min:                      # ESC deadband: the motor coasts
            self.i *= 0.5
            return 0.0, 0.0, 0.0
        if u < 0.0:                                    # ESC brake (no reverse: this is a drift test)
            self.i = 0.0
            return 0.0, 0.0, float(min(self.brake_max, -u * 2.0)) if v_w > 0.1 else 0.0
        e = u * self.v0 - v_w
        raw = self.kp * e + self.i
        if (0.0 < raw < 1.0) or (raw >= 1.0 and e < 0.0) or (raw <= 0.0 and e > 0.0):   # anti-windup
            self.i = min(max(self.i + self.ki * e * dt, 0.0), 1.0)
        thr = min(max(self.kp * e + self.i, 0.0), 1.0)
        pb = min(max(-self.kb * (e + 0.3), 0.0), self.brake_max) if e < -0.3 else 0.0
        return thr, 0.0, pb


def parse_wheels(resp: Any) -> dict[str, float] | None:
    """Lua answer -> {wheel name: surface speed (m/s)}; None if it is not a usable answer."""
    if not isinstance(resp, dict) or len(resp) < 2:
        return None
    out = {}
    for name, val in resp.items():
        try:
            av, rad = float(val[0]), float(val[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        out[str(name)] = av * rad
    return out if any(abs(r) > 0 for r in out.values()) or len(out) >= 4 else None


def axle_names(names: list[str]) -> tuple[list[str], list[str]]:
    """(front, rear) wheel names from BeamNG's wheel names (FL, FR, RL, RR, RL1, ...)."""
    front = [n for n in names if n[:1].upper() == "F"]
    rear = [n for n in names if n[:1].upper() == "R"]
    return front, rear


def lock_from_curvature(yaw_rate: float, speed: float, scale: float, wheelbase: float, steer_input: float) -> float:
    """RC-equivalent road-wheel angle (rad) of full BeamNG steering, from a slow steady turn: a curvature
    ``yaw_rate / speed`` of the BeamNG car is ``scale`` times smaller than the trained car's at the same
    input, and a slow turn has ``tan(angle) = curvature x wheelbase``."""
    kappa = abs(yaw_rate) / max(abs(speed), 1e-6) * scale
    return math.atan(kappa * wheelbase) / max(abs(steer_input), 1e-6)


_REF_CACHE: dict[str, float | None] = {}


def reference_turn(meta: dict, throttle: float, steer: float, seconds: float = 3.0, settle: float = 1.2) -> float | None:
    """Curvature (1/m, yaw rate / forward speed) of the trained car in the calibration turn: DriftSim's
    nominal training car at ``throttle``, straight for ``seconds`` and then ``steer`` for ``seconds``,
    median over the turn after ``settle`` s. None if DriftSim cannot run the car."""
    import json
    key = json.dumps([meta.get("training", {}).get("env"), round(throttle, 4), steer, seconds, settle], sort_keys=True, default=str)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = _reference_turn(meta, throttle, steer, seconds, settle)
    return _REF_CACHE[key]


def _reference_turn(meta: dict, throttle: float, steer: float, seconds: float, settle: float) -> float | None:
    try:
        from ..rl.config import EnvConfig
        from ..rl.env import nominal_params
        from ..sim import state as S
        from ..sim.vehicle import Vehicle
        car = Vehicle(nominal_params(EnvConfig(**meta["training"]["env"])))
    except Exception:
        return None
    s = car.initial_state()
    n = int(round(seconds / car.control_dt))
    k_vals = []
    for k in range(2 * n):
        s, _ = car.step(s, np.array([0.0 if k < n else steer, throttle]), want_info=False)
        if k >= n + int(settle / car.control_dt) and s[S.VX] > 0.05:
            k_vals.append(s[S.R] / s[S.VX])
    return abs(float(np.median(k_vals))) if k_vals else None


def _esc_off(mode: Any) -> bool:
    m = str(mode).lower()
    return "off" in m or m in ("disabled", "0", "false")


# ----------------------------------------------------------------------------- the driver
class BeamNGCar(CarInterface):
    """BeamNG as the car. ``throttle_mode`` "motor" (default) emulates the RC motor on BeamNG's engine
    (``MotorEmulator``) in a fixed gear (``gear``, default 2) with BeamNG's driving assists switched off
    (``assists="off"``: ESC / traction control fight a drift); "direct" sends the throttle as is (x
    ``throttle_gain``) with BeamNG's own gearbox. ``wheel_key`` "auto" reads the rear wheels' speeds in
    the vehicle's Lua VM (the Electrics ``wheelspeed`` is a speedometer value, not the driven wheels),
    falling back to ``wheelspeed``. ``calibrate`` measures the steering sign, the steering gain (the
    RC-equivalent lock, unless ``steer_lock_deg`` is given) and the wheel-speed factor at the start."""

    realtime = False

    def __init__(self, meta: dict, home: str | None = None, host: str = "localhost", port: int = 25252,
                 user: str | None = None, level: str = "smallgrid", model: str = "etk800", part_config: str | None = None,
                 scale: Any = "auto", steer_lock_deg: float | None = None, throttle_gain: float = 1.0,
                 launch: bool = True, wheel_key: str = "auto", gear: Any = None, steps_per_second: int = 200,
                 spawn: tuple = (0.0, 0.0, 0.3), throttle_mode: str = "motor", assists: str = "off",
                 calibrate: bool = True, motor_gains: tuple | None = None, substeps: int = 4):
        try:
            import beamngpy
            del beamngpy
        except ImportError as exc:
            raise RuntimeError("BeamNG needs the beamngpy package: pip install beamngpy (matching your BeamNG)") from exc
        if throttle_mode not in ("motor", "direct"):
            raise ValueError("throttle_mode must be motor or direct")
        if assists not in ("off", "keep"):
            raise ValueError("assists must be off or keep")
        self.meta = meta
        if launch and not home:
            home = find_beamng_home()
            if home is None:
                raise RuntimeError("BeamNG was not found: pass its install folder (--beamng-home, e.g. "
                                   "C:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive) or set BNG_HOME")
        self.home, self.host, self.port, self.user, self.launch = home, host, int(port), user, bool(launch)
        self.level, self.model, self.part_config = level, model, part_config
        self.scale_opt = scale
        self.lock_given = math.radians(float(steer_lock_deg)) if steer_lock_deg else None
        self.car_lock = self.lock_given
        self.rc_lock = math.radians(meta["car"]["steer_max_deg"])
        self.wheelbase = float(meta["car"].get("wheelbase", 0.257))
        self.throttle_gain, self.wheel_key = float(throttle_gain), wheel_key
        self.motor = throttle_mode == "motor"
        self.gear_opt = gear if gear not in ("", None) else (2 if self.motor else "auto")
        self.assists, self.calibrate = assists, bool(calibrate)
        m = rc_motor(meta)
        kw = dict(zip(("kp", "ki", "kb"), motor_gains)) if motor_gains else {}
        self.emu = MotorEmulator(m["v0"], m["thr_min"], **kw)
        self.sps, self.spawn, self.substeps = int(steps_per_second), spawn, max(1, int(substeps))
        self.sign = 1.0
        self.log = print                              # the GUI replaces this to show the messages
        self.prev: dict | None = None
        self.cmd = (0.0, 0.0)
        self.steer_in = 0.0
        self.t = 0.0
        self.lua = False
        self.wheel_factor = 1.0
        self.fixed_gear: int | None = None
        self.diag: dict = dict(throttle_mode=throttle_mode)
        self.stats = dict(steps=0, esc=0, tcs=0, steer_sat=0, gear_changes=0, pbrake=0.0)
        self._gear_seen: Any = None
        self.out = (0.0, 0.0, 0.0)
        self.v_rear = 0.0

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

    def _control(self, steering: float, throttle: float, brake: float, parkingbrake: float = 0.0) -> None:
        kw = dict(steering=float(steering), throttle=float(throttle), brake=float(brake), parkingbrake=float(parkingbrake))
        if self.fixed_gear is not None:
            kw["gear"] = int(self.fixed_gear)
        self.vehicle.control(**kw)

    def _wheels(self) -> dict[str, float] | None:
        try:
            return parse_wheels(self.vehicle.queue_lua_command(WHEEL_LUA, response=True))
        except Exception:
            return None

    def _rear_speed(self, el: dict | None = None) -> float:
        """Driven (rear) wheel surface speed in BeamNG m/s (calibrated)."""
        if self.lua:
            w = self._wheels()
            if w is not None:
                return self.wheel_factor * float(np.mean([w[n] for n in self.rear if n in w]))
        if el is None:
            _, el = self._poll()
        return self.wheel_factor * float(el.get(self.wheel_key_used, 0.0))

    # ---------------------------------------------------------------- start-up
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
        self.rc_dt = runtime.control_dt
        self.dt_car = runtime.control_dt * fr["time"]
        self.n_steps = max(1, int(round(self.dt_car * self.sps)))
        self.dt_car = self.n_steps / self.sps
        self.chunks = self._chunks(self.n_steps, self.substeps if self.motor else 1)
        self._setup_drivetrain()
        _, el = self._poll()
        self._setup_wheels(el)
        self.log(f"BeamNG: {self.model} on {self.level}, scale {self.scale:.2f} (speeds / {math.sqrt(self.scale):.2f}), "
                 f"control every {self.dt_car * 1000:.0f} ms ({self.n_steps} steps at {self.sps}/s), throttle "
                 f"{'emulating the RC motor (rear-wheel speed servo)' if self.motor else 'direct'}")
        if self.calibrate:
            self._calibrate()
        else:
            self.car_lock = self.car_lock or self.rc_lock
        self.diag.update(scale=round(self.scale, 3), control_ms=round(self.dt_car * 1000, 1), steer_sign=self.sign,
                         lock_rc_equiv_deg=round(math.degrees(self.car_lock), 1),
                         lock_trained_deg=round(math.degrees(self.rc_lock), 1), wheel_factor=round(self.wheel_factor, 3))
        if self.car_lock < 0.95 * self.rc_lock:
            self.log(f"BeamNG: WARNING full BeamNG steering is only {math.degrees(self.car_lock):.0f} deg RC-equivalent "
                     f"(trained {math.degrees(self.rc_lock):.0f} deg): the policy cannot use its full counter-steer; "
                     "pick a car / steering rack with more lock (drift parts)")
        self.prev = None
        self.emu.reset()
        self.read()                                   # first state for the finite differences
        self.stats = dict(steps=0, esc=0, tcs=0, steer_sat=0, gear_changes=0, pbrake=0.0)   # count the run only

    @staticmethod
    def _chunks(n: int, k: int) -> list[int]:
        k = max(1, min(int(k), n))
        base, extra = divmod(n, k)
        return [base + (1 if i < extra else 0) for i in range(k)]

    def _setup_drivetrain(self) -> None:
        if self.gear_opt != "auto":
            try:
                self.vehicle.set_shift_mode("realistic_manual_auto_clutch")
                self.fixed_gear = int(self.gear_opt)
                self.diag["gear"] = self.fixed_gear
            except Exception as exc:                  # older beamngpy / no API: keep BeamNG's shifting
                self.log(f"BeamNG: could not select manual gear {self.gear_opt} ({exc}); BeamNG shifts by itself")
                self.diag["gear"] = "auto (fixed gear failed)"
        else:
            self.diag["gear"] = "auto"
        self.diag["assists"] = self._assists_off() if self.assists == "off" else "kept"

    def _assists_off(self) -> str:
        """Switch ESC / traction control off: try the 'off' keys, then cycle through the car's modes."""
        v = self.vehicle
        try:
            mode0 = v.get_esc_mode()
        except Exception:
            self.log("BeamNG: this beamngpy cannot read the ESC mode; if the car has ESC/TCS, switch it off in the game")
            return "unknown"
        if str(mode0).lower() == "none":
            self.log("BeamNG: the car has no switchable ESC mode (or a fixed 'regular' ESC); if its ESC/TCS acts "
                     "(see esc/tcs in the summary), pick a configuration without ESC")
            return "none"
        if _esc_off(mode0):
            return str(mode0)
        for key in ("off", "Off", "OFF", "disabled"):
            try:
                v.set_esc_mode(key)
                if _esc_off(v.get_esc_mode()):
                    self.log(f"BeamNG: ESC mode {mode0} -> {key}")
                    return key
            except Exception:
                pass
        seen = [str(mode0)]
        for _ in range(8):
            try:
                v.cycle_esc_mode()
                m = str(v.get_esc_mode())
            except Exception:
                break
            if _esc_off(m):
                self.log(f"BeamNG: ESC mode {mode0} -> {m} (modes {seen + [m]})")
                return m
            if m in seen:
                break
            seen.append(m)
        self.log(f"BeamNG: WARNING could not switch the ESC off (modes {seen}); traction / stability control may "
                 "fight the slide")
        return f"on ({seen[-1]})"

    def _setup_wheels(self, el: dict) -> None:
        self.wheel_key_used = "wheelspeed" if self.wheel_key in ("auto", "lua") else self.wheel_key
        w = self._wheels() if self.wheel_key in ("auto", "lua") else None
        if w is not None:
            self.front, self.rear = axle_names(sorted(w))
            if self.rear:
                self.lua = True
                self.diag["wheels"] = f"Lua per wheel, rear = {self.rear}"
                return
        if self.wheel_key == "lua":
            raise RuntimeError("the per-wheel Lua query gave no rear wheels; use --wheel-key wheelspeed")
        if self.wheel_key_used not in el:
            keys = sorted(k for k in el if "wheel" in k.lower() or "speed" in k.lower())
            raise KeyError(f"electrics has no {self.wheel_key_used!r}; pick one of {keys} with --wheel-key")
        self.diag["wheels"] = f"Electrics {self.wheel_key_used!r} (not per wheel: may average all four wheels)"
        self.log("BeamNG: per-wheel speeds unavailable, using Electrics " + repr(self.wheel_key_used))

    def _calibrate(self) -> None:
        """Roll straight slowly (wheel-speed factor), then a slow steady turn at a fixed input (steering sign and
        gain), then stop. Speeds of about 0.8 m/s RC-equivalent keep the tires in their linear range."""
        sk = math.sqrt(self.scale)
        per_s = 1.0 / self.rc_dt
        v_cal, s_cal = 0.8, 0.4
        ratios, turn = [], []

        def drive_step(steer_in: float, u: float) -> dict:
            self.steer_in, self.cmd = steer_in, (0.0, u)
            self.advance(clock=False)
            st, el = self._poll()
            self.v_rear = self._rear_speed(el)
            b = body_readings(st["dir"], st["up"], st["vel"], self._cal_prev, self.dt_car)
            self._cal_prev = b
            return b

        self._cal_prev = None
        u_roll = v_cal / self.emu.v0 if self.motor else 0.25          # direct: x throttle_gain in advance()
        for k in range(int(3.0 * per_s)):                               # straight, slowly
            b = drive_step(0.0, u_roll if (self.motor or b_vx(self._cal_prev) < v_cal * sk) else 0.0)
            if k > int(1.5 * per_s) and b["vx"] > 0.5 * v_cal * sk:
                w = self._wheels() if self.lua else None
                ref = float(np.mean([w[n] for n in self.front])) if (w and self.front) else None
                if ref is None and not self.lua:                       # electrics: rear ~ ground speed when rolling slowly
                    _, el = self._poll()
                    ref = float(el.get(self.wheel_key_used, 0.0))
                if ref is not None and abs(ref) > 0.2 * v_cal * sk:
                    ratios.append(b["vx"] / ref)
        if ratios:
            f = float(np.median(ratios))
            if 0.5 < abs(f) < 2.0:
                self.wheel_factor = f
            else:
                self.log(f"BeamNG: wheel-speed factor {f:.2f} looks wrong; keeping 1")
        for k in range(int(3.0 * per_s)):                               # slow steady turn at +s_cal input
            b = drive_step(s_cal, u_roll if (self.motor or b["vx"] < v_cal * sk) else 0.0)
            if k > int(1.2 * per_s) and b["vx"] > 0.3 * v_cal * sk:
                turn.append((b["r"], b["vx"]))
        if turn:
            r = float(np.median([t[0] for t in turn]))
            vx = float(np.median([t[1] for t in turn]))
            if abs(r) > 0.01:
                self.sign = 1.0 if r > 0 else -1.0
                if self.lock_given is None:
                    # the BeamNG car must turn as much per input as the trained car does in the same slow turn
                    # (its spool makes it understeer, so plain kinematics would misjudge the gain)
                    u_ref = u_roll if self.motor else min(max(vx / sk / self.emu.v0, 0.03), 1.0)
                    k_ref = reference_turn(self.meta, u_ref, s_cal)
                    k_bng = abs(r) / max(vx, 1e-6) * self.scale
                    if k_ref:
                        self.car_lock = self.rc_lock * k_bng / k_ref
                        self.diag["calibration"] = "slow-turn curvature vs DriftSim's trained car"
                    else:
                        self.car_lock = lock_from_curvature(r, vx, self.scale, self.wheelbase, s_cal)
                        self.diag["calibration"] = "slow-turn curvature, kinematic"
        if self.car_lock is None:
            self.log("BeamNG: the steering test gave no turn; assuming the trained steering lock")
            self.car_lock = self.rc_lock
        for _ in range(int(2.5 * per_s)):                               # stop again
            self.steer_in = 0.0
            self.cmd = (0.0, 0.0)
            self._control(0.0, 0.0, 1.0)
            self._step(self.n_steps)
        self.log(f"BeamNG: steering sign {self.sign:+.0f} (a positive policy steer turns the car left), full BeamNG "
                 f"steering = {math.degrees(self.car_lock):.1f} deg RC-equivalent "
                 f"({'given' if self.lock_given else 'measured'}; trained {math.degrees(self.rc_lock):.0f}), "
                 f"wheel-speed factor {self.wheel_factor:.3f}")

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

    # ---------------------------------------------------------------- CarInterface
    def read(self) -> dict:
        st, el = self._poll()
        b = body_readings(st["dir"], st["up"], st["vel"], self.prev, self.dt_car)
        self.prev = b
        self.v_rear = self._rear_speed(el)
        self.last = (st, b, el)
        s = self.stats
        s["steps"] += 1
        s["esc"] += int(bool(el.get("esc_active", el.get("escActive", False))))
        s["tcs"] += int(bool(el.get("tcs_active", el.get("tcsActive", False))))
        g = el.get("gear")
        if self._gear_seen is not None and g is not None and g != self._gear_seen:
            s["gear_changes"] += 1
        self._gear_seen = g if g is not None else self._gear_seen
        return to_policy_readings(b, self.v_rear, self.scale)

    def write(self, steer: float, throttle: float) -> None:
        self.cmd = (steer, throttle)
        s_in = self.sign * steer * self.rc_lock / self.car_lock
        if abs(s_in) > 1.0:
            self.stats["steer_sat"] += 1
        self.steer_in = float(np.clip(s_in, -1.0, 1.0))

    def advance(self, clock: bool = True) -> None:
        u = self.cmd[1]
        if not self.motor:
            thr = max(u, 0.0) * self.throttle_gain
            self.out = (min(thr, 1.0), min(max(-u, 0.0), 1.0), 0.0)
            self._control(self.steer_in, *self.out)
            self._step(self.n_steps)
        else:
            sk = math.sqrt(self.scale)
            for j, n in enumerate(self.chunks):
                v_w = (self.v_rear if j == 0 else self._rear_speed()) / sk
                self.out = self.emu(u, v_w, n / self.sps / sk)
                self._control(self.steer_in, *self.out)
                self._step(n)
                self.stats["pbrake"] += self.out[2] * n / self.n_steps
        if clock:
            self.t += self.dt_car

    def truth(self) -> dict:
        st, b, _ = self.last
        v = math.hypot(b["vx"], b["vy"])
        fr = froude(self.scale)
        return dict(speed=v * fr["speed"], beta_deg=math.degrees(math.atan2(b["vy"], b["vx"])) if v > 0.2 else 0.0,
                    speed_beamng=v, x=float(st["pos"][0]), y=float(st["pos"][1]), t_beamng=round(self.t, 3),
                    bng_steer=round(self.steer_in, 4), bng_throttle=round(self.out[0], 4),
                    bng_brake=round(self.out[1], 4), bng_pbrake=round(self.out[2], 4))

    def diagnostics(self) -> dict:
        s = self.stats
        n = max(s["steps"], 1)
        return dict(self.diag, esc_active_share=round(s["esc"] / n, 3), tcs_active_share=round(s["tcs"] / n, 3),
                    steer_saturated_share=round(s["steer_sat"] / n, 3), gear_changes=s["gear_changes"],
                    rear_pbrake_mean=round(s["pbrake"] / n, 3))

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


def b_vx(b: dict | None) -> float:
    return 0.0 if b is None else float(b["vx"])


__all__ = ["beamngpy_installed", "BeamNGCar", "MotorEmulator", "body_frame", "body_readings", "froude",
           "to_policy_readings", "find_beamng_home", "is_beamng_home", "rc_motor", "parse_wheels", "axle_names",
           "lock_from_curvature", "reference_turn", "WHEEL_LUA"]
