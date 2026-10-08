"""The control loop that drives a car with an exported policy: the same loop for the real car (Jetson),
a software-in-the-loop check in DriftSim and a test in BeamNG.

    driftsim-drive --model rl_runs/<run>/final --car sim --grip 0.4 --seconds 10
    driftsim-drive --model final --car beamng --beamng-home "C:/BeamNG.tech.v0.39" --vehicle etk800
    driftsim-drive --model final --car mypackage.mycar:MyCar           # your hardware driver

A car driver implements ``CarInterface``: ``read()`` returns the readings in SI units (see
``deploy.runtime``), ``write(steer, throttle)`` sends the filtered commands, ``advance()`` moves a
simulator one control period (hardware: nothing, the loop keeps real time), ``armed()`` is the
driver's dead-man switch (False: neutral commands and the policy restarts when re-armed) and
``truth()`` optionally gives ground truth for the log. Every step is written to a CSV log.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .runtime import Command, PolicyRuntime


class CarInterface:
    """Base class of car drivers (simulated or real)."""

    realtime = True                                   # False: a simulator stepped by advance()

    def start(self, runtime: PolicyRuntime) -> None:
        pass

    def read(self) -> dict:
        raise NotImplementedError

    def write(self, steer: float, throttle: float) -> None:
        raise NotImplementedError

    def advance(self) -> None:
        pass

    def armed(self) -> bool:
        return True

    def done(self) -> bool:
        return False

    def truth(self) -> dict:
        return {}

    def stop(self) -> None:
        pass


class HardwareCar(CarInterface):
    """Template for the real car on the Jetson. Fill in your sensor and actuator code, e.g.:

    * readings: SparkFun OTOS (pip install sparkfun-qwiic-otos) for vel_x / vel_y in the body frame
      (rotate its field-frame velocity by its heading) and gyro_z; an IMU or the OTOS IMU for accel_x /
      accel_y; a hall sensor on the spur gear or rear axle for wheel_rear (m/s at the tire surface)
    * commands: a PCA9685 PWM board (pip install adafruit-circuitpython-servokit) or the Jetson's
      hardware PWM: steer -> servo pulse (1500 us +- your end points; check +1 = left), throttle -> ESC
    * armed(): a transmitter switch read through the receiver, or a heartbeat from your laptop
    The policy expects the safety filter's car geometry in policy.json (``car``); pass the real
    car's ``steer_max_deg`` / ``cg_to_front`` with --car-config if they differ."""

    def read(self) -> dict:
        raise NotImplementedError("HardwareCar.read: return dict(gyro_z, accel_x, accel_y, wheel_rear, vel_x, vel_y)")

    def write(self, steer: float, throttle: float) -> None:
        raise NotImplementedError("HardwareCar.write: send the servo and ESC pulses")

    def armed(self) -> bool:
        return False                                  # never drive until you implement a dead-man switch


class SimCar(CarInterface):
    """DriftSim as the car: the training physics (with the training randomization, latency and sensor
    noise) but driven through the runtime, to check the exported model and the loop end to end."""

    realtime = False

    def __init__(self, meta: dict, grip: float | None = None, seed: int = 0, grip_change: tuple | None = None,
                 drift_start: bool = False):
        from ..rl.config import EnvConfig
        from ..rl.env import DriftBatchEnv
        env_cfg = dict(meta["training"]["env"])
        cfg = EnvConfig(**env_cfg)
        rnd = dict(cfg.randomize)
        if grip is not None:
            rnd["surface.mu_scale"] = {"dist": "fixed", "value": float(grip)}
        safety = dict(cfg.safety.__dict__, enabled=False)          # the runtime filters, not the simulator
        cfg = cfg.replace(randomize=rnd, safety=safety, grip_change_prob=0.0, target_beta_jitter_deg=0.0,
                          target_speed_jitter=0.0, init_drift_prob=1.0 if drift_start else 0.0, episode_s=600.0)
        self.env = DriftBatchEnv(1, cfg, autoreset=False)
        self.seed, self.grip_change = int(seed), grip_change
        self.noise = dict(cfg.noise)
        self.rng = np.random.default_rng([self.seed, 0xCA2])
        self.cmd = np.zeros((1, 2))
        self.k = 0
        self.ended = False

    def start(self, runtime: PolicyRuntime) -> None:
        self.env.reset(seed=self.seed)
        self.env.target_beta[:] = runtime.target_beta
        self.env.target_speed[:] = runtime.target_speed
        self.ended = False
        self.k = 0

    def read(self) -> dict:
        from ..sim import state as S
        from ..sim.vehicle import derivatives_model
        e = self.env
        s = e.state[0]
        _, info = derivatives_model(e.state, e._applied(), e.model, want_info=True)
        n = lambda key: self.rng.normal(0.0, float(self.noise.get(key, 0.0)))  # noqa: E731
        Rw = float(np.asarray(e.model.Rw).reshape(-1)[0])
        w = s[S.OMEGA] * Rw
        return dict(gyro_z=s[S.R] + n("gyro"), accel_x=float(info["ax"][0]) + n("accel"),
                    accel_y=float(info["ay"][0]) + n("accel"), wheel_fl=w[0] + n("wheel"), wheel_fr=w[1] + n("wheel"),
                    wheel_rl=w[2] + n("wheel"), wheel_rr=w[3] + n("wheel"), vel_x=s[S.VX] + n("velocity"),
                    vel_y=s[S.VY] + n("velocity"))

    def write(self, steer: float, throttle: float) -> None:
        self.cmd = np.array([[steer, throttle]])

    def advance(self) -> None:
        if self.grip_change is not None and self.k == int(round(self.grip_change[1] / self.env.control_dt)):
            self.env.model.tm.mu_scale[:] = self.env.model.tm.mu_scale * float(self.grip_change[0])
        _, _, term, _, self.info = self.env.step(self.cmd)
        self.ended = bool(term[0])
        self.k += 1

    def done(self) -> bool:
        return self.ended

    def truth(self) -> dict:
        from ..sim import state as S
        s = self.env.state[0]
        v = math.hypot(s[S.VX], s[S.VY])
        return dict(x=s[S.X], y=s[S.Y], speed=v, beta_deg=math.degrees(math.atan2(s[S.VY], s[S.VX])) if v > 0.05 else 0.0,
                    yaw_rate=s[S.R], grip_true=float(np.asarray(self.env.last_grip)[0]),
                    reward=float(self.info["episode_return"][0]) if hasattr(self, "info") else 0.0)


def drive(runtime: PolicyRuntime, car: CarInterface, seconds: float, log_path: str | Path | None = None,
          verbose: bool = True, on_step: Callable[[dict], None] | None = None,
          stop: Callable[[], bool] | None = None) -> dict:
    """Run the loop for ``seconds`` (of car time); returns a summary and writes the CSV log.
    ``on_step(row)`` sees every logged step; ``stop()`` ends the run early (neutral commands)."""
    dt = runtime.control_dt
    n = max(1, int(round(seconds / dt)))
    car.start(runtime)
    runtime.reset()
    rows: list[dict] = []
    overruns, was_armed = 0, False
    t_next = time.perf_counter()
    try:
        for k in range(n):
            sensors = car.read()
            if car.armed():
                if not was_armed:
                    runtime.reset()
                cmd = runtime.step(sensors)
                was_armed = True
            else:
                cmd = Command(0.0, 0.0, 0.0, 0.0, None, 0.0)
                was_armed = False
            car.write(cmd.steer, cmd.throttle)
            car.advance()
            row = dict(t=round(k * dt, 3), **{f"s_{key}": round(float(v), 5) for key, v in sensors.items()},
                       steer=cmd.steer, throttle=cmd.throttle, raw_steer=cmd.raw_steer, raw_throttle=cmd.raw_throttle,
                       grip_est=cmd.grip, override=cmd.override, compute_ms=round(cmd.compute_ms, 3),
                       **{f"true_{key}": v for key, v in car.truth().items()})
            rows.append(row)
            if on_step is not None:
                on_step(row)
            if verbose and k % max(1, int(round(1.0 / dt))) == 0:
                tr = car.truth()
                print(f"t {k * dt:6.2f}s  v {math.hypot(sensors['vel_x'], sensors['vel_y']):5.2f} m/s  "
                      f"beta {math.degrees(math.atan2(sensors['vel_y'], abs(sensors['vel_x']))):6.1f} deg  "
                      f"cmd {cmd.steer:+.2f} {cmd.throttle:+.2f}  grip est {cmd.grip if cmd.grip is None else round(cmd.grip, 3)}"
                      + (f"  (true {tr['grip_true']:.3f})" if "grip_true" in tr else "")
                      + (f"  safety {cmd.override:.2f}" if cmd.override else ""), flush=True)
            if car.done() or (stop is not None and stop()):
                break
            if car.realtime:
                t_next += dt
                pause = t_next - time.perf_counter()
                if pause > 0:
                    time.sleep(pause)
                else:
                    overruns += 1
    finally:
        try:
            car.write(0.0, 0.0)
        finally:
            car.stop()
    if log_path is not None and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(log_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    return summarize(rows, runtime, car, overruns)


def summarize(rows: list[dict], runtime: PolicyRuntime, car: CarInterface, overruns: int = 0) -> dict:
    if not rows:
        return dict(steps=0)
    half = rows[len(rows) // 2:]
    src = "true_" if "true_beta_deg" in rows[0] else None
    if src:
        betas = np.array([abs(r["true_beta_deg"]) for r in half])
        speeds = np.array([r["true_speed"] for r in half])
    else:
        betas = np.array([abs(math.degrees(math.atan2(r["s_vel_y"], abs(r["s_vel_x"])))) for r in half])
        speeds = np.array([math.hypot(r["s_vel_x"], r["s_vel_y"]) for r in half])
    grips = [r["grip_est"] for r in half if r["grip_est"] is not None]
    out = dict(steps=len(rows), seconds=round(len(rows) * runtime.control_dt, 2), ended_early=bool(car.done()),
               target_beta_deg=round(math.degrees(runtime.target_beta), 1), target_speed=round(runtime.target_speed, 2),
               mean_abs_beta_deg_2nd_half=round(float(betas.mean()), 1), mean_speed_2nd_half=round(float(speeds.mean()), 2),
               override_share=round(float(np.mean([r["override"] > 0.01 for r in rows])), 4),
               max_compute_ms=round(max(r["compute_ms"] for r in rows), 3), realtime_overruns=overruns,
               grip_estimate_2nd_half=round(float(np.mean(grips)), 3) if grips else None)
    if "true_grip_true" in rows[0]:
        out["grip_true_2nd_half"] = round(float(np.mean([r["true_grip_true"] for r in half])), 3)
    return out


def load_car(spec: str, args: argparse.Namespace, meta: dict) -> CarInterface:
    if spec == "sim":
        gc = None
        if args.grip_change:
            f, t = args.grip_change.split("@")
            gc = (float(f), float(t))
        return SimCar(meta, grip=args.grip, seed=args.seed, grip_change=gc, drift_start=args.drift_start)
    if spec == "beamng":
        from .beamng import BeamNGCar
        return BeamNGCar(meta, home=args.beamng_home, host=args.beamng_host, port=args.beamng_port, user=args.beamng_user,
                         level=args.level, model=args.vehicle, part_config=args.part_config, scale=args.scale,
                         steer_lock_deg=args.steer_lock_deg, throttle_gain=args.throttle_gain, launch=not args.no_launch,
                         wheel_key=args.wheel_key, gear=args.gear)
    if spec == "hardware":
        return HardwareCar()
    mod, _, cls = spec.partition(":")
    if not cls:
        raise ValueError("--car must be sim, beamng, hardware or module:Class")
    return getattr(importlib.import_module(mod), cls)()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="driftsim-drive", description="Drive a car (simulated or real) with an exported policy.")
    ap.add_argument("--model", required=True, help="exported model: the <run>/final folder or final_model.zip")
    ap.add_argument("--car", default="sim", help="sim | beamng | hardware | module:Class")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=None, help="commanded |sideslip| (deg), hold task")
    ap.add_argument("--speed", type=float, default=None, help="commanded speed (m/s), hold task")
    ap.add_argument("--log", default=None, help="CSV log (default: <model>/drive_<car>.csv)")
    ap.add_argument("--car-config", default=None, help="JSON with the real car's steer_max_deg / cg_to_front")
    ap.add_argument("--safety", default=None, help="JSON overrides of the safety filter, e.g. '{\"v_max\": 2}'")
    g = ap.add_argument_group("sim")
    g.add_argument("--grip", type=float, default=None, help="surface grip multiplier (1 = P-tile; default: random as trained)")
    g.add_argument("--grip-change", default=None, help="FACTOR@SECONDS, e.g. 0.6@5 (a wet patch after 5 s)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--drift-start", action="store_true", help="start in a steady drift instead of parked")
    b = ap.add_argument_group("beamng")
    b.add_argument("--beamng-home", default=None, help="BeamNG.tech (or BeamNG.drive) install folder")
    b.add_argument("--beamng-host", default="localhost")
    b.add_argument("--beamng-port", type=int, default=25252)
    b.add_argument("--beamng-user", default=None, help="BeamNG user folder")
    b.add_argument("--no-launch", action="store_true", help="connect to a BeamNG that is already running")
    b.add_argument("--level", default="smallgrid", help="map (smallgrid: an endless flat grid)")
    b.add_argument("--vehicle", default="etk800", help="BeamNG vehicle model")
    b.add_argument("--part-config", default=None, help="vehicle configuration (.pc path), e.g. a rear-drive drift setup")
    b.add_argument("--scale", default="auto", help="size ratio BeamNG car / trained car (auto: from the bounding box; 1 = RC mod)")
    b.add_argument("--steer-lock-deg", type=float, default=None, help="road-wheel angle of full BeamNG steering (default: as trained)")
    b.add_argument("--throttle-gain", type=float, default=1.0, help="BeamNG throttle per unit policy throttle")
    b.add_argument("--wheel-key", default="wheelspeed", help="electrics value used as the driven-wheel speed")
    b.add_argument("--gear", type=int, default=None, help="force a gear (e.g. 1 or 2) if the car does not move")
    a = ap.parse_args(argv)
    car_cfg = json.loads(Path(a.car_config).read_text()) if a.car_config else None
    rt = PolicyRuntime(a.model, car=car_cfg, safety=json.loads(a.safety) if a.safety else None)
    rt.set_targets(a.beta, a.speed)
    print(rt.describe())
    car = load_car(a.car, a, rt.meta)
    log = a.log or str(rt.folder / f"drive_{a.car.replace(':', '_').replace('.', '_')}.csv")
    res = drive(rt, car, a.seconds, log)
    print(json.dumps(res, indent=1))
    print(f"log: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CarInterface", "HardwareCar", "SimCar", "drive", "summarize", "main"]
