"""Signals that can be exported, and the shard / table / manifest writers.

A batch is written to ``<out_dir>/<timestamp>_<name>/``:

    manifest.json        spec, versions, signal schema (names, units, shapes), shards, timing, status
    episodes.csv         one row per episode: sampled parameters + outcome metrics (drift time, spin, ...)
    shard_0000.npz       arrays episode_id (n,), t (T_rec,), and one array per signal:
                         (n, T_rec) for scalars, (n, T_rec, 4) for per-wheel signals (FL, FR, RL, RR)
    shard_0000.csv       long table: episode_id, t, then one column per scalar / wheel (name_fl ...)
    shard_0000.parquet   same columns as the CSV (only when pyarrow is installed)
    README.txt           how to load the data

Every recorded row is a state at a control-step boundary t_k = k * decimation * control_dt. The
actions on that row are the commands issued at t_k (``*_cmd``) and the ones reaching the car after
the control latency (``*_applied``); info signals (forces, accelerations, ...) are evaluated at the
recorded state with the applied action.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..sim import state as S

WHEELS = ("fl", "fr", "rl", "rr")
FORMATS = ("npz", "csv", "parquet")

# (name, unit, per_wheel, source, source key, description)
_SIG = lambda *a: dict(zip(("name", "unit", "per_wheel", "source", "src", "desc"), a))  # noqa: E731
SIGNAL_GROUPS: list[dict] = [
    dict(key="pose", label="Pose", default=True, signals=[
        _SIG("x", "m", False, "state", S.X, "world x (east)"),
        _SIG("y", "m", False, "state", S.Y, "world y (north)"),
        _SIG("yaw", "rad", False, "state", S.YAW, "heading, CCW from +x")]),
    dict(key="velocity", label="Body velocities", default=True, signals=[
        _SIG("vx", "m/s", False, "state", S.VX, "forward velocity (body frame)"),
        _SIG("vy", "m/s", False, "state", S.VY, "leftward velocity (body frame)"),
        _SIG("yaw_rate", "rad/s", False, "state", S.R, "yaw rate, + = turning left")]),
    dict(key="derived", label="Speed & sideslip", default=True, signals=[
        _SIG("speed", "m/s", False, "derived", "speed", "ground speed |v|"),
        _SIG("beta", "rad", False, "derived", "beta", "sideslip atan2(vy, vx); < 0 in a left-hand drift")]),
    dict(key="wheel_speeds", label="Wheel speeds", default=True, signals=[
        _SIG("omega", "rad/s", True, "state", S.OMEGA, "wheel angular speed")]),
    dict(key="steering", label="Steering angle", default=True, signals=[
        _SIG("delta", "rad", False, "state", S.DELTA, "actual steering angle (after the servo)")]),
    dict(key="actions", label="Actions (commanded & applied)", default=True, signals=[
        _SIG("steer_cmd", "-", False, "action", "cmd0", "steering command issued at t"),
        _SIG("throttle_cmd", "-", False, "action", "cmd1", "throttle command issued at t"),
        _SIG("steer_applied", "-", False, "action", "app0", "steering command reaching the car (after latency)"),
        _SIG("throttle_applied", "-", False, "action", "app1", "throttle command reaching the car (after latency)")]),
    dict(key="accel", label="Accelerations (IMU-like)", default=True, signals=[
        _SIG("ax", "m/s^2", False, "info", "ax", "longitudinal specific force (body frame, noise-free IMU)"),
        _SIG("ay", "m/s^2", False, "info", "ay", "lateral specific force (body frame, noise-free IMU)")]),
    dict(key="motor", label="Motor current", default=False, signals=[
        _SIG("i_motor", "A", False, "state", S.I_MOTOR, "motor current")]),
    dict(key="motor_torque", label="Motor torque & speed", default=False, signals=[
        _SIG("T_motor", "N m", False, "info", "T_motor", "net motor shaft torque"),
        _SIG("omega_m", "rad/s", False, "info", "omega_m", "motor shaft speed")]),
    dict(key="tire_temps", label="Tire temperatures", default=False, signals=[
        _SIG("t_tire", "degC", True, "state", S.T_TIRE, "tire tread temperature")]),
    dict(key="load_transfer", label="Load transfer", default=False, signals=[
        _SIG("dfz_long", "N", False, "state", S.DFZ_LONG, "longitudinal load transfer (+ = rear loaded)"),
        _SIG("dfz_lat", "N", False, "state", S.DFZ_LAT, "lateral load transfer (+ = right loaded)")]),
    dict(key="slip_states", label="Relaxed slips (states)", default=False, signals=[
        _SIG("kappa_lag", "-", True, "state", S.KAPPA_LAG, "relaxed slip ratio fed to the tire model"),
        _SIG("alpha_lag", "rad", True, "state", S.ALPHA_LAG, "relaxed slip angle fed to the tire model")]),
    dict(key="contamination", label="Tire contamination", default=False, signals=[
        _SIG("contam", "-", True, "state", S.CONTAM, "dust on the tread (0..1)")]),
    dict(key="tire_forces", label="Tire forces", default=False, signals=[
        _SIG("Fx", "N", True, "info", "Fx", "longitudinal tire force (wheel frame)"),
        _SIG("Fy", "N", True, "info", "Fy", "lateral tire force (wheel frame)"),
        _SIG("Fz", "N", True, "info", "Fz", "normal load")]),
    dict(key="slips", label="Instantaneous slips", default=False, signals=[
        _SIG("kappa", "-", True, "info", "kappa", "slip ratio"),
        _SIG("alpha", "rad", True, "info", "alpha", "slip angle")]),
    dict(key="grip", label="Effective grip", default=False, signals=[
        _SIG("mu_x", "-", True, "info", "mu_x", "peak longitudinal friction incl. temperature & condition"),
        _SIG("mu_y", "-", True, "info", "mu_y", "peak lateral friction incl. temperature & condition")]),
    dict(key="power", label="Slip power", default=False, signals=[
        _SIG("slip_power", "W", False, "info", "slip_power", "total frictional power of the four tires")]),
]
_BY_KEY = {g["key"]: g for g in SIGNAL_GROUPS}


def available_formats() -> list[str]:
    """NPZ and CSV always; Parquet when pyarrow is installed."""
    import importlib.util
    return ["npz", "csv"] + (["parquet"] if importlib.util.find_spec("pyarrow") else [])


def selected_signals(groups: list[str]) -> list[dict]:
    """Signal rows of the selected groups, in catalog order, without duplicates."""
    out, seen = [], set()
    for g in SIGNAL_GROUPS:
        if g["key"] in groups:
            for s in g["signals"]:
                if s["name"] not in seen:
                    seen.add(s["name"])
                    out.append(s)
    return out


def signal_columns(groups: list[str]) -> dict[str, list[str]]:
    """{signal: [column names]} (per-wheel signals expand to name_fl, name_fr, name_rl, name_rr)."""
    return {s["name"]: ([f"{s['name']}_{w}" for w in WHEELS] if s["per_wheel"] else [s["name"]])
            for s in selected_signals(groups)}


def needs_info(groups: list[str]) -> bool:
    return any(s["source"] == "info" for s in selected_signals(groups))


def schema(groups: list[str]) -> list[dict]:
    """Manifest entry per exported signal: name, unit, shape, column names, description."""
    cols = signal_columns(groups)
    return [dict(name=s["name"], unit=s["unit"], shape="per_wheel" if s["per_wheel"] else "scalar",
                 columns=cols[s["name"]], desc=s["desc"]) for s in selected_signals(groups)]


# ----------------------------------------------------------------------------- writers
def write_shard(out_dir: Path, shard: int, episode_ids: np.ndarray, t: np.ndarray,
                data: dict[str, np.ndarray], formats: list[str], float32: bool, compress: bool) -> list[str]:
    """Write one shard in every requested format; returns the file names."""
    dtype = np.float32 if float32 else np.float64
    arrays = {k: np.asarray(v, dtype=dtype) for k, v in data.items()}
    files = []
    stem = f"shard_{shard:04d}"
    if "npz" in formats:
        fn = out_dir / f"{stem}.npz"
        (np.savez_compressed if compress else np.savez)(fn, episode_id=np.asarray(episode_ids, dtype=np.int64),
                                                        t=np.asarray(t, dtype=np.float64), **arrays)
        files.append(fn.name)
    if "csv" in formats or "parquet" in formats:
        n, T = len(episode_ids), len(t)
        cols = ["episode_id", "t"]
        blocks = [np.repeat(np.asarray(episode_ids, dtype=np.float64), T)[:, None],
                  np.tile(np.asarray(t, dtype=np.float64), n)[:, None]]
        for name, arr in arrays.items():
            a = arr.reshape(n * T, -1)
            cols += [f"{name}_{w}" for w in WHEELS] if a.shape[1] == 4 and arr.ndim == 3 else [name]
            blocks.append(a.astype(np.float64))
        table = np.concatenate(blocks, axis=1)
        if "csv" in formats:
            fn = out_dir / f"{stem}.csv"
            fmt = ["%d", "%.6f"] + ["%.7g" if float32 else "%.12g"] * (table.shape[1] - 2)
            np.savetxt(fn, table, fmt=fmt, delimiter=",", header=",".join(cols), comments="")
            files.append(fn.name)
        if "parquet" in formats:
            import pyarrow as pa
            import pyarrow.parquet as pq
            cols_arrays = {c: table[:, j].astype(np.int64 if c == "episode_id" else (dtype if j > 1 else np.float64))
                           for j, c in enumerate(cols)}
            fn = out_dir / f"{stem}.parquet"
            pq.write_table(pa.table(cols_arrays), fn, compression="zstd" if compress else "snappy")
            files.append(fn.name)
    return files


def _cell(v: Any) -> Any:
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(f"{float(x):.6g}" for x in v)
    if isinstance(v, float):
        return f"{v:.9g}"
    return v


def write_episodes_table(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    front = [k for k in ("episode_id", "shard", "group", "tire", "surface") if k in keys]
    keys = front + [k for k in keys if k not in front]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["episode_id"]):
            w.writerow({k: _cell(v) for k, v in r.items()})


def write_json(path: Path, obj: Any) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    os.replace(tmp, path)


README = """RC drift simulator - batch export
==================================

manifest.json   the full spec (reproduce with: python -m rc_drift_sim.datagen manifest.json), signal
                schema with units, shard list, timing and status
episodes.csv    one row per episode: sampled parameters and outcomes
                (max_abs_beta_deg, drift_time_s, spun, max_speed, final_speed, distance, finite)
shard_XXXX.*    the time series

Load an NPZ shard in Python:

    import numpy as np
    d = np.load("shard_0000.npz")
    d["episode_id"]        # (n,)       episode ids in this shard (rows of episodes.csv)
    d["t"]                 # (T_rec,)   time stamps in s
    d["speed"]             # (n, T_rec) one array per scalar signal
    d["omega"]             # (n, T_rec, 4) per-wheel signals, wheel order FL, FR, RL, RR

CSV / Parquet shards are long tables: one row per (episode_id, t), per-wheel signals expanded into
name_fl, name_fr, name_rl, name_rr columns. Units are listed in manifest.json ("signals").
Angles are in radians in the data; parameter values in episodes.csv use the GUI units (angles in
degrees for *_deg keys).
"""
