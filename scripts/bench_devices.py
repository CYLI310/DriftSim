"""Raw simulation speed of every compute device on this machine, in control steps per second.

    python scripts/bench_devices.py                 # NumPy (one process) and every available GPU
    python scripts/bench_devices.py --compile       # also torch.compile on CUDA (slow first call)
    python scripts/bench_devices.py --sizes 4096 65536

One control step = 20 RK4 steps of 1 ms for one car. For batch exports, the CPU path runs one
NumPy process per core, so multiply its number by roughly the core count for a fair comparison
(a run on the Apple M4 reaches about 54k control steps/s on 9 worker processes).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from rc_drift_sim.datagen.runner import _stepper
from rc_drift_sim.sim import xp
from rc_drift_sim.sim.params import default_params
from rc_drift_sim.sim.vehicle import VehicleBatch


def bench(device: str, B: int, steps: int, use_compile: bool, precision: str) -> float:
    batch = VehicleBatch([default_params()] * B, check_stiffness=False)
    s = batch.initial_states(v=1.0)
    u = np.tile([0.2, 0.25], (B, 1))
    if device == "numpy":
        batch.step(s, u)
        t0 = time.perf_counter()
        for _ in range(steps):
            s, _ = batch.step(s, u)
        return steps * B / (time.perf_counter() - t0)
    import torch
    dtype = xp.torch_dtype(precision)
    model = xp.to_device(batch.model, device, dtype)
    step = _stepper(batch, model, use_compile, device)
    st = torch.as_tensor(s, dtype=dtype, device=device)
    ut = torch.as_tensor(u, dtype=dtype, device=device)
    sync = {"mps": lambda: torch.mps.synchronize(), "cuda": lambda: torch.cuda.synchronize()}.get(device, lambda: None)
    st = step(st, ut, 2 * batch.n_substeps)           # warm-up (and compile)
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        st = step(st, ut, batch.n_substeps)
    sync()
    if not bool(torch.isfinite(st).all()):
        raise RuntimeError(f"non-finite state on {device}")
    return steps * B / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 8192, 65536], help="cars per batch")
    ap.add_argument("--steps", type=int, default=10, help="control steps to time")
    ap.add_argument("--precision", default="float32", choices=xp.PRECISIONS, help="GPU precision")
    ap.add_argument("--compile", action="store_true", help="also time torch.compile (CUDA)")
    args = ap.parse_args()
    avail = xp.available_devices()
    runs = [("numpy", False)] + [(d, False) for d in ("mps", "cuda") if avail[d]]
    if args.compile and avail["cuda"]:
        runs.append(("cuda", True))
    print("devices:", ", ".join(d for d, ok in avail.items() if ok))
    for device, comp in runs:
        for B in args.sizes:
            r = bench(device, B, args.steps, comp, args.precision)
            label = device + (" + compile" if comp else "") + ("" if device == "numpy" else f" {args.precision}")
            print(f"{label:22s} {B:7d} cars: {r / 1e3:9.1f} k control steps/s", flush=True)


if __name__ == "__main__":
    main()
