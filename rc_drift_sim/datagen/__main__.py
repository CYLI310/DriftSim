"""Command-line batch generation: ``driftsim-datagen spec.json [--out DIR] [--workers N]``
(same as ``python -m rc_drift_sim.datagen ...``).

``spec.json`` may be a spec or a manifest.json from an earlier run (its "spec" entry is used), which
reproduces that dataset exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .runner import SpecError, run_batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="driftsim-datagen", description=__doc__)
    ap.add_argument("spec", help="batch spec JSON (or a manifest.json to reproduce a dataset)")
    ap.add_argument("--out", default=None, help="output root directory (default: export.out_dir)")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (0 = automatic)")
    args = ap.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    if "spec" in spec and "format_version" in spec:
        spec = spec["spec"]
    last = [0.0]

    def progress(p: dict) -> None:
        if time.time() - last[0] > 0.5 or p["stage"] != "simulating":
            last[0] = time.time()
            eta = f", ETA {p['eta_s']:.0f} s" if p.get("eta_s") else ""
            print(f"\r{p['stage']}: {p['done']:.0f}/{p['total']} episodes ({100 * p['fraction']:.1f} %), "
                  f"{p['episodes_per_s']:.1f} episodes/s{eta}   ", end="", flush=True)

    try:
        res = run_batch(spec, out_root=args.out, progress=progress, workers=args.workers)
    except SpecError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"\n{res['status']}: {res['episodes_written']} episodes in {res['seconds']:.1f} s -> {res['out_dir']} "
          f"({res['bytes'] / 1e6:.1f} MB)")
    for w in res["warnings"]:
        print("warning:", w)
    return 0 if res["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
