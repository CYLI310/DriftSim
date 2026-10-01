"""Generate a small dataset from Python and read it back.

    python examples/batch_export.py

Writes exports/<timestamp>_example/ (git-ignored). The same spec saved as JSON can be run with
``driftsim-datagen spec.json``; see docs/DATA_GENERATION.md for every option.
"""
import csv
import glob

import numpy as np

from rc_drift_sim import default_spec, run_batch


def main():
    spec = default_spec()
    spec.update(name="example", episodes=200, duration_s=3.0, seed=42)
    spec["params"] = {
        "tire": {"dist": "choice", "values": ["hard_plastic_drift", "rubber_onroad"]},
        "surface": {"dist": "choice", "values": ["epoxy_ptile", "dry_asphalt", "loose_dirt"]},
        "vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8},
        "condition.wear": {"dist": "uniform", "low": 0.0, "high": 0.6, "per_wheel": True},
        "actuators.latency": {"dist": "choice", "values": [0.02, 0.04]},
    }
    spec["maneuver"] = {"type": "random", "params": {"mirror_prob": {"dist": "fixed", "value": 0.5}}}
    spec["export"]["formats"] = ["npz", "csv"]

    summary = run_batch(spec, progress=lambda p: print(f"\r{p['done']:.0f}/{p['total']} episodes", end=""))
    print(f"\n{summary['status']}: {summary['episodes_written']} episodes in {summary['seconds']:.1f} s "
          f"-> {summary['out_dir']}")

    # read it back: one NPZ per shard, one row per episode in episodes.csv; all/ holds every episode,
    # not_spun/ the same files with only the episodes that did not spin out
    shard = np.load(sorted(glob.glob(f"{summary['out_dir']}/all/shard_*.npz"))[0])
    print("signals:", [k for k in shard.files if k not in ("episode_id", "t")])
    print("speed array:", shard["speed"].shape, "(episodes, time steps); wheel speeds:", shard["omega"].shape)
    rows = list(csv.DictReader(open(f"{summary['out_dir']}/all/episodes.csv")))
    spun = sum(r["spun"] == "True" for r in rows)
    print(f"{spun} of {len(rows)} episodes spun out; first row: tire={rows[0]['tire']}, surface={rows[0]['surface']}, "
          f"mass={float(rows[0]['vehicle.mass']):.2f} kg")


if __name__ == "__main__":
    main()
