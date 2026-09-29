"""Batch data generation: sample car/tire/surface/condition/maneuver variations, simulate them in
vectorized shards across worker processes, and export the time series (NPZ / CSV / Parquet).

    from rc_drift_sim.datagen import default_spec, run_batch
    spec = default_spec(); spec["episodes"] = 1000
    spec["params"]["vehicle.mass"] = {"dist": "uniform", "low": 1.4, "high": 1.8}
    summary = run_batch(spec)

Command line: ``driftsim-datagen spec.json [--out DIR] [--workers N]`` (or
``python -m rc_drift_sim.datagen ...``). The spec format is documented in docs/DATA_GENERATION.md.
"""
from .catalog import build_catalog
from .runner import SpecError, plan, preview, run_batch, simulate_episodes
from .spec import default_spec, estimate, validate_spec

__all__ = ["build_catalog", "default_spec", "validate_spec", "estimate", "plan", "run_batch", "preview",
           "simulate_episodes", "SpecError"]
