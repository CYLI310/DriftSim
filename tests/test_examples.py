"""The shipped examples and generated docs stay valid: every example spec validates and runs (scaled
down), and docs/PARAMETERS.md matches the catalog."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rc_drift_sim.datagen import run_batch, validate_spec

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "examples" / "specs").glob("*.json"))


def test_there_are_example_specs():
    assert len(SPECS) >= 4


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_example_spec_validates_and_runs(path, tmp_path):
    spec = json.loads(path.read_text())
    _, errors, _ = validate_spec(spec)
    assert errors == [], errors
    small = dict(spec, episodes=3, duration_s=0.2)
    res = run_batch(small, out_root=tmp_path, workers=1)
    assert res["status"] == "complete" and res["episodes_written"] == 3


def test_parameter_reference_is_up_to_date():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_param_reference.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
