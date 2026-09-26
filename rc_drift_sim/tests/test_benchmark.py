"""DESIGN.md section 10 item 11: physics throughput of the NumPy reference (informational).

Run with ``pytest -s -m benchmark`` to see the numbers. The assertion is deliberately loose
(> 2000 substeps/s); the printed rate is the useful output.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from rc_drift_sim.sim.params import TireCondition
from rc_drift_sim.tests.conftest import assert_all_finite
from rc_drift_sim.control.maneuvers import open_loop_drift_actions

MIN_SUBSTEPS_PER_S = 1500.0          # single env, NumPy (loose; the printed number matters)
MIN_BATCH_ENV_STEPS_PER_S = 5000.0   # 1024 envs in lock-step, 50 Hz control, 1 kHz RK4


@pytest.mark.benchmark
def test_physics_substeps_per_second(params, vehicle, run):
    duration = 2.0
    actions = open_loop_drift_actions(params, duration)
    s0 = vehicle.initial_state()
    run(vehicle, s0, actions[:10], 0.2)                       # warm-up (imports, caches)

    t0 = time.perf_counter()
    traj = run(vehicle, s0, actions, duration)
    elapsed = time.perf_counter() - t0
    assert_all_finite(traj, "benchmark rollout")

    n_control = int(np.asarray(traj.actions).shape[0])
    n_sub = n_control * params.sim.n_substeps
    rate = n_sub / elapsed
    print(f"\nNumPy reference: {n_sub} physics substeps ({n_control} control steps x "
          f"{params.sim.n_substeps}) in {elapsed:.3f} s -> {rate:,.0f} substeps/s "
          f"({1e6 * elapsed / n_sub:.1f} us/substep, {n_control / elapsed:,.0f} control steps/s, "
          f"real-time factor {duration / elapsed:.1f}x)")
    assert rate > MIN_SUBSTEPS_PER_S, f"{rate:.0f} substeps/s is below the loose floor {MIN_SUBSTEPS_PER_S}"


@pytest.mark.benchmark
def test_derivatives_call_rate(params, vehicle):
    """Micro-benchmark of one derivatives() evaluation (4 of them per RK4 substep)."""
    from rc_drift_sim.sim import surface as surface_mod
    from rc_drift_sim.sim.vehicle import derivatives
    surf4 = surface_mod.uniform_surface(params.surface)
    cond = TireCondition.neutral(params.sim.ambient_temp)
    s = vehicle.initial_state(v=2.0)
    u = np.array([0.3, 0.5])
    for _ in range(50):
        derivatives(s, u, params, surf4, cond)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        ds, info = derivatives(s, u, params, surf4, cond)
    elapsed = time.perf_counter() - t0
    assert np.all(np.isfinite(ds))
    print(f"\nderivatives(): {1e6 * elapsed / n:.1f} us per call -> {n / elapsed:,.0f} calls/s")


@pytest.mark.benchmark
def test_batched_env_steps_per_second(params):
    """Throughput of the vectorized NumPy core: B envs stepped in lock-step (one array op per
    physics op for all envs). Reported in control steps (env-steps) and physics substeps per second."""
    from rc_drift_sim.sim.vehicle import Vehicle
    v = Vehicle(params)
    rng = np.random.default_rng(0)
    print()
    for B in (1, 64, 1024, 4096):
        s = np.broadcast_to(v.initial_state(v=1.5), (B, v.initial_state().size)).copy()
        u = np.column_stack([rng.uniform(-1, 1, B), rng.uniform(0, 0.6, B)])
        v.step(s, u, want_info=False)                                     # warm-up
        reps = max(2, int(200 / max(B / 64, 1)))
        t0 = time.perf_counter()
        for _ in range(reps):
            s, _ = v.step(s, u, want_info=False)
        el = (time.perf_counter() - t0) / reps
        rate = B / el
        print(f"B = {B:5d}: {1e3 * el:8.2f} ms per control step -> {rate:10,.0f} env-steps/s, "
              f"{rate * params.sim.n_substeps:12,.0f} substeps/s")
        assert np.all(np.isfinite(s))
    assert rate > 0
    if B >= 1024:
        assert rate > MIN_BATCH_ENV_STEPS_PER_S
