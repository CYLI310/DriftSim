"""Drift tests.

* DESIGN.md section 10 item 10: a hand-tuned open-loop schedule produces a sustained drift on the
  paved track (|beta| > 20 deg for >= 1.5 s continuously, never |beta| >= 80 deg while moving,
  v > 0.8 m/s in the drift window, no NaN).
* Drift equilibria exist, are open-loop unstable, and can be held by full-state feedback through the
  real actuator chain (servo, motor, 20 ms latency) - including capture from a real drift entry and
  under model mismatch. This is the capability the RL policy has to learn.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_drift_sim.control.lqr import LQRDriftPolicy
from rc_drift_sim.control.maneuvers import DRIFT_SCHEDULE, LQR_ENTRY, open_loop_drift_actions
from rc_drift_sim.sim import equilibrium as EQ
from rc_drift_sim.sim import state as S
from rc_drift_sim.sim.vehicle import Vehicle
from rc_drift_sim.tests.conftest import (assert_all_finite, longest_true_run, traj_beta, traj_dt,
                                         traj_speed, traj_states)

BETA_DRIFT_DEG = 20.0
BETA_SPIN_DEG = 80.0
MIN_DRIFT_S = 1.5
MIN_SPEED = 0.8
MOVING = 0.05


def _print_table(t, v, beta_deg, r, delta_deg, actions, every_s=0.25):
    dt = t[1] - t[0]
    stride = max(int(round(every_s / dt)), 1)
    print("\n   t[s]   v[m/s]  beta[deg]  r[deg/s]  delta[deg]  steer   thr")
    for k in range(0, len(t), stride):
        a = actions[min(k, len(actions) - 1)]
        print(f"  {t[k]:5.2f}  {v[k]:7.3f}  {beta_deg[k]:9.2f}  {np.degrees(r[k]):8.1f}  {delta_deg[k]:10.2f}"
              f"  {a[0]:5.2f} {a[1]:5.3f}")


# ----------------------------------------------------------------------------- open loop (item 10)
def test_open_loop_drift_is_sustained(params, vehicle, run):
    duration = 4.0
    actions = open_loop_drift_actions(params, duration)
    assert actions.shape == (int(round(duration / params.sim.control_dt)), 2)
    traj = run(vehicle, vehicle.initial_state(), actions, duration)
    assert_all_finite(traj, "drift rollout")

    st = traj_states(traj)
    t = np.asarray(traj.t, dtype=float)
    dt = traj_dt(traj)
    v = traj_speed(traj)
    beta_deg = np.degrees(traj_beta(traj))
    r = st[:, S.R]
    _print_table(t, v, beta_deg, r, np.degrees(st[:, S.DELTA]), np.asarray(traj.actions))

    moving = v > MOVING
    drifting = moving & (np.abs(beta_deg) > BETA_DRIFT_DEG) & (np.abs(beta_deg) < BETA_SPIN_DEG) & (v > MIN_SPEED)
    start, length = longest_true_run(drifting)
    drift_s = max(length - 1, 0) * dt
    window = slice(start, start + length)
    print(f"\nschedule: {DRIFT_SCHEDULE}")
    print(f"longest continuous drift window: {drift_s:.2f} s (t = {t[start]:.2f} .. "
          f"{t[start + max(length - 1, 0)]:.2f} s)")
    if length:
        print(f"inside the window: beta {beta_deg[window].mean():+.1f} deg (min {beta_deg[window].min():+.1f}), "
              f"yaw rate {np.degrees(r[window]).mean():+.0f} deg/s, speed {v[window].mean():.2f} m/s "
              f"(min {v[window].min():.2f})")

    assert drift_s >= MIN_DRIFT_S, (
        f"drift held for only {drift_s:.2f} s (need >= {MIN_DRIFT_S} s with {BETA_DRIFT_DEG} < |beta| < "
        f"{BETA_SPIN_DEG} deg and v > {MIN_SPEED} m/s); re-tune with `python scripts/tune_drift.py`")
    assert np.all(np.abs(beta_deg[moving]) < BETA_SPIN_DEG), (
        f"spin-out: |beta| reached {np.max(np.abs(beta_deg[moving])):.1f} deg")
    # a left-hand drift (entry steer left, counter-steer right) has beta < 0 and r > 0
    assert beta_deg[window].mean() < 0 and r[window].mean() > 0, "drift has the wrong direction"


def test_drift_schedule_expands_to_expected_steps(params):
    dt = params.sim.control_dt
    acts = open_loop_drift_actions(params, 4.0)
    k = 0
    for seg_duration, steer, thr in DRIFT_SCHEDULE:
        n = min(int(round(seg_duration / dt)), acts.shape[0] - k)
        assert np.all(acts[k:k + n] == [steer, thr]), f"segment {(seg_duration, steer, thr)} not held"
        k += n
    assert k == acts.shape[0]
    assert np.all(np.abs(acts) <= 1.0)


# ----------------------------------------------------------------------------- drift equilibria
@pytest.fixture(scope="module")
def drift_trim(params):
    v = Vehicle(params)
    trim = EQ.solve_trim(v, 1.5, beta=np.radians(-30.0))
    assert trim.success, f"no drift equilibrium found: {trim.summary()}"
    return v, trim


def test_drift_equilibrium_is_open_loop_unstable(drift_trim):
    """The rear-saturated, counter-steered drift trim exists and has exactly one unstable real mode."""
    v, trim = drift_trim
    print("\n" + trim.summary())
    kappa_r, alpha_f, alpha_r = trim.info["kappa"][2], np.degrees(trim.info["alpha"][0]), np.degrees(trim.info["alpha"][2])
    print(f"rear slip ratio {kappa_r:+.2f}, front slip angle {alpha_f:+.1f} deg, rear slip angle {alpha_r:+.1f} deg")
    ev = EQ.open_loop_eigenvalues(v, trim)
    unstable = ev[ev.real > 1e-6]
    print(f"unstable eigenvalues: {np.round(unstable, 3)} 1/s")
    assert trim.u[0] < -0.2, "a left-hand drift needs counter-steer (steer right)"
    assert trim.r > 0 and 0.2 * 9.81 < trim.ay < 0.45 * 9.81, "drift should run near the grip limit (~0.3 g)"
    assert kappa_r > 0.2, "the rear axle must be spinning in a drift"
    assert abs(alpha_r) > 25.0 > abs(alpha_f), "rear saturated (large slip angle), front in its useful range"
    assert len(unstable) == 1 and abs(unstable[0].imag) < 1e-9 and 1.0 < unstable[0].real < 10.0, (
        "a RWD drift equilibrium should have exactly one real unstable mode (Hindiyeh & Gerdes)")


def _lqr_drift_run(vehicle, trim, entry, design_vehicle=None, duration=8.0):
    """Launch + throttle stab, then the LQR (designed on ``design_vehicle``), through the latency."""
    policy = LQRDriftPolicy(vehicle, trim, entry, design_vehicle=design_vehicle)
    traj = vehicle.rollout(vehicle.initial_state(), policy, duration=duration)
    assert_all_finite(traj, "LQR drift rollout")
    return np.degrees(traj.beta[1:]), traj.speed[1:], policy.t_switch


def test_lqr_holds_drift_from_real_entry(drift_trim):
    """Full-state feedback captures the drift right after a throttle-stab entry and holds it."""
    v, trim = drift_trim
    betas, speeds, t_sw = _lqr_drift_run(v, trim, LQR_ENTRY)
    after = slice(int(round(t_sw / v.control_dt)), None)
    last = slice(-int(round(3.0 / v.control_dt)), None)
    print(f"\nLQR on from t = {t_sw:.2f} s; worst |beta| afterwards {np.abs(betas[after]).max():.1f} deg; "
          f"last 3 s: beta {betas[last].mean():+.2f} +- {betas[last].std():.2f} deg, v {speeds[last].mean():.2f} m/s")
    assert np.abs(betas[after]).max() < BETA_SPIN_DEG, "spun out after the controller took over"
    assert abs(betas[last].mean() - np.degrees(trim.beta)) < 1.5 and betas[last].std() < 1.0
    assert np.all(speeds[last] > MIN_SPEED)


@pytest.mark.parametrize("label,change", [
    ("mu -10%", lambda p: dataclasses.replace(p, surface=dataclasses.replace(p.surface, mu_scale=p.surface.mu_scale * 0.9))),
    ("mu +10%", lambda p: dataclasses.replace(p, surface=dataclasses.replace(p.surface, mu_scale=p.surface.mu_scale * 1.1))),
    ("mass +15%", lambda p: dataclasses.replace(p, vehicle=dataclasses.replace(p.vehicle, mass=p.vehicle.mass * 1.15))),
    ("latency 40 ms", lambda p: dataclasses.replace(p, actuators=dataclasses.replace(p.actuators, latency=0.04))),
    ("servo 300 deg/s", lambda p: dataclasses.replace(p, actuators=dataclasses.replace(p.actuators, servo_rate=np.radians(300.0)))),
])
def test_lqr_drift_survives_model_mismatch(drift_trim, label, change):
    """The controller is designed on the nominal car and run on a perturbed one: the drift must hold
    (the steady sideslip may shift; there is no integral action)."""
    v_nom, trim = drift_trim
    v_real = Vehicle(change(v_nom.params))
    betas, speeds, _ = _lqr_drift_run(v_real, trim, LQR_ENTRY, design_vehicle=v_nom)
    last = slice(-int(round(3.0 / v_nom.control_dt)), None)
    print(f"\n{label}: beta {betas[last].mean():+.1f} +- {betas[last].std():.2f} deg, v {speeds[last].mean():.2f} m/s")
    assert np.all((np.abs(betas[last]) > BETA_DRIFT_DEG) & (np.abs(betas[last]) < BETA_SPIN_DEG))
    assert np.all(speeds[last] > MIN_SPEED)
