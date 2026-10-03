"""Drift RL environments: Gymnasium compliance, physics identical to the simulator, reward sanity
(the model-based drift controller must beat open-loop, idle and random driving), randomization,
autoreset and the PyTorch backend."""
from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
import pytest

import rc_drift_sim.rl as rl
from rc_drift_sim.control.maneuvers import open_loop_drift_actions
from rc_drift_sim.sim import state as S
from rc_drift_sim.sim.actuators import ActionDelay
from rc_drift_sim.sim.params import default_params
from rc_drift_sim.sim.vehicle import Vehicle, derivatives_model
from rc_drift_sim.sim.xp import available_devices


@pytest.mark.parametrize("env_id", ["DriftSim/DriftHold-v0", "DriftSim/DriftTrack-v0"])
def test_gymnasium_env_checker_and_determinism(env_id):
    from gymnasium.utils.env_checker import check_env
    env = gym.make(env_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # only "Box bounds are infinite"
        check_env(env.unwrapped)
    o1, _ = env.reset(seed=7)
    a = np.array([0.3, 0.4], dtype=np.float32)
    s1 = [env.step(a)[0] for _ in range(5)]
    o2, _ = env.reset(seed=7)
    s2 = [env.step(a)[0] for _ in range(5)]
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(np.array(s1), np.array(s2))
    assert env.observation_space.shape == (len(env.unwrapped.obs_names),)


def test_env_physics_equals_the_simulator_with_latency():
    """No noise, no randomization: the env state is the simulator's, commands delayed by the latency."""
    env = rl.DriftBatchEnv(1, rl.EnvConfig(noise={}), autoreset=False)
    env.reset(seed=0)
    car = Vehicle(default_params())
    s = car.initial_state()
    delay = ActionDelay(car.params.actuators.latency, car.control_dt)
    actions = open_loop_drift_actions(car.params, duration=2.0)
    for a in actions:
        env.step(a[None])
        s, _ = car.step(s, delay.push(a), want_info=False)
    np.testing.assert_array_equal(env.state[0], s)


def test_mirrored_physics_is_the_mirror_image():
    """Mirroring the state and the steering mirrors the derivative (left/right symmetric car)."""
    car = Vehicle(default_params())
    rng = np.random.default_rng(0)
    s = car.initial_state(v=1.5, beta=-0.4, yaw_rate=1.2)
    s[S.DELTA], s[S.I_MOTOR], s[S.DFZ_LAT] = -0.2, 8.0, 1.5
    s[S.KAPPA_LAG], s[S.ALPHA_LAG] = rng.uniform(-0.3, 0.6, 4), rng.uniform(-0.5, 0.5, 4)
    u = np.array([-0.4, 0.3])
    ds = derivatives_model(s, u, car.model)[0]
    ds_m = derivatives_model(rl.mirror_state(s), u * [-1.0, 1.0], car.model)[0]
    np.testing.assert_allclose(ds_m, rl.mirror_state(ds), rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(rl.mirror_state(rl.mirror_state(s)), s)


def _open_loop(env):
    ol = open_loop_drift_actions(default_params(), duration=env.cfg.episode_s)
    return ol[np.minimum(env.t_step, len(ol) - 1)]


def test_hold_reward_ranks_the_drift_controller_first():
    env = rl.DriftBatchEnv(8, rl.EnvConfig(task="hold", episode_s=5.0))
    rng = np.random.default_rng(1)
    res = {name: rl.evaluate(env, pol, seed=3) for name, pol in [
        ("lqr", rl.LQRDriftBaseline(env)), ("open_loop", _open_loop), ("zero", rl.zero_policy),
        ("random", lambda e: rl.random_policy(e, rng))]}
    for k, v in res.items():
        print(f"\n{k:9s} return {v['mean_return']:7.1f}  early end {v['early_end_share']:.2f}  mean|beta| {v['mean_abs_beta_deg']:.1f}")
    assert res["lqr"]["early_end_share"] == 0.0 and res["lqr"]["mean_abs_beta_deg"] > 15.0
    assert res["lqr"]["mean_return"] > 2 * res["open_loop"]["mean_return"] > 0.0
    assert res["zero"]["mean_return"] < 1.0 and res["random"]["early_end_share"] > 0.5


def test_track_reward_needs_progress_and_ends_off_the_line():
    env = rl.DriftBatchEnv(4, rl.EnvConfig(task="track", episode_s=3.0))
    assert rl.evaluate(env, rl.zero_policy, seed=0)["mean_return"] < 0.1, "parking on the line earns (almost) nothing"
    res = rl.evaluate(env, lambda e: np.tile([0.0, 0.5], (e.num_envs, 1)), seed=0)   # straight on: leaves the circle
    assert res["early_end_share"] == 1.0 and res["mean_length"] < 150


def test_randomization_draws_a_new_car_per_episode():
    rnd = {"vehicle.mass": {"dist": "uniform", "low": 1.2, "high": 2.0},
           "actuators.latency": {"dist": "choice", "values": [0.0, 0.04]}}
    env = rl.DriftBatchEnv(16, rl.EnvConfig(task="hold", episode_s=0.2, randomize=rnd))
    m0 = 1.0 / env.model.inv_m.copy()
    assert 1.2 <= m0.min() and m0.max() <= 2.0 and np.unique(np.round(m0, 6)).size == 16
    assert set(np.unique(env._delay)) == {0, 2}
    for _ in range(env.max_steps):                      # every car finishes once and is redrawn
        env.step(np.zeros((16, 2)))
    m1 = 1.0 / env.model.inv_m
    assert not np.allclose(m0, m1) and 1.2 <= m1.min() and m1.max() <= 2.0
    again = rl.DriftBatchEnv(16, rl.EnvConfig(task="hold", episode_s=0.2, randomize=rnd))
    np.testing.assert_array_equal(1.0 / again.model.inv_m, m0)


def test_vector_env_autoreset_reports_final_observations():
    venv = gym.make_vec("DriftSim/DriftHold-v0", num_envs=4, vectorization_mode="vector_entry_point", episode_s=0.1)
    obs, _ = venv.reset(seed=0)
    for k in range(5):
        obs, r, te, tr, info = venv.step(np.zeros((4, 2), dtype=np.float32))
    assert tr.all() and info["_final_obs"].all() and obs.shape == (4, venv.single_observation_space.shape[0])
    assert all(info["final_info"][i]["episode"]["l"] == 5 for i in range(4))
    assert all(info["final_obs"][i].shape == obs[i].shape for i in range(4))


def test_drift_start_curriculum_begins_in_a_drift():
    env = rl.DriftBatchEnv(8, rl.EnvConfig(task="hold", init_drift_prob=1.0, noise={}))
    beta = np.degrees(np.arctan2(env.state[:, S.VY], env.state[:, S.VX]))
    assert np.allclose(np.abs(beta), 30.0, atol=0.5) and np.any(beta > 0) and np.any(beta < 0)
    assert np.all(beta * env.state[:, S.R] < 0), "rotating into the slide"


def test_torch_backend_matches_numpy():
    torch = pytest.importorskip("torch")
    cfg = rl.EnvConfig(task="track", noise={}, episode_s=0.6, init_drift_prob=0.5,
                       randomize={"vehicle.mass": {"dist": "uniform", "low": 1.3, "high": 1.9},
                                  "surface": {"dist": "choice", "values": ["epoxy_ptile", "loose_dirt"]}})
    e_np = rl.DriftBatchEnv(4, cfg)
    e_t = rl.DriftBatchEnv(4, cfg, device="torch-cpu", precision="float64")
    rng = np.random.default_rng(0)
    for _ in range(45):                                # includes automatic resets
        a = rng.uniform(-0.6, 0.6, (4, 2)) + [0.0, 0.3]
        o1, r1, te1, tr1, _ = e_np.step(a)
        o2, r2, te2, tr2, _ = e_t.step(torch.as_tensor(a))
        np.testing.assert_allclose(o2.numpy(), o1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(r2.numpy(), r1, atol=1e-9)
        assert np.array_equal(te1, te2.numpy()) and np.array_equal(tr1, tr2.numpy())


@pytest.mark.skipif(not available_devices()["mps"], reason="no Apple GPU")
def test_mps_env_runs():
    import torch
    env = rl.DriftBatchEnv(256, rl.EnvConfig(task="hold"), device="mps")
    a = torch.tensor([0.2, 0.4], device="mps").repeat(256, 1)
    for _ in range(10):
        obs, r, te, tr, info = env.step(a)
    assert obs.device.type == "mps" and obs.dtype == torch.float32 and bool(torch.isfinite(obs).all())


def test_config_validation():
    with pytest.raises(ValueError):
        rl.EnvConfig(task="donut").validate()
    with pytest.raises(ValueError):
        rl.DriftBatchEnv(2, rl.EnvConfig(randomize={"drivetrain.layout": {"dist": "fixed", "value": "awd_spool"}}))
    with pytest.raises(ValueError):
        rl.DriftBatchEnv(2, rl.EnvConfig(randomize={"vehicle.mass": {"dist": "uniform", "low": 2, "high": 1}}))
    assert math.isclose(rl.DriftBatchEnv(1).control_dt, 0.02)


def test_ppo_trains_saves_and_reloads(tmp_path):
    pytest.importorskip("torch")
    from rc_drift_sim.rl.ppo import PPOConfig, load_policy, train
    cfg = rl.EnvConfig(task="hold", episode_s=0.5)
    model, hist = train(cfg, PPOConfig(total_steps=2 * 32 * 16, num_envs=32, rollout=16, epochs=2, minibatches=2),
                        out=tmp_path, log=None)
    assert len(hist) == 2 and hist[-1]["episodes"] > 0 and np.isfinite(hist[-1]["value_loss"])
    loaded, cfg2 = load_policy(tmp_path / "policy.pt")
    assert cfg2.task == "hold" and (tmp_path / "log.jsonl").read_text().count("\n") == 2
    env = rl.DriftBatchEnv(4, cfg2)
    a1, a2 = model.act(env.last_obs), loaded.act(env.last_obs)
    np.testing.assert_allclose(a1.numpy(), a2.numpy(), atol=1e-6)
    assert a1.shape == (4, 2) and float(a1.abs().max()) <= 1.0
