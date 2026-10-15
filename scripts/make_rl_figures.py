"""Milestone 5 figures and scores: learned drift policies against the reference controllers.

    python scripts/make_rl_figures.py --hold runs/hold --track runs/track [--out docs/images/m5]

Each run folder is a ``driftsim-train`` output (``log.jsonl``, ``policy.pt``). The scores use the
reference evaluation of docs/RL.md (nominal car, the run's own observations, 32 cars, parked start,
8 s episodes, default reward weights, deterministic actions) and go to ``<out>/scores.txt``.
"""
from __future__ import annotations

import argparse
import json
import math
import types
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import rc_drift_sim.rl as rl  # noqa: E402
from rc_drift_sim.rl.baselines import LQRDriftBaseline, evaluate, random_policy, zero_policy  # noqa: E402
from rc_drift_sim.rl.ppo import load_policy  # noqa: E402
from rc_drift_sim.rl.visual import record_rollout  # noqa: E402
from rc_drift_sim.sim import state as S  # noqa: E402
from rc_drift_sim.sim.params import default_params  # noqa: E402
from rc_drift_sim.sim.xp import to_numpy  # noqa: E402
from rc_drift_sim.viz import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def reference_config(cfg: rl.EnvConfig) -> rl.EnvConfig:
    """The reference evaluation of a run's config: same observations, nominal everything else."""
    return cfg.replace(episode_s=8.0, reward=rl.RewardWeights(), randomize={}, init_drift_prob=0.0, init_speed=0.0,
                       grip_change_prob=0.0, target_beta_jitter_deg=0.0, target_speed_jitter=0.0)


def score(env: rl.DriftBatchEnv, make_policy, seed: int = 0) -> dict:
    """``rl.evaluate`` plus the second-half drift quality (|sideslip|, speed, share near the target).
    ``make_policy(env)`` builds a fresh policy for each pass (the LQR baseline keeps per-car state)."""
    res = evaluate(env, make_policy(env), seed=seed)
    policy = make_policy(env)
    env.reset(seed=seed)
    betas, speeds = [], []
    for k in range(env.max_steps):
        _, _, _, _, info = env.step(policy(env))
        if k >= env.max_steps // 2:
            betas.append(np.abs(to_numpy(info["beta_deg"])))
            speeds.append(to_numpy(info["speed"]))
    b = np.array(betas)
    res.update(beta_2nd=float(b.mean()), speed_2nd=float(np.mean(speeds)),
               near_target=float((np.abs(b - env.cfg.target_beta_deg) < 10.0).mean()))
    return res


def learned(model):
    return lambda env: (lambda e: to_numpy(model.act(e.last_obs)).astype(np.float64))


def read_log(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]


def as_traj(data: dict) -> types.SimpleNamespace:
    """A ``record_rollout`` episode as the (t, states) the top-down renderer draws."""
    s = data["series"]
    n = len(s["x"])
    st = np.zeros((n, S.NS))
    st[:, S.X], st[:, S.Y], st[:, S.YAW] = s["x"], s["y"], np.radians(s["yaw_deg"])
    beta, v = np.radians(s["beta_deg"]), np.asarray(s["speed"])
    st[:, S.VX], st[:, S.VY] = v * np.cos(beta), v * np.sin(beta)
    st[:, S.DELTA] = np.radians(s["delta_deg"])
    return types.SimpleNamespace(t=np.asarray(s["t"]), states=st, info=None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hold", type=Path, required=True)
    ap.add_argument("--track", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "images" / "m5")
    a = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    a.out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    log = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    fmt = lambda name, r: (f"{name:44s} {r['mean_return']:6.0f}   early end {100 * r['early_end_share']:3.0f} %   "  # noqa: E731
                           f"2nd half |beta| {r['beta_2nd']:4.1f} deg, speed {r['speed_2nd']:.2f} m/s, "
                           f"within 10 deg of target {100 * r['near_target']:3.0f} %")
    runs = {"hold": a.hold, "track": a.track}
    refs: dict[str, dict] = {}
    log("RC drift sim - Milestone 5: reference evaluation (32 cars, parked start, 8 s, nominal car)")
    for task, run in runs.items():
        if run is None:
            continue
        model, cfg = load_policy(run / "policy.pt")
        ecfg = reference_config(cfg)
        env = rl.DriftBatchEnv(32, ecfg)
        log(f"\n[{task}] {run.name}: obs {cfg.obs}, history {cfg.history}, "
            f"{read_log(run)[-1]['env_steps'] / 1e6:.0f}M env steps")
        r_pol = score(env, learned(model))
        log(fmt("PPO policy (sensors, deterministic)", r_pol))
        refs[task] = {"PPO": r_pol["mean_return"]}
        if task == "hold":
            r_lqr = score(env, LQRDriftBaseline)
            log(fmt("LQR drift controller (true state)", r_lqr))
            refs[task]["LQR (true state)"] = r_lqr["mean_return"]
        rng = np.random.default_rng(0)
        for name, make in (("no input", lambda env: zero_policy),
                           ("uniform random", lambda env: (lambda e: random_policy(e, rng)))):
            r = score(env, make)
            log(fmt(name, r))
            refs[task][name] = r["mean_return"]
    (a.out / "scores.txt").write_text("\n".join(lines) + "\n")

    # learning curves
    tasks = [t for t in runs if runs[t] is not None]
    fig, axs = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), squeeze=False)
    for ax, task in zip(axs[0], tasks):
        rows = read_log(runs[task])
        steps = np.array([r["env_steps"] for r in rows]) / 1e6
        ax.plot(steps, [r["episode_return"] for r in rows], label="PPO (training episodes)", color="C0")
        for (name, val), c in zip([(k, v) for k, v in refs[task].items() if k != "PPO"], ("C3", "0.5", "0.75")):
            ax.axhline(val, linestyle="--", color=c, linewidth=1, label=name)
        ax.set_title(f"{task}: mean episode return")
        ax.set_xlabel("environment steps (millions)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out / "learning_curves.png", dpi=90)
    plt.close(fig)

    # the learned drift next to the LQR one (hold) and round the circle (track)
    p = default_params()
    model, cfg = load_policy(a.hold / "policy.pt")
    ecfg = reference_config(cfg)
    ep_pol = record_rollout(ecfg, model.act)
    ep_lqr = record_rollout(rl.EnvConfig(task="hold"), "lqr")
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for ep, label, style in ((ep_lqr, "LQR (true state)", "--"), (ep_pol, "PPO (sensors)", "-")):
        s = ep["series"]
        axs[0].plot(s["t"], np.abs(s["beta_deg"]), style, label=label)
        axs[1].plot(s["t"], s["speed"], style)
        axs[2].plot(s["t"][1:], s["steer"], style, color="C2" if style == "-" else "C2", alpha=1 if style == "-" else 0.5)
        axs[2].plot(s["t"][1:], s["throttle"], style, color="C1", alpha=1 if style == "-" else 0.5)
    axs[0].axhline(ecfg.target_beta_deg, color="0.6", linewidth=0.8)
    axs[1].axhline(ecfg.target_speed, color="0.6", linewidth=0.8)
    axs[0].set_ylabel("|sideslip| (deg)")
    axs[0].legend(fontsize=7)
    axs[1].set_ylabel("speed (m/s)")
    axs[2].set_ylabel("steer / throttle")
    axs[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(a.out / "hold_timeseries.png", dpi=90)
    plt.close(fig)
    fig = render.snapshot(as_traj(ep_pol), p, None, n_poses=7, title=f"PPO drift from sensors (return {ep_pol['episode_return']:.0f})")
    fig.savefig(a.out / "hold_snapshot.png", dpi=80)
    plt.close(fig)
    if a.track is not None:
        model, cfg = load_policy(a.track / "policy.pt")
        ep = record_rollout(reference_config(cfg), model.act)
        c, R = ep["track"]["center"], ep["track"]["radius"]
        th = np.linspace(0, 2 * math.pi, 200)
        circle = np.stack([c[0] + R * np.cos(th), c[1] + R * np.sin(th)], axis=1)
        fig = render.snapshot(as_traj(ep), p, None, n_poses=8, path=circle,
                              title=f"PPO on the circle task (return {ep['episode_return']:.0f})")
        fig.savefig(a.out / "track_snapshot.png", dpi=80)
        plt.close(fig)
    print(f"saved figures and scores.txt to {a.out}")


if __name__ == "__main__":
    main()
