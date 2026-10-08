"""Compact PPO for the drift environments: a working starting point for Milestone 5.

    python -m rc_drift_sim.rl.ppo --task hold --envs 1024 --steps 20000000 --out runs/hold
    python -m rc_drift_sim.rl.ppo --task track --device mps --envs 8192 --out runs/track_gpu
    driftsim-train ...                                   (the same, installed as a command)

Clipped-surrogate PPO with GAE on ``DriftBatchEnv`` (all cars in one vectorized simulation): a Gaussian
MLP policy with a state-independent log std, an MLP critic and running observation normalization
(stored with the weights). Episodes cut by the time limit are bootstrapped with the critic's value of
their final observation. The policy runs where the environment runs (CPU for the NumPy physics, the
GPU for ``--device mps|cuda``). Each iteration logs to ``<out>/log.jsonl`` and saves ``<out>/policy.pt``;
``load_policy(path)`` restores the policy and the environment config it was trained on.

Self-adaptation (``PPOConfig.grip_estimator``): a second network reads the same sensor history and
estimates the grip (friction coefficient), trained by regression on the simulator's true grip; its
estimate feeds the policy and the safety filter, on the car as in training. ``privileged_critic``
gives the critic the simulator state (training only; the policy never sees it). With ``eval_every``
the policy is scored on a grip sweep (``rl.evaluation``) and the best one is kept as ``best.pt``.

    driftsim-train --preset safe-adaptive --device cuda --out runs/final     # the recommended setup
    driftsim-train --config settings.json --device cuda                      # settings saved from the GUI
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .config import EnvConfig
from .env import DriftBatchEnv
from .evaluation import DEFAULT_LEVELS, GripSweep

GRIP_MEAN, GRIP_STD = 0.3, 0.15          # scaling of the grip estimate (friction coefficient)


@dataclass
class PPOConfig:
    total_steps: int = 20_000_000       # environment steps (cars x control steps)
    num_envs: int = 1024
    rollout: int = 32                   # control steps per car per iteration
    epochs: int = 5
    minibatches: int = 8
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    entropy: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    init_log_std: float = -0.5
    hidden: tuple = field(default=(256, 256))
    seed: int = 0
    lr_schedule: str = "constant"       # constant | linear (decays to 0 at the end)
    grip_estimator: bool = False        # learn a grip estimate from the sensor history, feed it to policy + filter
    estimator_hidden: tuple = field(default=(128, 64))
    estimator_coef: float = 1.0         # weight of the grip-regression loss
    privileged_critic: bool = False     # the critic also sees the simulator state (training only)
    eval_every: int = 0                 # iterations between grip-sweep evaluations (0 = off); keeps best.pt
    eval_cars: int = 8                  # cars per grip level in an evaluation
    eval_grip_levels: tuple = field(default=DEFAULT_LEVELS)


class RunningNorm(nn.Module):
    """Observation normalization with running mean / variance (parallel Welford update)."""

    def __init__(self, n: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def _mlp(sizes: list[int], out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, out_gain if last else 2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin] if last else [lin, nn.Tanh()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Policy (Gaussian MLP), critic and, optionally, the grip estimator and a privileged critic input.

    ``estimator`` (hidden sizes, empty = none): an MLP from the normalized observation (the sensor
    history) to the grip; its estimate, detached, is an extra input of the policy. ``n_priv`` > 0:
    the critic also gets the (normalized) privileged vector."""

    def __init__(self, n_obs: int, n_act: int = 2, hidden: tuple = (256, 256), init_log_std: float = -0.5,
                 n_priv: int = 0, estimator: tuple = ()):
        super().__init__()
        self.norm = RunningNorm(n_obs)
        self.n_priv = int(n_priv)
        self.priv_norm = RunningNorm(self.n_priv) if self.n_priv else None
        self.est = _mlp([n_obs, *estimator, 1], 1.0) if estimator else None
        self.actor = _mlp([n_obs + (1 if self.est is not None else 0), *hidden, n_act], 0.01)
        self.critic = _mlp([n_obs + self.n_priv, *hidden, 1], 1.0)
        self.log_std = nn.Parameter(torch.full((n_act,), float(init_log_std)))

    def grip(self, obs_n: torch.Tensor) -> torch.Tensor | None:
        """Estimated friction coefficient (B,) from normalized observations (None without estimator)."""
        if self.est is None:
            return None
        return GRIP_MEAN + GRIP_STD * self.est(obs_n).squeeze(-1)

    def _actor_in(self, obs_n: torch.Tensor, grip: torch.Tensor | None) -> torch.Tensor:
        if self.est is None:
            return obs_n
        g = self.grip(obs_n) if grip is None else grip
        g = torch.clamp(g.detach(), 0.0, 1.5)
        return torch.cat([obs_n, ((g - GRIP_MEAN) / GRIP_STD)[:, None]], dim=1)

    def dist(self, obs_n: torch.Tensor, grip: torch.Tensor | None = None) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(self._actor_in(obs_n, grip)), self.log_std.exp())

    def value(self, obs_n: torch.Tensor, priv_n: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([obs_n, priv_n], dim=1) if self.n_priv else obs_n
        return self.critic(x).squeeze(-1)

    def _obs(self, obs: Any) -> torch.Tensor:
        p = next(self.parameters())
        return torch.as_tensor(np.asarray(obs) if not isinstance(obs, torch.Tensor) else obs).to(p.device, torch.float32)

    @torch.no_grad()
    def act(self, obs: Any, deterministic: bool = True) -> torch.Tensor:
        """Action in [-1, 1] for raw observations (NumPy or tensor), e.g. for evaluation or deployment."""
        return self.act_with_grip(obs, deterministic)[0]

    @torch.no_grad()
    def act_with_grip(self, obs: Any, deterministic: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(action in [-1, 1], grip estimate or None) for raw observations; pass the grip to
        ``env.step(actions, grip=...)`` so the safety filter uses it, as on the car."""
        on = self.norm(self._obs(obs))
        g = self.grip(on)
        d = self.dist(on, g)
        return torch.clamp(d.mean if deterministic else d.sample(), -1.0, 1.0), g

    def policy_function(self) -> Callable[[Any], Any]:
        """``obs -> (actions, grip)`` with the grip only when the model has an estimator."""
        return self.act_with_grip if self.est is not None else self.act


def _to(x: Any, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def train(env_cfg: EnvConfig, ppo: PPOConfig | None = None, device: str = "cpu", out: str | Path | None = None,
          log: Callable[[dict], None] | None = print, stop: Callable[[], bool] | None = None,
          on_iteration: Callable[[dict, "ActorCritic"], None] | None = None,
          precision: str = "float32") -> tuple[ActorCritic, list[dict]]:
    """Train a policy; returns ``(model, history)`` (one dict per iteration).

    ``stop()`` is checked every control step (training ends early, keeping the last saved policy);
    ``on_iteration(row, model)`` runs after each iteration (the GUI records policy snapshots there).
    Rows of evaluated iterations carry ``row["eval"]`` (``rl.evaluation`` result, ``best`` flag)."""
    ppo = ppo or PPOConfig()
    torch.manual_seed(ppo.seed)
    env = DriftBatchEnv(ppo.num_envs, env_cfg, device=device, precision=precision, privileged=ppo.privileged_critic)
    tdev = env.device or "cpu"
    B, R = env.num_envs, ppo.rollout
    model = ActorCritic(env.n_obs, 2, tuple(ppo.hidden), ppo.init_log_std,
                        n_priv=env.n_priv if ppo.privileged_critic else 0,
                        estimator=tuple(ppo.estimator_hidden) if ppo.grip_estimator else ()).to(tdev)
    opt = torch.optim.Adam(model.parameters(), lr=ppo.lr, eps=1e-5)
    out = Path(out) if out is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    obs = _to(env.last_obs, tdev)                     # the env started its episode sequence from env_cfg.seed
    priv = _to(env.last_priv, tdev) if model.n_priv else None
    n_iter = max(1, ppo.total_steps // (B * R))
    shapes = dict(obs=(env.n_obs,), act=(2,), logp=(), rew=(), done=(), val=(), grip=(), over=())
    if model.n_priv:
        shapes["priv"] = (env.n_priv,)
    buf = {k: torch.zeros((R, B) + s, device=tdev) for k, s in shapes.items()}
    sweep = GripSweep(env_cfg, ppo.eval_grip_levels, ppo.eval_cars, device=device, precision=precision) \
        if ppo.eval_every else None
    best_score = -float("inf")
    history: list[dict] = []
    t_start, steps = time.time(), 0
    for it in range(1, n_iter + 1):
        ep_returns: list[float] = []
        ep_lengths: list[float] = []
        ep_early: list[float] = []
        grip_err: list[float] = []
        if stop is not None and stop():
            break
        if ppo.lr_schedule == "linear":
            for grp in opt.param_groups:
                grp["lr"] = ppo.lr * max(1.0 - (it - 1) / n_iter, 0.02)
        for t in range(R):
            if stop is not None and stop():
                break
            with torch.no_grad():
                model.norm.update(obs)
                on = model.norm(obs)
                pn = None
                if model.n_priv:
                    model.priv_norm.update(priv)
                    pn = model.priv_norm(priv)
                    buf["priv"][t] = pn
                g = model.grip(on)
                d = model.dist(on, g)
                a = d.sample()
                buf["obs"][t], buf["act"][t] = on, a
                buf["logp"][t], buf["val"][t] = d.log_prob(a).sum(-1), model.value(on, pn)
                g_true = _to(env.last_grip, tdev)
                buf["grip"][t] = g_true
                if g is not None:
                    grip_err.append(float((g - g_true).abs().mean()))
            g_env = None if g is None else (g if env.device is not None else g.cpu().numpy().astype(np.float64))
            o2, r, te, tr, info = env.step(a if env.device is not None else a.cpu().numpy().astype(np.float64), grip=g_env)
            r, te, tr = _to(r, tdev).clone(), _to(te, tdev), _to(tr, tdev)
            if "override" in info:
                buf["over"][t] = (_to(info["override"], tdev) > 0.01).float()
            if "final_idx" in info:                      # bootstrap episodes cut by the time limit
                idx = torch.as_tensor(info["final_idx"], device=tdev)
                cut = tr[idx] > 0
                if bool(cut.any()):
                    with torch.no_grad():
                        fp = model.priv_norm(_to(info["final_priv"], tdev)[cut]) if model.n_priv else None
                        v_last = model.value(model.norm(_to(info["final_obs"], tdev)[cut]), fp)
                    r[idx[cut]] += ppo.gamma * v_last
                ret = _to(info["episode_return"], tdev)[idx]
                ep_returns += ret.tolist()
                ep_lengths += _to(info["episode_length"], tdev)[idx].tolist()
                ep_early += te[idx].tolist()
            buf["rew"][t], buf["done"][t] = r, torch.clamp(te + tr, 0.0, 1.0)
            obs = _to(o2, tdev)
            if model.n_priv:
                priv = _to(env.last_priv, tdev)
        if stop is not None and stop():               # stopped inside the rollout: discard the partial batch
            break
        steps += B * R
        with torch.no_grad():                            # GAE
            next_v = model.value(model.norm(obs), model.priv_norm(priv) if model.n_priv else None)
            adv = torch.zeros_like(buf["rew"])
            last = torch.zeros(B, device=tdev)
            for t in reversed(range(R)):
                nv = next_v if t == R - 1 else buf["val"][t + 1]
                nonterm = 1.0 - buf["done"][t]
                delta = buf["rew"][t] + ppo.gamma * nv * nonterm - buf["val"][t]
                last = delta + ppo.gamma * ppo.lam * nonterm * last
                adv[t] = last
            ret_t = adv + buf["val"]
        flat = {k: v.reshape((R * B,) + v.shape[2:]) for k, v in buf.items()}
        flat_adv, flat_ret = adv.reshape(-1), ret_t.reshape(-1)
        n = R * B
        mb = max(1, n // ppo.minibatches)
        kls, clipfracs, est_losses = [], [], []
        for _ in range(ppo.epochs):
            perm = torch.randperm(n, device=tdev)
            for s0 in range(0, n, mb):
                j = perm[s0:s0 + mb]
                o_j = flat["obs"][j]
                g_j = model.grip(o_j)
                d = model.dist(o_j, g_j)
                logp = d.log_prob(flat["act"][j]).sum(-1)
                ratio = torch.exp(logp - flat["logp"][j])
                a_j = flat_adv[j]
                a_j = (a_j - a_j.mean()) / (a_j.std() + 1e-8)
                pg = torch.max(-a_j * ratio, -a_j * torch.clamp(ratio, 1 - ppo.clip, 1 + ppo.clip)).mean()
                v_loss = 0.5 * ((model.value(o_j, flat["priv"][j] if model.n_priv else None) - flat_ret[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + ppo.value_coef * v_loss - ppo.entropy * ent
                if g_j is not None:
                    est_loss = ((g_j - flat["grip"][j]) ** 2).mean()
                    loss = loss + ppo.estimator_coef * est_loss / GRIP_STD ** 2
                    est_losses.append(float(est_loss.detach()))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kls.append(float(((ratio - 1) - torch.log(ratio)).mean()))
                    clipfracs.append(float(((ratio - 1).abs() > ppo.clip).float().mean()))
        row = dict(iteration=it, env_steps=steps, seconds=round(time.time() - t_start, 1),
                   steps_per_s=round(steps / max(time.time() - t_start, 1e-9)),
                   reward_per_step=round(float(buf["rew"].mean()), 4),
                   episodes=len(ep_returns), episode_return=round(float(np.mean(ep_returns)), 2) if ep_returns else None,
                   episode_length=round(float(np.mean(ep_lengths)), 1) if ep_lengths else None,
                   spin_share=round(float(np.mean(ep_early)), 3) if ep_early else None,
                   override_share=round(float(buf["over"].mean()), 4) if env_cfg.safety.enabled else None,
                   grip_error=round(float(np.mean(grip_err)), 4) if grip_err else None,
                   value_loss=round(float(v_loss.detach()), 4), approx_kl=round(float(np.mean(kls)), 5),
                   clip_frac=round(float(np.mean(clipfracs)), 3), action_std=[round(float(x), 3) for x in model.log_std.detach().exp()],
                   lr=float(f"{opt.param_groups[0]['lr']:.3g}"))
        if out is not None:
            save_policy(model, out / "policy.pt", env_cfg, ppo, env.obs_names, env, iteration=it)
        if sweep is not None and (it % ppo.eval_every == 0 or it == n_iter):
            model.eval()
            res = sweep.run(model.policy_function())
            model.train()
            res["best"] = res["score"] > best_score
            if res["best"]:
                best_score = res["score"]
                if out is not None:
                    save_policy(model, out / "best.pt", env_cfg, ppo, env.obs_names, env, iteration=it, evaluation=res)
            row["eval"] = res
        history.append(row)
        if log is not None:
            log(row)
        if out is not None:
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        if on_iteration is not None:
            on_iteration(row, model)
    return model, history


def save_policy(model: ActorCritic, path: str | Path, env_cfg: EnvConfig, ppo: PPOConfig, obs_names: list[str],
                env: DriftBatchEnv | None = None, iteration: int | None = None, evaluation: dict | None = None) -> None:
    """Weights + everything needed to rebuild and deploy the policy; written atomically."""
    extra = {}
    if env is not None:
        extra = dict(priv_names=list(env.priv_names), control_dt=env.control_dt)
        if env_cfg.obs == "sensors":
            extra["obs_spec"] = env.obs_spec()
    ck = dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, env_config=env_cfg.to_dict(),
              ppo=dataclasses.asdict(ppo), obs_names=list(obs_names), iteration=iteration, evaluation=evaluation,
              arch=dict(n_priv=model.n_priv, estimator=list(ppo.estimator_hidden) if model.est is not None else []),
              **extra)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(path)


def _ppo_from(d: dict) -> PPOConfig:
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    d = {k: v for k, v in d.items() if k in known}
    for k in ("hidden", "estimator_hidden", "eval_grip_levels"):
        if k in d:
            d[k] = tuple(d[k])
    return PPOConfig(**d)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ActorCritic, EnvConfig, PPOConfig, dict]:
    """(model, env config, PPO config, the raw checkpoint dict) of a saved policy."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = EnvConfig(**ck["env_config"])
    ppo = _ppo_from(ck["ppo"])
    arch = ck.get("arch") or {}
    model = ActorCritic(len(ck["obs_names"]), 2, ppo.hidden, ppo.init_log_std, n_priv=int(arch.get("n_priv", 0)),
                        estimator=tuple(arch.get("estimator", ())))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg, ppo, ck


def load_policy(path: str | Path, device: str = "cpu") -> tuple[ActorCritic, EnvConfig]:
    model, cfg, _, _ = load_checkpoint(path, device)
    return model, cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="driftsim-train", description="Train a drift policy with PPO.")
    ap.add_argument("--preset", default=None, help="a ready-made setup, e.g. safe-adaptive (see rl.presets)")
    ap.add_argument("--config", default=None, help="settings JSON saved from the GUI's RL training page")
    ap.add_argument("--task", default=None, choices=["hold", "track"])
    ap.add_argument("--steps", type=float, default=None, help="environment steps (cars x control steps)")
    ap.add_argument("--envs", type=int, default=None, help="cars simulated together")
    ap.add_argument("--device", default=None, help="cpu (NumPy physics), mps, cuda or auto")
    ap.add_argument("--obs", default=None, choices=["sensors", "full"])
    ap.add_argument("--randomize", default=None, help="JSON file with a datagen 'params' dict (domain randomization)")
    ap.add_argument("--init-drift-prob", type=float, default=None, help="share of episodes starting in a drift")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="runs/ppo", help="folder for log.jsonl, policy.pt and best.pt")
    ap.add_argument("--export", action="store_true", help="export the final model (best.pt, else policy.pt) at the end")
    a = ap.parse_args(argv)
    from .catalog import build_configs
    from .presets import preset_body
    body: dict = {}
    if a.preset:
        body = preset_body(a.preset)
    if a.config:
        body = json.loads(Path(a.config).read_text())
    body = json.loads(json.dumps(body))                  # private copy
    env_part, ppo_part, run_part = body.setdefault("env", {}), body.setdefault("ppo", {}), body.setdefault("run", {})
    for key, val in (("task", a.task), ("obs", a.obs), ("init_drift_prob", a.init_drift_prob), ("seed", a.seed)):
        if val is not None:
            env_part[key] = val
    if a.steps is not None:
        ppo_part["total_steps"] = int(a.steps)
    if a.envs is not None:
        ppo_part["num_envs"] = a.envs
    if a.seed is not None:
        ppo_part["seed"] = a.seed
    if a.device is not None:
        run_part["device"] = a.device
    run_part.setdefault("device", "cpu")
    if a.randomize:
        rnd = json.loads(Path(a.randomize).read_text())
        body["randomize"] = rnd.get("params", rnd)        # accept a whole datagen spec too
    cfg, ppo_kw, run, errors = build_configs(body)
    if errors:
        print("invalid settings:\n  " + "\n  ".join(errors))
        return 2
    ppo = PPOConfig(**ppo_kw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(body, indent=1))

    def show(row: dict) -> None:
        extra = ""
        if row.get("override_share") is not None:
            extra += f"  safety {100 * row['override_share']:.1f}%"
        if row.get("grip_error") is not None:
            extra += f"  grip err {row['grip_error']:.3f}"
        if row.get("eval"):
            e = row["eval"]
            extra += f"  | eval {e['score']} spins {100 * e['spin_share']:.0f}%" + (" (best)" if e["best"] else "")
        print(f"it {row['iteration']:4d}  steps {row['env_steps'] / 1e6:7.2f}M  {row['steps_per_s'] / 1e3:6.1f}k/s  "
              f"return {row['episode_return']}  len {row['episode_length']}  r/step {row['reward_per_step']:+.3f}  "
              f"kl {row['approx_kl']:.4f}  std {row['action_std']}{extra}", flush=True)
    train(cfg, ppo, device=run["device"], precision=run["precision"], out=out, log=show)
    print(f"saved {out / 'policy.pt'}" + (f" and {out / 'best.pt'}" if (out / "best.pt").is_file() else ""))
    if a.export:
        from .export import export_final
        res = export_final(out)
        print(f"exported the final model to {res['folder']} ({res['zip']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
