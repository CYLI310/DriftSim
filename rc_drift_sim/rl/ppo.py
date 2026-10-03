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
    def __init__(self, n_obs: int, n_act: int = 2, hidden: tuple = (256, 256), init_log_std: float = -0.5):
        super().__init__()
        self.norm = RunningNorm(n_obs)
        self.actor = _mlp([n_obs, *hidden, n_act], 0.01)
        self.critic = _mlp([n_obs, *hidden, 1], 1.0)
        self.log_std = nn.Parameter(torch.full((n_act,), float(init_log_std)))

    def dist(self, obs_n: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs_n), self.log_std.exp())

    def value(self, obs_n: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_n).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: Any, deterministic: bool = True) -> torch.Tensor:
        """Action in [-1, 1] for raw observations (NumPy or tensor), e.g. for evaluation or deployment."""
        p = next(self.parameters())
        o = torch.as_tensor(np.asarray(obs) if not isinstance(obs, torch.Tensor) else obs).to(p.device, torch.float32)
        d = self.dist(self.norm(o))
        return torch.clamp(d.mean if deterministic else d.sample(), -1.0, 1.0)


def _to(x: Any, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def train(env_cfg: EnvConfig, ppo: PPOConfig | None = None, device: str = "cpu", out: str | Path | None = None,
          log: Callable[[dict], None] | None = print) -> tuple[ActorCritic, list[dict]]:
    """Train a policy; returns ``(model, history)`` (one dict per iteration)."""
    ppo = ppo or PPOConfig()
    torch.manual_seed(ppo.seed)
    env = DriftBatchEnv(ppo.num_envs, env_cfg, device=device)
    tdev = env.device or "cpu"
    B, R = env.num_envs, ppo.rollout
    model = ActorCritic(env.n_obs, 2, tuple(ppo.hidden), ppo.init_log_std).to(tdev)
    opt = torch.optim.Adam(model.parameters(), lr=ppo.lr, eps=1e-5)
    out = Path(out) if out is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    obs = _to(env.reset(seed=ppo.seed), tdev)
    n_iter = max(1, ppo.total_steps // (B * R))
    buf = {k: torch.zeros((R, B) + s, device=tdev) for k, s in
           dict(obs=(env.n_obs,), act=(2,), logp=(), rew=(), done=(), val=()).items()}
    history: list[dict] = []
    t_start, steps = time.time(), 0
    for it in range(1, n_iter + 1):
        ep_returns: list[float] = []
        ep_lengths: list[float] = []
        for t in range(R):
            with torch.no_grad():
                model.norm.update(obs)
                on = model.norm(obs)
                d = model.dist(on)
                a = d.sample()
                buf["obs"][t], buf["act"][t] = on, a
                buf["logp"][t], buf["val"][t] = d.log_prob(a).sum(-1), model.value(on)
            o2, r, te, tr, info = env.step(a if env.device is not None else a.cpu().numpy().astype(np.float64))
            r, te, tr = _to(r, tdev).clone(), _to(te, tdev), _to(tr, tdev)
            if "final_idx" in info:                      # bootstrap episodes cut by the time limit
                idx = torch.as_tensor(info["final_idx"], device=tdev)
                cut = tr[idx] > 0
                if bool(cut.any()):
                    with torch.no_grad():
                        v_last = model.value(model.norm(_to(info["final_obs"], tdev)[cut]))
                    r[idx[cut]] += ppo.gamma * v_last
                ret = _to(info["episode_return"], tdev)[idx]
                ep_returns += ret.tolist()
                ep_lengths += _to(info["episode_length"], tdev)[idx].tolist()
            buf["rew"][t], buf["done"][t] = r, torch.clamp(te + tr, 0.0, 1.0)
            obs = _to(o2, tdev)
        steps += B * R
        with torch.no_grad():                            # GAE
            next_v = model.value(model.norm(obs))
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
        kls, clipfracs = [], []
        for _ in range(ppo.epochs):
            perm = torch.randperm(n, device=tdev)
            for s0 in range(0, n, mb):
                j = perm[s0:s0 + mb]
                d = model.dist(flat["obs"][j])
                logp = d.log_prob(flat["act"][j]).sum(-1)
                ratio = torch.exp(logp - flat["logp"][j])
                a_j = flat_adv[j]
                a_j = (a_j - a_j.mean()) / (a_j.std() + 1e-8)
                pg = torch.max(-a_j * ratio, -a_j * torch.clamp(ratio, 1 - ppo.clip, 1 + ppo.clip)).mean()
                v_loss = 0.5 * ((model.value(flat["obs"][j]) - flat_ret[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + ppo.value_coef * v_loss - ppo.entropy * ent
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
                   value_loss=round(float(v_loss.detach()), 4), approx_kl=round(float(np.mean(kls)), 5),
                   clip_frac=round(float(np.mean(clipfracs)), 3), action_std=[round(float(x), 3) for x in model.log_std.detach().exp()])
        history.append(row)
        if log is not None:
            log(row)
        if out is not None:
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            save_policy(model, out / "policy.pt", env_cfg, ppo, env.obs_names)
    return model, history


def save_policy(model: ActorCritic, path: str | Path, env_cfg: EnvConfig, ppo: PPOConfig, obs_names: list[str]) -> None:
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, env_config=env_cfg.to_dict(),
                    ppo=dataclasses.asdict(ppo), obs_names=list(obs_names)), path)


def load_policy(path: str | Path, device: str = "cpu") -> tuple[ActorCritic, EnvConfig]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = EnvConfig(**ck["env_config"])
    ppo = PPOConfig(**{**ck["ppo"], "hidden": tuple(ck["ppo"]["hidden"])})
    model = ActorCritic(len(ck["obs_names"]), 2, ppo.hidden, ppo.init_log_std)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="driftsim-train", description="Train a drift policy with PPO.")
    ap.add_argument("--task", default="hold", choices=["hold", "track"])
    ap.add_argument("--steps", type=float, default=20e6, help="environment steps (cars x control steps)")
    ap.add_argument("--envs", type=int, default=1024, help="cars simulated together")
    ap.add_argument("--device", default="cpu", help="cpu (NumPy physics), mps, cuda or auto")
    ap.add_argument("--obs", default="sensors", choices=["sensors", "full"])
    ap.add_argument("--randomize", default=None, help="JSON file with a datagen 'params' dict (domain randomization)")
    ap.add_argument("--init-drift-prob", type=float, default=0.0, help="share of episodes starting in a drift")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ppo", help="folder for log.jsonl and policy.pt")
    a = ap.parse_args(argv)
    rnd = json.loads(Path(a.randomize).read_text()) if a.randomize else {}
    rnd = rnd.get("params", rnd)                      # accept a whole datagen spec too
    cfg = EnvConfig(task=a.task, obs=a.obs, randomize=rnd, init_drift_prob=a.init_drift_prob, seed=a.seed)
    ppo = PPOConfig(total_steps=int(a.steps), num_envs=a.envs, seed=a.seed)

    def show(row: dict) -> None:
        print(f"it {row['iteration']:4d}  steps {row['env_steps'] / 1e6:7.2f}M  {row['steps_per_s'] / 1e3:6.1f}k/s  "
              f"return {row['episode_return']}  len {row['episode_length']}  r/step {row['reward_per_step']:+.3f}  "
              f"kl {row['approx_kl']:.4f}  std {row['action_std']}", flush=True)
    train(cfg, ppo, device=a.device, out=a.out, log=show)
    print(f"saved {Path(a.out) / 'policy.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
