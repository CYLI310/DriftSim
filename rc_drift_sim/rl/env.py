"""Batched drift RL environment: B cars stepped together, on NumPy (the float64 reference) or on a
GPU through PyTorch. This is the core the Gymnasium wrappers (``rl.gym_env``) are built on; GPU
training loops can use it directly (observations, rewards and flags stay torch tensors on the device).

One step = one control period (``sim.control_dt``, 20 ms): the commanded action enters a per-car
latency line (``actuators.latency``, as in the batch export), then the physics runs ``n_substeps``
RK4 steps. Rewards are computed from the true state; observations can be noisy sensor readings
(``obs="sensors"``) or the simulator state (``obs="full"``, for asymmetric critics or debugging).

    env = DriftBatchEnv(256, EnvConfig(task="hold", randomize={"vehicle.mass": {...}}))
    obs = env.reset(seed=0)                      # (256, n_obs)
    obs, reward, terminated, truncated, info = env.step(actions)    # actions (256, 2) in [-1, 1]

Finished cars are reset automatically (``autoreset=True``): the returned observation is the first one
of the next episode and ``info["final_obs"]`` holds the last one of the finished episode, for the cars
in ``info["done"]``. Each new episode draws its car from ``config.randomize`` with the batch-export
sampler, so episode ``k`` of seed ``s`` is the same car in every run.

Safe, self-adapting sliding: with ``config.safety.enabled`` the commands pass through
``deploy.safety.safety_filter`` (fed with the same noisy readings as the policy and, if given,
``step(actions, grip=...)``, the policy's grip estimate) and the change it made is penalized;
``config.history`` > 1 stacks the last readings into the observation; ``info["grip"]`` is the true
effective friction coefficient (the target of a grip estimator) and, with ``privileged=True``,
``info["priv"]`` / ``env.last_priv`` the simulator state for an asymmetric critic.
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any

import numpy as np

from ..datagen.sampling import Episode, Sampler
from ..datagen.spec import default_spec, validate_spec
from ..sim import integrator
from ..sim import state as S
from ..sim.vehicle import VehicleBatch, derivatives_model, rhs_model
from ..deploy.safety import safety_filter
from ..sim.xp import TORCH, resolve_device, to_device, to_numpy, torch_dtype
from .config import MAX_LATENCY_STEPS, EnvConfig

MOVING = 0.3                     # m/s: below this the sideslip is not meaningful (no reward, no spin)
_LEFT, _RIGHT = [0, 2], [1, 3]   # wheel columns (FL, RL) and (FR, RR)


# ----------------------------------------------------------------------------- helpers
def mirror_state(s: np.ndarray) -> np.ndarray:
    """Left-right mirror image of states (..., NS): lateral quantities change sign, left and right
    wheels swap. Turns a left-hand drift into the same right-hand drift."""
    m = np.array(s, dtype=np.float64, copy=True)
    for k in (S.Y, S.YAW, S.VY, S.R, S.DELTA, S.DFZ_LAT):
        m[..., k] = -m[..., k]
    for sl, sign in ((S.OMEGA, 1.0), (S.T_TIRE, 1.0), (S.KAPPA_LAG, 1.0), (S.ALPHA_LAG, -1.0), (S.CONTAM, 1.0)):
        w = m[..., sl].copy()
        m[..., sl] = sign * w[..., [1, 0, 3, 2]]
    return m


def _assign_rows(dst: Any, src: Any, idx: Any, B: int) -> None:
    """Copy the per-car rows of a compiled sub-batch model into the batch model (in place)."""
    if isinstance(dst, np.ndarray):
        if dst.ndim >= 1 and dst.shape[0] == B and src.shape[0] == len(idx):
            dst[idx] = src
        return
    if type(dst).__module__.startswith("torch"):
        if dst.ndim >= 1 and dst.shape[0] == B and len(src) == len(idx):
            import torch
            dst[idx] = torch.as_tensor(np.asarray(src), dtype=dst.dtype, device=dst.device)
        return
    if dataclasses.is_dataclass(dst) and not type(dst).__module__.endswith(".params"):
        for f in dataclasses.fields(dst):
            _assign_rows(getattr(dst, f.name), getattr(src, f.name), idx, B)
    elif isinstance(dst, tuple) and hasattr(dst, "_fields"):
        for a, b in zip(dst, src):
            _assign_rows(a, b, idx, B)


def nominal_params(cfg: EnvConfig):
    """The nominal car of a config: its fixed randomize values applied, the random ones at their defaults."""
    spec = default_spec()
    spec["params"] = {k: v for k, v in cfg.randomize.items() if isinstance(v, dict) and v.get("dist") == "fixed"}
    spec, errors, _ = validate_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    return Sampler(spec).episode(0).params


def _car_flags(ep: Episode) -> dict[str, bool]:
    """The per-car configuration flags the batch model combines (any / all)."""
    p = ep.params
    return dict(has_plow=bool(np.any(np.asarray(p.surface.loose_drag) != 0.0)), has_pkx3=float(p.tire.pkx3) != 0.0,
                has_drag_brake=float(p.drivetrain.drag_brake) != 0.0, parallel_steer=float(p.vehicle.ackermann) == 0.0)


# ----------------------------------------------------------------------------- environment
class DriftBatchEnv:
    """B drift-RL environments in one vectorized simulation (see the module docstring)."""

    def __init__(self, num_envs: int = 1, config: EnvConfig | None = None, device: str = "cpu",
                 precision: str = "float32", autoreset: bool = True, privileged: bool = False, **overrides: Any):
        """device: "cpu" (NumPy float64 reference), "mps", "cuda", "auto" (best GPU, else NumPy) or
        "torch-cpu"; precision applies to the PyTorch devices. privileged: also compute the
        simulator-state vector ``last_priv`` / ``info["priv"]`` (for an asymmetric critic)."""
        cfg = config if config is not None else EnvConfig()
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        cfg.validate()
        self.cfg, self.num_envs, self.autoreset = cfg, int(num_envs), bool(autoreset)
        self.privileged = bool(privileged)
        if device == "torch-cpu":                    # PyTorch on the CPU (tests, debugging)
            self.device = "cpu"
        else:
            dev = None if device == "cpu" else resolve_device(device)
            self.device = None if dev == "cpu" else dev
        self.xp = np if self.device is None else TORCH
        self.dtype = np.float64 if self.device is None else torch_dtype(precision)
        spec = default_spec()
        spec.update(name="rl", seed=int(cfg.seed), episodes=1, duration_s=float(cfg.episode_s))
        spec["params"] = dict(cfg.randomize)
        self._spec, errors, _ = validate_spec(spec)
        if errors:
            raise ValueError("invalid randomize: " + "; ".join(errors))
        self._use_init = any(k.startswith("init.") for k in cfg.randomize)
        self._trim = None
        self._trim_tried = False
        self.H = int(cfg.history)
        self.dyn_names, self.task_names = self._feature_names()
        self.obs_names = self._obs_names()
        self.n_obs = len(self.obs_names)
        self.priv_names = self._priv_names()
        self.n_priv = len(self.priv_names)
        self.reset(seed=cfg.seed)

    # ---------------------------------------------------------------- arrays
    def _arr(self, a, dtype=None):
        if self.device is None:
            return np.asarray(to_numpy(a), dtype=np.float64 if dtype is None else dtype)
        import torch
        dtype = self.dtype if dtype is None else dtype
        if isinstance(a, torch.Tensor):
            return a.to(device=self.device, dtype=dtype)
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)

    def _iarr(self, a):
        """Integer array (per-car step counters and latencies)."""
        if self.device is None:
            return np.asarray(a, dtype=np.int64)
        import torch
        return torch.as_tensor(np.asarray(a), dtype=torch.long, device=self.device)

    def _idx(self, idx: np.ndarray):
        if self.device is None:
            return idx
        import torch
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    def _noise(self, key: str, shape: tuple) -> Any:
        std = float(self.cfg.noise.get(key, 0.0))
        if std == 0.0:
            return 0.0
        if self.device is None:
            return self._rng.normal(0.0, std, shape)
        import torch
        return torch.randn(shape, generator=self._tgen, dtype=self.dtype, device=self._tgen_device).to(self.device) * std

    # ---------------------------------------------------------------- episodes
    def _draw(self) -> Episode:
        ep = self._sampler.episode(self._episode_counter)
        self._episode_counter += 1
        return ep

    def reset(self, seed: int | None = None):
        """Start new episodes in every car (``seed`` restarts the episode sequence); returns obs."""
        if seed is not None:
            self._spec = dict(self._spec, seed=int(seed))
            self._episode_counter = 0
            self._rng = np.random.default_rng([int(seed), 0x0B5])
            if self.device is not None:
                import torch
                self._tgen_device = self.device if self.device == "cuda" else "cpu"
                self._tgen = torch.Generator(device=self._tgen_device)
                self._tgen.manual_seed(int(seed))
        self._sampler = Sampler(self._spec)
        B = self.num_envs
        eps = [self._draw() for _ in range(B)]
        self._signature = eps[0].signature
        for e in eps:
            if e.signature != self._signature:
                raise ValueError("randomize must not change structural settings across episodes")
        vb = VehicleBatch([e.params for e in eps], conds=[e.cond for e in eps], check_stiffness=False)
        self.dt, self.control_dt, self.n_substeps = vb.dt, vb.control_dt, vb.n_substeps
        self.method = str(vb.params.sim.integrator)
        self.max_steps = max(1, int(round(self.cfg.episode_s / self.control_dt)))
        self.model = vb.model if self.device is None else to_device(vb.model, self.device, self.dtype)
        self._flags = {k: np.array([_car_flags(e)[k] for e in eps]) for k in _car_flags(eps[0])}
        z = lambda *shape: self._arr(np.zeros(shape))  # noqa: E731
        self.state = z(B, S.NS)
        self._hist = z(MAX_LATENCY_STEPS + 1, B, 2)
        self._ptr = 0
        self._delay = self._iarr(np.zeros(B))
        self._ar = self._idx(np.arange(B))
        self.prev_action = z(B, 2)
        self.t_step = self._iarr(np.zeros(B))
        self._t_cpu = np.zeros(B, dtype=np.int64)          # CPU copies: no GPU sync for the grip changes
        self._gc_cpu = np.full(B, -1, dtype=np.int64)
        self.direction = z(B)
        self.center = z(B, 2)
        self.target_beta = z(B)
        self.target_speed = z(B)
        self.override = z(B)
        self._gc_step = self._iarr(np.full(B, -1))
        self._gc_factor = z(B)
        self._steer_max = self.model.am.steer_max * 1.0
        self._lf = self.model.xi[:, 0] * 1.0
        self.episode_return = z(B)
        self.episode_id = np.zeros(B, dtype=np.int64)
        self._writable_grip()
        self._start_rows(np.arange(B), eps, vb)
        _, info = derivatives_model(self.state, self._applied(), self.model, want_info=True)
        dyn, task, self._meas = self._features(self.state, info)
        self._hist_obs = self._arr(np.zeros((B, self.H, len(self.dyn_names))))
        self._hist_obs[:] = dyn[:, None, :]
        self.last_obs = self._assemble(task)
        self.last_grip, self.last_priv = self._privileged(self.state, info)
        return self.last_obs

    def _start_rows(self, idx: np.ndarray, eps: list[Episode], vb: VehicleBatch | None = None) -> None:
        """Begin episodes ``eps`` in cars ``idx``: car model rows, initial state, latency, task state."""
        k = len(idx)
        if vb is None:
            for e in eps:
                if e.signature != self._signature:
                    raise ValueError("randomize must not change structural settings across episodes")
            vb = VehicleBatch([e.params for e in eps], conds=[e.cond for e in eps], check_stiffness=False)
            _assign_rows(self.model, vb.model, self._idx(idx), self.num_envs)
            for e, i in zip(eps, idx):
                for name, val in _car_flags(e).items():
                    self._flags[name][i] = val
            self._refresh_flags()
            ii = self._idx(idx)
            self._steer_max[ii] = self._arr(np.asarray(vb.model.am.steer_max, dtype=np.float64).reshape(k))
            self._lf[ii] = self._arr(np.asarray(vb.model.xi, dtype=np.float64)[:, 0])
        cfg = self.cfg
        rngs = [np.random.default_rng([int(self._spec["seed"]), int(e.id), 0x5EED]) for e in eps]
        direction = np.array([(1.0 if r.random() < 0.5 else -1.0) if (cfg.task == "track" and cfg.track_both_ways)
                              or cfg.task == "hold" else 1.0 for r in rngs])
        if self._use_init:
            init = {key: np.array([e.init[key] for e in eps]) for key in ("x", "y", "yaw", "v", "beta", "yaw_rate")}
        else:
            init = dict(x=np.zeros(k), y=np.zeros(k), yaw=np.zeros(k), v=np.full(k, cfg.init_speed),
                        beta=np.zeros(k), yaw_rate=np.zeros(k))
        s0 = vb.initial_states(**init)
        drift = np.array([r.random() < cfg.init_drift_prob for r in rngs]) if cfg.init_drift_prob > 0 else np.zeros(k, bool)
        if drift.any() and self._drift_trim() is not None:
            trim_s = self._trim.s
            for j in np.flatnonzero(drift):
                st = trim_s.copy() if direction[j] > 0 else mirror_state(trim_s)
                st[[S.X, S.Y, S.YAW]] = s0[j, [S.X, S.Y, S.YAW]]
                if cfg.task == "track":                 # the velocity, not the nose, points along the line
                    st[S.YAW] = -math.atan2(st[S.VY], st[S.VX])
                st[S.T_TIRE], st[S.CONTAM] = s0[j, S.T_TIRE], s0[j, S.CONTAM]
                s0[j] = st
        center = np.zeros((k, 2))
        if cfg.task == "track":
            center[:, 1] = direction * cfg.track_radius   # the line starts at the origin heading +x
        tb = np.full(k, math.radians(cfg.target_beta_deg))
        ts = np.full(k, float(cfg.target_speed))
        gc_step, gc_factor = np.full(k, -1, dtype=np.int64), np.ones(k)
        for j, r in enumerate(rngs):                        # drawn after the draws above: same starts as before
            if cfg.target_beta_jitter_deg > 0:
                tb[j] += math.radians(cfg.target_beta_jitter_deg) * r.uniform(-1.0, 1.0)
            if cfg.target_speed_jitter > 0:
                ts[j] += cfg.target_speed_jitter * r.uniform(-1.0, 1.0)
            if cfg.grip_change_prob > 0 and r.random() < cfg.grip_change_prob:
                gc_step[j] = int(r.uniform(0.25, 0.75) * self.max_steps)
                gc_factor[j] = math.exp(r.uniform(math.log(cfg.grip_change_min), math.log(cfg.grip_change_max)))
        lat = np.array([float(e.params.actuators.latency) for e in eps])
        delay = np.floor(lat / self.control_dt + 0.5).astype(np.int64)
        if np.any(delay > MAX_LATENCY_STEPS):
            raise ValueError(f"actuators.latency above {MAX_LATENCY_STEPS * self.control_dt:.3f} s is not supported")
        ii = self._idx(idx)
        self.state[ii] = self._arr(s0)
        self._hist[:, ii] = 0.0
        self._delay[ii] = self._iarr(delay)
        self.prev_action[ii] = 0.0
        self.t_step[ii] = 0
        self._t_cpu[idx] = 0
        self._gc_cpu[idx] = gc_step
        self.direction[ii] = self._arr(direction)
        self.center[ii] = self._arr(center)
        self.target_beta[ii] = self._arr(tb)
        self.target_speed[ii] = self._arr(ts)
        self._gc_step[ii] = self._iarr(gc_step)
        self._gc_factor[ii] = self._arr(gc_factor)
        self.override[ii] = 0.0
        self.episode_return[ii] = 0.0
        self.episode_id[idx] = [e.id for e in eps]

    def _refresh_flags(self) -> None:
        m = self.model
        tm = dataclasses.replace(m.tm, has_plow=bool(self._flags["has_plow"].any()),
                                 has_pkx3=bool(self._flags["has_pkx3"].any()))
        dm = dataclasses.replace(m.dm, has_drag_brake=bool(self._flags["has_drag_brake"].any()))
        self.model = dataclasses.replace(m, tm=tm, dm=dm, parallel_steer=bool(self._flags["parallel_steer"].all()))

    def _writable_grip(self) -> None:
        """Own, writable per-car peak-friction multipliers (the mid-episode grip changes write them)."""
        tm = self.model.tm
        mu = tm.mu_scale
        if isinstance(mu, np.ndarray):
            mu = np.array(np.broadcast_to(mu, (self.num_envs, 4)), dtype=np.float64)
        else:
            mu = mu.expand(self.num_envs, 4).clone()
        self.model = dataclasses.replace(self.model, tm=dataclasses.replace(tm, mu_scale=mu))

    def _apply_grip_changes(self) -> None:
        hit = self._t_cpu == self._gc_cpu
        if hit.any():
            ii = self._idx(np.flatnonzero(hit))
            self.model.tm.mu_scale[ii] = self.model.tm.mu_scale[ii] * self._gc_factor[ii][:, None]

    def _drift_trim(self):
        """Steady drift of the first car at the target speed and sideslip (left-hand), for drift starts."""
        if not self._trim_tried:
            self._trim_tried = True
            from ..sim.equilibrium import solve_trim
            from ..sim.vehicle import Vehicle
            trim = solve_trim(Vehicle(nominal_params(self.cfg)), self.cfg.target_speed,
                              beta=-math.radians(self.cfg.target_beta_deg))
            if trim.success:
                self._trim = trim
            else:
                warnings.warn("no steady drift at the target speed and sideslip: init_drift_prob is ignored",
                              RuntimeWarning, stacklevel=3)
        return self._trim

    # ---------------------------------------------------------------- stepping
    def _applied(self):
        D = self._hist.shape[0]
        return self._hist[(self._ptr - self._delay) % D, self._ar]

    def step(self, action: Any, grip: Any = None):
        """Advance one control period. ``grip`` (B,): the policy's friction estimate for the safety
        filter (None: the filter's nominal grip)."""
        xp, w, cfg = self.xp, self.cfg.reward, self.cfg
        a_pol = xp.clip(self._arr(action), -1.0, 1.0)
        if cfg.safety.enabled:
            vx, vy, r = self._meas
            g = None if grip is None else self._arr(grip).reshape(-1)
            a, self.override = safety_filter(xp, a_pol, self.prev_action, vx, vy, r, g, cfg.safety,
                                             self._steer_max, self._lf)
            filt = xp.sum((a - a_pol) ** 2, axis=1)
        else:
            a, filt = a_pol, None
        if cfg.grip_change_prob > 0:
            self._apply_grip_changes()
        D = self._hist.shape[0]
        self._hist[self._ptr] = a
        u = self._applied()
        self._ptr = (self._ptr + 1) % D
        s = integrator.integrate(rhs_model, self.state, self.dt, self.n_substeps, self.method, u, self.model)
        finite = xp.all(xp.isfinite(s), axis=1)
        if not bool(finite.all()):                 # keep the batch finite; the episode ends below
            s = xp.where(finite[:, None], s, self.state)
        _, info = derivatives_model(s, u, self.model, want_info=True)
        self.state = s
        self.t_step = self.t_step + 1
        self._t_cpu += 1
        reward, failed, terms = self._reward(s, a)
        if filt is not None:
            reward = reward - w.intervention * filt
            terms.update(override=self.override, intervention=filt)
        terminated = failed | ~finite
        reward = xp.where(terminated, reward - w.spin, reward)
        truncated = (self.t_step >= self.max_steps) & ~terminated
        self.prev_action = a
        self.episode_return = self.episode_return + reward
        dyn, task, self._meas = self._features(s, info)
        self._hist_obs = xp.concatenate([self._hist_obs[:, 1:], dyn[:, None, :]], axis=1)
        obs = self._assemble(task)
        grip, priv = self._privileged(s, info)
        done = terminated | truncated
        out = dict(terms, done=done, episode_return=self.episode_return, episode_length=self.t_step, action=a,
                   grip=grip)
        if self.autoreset and bool(done.any()):
            idx = np.flatnonzero(to_numpy(done))
            ii = self._idx(idx)
            out["final_obs"] = obs[ii]
            out["final_idx"] = idx
            if priv is not None:
                out["final_priv"] = priv[ii]
            out["episode_return"] = self.episode_return * 1.0          # values of the finished episodes
            out["episode_length"] = self.t_step * 1
            self._start_rows(idx, [self._draw() for _ in idx])
            _, info0 = derivatives_model(self.state, self._applied(), self.model, want_info=True)
            dyn0, task0, meas0 = self._features(self.state, info0)
            self._hist_obs[ii] = dyn0[ii][:, None, :]
            self._meas = tuple(xp.where(done, m0, m) for m0, m in zip(meas0, self._meas))
            obs = self._assemble(task0)                    # unchanged rows: same state, same task features
            grip0, priv0 = self._privileged(self.state, info0)
            grip = xp.where(done, grip0, grip)
            if priv is not None:
                priv = xp.where(done[:, None], priv0, priv)
        self.last_obs, self.last_grip, self.last_priv = obs, grip, priv
        return obs, reward, terminated, truncated, out

    # ---------------------------------------------------------------- reward
    def _reward(self, s, a):
        xp, cfg, w = self.xp, self.cfg, self.cfg.reward
        vx, vy, r = s[:, S.VX], s[:, S.VY], s[:, S.R]
        v = xp.hypot(vx, vy)
        beta = xp.arctan2(vy, vx)
        babs = xp.abs(beta)
        gate = xp.clip((v - MOVING) / 0.5, 0.0, 1.0)
        spun = (v > MOVING) & (babs > math.radians(cfg.spin_beta_deg))
        rate = xp.sum((a - self.prev_action) ** 2, axis=1)
        if cfg.task == "hold":
            into = (beta * r < 0.0) * 1.0                     # rotating into the slide
            r_beta = xp.exp(-((babs - self.target_beta) / math.radians(w.beta_sigma_deg)) ** 2) * gate * into
            r_speed = xp.exp(-((v - self.target_speed) / w.speed_sigma) ** 2)
            reward = w.beta * r_beta + w.speed * r_speed - w.action_rate * rate
            terms = dict(r_beta=r_beta, r_speed=r_speed, beta_deg=beta * (180.0 / math.pi), speed=v, spun=spun)
            return reward, spun, terms
        e_y, head_err, tx, ty = self._track_geometry(s)
        yaw = s[:, S.YAW]
        v_along = (vx * xp.cos(yaw) - vy * xp.sin(yaw)) * tx + (vx * xp.sin(yaw) + vy * xp.cos(yaw)) * ty
        progress = xp.clip(v_along / self.target_speed, 0.0, 1.0)      # no credit for parking on the line
        r_track = xp.exp(-(e_y / w.track_sigma) ** 2) * progress
        r_speed = xp.exp(-((v_along - self.target_speed) / w.speed_sigma) ** 2)
        into = (beta * self.direction < 0.0) * 1.0
        r_drift = xp.clip((babs - math.radians(w.drift_min_deg)) / math.radians(10.0), 0.0, 1.0) * into * gate
        reward = w.track * r_track + w.speed * r_speed + w.drift * r_drift - w.action_rate * rate
        off = xp.abs(e_y) > cfg.max_track_error
        terms = dict(r_track=r_track, r_speed=r_speed, r_drift=r_drift, track_error=e_y,
                     beta_deg=beta * (180.0 / math.pi), speed=v, spun=spun, off_track=off)
        return reward, spun | off, terms

    def _track_geometry(self, s):
        """Signed distance to the circle (+ = left of the travel direction), body heading error to the
        tangent (rad) and the tangent unit vector."""
        xp, R = self.xp, self.cfg.track_radius
        relx, rely = s[:, S.X] - self.center[:, 0], s[:, S.Y] - self.center[:, 1]
        dist = xp.hypot(relx, rely)
        e_y = self.direction * (R - dist)
        tang = xp.arctan2(rely, relx) + self.direction * (math.pi / 2)
        d = s[:, S.YAW] - tang
        head_err = xp.arctan2(xp.sin(d), xp.cos(d))
        return e_y, head_err, xp.cos(tang), xp.sin(tang)

    # ---------------------------------------------------------------- observations
    def _feature_names(self) -> tuple[list[str], list[str]]:
        """Per-step (dynamic) features, stacked over the history, and task features (current only)."""
        if self.cfg.obs == "sensors":
            front = ["wheel_fl", "wheel_fr"] if self.cfg.front_wheel_speeds else []
            dyn = ["gyro_z", "accel_x", "accel_y", *front, "wheel_rl", "wheel_rr", "vel_x", "vel_y"]
        else:
            dyn = ["vx", "vy", "yaw_rate", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr", "steer_angle",
                   "motor_current", "load_long", "load_lat", *[f"kappa_{w}" for w in ("fl", "fr", "rl", "rr")],
                   *[f"alpha_{w}" for w in ("fl", "fr", "rl", "rr")], *[f"tire_temp_{w}" for w in ("fl", "fr", "rl", "rr")]]
        dyn += ["prev_steer", "prev_throttle"]
        task = ["target_beta", "target_speed"] if self.cfg.task == "hold" else \
            ["track_error", "heading_err_sin", "heading_err_cos", "direction"]
        return dyn, task

    def _obs_names(self) -> list[str]:
        if self.H == 1:
            return self.dyn_names + self.task_names
        return [f"{n}[t-{k}]" for k in range(self.H - 1, -1, -1) for n in self.dyn_names] + self.task_names

    def _priv_names(self) -> list[str]:
        return ["vx", "vy", "yaw_rate", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr", "steer_angle",
                "motor_current", "load_long", "load_lat", *[f"kappa_{w}" for w in ("fl", "fr", "rl", "rr")],
                *[f"alpha_{w}" for w in ("fl", "fr", "rl", "rr")], *[f"tire_temp_{w}" for w in ("fl", "fr", "rl", "rr")],
                "grip_lat", "grip_long", "latency"]

    def obs_spec(self) -> dict:
        """How the observation is built from the car's readings (for the on-car runtime): the per-step
        features with their source signal and scale (feature = signal x scale), the history length and
        the task features. Sensors only; ``obs="full"`` is not deployable."""
        scale = dict(gyro_z=0.2, accel_x=1 / 9.81, accel_y=1 / 9.81, wheel_fl=1 / 3, wheel_fr=1 / 3, wheel_rl=1 / 3,
                     wheel_rr=1 / 3, vel_x=1 / 3, vel_y=1 / 3, prev_steer=1.0, prev_throttle=1.0,
                     target_beta=1.0, target_speed=1 / 3, track_error=1.0, heading_err_sin=1.0, heading_err_cos=1.0,
                     direction=1.0)
        units = dict(gyro_z="rad/s", accel_x="m/s^2", accel_y="m/s^2", vel_x="m/s", vel_y="m/s", prev_steer="command",
                     prev_throttle="command", target_beta="rad", target_speed="m/s", track_error="m (clipped to +-2)",
                     heading_err_sin="-", heading_err_cos="-", direction="+1 ccw / -1 cw")
        if self.cfg.obs != "sensors":
            raise ValueError("only obs='sensors' policies can run on the car")
        feat = lambda n: dict(name=n, scale=scale[n], unit=units.get(n, "m/s"))  # noqa: E731
        return dict(history=self.H, dynamic=[feat(n) for n in self.dyn_names], task=[feat(n) for n in self.task_names],
                    layout="dynamic features oldest to newest, then the task features", control_dt=self.control_dt)

    def _features(self, s, info):
        """(dynamic (B, n_dyn), task (B, n_task), measured (vx, vy, r)). Sensors: gyro (rad/s / 5), IMU
        specific force (/ g), wheel surface speeds (m/s / 3, the front encoders and the rear from the
        motor) and the velocity estimate (m/s / 3) of an optical-flow sensor or an overhead tracker, all
        with noise; the safety filter gets the same noisy readings."""
        xp, B = self.xp, self.num_envs
        Rw = self.model.Rw
        if self.cfg.obs == "sensors":
            r = s[:, S.R] + self._noise("gyro", (B,))
            ax = info["ax"] + self._noise("accel", (B,))
            ay = info["ay"] + self._noise("accel", (B,))
            wheels = s[:, S.OMEGA] * Rw + self._noise("wheel", (B, 4))
            if not self.cfg.front_wheel_speeds:
                wheels = wheels[:, 2:]
            vel = s[:, S.VX:S.VY + 1] + self._noise("velocity", (B, 2))
            cols = [(r / 5.0)[:, None], (ax / 9.81)[:, None], (ay / 9.81)[:, None], wheels / 3.0, vel / 3.0]
            meas = (vel[:, 0], vel[:, 1], r)
        else:
            cols = [s[:, S.VX:S.VY + 1] / 3.0, s[:, S.R:S.R + 1] / 5.0, s[:, S.OMEGA] * Rw / 3.0,
                    s[:, S.DELTA:S.DELTA + 1] / 0.6, s[:, S.I_MOTOR:S.I_MOTOR + 1] / 20.0,
                    s[:, S.DFZ_LONG:S.DFZ_LAT + 1] / 5.0, xp.clip(s[:, S.KAPPA_LAG], -5.0, 5.0) / 2.0,
                    s[:, S.ALPHA_LAG], (s[:, S.T_TIRE] - 40.0) / 40.0]
            meas = (s[:, S.VX], s[:, S.VY], s[:, S.R])
        cols.append(self.prev_action)
        dyn = xp.concatenate(cols, axis=1)
        if self.cfg.task == "hold":
            task = xp.stack([self.target_beta, self.target_speed / 3.0], axis=1)
        else:
            e_y, head, _, _ = self._track_geometry(s)
            task = xp.stack([xp.clip(e_y, -2.0, 2.0), xp.sin(head), xp.cos(head), self.direction], axis=1)
        return dyn, task, meas

    def _assemble(self, task):
        xp, B = self.xp, self.num_envs
        obs = xp.concatenate([self._hist_obs.reshape(B, -1), task], axis=1)
        return obs if self.device is not None else obs.astype(np.float32)

    def _privileged(self, s, info):
        """(true grip (B,): mean effective peak lateral friction coefficient, privileged vector or None)."""
        xp = self.xp
        grip = xp.sum(info["mu_y"], axis=1) * 0.25
        if not self.privileged:
            return grip, None
        Rw = self.model.Rw
        priv = xp.concatenate([s[:, S.VX:S.VY + 1] / 3.0, s[:, S.R:S.R + 1] / 5.0, s[:, S.OMEGA] * Rw / 3.0,
                               s[:, S.DELTA:S.DELTA + 1] / 0.6, s[:, S.I_MOTOR:S.I_MOTOR + 1] / 20.0,
                               s[:, S.DFZ_LONG:S.DFZ_LAT + 1] / 5.0, xp.clip(s[:, S.KAPPA_LAG], -5.0, 5.0) / 2.0,
                               s[:, S.ALPHA_LAG], (s[:, S.T_TIRE] - 40.0) / 40.0,
                               grip[:, None], (xp.sum(info["mu_x"], axis=1) * 0.25)[:, None],
                               (self._delay * 1.0 / MAX_LATENCY_STEPS)[:, None] if self.device is None
                               else (self._delay.to(self.dtype) / MAX_LATENCY_STEPS)[:, None]], axis=1)
        return grip, priv if self.device is not None else priv.astype(np.float32)


__all__ = ["DriftBatchEnv", "mirror_state", "nominal_params"]
