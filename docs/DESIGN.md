# RC Drift Sim — Design Contract (Milestone 1, as built)

This document is the single source of truth for conventions, units, the state layout and the
module interfaces. Implementers of different modules work against this contract. Concrete field
names live in `rc_drift_sim/sim/params.py` and `rc_drift_sim/sim/state.py` — read both.

Milestone 1 scope: NumPy reference physics (4-wheel planar model, Pacejka tire with combined
slip, tire condition and temperature effects, loose-surface plowing, drivetrain, actuators,
RK4/semi-implicit integrator) on a uniform or per-wheel surface, a batch-capable fast core, a
trim / stability / LQR analysis module, a stiffness guard, a matplotlib top-down visualizer +
time-series plots, and the pytest suite. Surface MAPS, roughness, sensors, the Gym env, JAX and
training come in later milestones; every interface already carries the hooks they need.

## 0. Repository layout

```
rc_drift_sim/
  configs/              vehicle.yaml, tires.yaml, surfaces.yaml (all default values)
  sim/                  params, state, tire, surface, drivetrain, actuators, vehicle, integrator,
                        equilibrium (trims, linearization), stability (stiffness guard)
  control/              maneuvers (open-loop schedules), lqr (TrimLQR, LQRDriftPolicy)
  datagen/              batch dataset generation (catalog, spec, sampling, inputs, runner, export)
  app/                  web GUI: server.py (stdlib HTTP server + JSON API), static/ (page, no build step)
  viz/                  render (top-down, animations), plots (time series, tire curves, g-g)
examples/               quickstart.py, batch_export.py, specs/*.json
scripts/                make_figures.py (docs/images/m1), tune_drift.py, make_param_reference.py
tests/                  pytest suite (helpers.py holds shared helper functions)
docs/                   DESIGN.md (this contract), DATA_GENERATION.md, PARAMETERS.md (generated), images/
```
Dependency direction: `sim` depends on nothing else in the package; `control`, `viz` and `datagen`
depend on `sim`; `app` depends on `datagen`; `tests`, `examples` and `scripts` depend on everything;
nothing depends on them.
The Gymnasium env, training and system identification packages are added with Milestones 4-8.

## 1. Conventions

- SI units everywhere internally: m, s, kg, N, rad, A, V, degC. YAML may use `*_deg` keys
  (degrees) — the loader converts to radians and strips the suffix.
- World frame: x east, y north, z up. Yaw `psi` is CCW-positive measured from +x.
- Body frame: x forward, y left, z up. Positive yaw rate `r` = turning left.
- Wheel order everywhere: index 0 = FL, 1 = FR, 2 = RL, 3 = RR. Per-wheel arrays have shape (4,).
- Wheel contact positions in the body frame (a = `cg_to_front`, b = `cg_to_rear`, w = `track_width`):
  FL (+a, +w/2), FR (+a, -w/2), RL (-b, +w/2), RR (-b, -w/2).
- Steering angle `delta` positive = wheels turned left. Only the front wheels steer.
  Per-wheel steer angles come from `actuators.ackermann_angles` (parallel steering when
  `ackermann = 0`, full Ackermann when 1). Rear steer angles are 0.
- Vehicle sideslip `beta = atan2(vy, vx)` (positive = nose pointing right of the velocity, i.e.
  the car is sliding to the left of where it points).
- Wheel-frame velocities: `v_i = (vx - r*y_i, vy + r*x_i)` in the body frame, rotated by
  `-delta_i` into the wheel frame giving `(vxw_i, vyw_i)`.
- Slip ratio  `kappa_i = (omega_i * R - vxw_i) / max(|vxw_i|, v_eps)`; positive = driving.
  Positive kappa produces positive (forward) longitudinal force.
- Slip angle   `alpha_i = atan2(vyw_i, max(|vxw_i|, v_eps))` in the wheel frame; positive = the
  contact patch velocity points LEFT of the wheel heading. The lateral force opposes it:
  `Fy_i = -MF_y(alpha_i)` (a positive slip angle gives a force to the right). This is the
  ISO-style convention; racer.nl's SAE description has the opposite sign for Fy — do not mix.
- Load `Fz_i >= 0` always (clamped). A lifted wheel produces no force.
- `v_eps = 0.3 m/s` regularizes every slip denominator (`SimParams.v_eps`).

## 2. State vector

Flat float64 array, length `NS = 30`, indices in `state.py`; any leading batch shape
(..., NS) is allowed everywhere in the physics:

| idx   | name       | meaning                                                              |
|-------|------------|----------------------------------------------------------------------|
| 0,1,2 | X, Y, YAW  | world pose                                                           |
| 3,4,5 | VX, VY, R  | body-frame velocities, yaw rate                                      |
| 6-9   | OMEGA      | wheel angular speeds (rad/s), FL FR RL RR                            |
| 10    | DELTA      | actual steering angle of the virtual center front wheel (rad)        |
| 11    | I_MOTOR    | motor current (A)                                                    |
| 12-15 | T_TIRE     | tire temperatures (degC)                                             |
| 16    | DFZ_LONG   | lagged longitudinal load transfer (N), + = rear axle more loaded     |
| 17    | DFZ_LAT    | lagged lateral load transfer (N), + = right side more loaded         |
| 18-21 | KAPPA_LAG  | relaxed (lagged) slip ratio per wheel — the value fed to the MF      |
| 22-25 | ALPHA_LAG  | relaxed (lagged) slip angle per wheel (rad) — the value fed to the MF|
| 26-29 | CONTAM     | tire contamination 0..1 (dust on the tread), decays with distance    |

All 30 entries are integrated by the ODE solver. Discrete things (control latency buffer,
control-rate zero-order hold) live outside the ODE in the step wrapper.

Inputs `u` = `(steer_cmd, throttle_cmd)`, both in [-1, 1], already delayed by the latency buffer.

## 3. Parameters

See `params.py`: `VehicleParams`, `TireParams` (one compound), `TireCondition` (per wheel),
`SurfaceParams` (one surface), `DrivetrainParams`, `ActuatorParams`, `SimParams`, bundled in
`Params`. `default_params()` returns the hard-plastic drift tire on the epoxy/P-tile track with
the YAML defaults. All dataclasses are frozen; use `dataclasses.replace` to vary them.

Per-wheel surface: functions take `surf` = a `SurfaceParams` whose numeric fields are arrays of
shape (4,) (one value per wheel). `surface.uniform_surface(sp)` broadcasts a scalar surface to
four wheels. Milestone 2 replaces this with a map lookup; the tire and vehicle code must already
be written per-wheel.

## 4. Tire model (`sim/tire.py`)

Source: Pacejka Magic Formula, MF5.2-style normalized-load parameterization (see
http://www.racer.nl/reference/pacejka.htm for the formula, the B = K/(C*D) guard and the
combined-slip options). We use dimensionless coefficients with a nominal wheel load
`fz0` (about 4 N for a 1.6 kg car) instead of the kN-scaled Pacejka89 a/b/c sets.

Core formula (racer.nl, "Pacejka96"):

    mf(x, B, C, D, E, Sh=0, Sv=0):  xs = x + Sh;  Bx = B*xs
        return D * sin(C * atan(Bx - E * (Bx - atan(Bx)))) + Sv

Properties we rely on: D is the peak, B*C*D is the slope at the origin (stiffness), the
asymptote at large slip is D*sin(pi*C/2). C > 1 gives a peak followed by a drop; C = 1 rises
monotonically to D; C < 1 keeps building without a peak (this is how loose surfaces are made:
they blend C toward a target <= 1 instead of just scaling mu). E must be clamped to <= 1.

Coefficients per compound, `dfz = (Fz - fz0)/fz0`:

    Longitudinal (pure):  Cx = pcx1
                          mu_x = (pdx1 + pdx2*dfz) * lam_mu_x         Dx = mu_x * Fz
                          Ex = min(pex1 + pex2*dfz, 1)
                          Kx = Fz * (pkx1 + pkx2*dfz) * exp(pkx3*dfz) * lam_k_x
                          Bx = Kx / (Cx*Dx + eps)
    Lateral (pure):       Cy = pcy1
                          mu_y = (pdy1 + pdy2*dfz) * lam_mu_y         Dy = mu_y * Fz
                          Ey = min(pey1 + pey2*dfz, 1)
                          Ky = pky1 * fz0 * sin(2*atan(Fz/(pky2*fz0))) * lam_k_y
                          By = Ky / (Cy*Dy + eps)
    (eps = 1e-6; Fz clamped >= 0; when Fz == 0 all forces are 0.)

Surface modification (applied to the coefficients, per wheel):
    mu   *= surf.mu_scale * (1 + tire.loose_affinity * surf.looseness)
    K    *= surf.stiffness_scale
    C     = C + surf.shape_blend * (surf.shape_c_target - C)
    E     = min(E + surf.curvature_e_shift, 1)
    slip input to the MF is divided by (1 + surf.peak_slip_shift) (moves the peak to larger slip;
    it also lowers the effective initial stiffness by the same factor — intentional, loose
    surfaces are softer).

Tire condition and temperature (`tire.condition_scales`, per wheel, all clipped to [0, 1]):
    lam_mu = temperature_factor(T) * (1 - wear_mu_drop*wear) * (1 - wet_mu_drop*wetness)
             * (1 - contamination_mu_drop*contamination)
    lam_k  = (1 + wear_k_gain*wear) * (1 - wet_k_drop*wetness)      (worn tread is stiffer)
    C     += wet_c_gain*wetness                                     (sharper post-peak drop when wet)
    temperature_factor(T) = 1 - temp_mu_drop * (1 - exp(-((T - t_opt)/t_width)^2))
The YAML mu values are the OPTIMAL-temperature values (factor 1 at t_opt). Wear and wetness are
static per episode (folded into the precomputed TireModel); temperature (T_TIRE) and
contamination (CONTAM) are states and act dynamically. d(CONTAM)/dt = -|v_wheel|/decay_dist * CONTAM.

Plowing (loose surfaces, `tire.plowing_force`), added to the wheel-frame lateral force in
vehicle.py, NOT bounded by mu*Fz (it is a soil reaction, not friction):
    q = vyw / PLOW_V_REF (1 m/s),  F_plow = -loose_drag * Fz * q / sqrt(1 + (q/PLOW_CAP)^2),  PLOW_CAP = 2

Combined slip — `TireParams.combined_mode`:
- `"similarity"` (default, Bakker/Nyborg/Pacejka-style normalized slip; guarantees the
  resultant never exceeds max(Dx, Dy)):
      sx = Bx*Cx*kappa,  sy = By*Cy*tan(alpha)      (stiffness-normalized slips)
      rho = sqrt(sx^2 + sy^2), rho_reg = max(rho, 1e-9)
      Fx = mf(rho/(Bx*Cx), Bx, Cx, Dx, Ex) * sx/rho_reg
      Fy = -mf(rho/(By*Cy), By, Cy, Dy, Ey) * sy/rho_reg
  Pure slip is recovered exactly when the other slip is zero.
- `"mf_weighting"` (MF5.2 cosine weighting functions from racer.nl, with Shxa = Shyk = 0 and
  no kappa-induced side force):
      Gxa = cos(rcx1 * atan(rbx1 * cos(atan(rbx2*kappa)) * alpha))
      Gyk = cos(rcy1 * atan(rby1 * cos(atan(rby2*(alpha - rby3))) * kappa))
      Fx = Gxa * Fx0(kappa),  Fy = Gyk * Fy0(alpha)
  followed by a safety ellipse clamp so that (Fx/Dx)^2 + (Fy/Dy)^2 <= 1.
- `"ellipse"` (racer.nl simple method): Fx = Fx0(kappa), Fy = Fy0(alpha) * sqrt(max(0, 1 - (Fx/Dx)^2)).

Invariant that tests check on every compound and surface: sqrt(Fx^2 + Fy^2) <= max(mu_x, mu_y)*Fz*(1+1e-6).

Relaxation length: the MF is evaluated on the lagged slips `KAPPA_LAG`, `ALPHA_LAG`, integrated in
vehicle.py as
    d(kappa_lag)/dt = (v_reg / relax_x) * (kappa - kappa_lag)
    d(alpha_lag)/dt = (v_reg / relax_y) * (alpha - alpha_lag),   v_reg = max(|vxw|, v_eps)
with `relax_x`, `relax_y` in meters (about 0.02-0.05 m at RC scale).

Low-speed model: below `SimParams.v_low` the MF is blended with a saturated viscous model of the
contact-patch slip velocity `(vsx, vsy) = (omega*R - vxw, -vyw)`... precisely:
    F_low = mu * Fz * v_slip / sqrt(|v_slip|^2 + v_c_low^2)   with v_slip = (omega*R - vxw, -vyw)
    (x-component uses mu_x, y-component uses mu_y; the sign makes it oppose slip velocity)
    w = smoothstep(|v_wheel|; v_low_start, v_low)    (|v_wheel| = hypot(vxw, vyw))
    F = w * F_MF + (1 - w) * F_low
At rest (v_slip = 0) the force is exactly 0, so a car at rest stays at rest. The MF branch's
slip denominators are regularized with v_eps, so both branches are finite everywhere.

Thermal state (M1 dynamics, no effect on grip until M2):
    dT/dt = (|Fx*vsx| + |Fy*vsy|) / heat_capacity - (cool_coeff + cool_speed_coeff*|v_wheel|) * (T - T_ambient) / heat_capacity

Implementation: the math lives once in array functions on a precomputed `TireModel`
(`tire_model(tp, surf, cond)`, `coefficients(tm, Fz, T, contam)`, `forces_model(...)`); the
dict-returning contract API below wraps them. `sin(2 atan z)` in Ky is evaluated as the exact
identity `2z/(1+z^2)`.

Public functions (all vectorized over the wheel axis and any leading batch axes):
    mf(x, B, C, D, E, Sh=0.0, Sv=0.0)
    tire_coefficients(tp, surf, cond, Fz, T_tire) -> dict(Dx, Cx, Bx, Ex, Kx, mu_x, Dy, Cy, By, Ey, Ky, mu_y, slip_scale)
    pure_slip_forces(coef, kappa, alpha) -> (Fx0, Fy0)
    combined_forces(coef, kappa, alpha, mode) -> (Fx, Fy)
    tire_forces(tp, surf, cond, kappa_lag, alpha_lag, vxw, vyw, omega, Fz, T_tire, sim, R=None, contam=None)
        -> (Fx, Fy, info)
    condition_scales(tp, cond, T_tire, contam=None) -> dict(lam_mu_x, lam_mu_y, lam_k_x, lam_k_y, dC)
    temperature_factor(tp, T), plowing_force(loose_drag, vyw, Fz)
        does coefficients + combined slip + low-speed blend; info carries mu_x, mu_y, w_low, F_low, etc.
    slips(vxw, vyw, omega, R, v_eps) -> (kappa, alpha)  (instantaneous, before relaxation)

## 5. Vehicle model (`sim/vehicle.py`)

Fast core: `compile_model(params, surf, cond) -> VehicleModel` precomputes every constant once;
`derivatives_model(s, u, vm, want_info=False) -> (ds, info|None)` is batch-capable (s (..., NS),
u (..., 2)) and builds the diagnostics dict only on request (the RK4 inner stages skip it).
Verified identical (7e-14 relative) to the pre-refactor per-state implementation on 25,920 random
states x 432 configurations.

`derivatives(s, u, p: Params, surf, cond) -> (ds, info)` — contract API (uses a cached compiled
model), pure function, no mutation of `s`.
`info` is a dict of per-wheel and body quantities (Fx, Fy, Fz, kappa, alpha, delta_i, body-frame
forces, a_x, a_y, beta, T_motor, omega_m ...) used by tests and the visualizer.

Loads:
    Fz_static = [m g b/(2L), m g b/(2L), m g a/(2L), m g a/(2L)]
    a_x = sum(F_body_x)/m,  a_y = sum(F_body_y)/m   (total body-frame specific force incl. aero)
    dFz_long_target = m * a_x * h / L          (+ = rear gains)
    dFz_lat_target  = m * a_y * h / w          (+ a_y = left turn -> right wheels gain)
    d(DFZ_LONG)/dt = (dFz_long_target - DFZ_LONG)/load_transfer_tau, same for DFZ_LAT
    Fz = Fz_static + [-1,-1,+1,+1]*DFZ_LONG/2 + [-rf, +rf, -(1-rf), +(1-rf)]*DFZ_LAT,  rf = roll_stiffness_front
    Fz = max(Fz, 0)
Fz uses the lag STATES; the targets use the forces computed in the same evaluation (no algebraic
loop). Milestone 2 adds roughness noise and slope terms here.

Body:
    per-wheel tire force in the wheel frame (Fx_i, Fy_i) -> body frame with delta_i:
        Fbx_i = Fx_i cos(delta_i) - Fy_i sin(delta_i);  Fby_i = Fx_i sin(delta_i) + Fy_i cos(delta_i)
    aero: F_aero = -0.5 * rho * CdA * |v| * (vx, vy)
    dvx = (sum Fbx_i + F_aero_x)/m + vy*r
    dvy = (sum Fby_i + F_aero_y)/m - vx*r
    dr  = sum(x_i*Fby_i - y_i*Fbx_i) / Iz
    dX = vx cos(psi) - vy sin(psi);  dY = vx sin(psi) + vy cos(psi);  dpsi = r

Wheels: resisting torque per wheel `tau_i = R*Fx_i + T_rr_i` with rolling resistance
    T_rr_i = surf.rolling_resistance_i * Fz_i * R * tanh(omega_i*R / v_rr),  v_rr = 0.05 m/s
Then `domega = drivetrain.wheel_accelerations(dp, vp, omega, T_motor, tau)`.

Steering: `delta_target = actuators.steering_target(ap, steer_cmd, r)`,
`d(DELTA)/dt = actuators.steering_rate(ap, DELTA, delta_target)`.

Motor: `omega_m = drivetrain.shaft_speed(dp, omega)`, `di/dt = drivetrain.current_derivative(dp, i, omega_m, thr)` with `thr = actuators.throttle_command(ap, throttle_cmd)`, `T_motor = drivetrain.motor_torque(dp, i, omega_m)`.

Yaw inertia: `VehicleParams.yaw_inertia` if given, else `mass*(body_length^2 + track_width^2)/12`.

`Vehicle` wrapper class:
    Vehicle(params, surf=None, cond=None)
    .initial_state(x=0, y=0, yaw=0, v=0, beta=0, yaw_rate=0) -> s   (every wheel rolling without
        slip, temps at cond.temp0, contamination at cond.contamination, others 0)
    .step(s, action, n_sub=None, want_info=True) -> (s_new, info)   one control period; s may be (B, NS)
        (zero-order hold on the action; action is already latency-delayed by the caller)
    .rollout_batch(s0 (B, NS), actions (T, B, 2), record_info=False) -> (states (T+1, B, NS), info)

Per-car parameter batches: `compile_model_batch(params_list, surfs, conds)` stacks B cars with
different numeric parameters into one `VehicleModel`. Every model field is used either on per-wheel
quantities or on per-car scalars, so fields become (B, 1) / (B, 4) or (B,) arrays and
`derivatives_model` runs unchanged on (B, NS) states. `VehicleBatch(params_list, surfs, conds)`
wraps it (`initial_states`, `step`). Structural settings (`STRUCTURAL`: combined-slip mode,
layout, reverse enable, dt, control_dt, integrator) must be shared. Each car's result is
bit-identical to simulating it alone (tests/test_batch.py); `rc_drift_sim.datagen` builds on this.
    Vehicle() runs sim.stability.check_stiffness and warns when dt is too large for the config.
    .rollout(s0, actions (T,2) or callable(t, s, info)->action, control_dt) -> Trajectory
        Trajectory holds arrays: t, states (T+1, NS), actions (T,2), and stacked info dict.

## 6. Drivetrain (`sim/drivetrain.py`)

Motor (brushless treated as DC-equivalent):
    Ke = Kt = 60/(2*pi*kv)         [V s/rad = N m/A]
    L  = R_m * tau_e
    V_batt = V0 - R_batt * |i| * |thr|                       (sag under load)
    ESC:  thr in [-1,1] after deadband/expo (actuators.throttle_command).
          reverse_enabled:  V_cmd = thr * V_batt, g = min(|thr|/thr_min, 1)
          not reverse_enabled and thr < 0: V_cmd = 0, g = |thr|   (proportional brake by shorting)
          drag brake at neutral:  g_eff = g + (1 - g) * drag_brake
    Current limit (ESC clamps duty):  V_cmd = clip(V_cmd, Ke*omega_m - i_max*R_m, Ke*omega_m + i_max*R_m)
    di/dt = (g_eff * (V_cmd - Ke*omega_m) - R_m * i) / L
    T_motor = Kt*i - b_m*omega_m - Tc_m * tanh(omega_m / 5.0)

Layouts (`DrivetrainParams.layout`): "rwd", "awd_spool", "awd_overdrive". Everything is expressed
through per-wheel gear ratios and masks so the same code path handles all three (JAX-friendly):
    G = [Gf, Gf, Gr, Gr] with Gr = gear_ratio; Gf = 0 for rwd, Gr for awd_spool,
        Gr/overdrive_front for awd_overdrive (front wheels spin overdrive_front times faster).
    driven = G > 0 (as float mask).
    omega_m (shaft speed) = mean over driven wheels of G_i*omega_i.
    Reflected motor inertia: I_i = wheel_inertia + driven_i * motor_inertia * G_i^2 / n_driven.

Differentials via constraint torque + viscous coupling (open / LSD / locked in one formula):
    Axle (L, R) with axle torque T_axle (already multiplied by the ratio) and resisting torques tau_L, tau_R:
        T_c = clip(lock * (tau_L - tau_R)/2, -T_max, +T_max) + visc * (omega_R - omega_L)
        T_L = T_axle/2 + T_c,   T_R = T_axle/2 - T_c
    lock = 1, T_max = inf reproduces the exact locked-axle acceleration (equal inertias);
    lock = 0, visc = 0 is an open diff; intermediate lock with finite T_max approximates an LSD.
    `visc` also serves as Baumgarte-style stabilization for spools (keep I/visc >= 5 ms).
    UNITS: rear/front diff visc act at the wheels; center_visc acts at the MOTOR SHAFT and is
    therefore multiplied by G^2 at the wheels (default 5e-4; the original 0.02 made AWD diverge).
    Center coupling for AWD is the same formula applied between the two axles with speeds and
    torques referred to the motor shaft (omega_ref = G_axle * mean(omega_axle), tau_ref = sum(tau)/G_axle):
        T_f_shaft = T_motor/2 + T_cc,  T_r_shaft = T_motor/2 - T_cc    (then multiply by G_axle)
    For rwd: T_r_shaft = T_motor, front wheels free (domega = -tau/I_w).
    domega_i = (T_i - tau_i) / I_i.

Public functions:
    gear_ratios(dp) -> G (4,), driven mask
    shaft_speed(dp, omega) -> omega_m
    current_derivative(dp, i, omega_m, thr) -> di/dt
    motor_torque(dp, i, omega_m) -> T
    wheel_accelerations(dp, vp, omega, T_motor, tau) -> domega (4,)
    wheel_inertias(dp, vp) -> I (4,)

## 7. Actuators (`sim/actuators.py`)

    throttle_command(ap, thr_cmd) -> thr   (deadband, optional expo)
    steering_target(ap, steer_cmd, yaw_rate) -> delta_target:
        c = deadband(steer_cmd); delta = c*steer_max + steer_offset
        if gyro_enabled: delta -= clip(gyro_gain*yaw_rate, -gyro_max_correction, +gyro_max_correction)
        clip to [-steer_max, steer_max]
    steering_rate(ap, delta, delta_target) -> clip((delta_target - delta)/servo_tau, -servo_rate, servo_rate)
    ackermann_angles(vp, delta) -> (delta_FL, delta_FR):
        parallel: both = delta
        full Ackermann: inner = atan(L / (L/tan(delta) - w/2)), outer = atan(L/(L/tan(delta) + w/2))
        (guard tan(delta) ~ 0), blended by vp.ackermann.
    class ActionDelay(latency_s, control_dt, n_actions=2): FIFO of whole control steps
        .reset(action0), .push(action) -> delayed action
    (Throttle lag is the motor electrical time constant `motor_tau_e` in Milestone 1.)

## 8. Integrator (`sim/integrator.py`)

    rk4_step(f, s, dt, *args) -> s_new            f(s, *args) -> ds (ignores info)
    semi_implicit_euler_step(f, s, dt, *args)     velocities/other states explicit Euler, then the
                                                  pose (X, Y, YAW) integrated with the NEW velocities
    step(f, s, dt, method, *args)
    integrate(f, s, dt, n_steps, method, *args) -> s_new
Physics dt = 0.001 s, control dt = 0.02 s (50 Hz) => 20 substeps. Both configurable in SimParams.
Benchmark result (M1): RK4 at 1 ms is converged (< 0.04 cm vs dt = 0.25 ms on an aggressive 7 s
maneuver, every compound x surface). Semi-implicit Euler at 1 ms is stable but NOT accurate enough
(up to 30 cm error, front-wheel chatter: the light free front wheel is a fast, poorly damped mode
for explicit Euler), so RK4 stays the default; the 4x cheaper Euler would need an implicit
wheel/tire-slip update to be usable (candidate for the JAX port).
No in-place mutation of state arrays; write NumPy in a JAX-portable style (np.where instead of
Python if/else on array VALUES; Python control flow only on configuration).
The array core (`*_m` functions, `derivatives_model`, the integrators) calls `xp = namespace(...)`
from `sim/xp.py` and uses NumPy-named functions on it: the `numpy` module for NumPy arrays (the
float64 reference, unchanged) or a thin PyTorch adapter for tensors (Apple MPS / NVIDIA CUDA).
Configuration checks that used to test model values are boolean model fields
(`VehicleModel.parallel_steer`, `TireModel.has_pkx3`, `DrivetrainModel.has_drag_brake`), so the
GPU path never reads a value back from the device.

## 9. Visualization (`viz/`)

`render.py`: `draw_car(ax, state, info_step, params, artists=None)` (car body, wheels rotated by
their steer angle and colored by combined slip, per-wheel force arrows, velocity arrow;
updatable artists), `render_frame(ax, traj, k, params, surf_map=None, path=None, trail=True,
world_lim=None)`, `animate(traj, params, out_path, fps=25, stride=2, follow=False, ...)` (GIF via
Pillow, MP4 via ffmpeg when available), `snapshot(traj, params, out_path, ks=None)`,
`live_view(...)` (optional pygame).
`plots.py`: `plot_timeseries(traj, params, out_path)` (speed and sideslip, yaw rate and a_y,
steering angle vs target vs command, throttle and motor current, per-wheel slip ratio, slip
angle, Fz, tire temperature), `plot_tire_curves(tire, surfaces, out_path, Fz=4, T_tire=25)`,
`plot_gg(traj, out_path)`.
`scripts/make_figures.py` regenerates everything in `docs/images/m1/`.

## 9b. Trim and stability analysis (`sim/equilibrium.py`), LQR control (`control/lqr.py`)

`solve_trim(vehicle, v, beta=None | r=None, guess=None, ...) -> Trim` solves for a steady turn
(all derivatives zero except the pose; 23 dynamic states + 2 inputs vs 23 equations + speed +
sideslip-or-yaw-rate) with least squares, multi-start, and a near-straight anchor;
`continuation(vehicle, v, values, key)` follows a family (warm starts with step-size control,
cold multi-start only as a last resort). Each least-squares solve is capped at `MAX_NFEV = 400`
evaluations (warm steps converge in <= ~60, cold drift solves in up to ~140), so a point beyond
the end of a branch fails fast. `open_loop_eigenvalues`,
`linearize` (continuous) and `discrete_model` (one control period through the real integrator).
`control/lqr.py`: `lqr_gain` (with input-delay augmentation), `TrimLQR` (callable policy holding a
trim) and `LQRDriftPolicy` (launch, throttle-stab entry, then `TrimLQR`, latency applied inside).
`control/maneuvers.py`: open-loop schedules (`DRIFT_SCHEDULE`, constant inputs, lane change) and
the LQR entry parameters.
Findings: RWD drift equilibria (rear axle spinning, front counter-steered) exist at ~0.3 g on
P-tile and are open-loop UNSTABLE with one real eigenvalue of +1..+5.5 1/s (Hindiyeh & Gerdes);
full-state LQR through the 20 ms latency captures the drift right after a throttle-stab entry and
holds it under mu +-10 %, mass +15 %, 2x latency and a half-speed servo.

## 9c. Stiffness guard (`sim/stability.py`)

`stiffness_report(params)` estimates the fast rates (electrical, diff and center couplings,
low-speed tire, tire-wheel relaxation mode, servo, load transfer) and compares lambda*dt with the
integrator limit (RK4 2.785, Euler 2.0; the saturating low-speed tire mode is allowed 2x).
`Vehicle()` warns on a too-stiff configuration; a test sweeps every compound x surface x layout.

## 10. Tests (`tests/`)

Required checks (the "item N" references in the test docstrings):

| # | check | test |
|---|-------|------|
| 1 | car at rest stays at rest | test_vehicle::test_rest_stays_at_rest |
| 2 | straight-line coasting decays correctly | test_vehicle::test_straight_line_coasting_decays_correctly |
| 3 | low-speed behavior is stable (no NaN near zero velocity) | test_vehicle::test_low_speed_is_stable |
| 4 | steady-state cornering matches under/oversteer theory | test_vehicle::test_steady_state_understeer_gradient_matches_bicycle_theory |
| 5 | tire force never exceeds mu*Fz on any surface | test_tire::test_force_never_exceeds_mu_fz |
| 6 | loose surfaces have no sharp peak, paved ones do | test_tire::test_paved_surfaces_have_a_sharp_peak, test_loose_surfaces_have_no_peak |
| 7 | RK4 and semi-implicit Euler agree and converge | test_vehicle::test_rk4_and_semi_implicit_euler_agree_and_converge |
| 8 | mirror symmetry | test_vehicle::test_mirror_symmetry |
| 9 | locked vs open vs limited-slip differential | test_drivetrain::test_locked_rear_diff_keeps_wheels_together_open_diff_does_not |
| 10 | open-loop input produces a sustained drift | test_drift::test_open_loop_drift_is_sustained |
| 11 | benchmark reports steps per second | test_benchmark |
| 12 | YAML configs round-trip into the dataclasses | test_config |


`pytest` (about 1 minute on a laptop CPU; `-s` prints the measured physics numbers):
- test_config: YAML round trip, dataclass fields, per-wheel surface broadcasting.
- test_tire: Magic Formula vs the racer.nl formula (hand-computed), peak/slope/asymptote
  properties, sign conventions, |F| <= mu*Fz on a dense grid for every compound x surface x mode
  and through the full tire_forces path with random conditions and the low-speed blend, zero force
  at Fz = 0 and at rest, extreme inputs finite, similarity == pure slip, spinning tire loses
  lateral grip, dry-asphalt peaks in 5-20 deg, paved surfaces peak sharply (force at 3x peak slip
  < 0.97 of peak), loose surfaces rise monotonically to 80 deg, loose != scaled pavement, drift
  tire flatter than rubber, pin tire prefers loose ground, mu_scale, per-wheel surfaces, the
  temperature window, wear, asymmetric wear, wetness, contamination, plowing, thermal rate.
- test_drivetrain: gear ratios, reflected inertia counted once (energy), locked vs LSD vs open
  diff under asymmetric torque, torque-split formulas, AWD overdrive ratio, free front wheels,
  no-load speed and current limit, battery sag, coast vs drag brake, brake-only ESC never
  reverses, batch consistency, ActionDelay, servo rate limit/saturation/deadband/symmetry, gyro
  sign and clipping, exact Ackermann geometry.
- test_vehicle: rest stays at rest; coasting deceleration within 5 % of the analytic prediction
  (rolling + aero + motor friction through the gearing, over the effective mass incl. rotating
  inertia); low-speed stability on four tire/surface pairs; smooth reverse zero crossing;
  understeer gradient from trims within 15 % of the linear bicycle model for a front-heavy
  (understeer) and a rear-heavy (oversteer) car; the spool adds understeer; RK4 vs semi-implicit
  Euler agreement and convergence; mirror symmetry; batched rollout == individual rollouts;
  split-mu braking yaws toward the grippy side; plowing slows a sliding car; all three layouts;
  AWD launches harder; the gyro lowers yaw-rate gain and saves a spin; load-transfer signs and
  magnitudes; tires heat in a drift and cool exponentially; contamination wears off with distance.
- test_drift: open-loop drift (|beta| in (20, 80) deg, v > 0.8 m/s for >= 1.5 s, never spinning),
  drift equilibrium exists with exactly one unstable real mode, LQR captures and holds a drift from
  a real entry, and holds it under five kinds of model mismatch.
- test_stability: every shipped configuration is stable at dt = 1 ms; a too-stiff coupling is
  caught and really diverges.
- test_benchmark (marked): single-env substeps/s and batched env-steps/s.

The open-loop drift schedule is sensitive to the physics by nature (unstable equilibrium);
`python scripts/tune_drift.py` re-derives it in one batched run after model changes.
