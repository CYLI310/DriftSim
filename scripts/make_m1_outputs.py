"""Regenerate the Milestone 1 figures, animations and summary into outputs/m1/.

    python scripts/make_m1_outputs.py

Everything here is deterministic. Files are kept small (GIFs at dpi <= 80).
"""
from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rc_drift_sim.control.lqr import LQRDriftPolicy  # noqa: E402
from rc_drift_sim.control.maneuvers import DRIFT_SCHEDULE, LQR_ENTRY, open_loop_drift_actions  # noqa: E402
from rc_drift_sim.sim import equilibrium as EQ  # noqa: E402
from rc_drift_sim.sim import state as S  # noqa: E402
from rc_drift_sim.sim import tire as tire_mod  # noqa: E402
from rc_drift_sim.sim.params import load_surfaces, load_tires  # noqa: E402
from rc_drift_sim.sim.stability import stiffness_report  # noqa: E402
from rc_drift_sim.sim.vehicle import Vehicle, make_vehicle  # noqa: E402
from rc_drift_sim.viz import plots, render  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "m1"


def longest_drift(traj) -> tuple[float, slice]:
    beta = np.degrees(traj.beta)
    ok = (traj.speed > 0.8) & (np.abs(beta) > 20) & (np.abs(beta) < 80)
    best, cur, end = 0, 0, 0
    for k, o in enumerate(ok):
        cur = cur + 1 if o else 0
        if cur > best:
            best, end = cur, k
    dt = float(traj.t[1] - traj.t[0])
    return max(best - 1, 0) * dt, slice(end - best + 1, end + 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore", RuntimeWarning)
    lines: list[str] = []
    log = lambda s="": (print(s), lines.append(s))  # noqa: E731
    t_start = time.time()

    v = make_vehicle()
    p = v.params
    log("RC drift sim - Milestone 1 summary")
    log(f"car: {p.vehicle.mass} kg, wheelbase {p.vehicle.wheelbase} m, track {p.vehicle.track_width} m, "
        f"CG height {p.vehicle.cg_height} m, Iz {p.vehicle.iz:.4f} kg m^2 (estimated)")
    log(f"tire {p.tire.name} on {p.surface.name}; drivetrain {p.drivetrain.layout}, spool rear; "
        f"physics {1 / p.sim.dt:.0f} Hz {p.sim.integrator}, control {1 / p.sim.control_dt:.0f} Hz, latency "
        f"{p.actuators.latency * 1e3:.0f} ms")
    log()

    # --- open-loop drift
    acts = open_loop_drift_actions(p, 4.0)
    traj = v.rollout(v.initial_state(), acts)
    dur, win = longest_drift(traj)
    beta = np.degrees(traj.beta)
    log(f"[open-loop drift] schedule {DRIFT_SCHEDULE}")
    log(f"  sustained drift {dur:.2f} s; in the window: beta {beta[win].mean():+.1f} deg "
        f"(range {beta[win].min():+.1f}..{beta[win].max():+.1f}), yaw rate {np.degrees(traj.yaw_rate[win]).mean():+.0f} deg/s, "
        f"speed {traj.speed[win].mean():.2f} m/s")
    render.animate(traj, p, OUT / "drift.gif", fps=25, stride=2, dpi=70)
    render.snapshot(traj, p, OUT / "drift_snapshot.png")
    plots.plot_timeseries(traj, p, OUT / "drift_timeseries.png")
    plt.close("all")

    # --- drift equilibrium + LQR
    trim = EQ.solve_trim(v, 1.5, beta=np.radians(-30.0))
    ev = EQ.open_loop_eigenvalues(v, trim)
    log()
    log(f"[drift equilibrium] {trim.summary()}")
    log(f"  rear slip ratio {trim.info['kappa'][2]:+.2f}, slip angles front {np.degrees(trim.info['alpha'][0]):+.1f} / "
        f"rear {np.degrees(trim.info['alpha'][2]):+.1f} deg; unstable eigenvalue {ev[0].real:+.2f} 1/s "
        f"(divergence time constant {1 / ev[0].real:.2f} s)")
    pol = LQRDriftPolicy(v, trim, LQR_ENTRY)
    traj_lqr = v.rollout(v.initial_state(), pol, duration=8.0)
    beta_l = np.degrees(traj_lqr.beta)
    k_sw = int(round(pol.t_switch / v.control_dt))
    last = slice(-150, None)
    log(f"[LQR drift] controller on at t = {pol.t_switch:.2f} s; worst |beta| afterwards {np.abs(beta_l[k_sw:]).max():.1f} deg; "
        f"last 3 s beta {beta_l[last].mean():+.2f} +- {beta_l[last].std():.2f} deg, speed {traj_lqr.speed[last].mean():.2f} m/s, "
        f"yaw rate {np.degrees(traj_lqr.yaw_rate[last]).mean():+.0f} deg/s")
    render.animate(traj_lqr, p, OUT / "lqr_drift.gif", fps=25, stride=3, dpi=70)
    plots.plot_timeseries(traj_lqr, p, OUT / "lqr_drift_timeseries.png")
    plots.plot_gg(traj_lqr, OUT / "lqr_drift_gg.png")
    plt.close("all")

    # --- drift equilibrium family
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    betas = np.radians(np.arange(-15, -47, -2.5))
    log()
    log("[drift equilibria] speed, sideslip -> steer, throttle, yaw rate, unstable eigenvalue")
    for speed in (1.0, 1.5, 2.0):
        fam = EQ.continuation(v, speed, betas, key="beta")
        good = [t for t in fam if t.success]
        b = np.degrees([t.beta for t in good])
        lam = [EQ.open_loop_eigenvalues(v, t)[0].real for t in good]
        axs[0].plot(b, [t.u[0] for t in good], "o-", ms=3, label=f"{speed} m/s")
        axs[1].plot(b, [t.u[1] for t in good], "o-", ms=3)
        axs[2].plot(b, np.degrees([t.r for t in good]), "o-", ms=3)
        axs[3].plot(b, lam, "o-", ms=3)
        mid = good[len(good) // 2]
        log(f"  v={speed}: {len(good)}/{len(fam)} trims; e.g. {mid.summary()}")
    for ax, ttl in zip(axs, ("steer command (counter-steer < 0)", "throttle command", "yaw rate [deg/s]",
                             "unstable eigenvalue [1/s]")):
        ax.set_xlabel("sideslip beta [deg]")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("Steady drift equilibria (hard plastic tire on P-tile): all open-loop unstable", fontsize=11)
    fig.savefig(OUT / "drift_equilibria.png", dpi=90)
    plt.close(fig)

    # --- understeer gradient vs theory
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ays = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8])
    log()
    log("[understeer gradient] open rear diff, trims at 1.5 m/s; measured vs linear bicycle theory")
    for a, col in ((0.090, "C0"), (0.133, "C1"), (0.170, "C3")):
        pp = dataclasses.replace(p, vehicle=dataclasses.replace(p.vehicle, cg_to_front=a),
                                 drivetrain=dataclasses.replace(p.drivetrain, rear_diff_lock=0.0, rear_diff_visc=0.0))
        vv = Vehicle(pp)
        trims = EQ.continuation(vv, 1.5, ays / 1.5, key="r")
        und = np.degrees([t.s[S.DELTA] - pp.vehicle.wheelbase / t.radius for t in trims])
        c = tire_mod.tire_coefficients(pp.tire, vv.surf, None, pp.vehicle.static_loads(), None)
        Ky = np.asarray(c["Ky_eff"]) * np.ones(4)
        K = pp.vehicle.mass / pp.vehicle.wheelbase * (pp.vehicle.cg_to_rear / Ky[:2].sum() - a / Ky[2:].sum())
        K_meas = np.polyfit(ays[:8], np.radians(und[:8]), 2)[1]
        ax.plot(ays, und, "o", color=col, ms=4, label=f"a = {a:.3f} m (4-wheel trims)")
        ax.plot(ays, np.degrees(K * ays), "-", color=col, lw=1, label=f"bicycle K_us = {K * 1e3:+.2f} mrad/(m/s^2)")
        log(f"  CG {a:.3f} m behind front axle: measured {K_meas * 1e3:+.3f}, theory {K * 1e3:+.3f} mrad/(m/s^2)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lateral acceleration [m/s^2]")
    ax.set_ylabel("delta - L/R [deg]  (> 0 understeer)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title("Steady-state cornering: model vs linear theory", fontsize=10)
    fig.savefig(OUT / "understeer.png", dpi=90)
    plt.close(fig)

    # --- tire curves
    tires, surfs = load_tires(), load_surfaces()
    tp = tires["hard_plastic_drift"]
    plots.plot_tire_curves(tp, [surfs[s] for s in ("epoxy_ptile", "dry_asphalt", "packed_dirt", "loose_dirt", "sand", "ice")],
                           OUT / "tire_curves_surfaces.png", T_tire=tp.t_opt, alpha_max_deg=80.0, kappa_max=1.5)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    alpha = np.radians(np.linspace(0, 60, 400))
    for tn, t in tires.items():
        for ax, sn in zip(axs, ("dry_asphalt", "loose_dirt")):
            c = tire_mod.tire_coefficients(t, surfs[sn], None, 4.0, None)
            ax.plot(np.degrees(alpha), np.abs(tire_mod.pure_slip_forces(c, 0.0, alpha)[1]) / 4.0, label=tn)
    for ax, sn in zip(axs, ("dry asphalt", "loose dirt")):
        ax.set_title(f"lateral friction coefficient on {sn}, Fz = 4 N", fontsize=10)
        ax.set_xlabel("slip angle [deg]")
        ax.set_ylabel("|Fy| / Fz")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.savefig(OUT / "tire_compounds.png", dpi=90)
    plt.close("all")

    # --- coasting
    coast = v.rollout(v.initial_state(v=3.0), np.zeros((150, 2)))
    plots.plot_timeseries(coast, p, OUT / "coast_timeseries.png")
    plt.close("all")
    log()
    log(f"[coast] 3 m/s -> {coast.speed[-1]:.2f} m/s after 3 s (motor friction through the gearing dominates)")

    # --- integrators, stiffness, speed
    rk = stiffness_report(p)
    log()
    log("[stiffness] " + rk.table().replace("\n", "\n  "))
    s_b = np.broadcast_to(v.initial_state(v=1.5), (1024, S.NS)).copy()
    u_b = np.tile([0.2, 0.25], (1024, 1))
    v.step(s_b, u_b, want_info=False)
    t0 = time.perf_counter()
    for _ in range(5):
        s_b, _ = v.step(s_b, u_b, want_info=False)
    el = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    v.rollout(v.initial_state(), acts[:50])
    el1 = (time.perf_counter() - t0) / 50
    log()
    log(f"[speed, NumPy reference on this CPU] single env: {1 / el1:,.0f} env-steps/s "
        f"({p.sim.n_substeps / el1:,.0f} RK4 substeps/s); 1024 envs batched: {1024 / el:,.0f} env-steps/s "
        f"({1024 * p.sim.n_substeps / el:,.0f} substeps/s)")
    log()
    log(f"generated in {time.time() - t_start:.0f} s")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
