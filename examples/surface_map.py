"""Surface maps (Milestone 2): drive the same drift on a uniform track and on a mixed-surface map.

    python examples/surface_map.py            # images -> examples/output/ (git-ignored)
    python examples/surface_map.py --docs     # regenerate docs/images/m2/

The map is a P-tile drift pad with a loose-dirt patch, a gravel strip, a wet-asphalt patch and a
gentle bank on the drift circle. The LQR drift controller is designed on the uniform track and
then held on the map, where each wheel feels its own surface, friction patch, bumps and slope.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from rc_drift_sim import make_vehicle  # noqa: E402
from rc_drift_sim.control.lqr import LQRDriftPolicy  # noqa: E402
from rc_drift_sim.control.maneuvers import LQR_ENTRY  # noqa: E402
from rc_drift_sim.sim import equilibrium as EQ  # noqa: E402
from rc_drift_sim.sim.surface_map import SurfaceMap  # noqa: E402
from rc_drift_sim.viz import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def mixed_track() -> SurfaceMap:
    """5 m x 5 m P-tile pad, 5 cm cells, with patches placed on the drift circle."""
    names = ["epoxy_ptile", "loose_dirt", "gravel", "wet_asphalt"]

    def labels(x, y):
        lab = np.zeros(x.shape, dtype=int)
        lab[(x - 2.3) ** 2 + (y - 1.5) ** 2 < 0.3 ** 2] = 1          # dirt patch
        lab[(x < 1.0) & (y > 1.1) & (y < 1.6)] = 2                    # gravel strip
        lab[(x > 1.3) & (x < 1.8) & (y < 0.6)] = 3                    # wet patch
        return lab

    def height(x, y):
        return 0.03 * np.clip(y - 1.6, 0.0, None)                     # 3 % bank beyond y = 1.6 m

    return SurfaceMap.from_function(labels, names, (-1, 4, -1.5, 3.5), 0.05, height_fn=height, seed=1)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    smap = mixed_track()
    flat = make_vehicle()
    car = make_vehicle(surface_map=smap)
    trim = EQ.solve_trim(flat, 1.5, beta=np.radians(-30.0))           # designed on the uniform track
    ref = flat.rollout(flat.initial_state(), LQRDriftPolicy(flat, trim, LQR_ENTRY), duration=8.0)
    tr = car.rollout(car.initial_state(), LQRDriftPolicy(car, trim, LQR_ENTRY, design_vehicle=flat),
                     duration=8.0)

    names = smap.names
    under = smap.surface_at(tr.xy[1:, 0], tr.xy[1:, 1])
    for k, name in enumerate(names):
        print(f"{name:12s} {np.mean(under == k) * 100:5.1f} % of the run")
    last = slice(-150, None)
    for label, x in (("uniform track", ref), ("mixed map", tr)):
        b = np.degrees(x.beta[last])
        print(f"{label:13s} last 3 s: sideslip {b.mean():+.1f} +- {b.std():.1f} deg, speed {x.speed[last].mean():.2f} m/s")

    # 1. top view: map, both traces, a few car poses
    fig = render.snapshot(tr, car.params, None, surf_map=smap, n_poses=6, world_lim=smap.extent,
                          title="LQR drift: uniform P-tile (dashed) vs mixed map")
    ax = fig.axes[0]
    ax.plot(ref.xy[:, 0], ref.xy[:, 1], "--", color="white", linewidth=1.0, zorder=1)
    ax.legend(handles=[Patch(color=s.color, label=s.name) for s in smap.surfaces], loc="lower left",
              fontsize=7, framealpha=0.8)
    fig.savefig(out / "surface_map_snapshot.png", dpi=90)
    plt.close(fig)

    # 2. what the tires felt
    t = tr.t[1:]
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for i, w in enumerate(("FL", "FR", "RL", "RR")):
        axs[0].plot(t, tr.info["mu_surface"][:, i], label=w, linewidth=1)
        axs[1].plot(t, tr.info["Fz"][:, i], linewidth=0.8)
    axs[0].set_ylabel("surface mu factor")
    axs[0].legend(ncol=4, fontsize=7)
    axs[1].set_ylabel("wheel load Fz (N)")
    axs[2].plot(ref.t[1:], np.degrees(np.abs(ref.beta[1:])), "--", label="uniform P-tile")
    axs[2].plot(t, np.degrees(np.abs(tr.beta[1:])), label="mixed map")
    axs[2].set_ylabel("|sideslip| (deg)")
    axs[2].set_xlabel("time (s)")
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "surface_map_timeseries.png", dpi=90)
    plt.close(fig)
    print(f"saved {out / 'surface_map_snapshot.png'} and {out / 'surface_map_timeseries.png'}")


if __name__ == "__main__":
    main(ROOT / "docs" / "images" / "m2" if "--docs" in sys.argv else Path(__file__).resolve().parent / "output")
