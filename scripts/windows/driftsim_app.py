"""Entry point of the DriftSim executable (built with PyInstaller, see build_exe.ps1).

    DriftSim.exe                 start the dataset GUI and open it in the browser; close the window to stop
    DriftSim.exe --port 9000     any option of ``driftsim-gui`` (``--out``, ``--no-browser``, ...)
    DriftSim.exe --self-test     simulate a small batch on 2 worker processes and exit (build check)

Datasets go to Documents\\DriftSim\\exports unless ``--out`` is given (inside the bundle there is no
repository folder to write to).
"""
from __future__ import annotations

import multiprocessing
import sys
import tempfile
from pathlib import Path


def export_dir() -> Path:
    d = Path.home() / "Documents" / "DriftSim" / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def self_test() -> int:
    """Run 16 short episodes on 2 worker processes (exercises the frozen multiprocessing start-up,
    the bundled configs and the exporter) and report the result."""
    from rc_drift_sim.datagen import default_spec, run_batch
    from rc_drift_sim.sim.xp import available_devices
    spec = default_spec()
    spec.update(name="self_test", episodes=16, duration_s=1.0, seed=1)
    spec["params"] = {"vehicle.mass": {"dist": "uniform", "low": 1.4, "high": 1.8}}
    spec["maneuver"] = {"type": "random", "params": {}}
    spec["export"]["shard_episodes"] = 8
    with tempfile.TemporaryDirectory() as tmp:
        res = run_batch(spec, out_root=tmp, workers=2)
        ok = res["status"] == "complete" and res["episodes_written"] == 16 and "all/episodes.csv" in res["files"]
        print(f"self-test: {res['status']}, {res['episodes_written']} episodes, {len(res['files'])} files, "
              f"devices {[d for d, on in available_devices().items() if on]}")
    from rc_drift_sim.app.rl_runs import available
    rl_ok, why = available()
    if rl_ok:                                   # bundled PyTorch: one tiny PPO iteration end to end
        from rc_drift_sim.rl import EnvConfig
        from rc_drift_sim.rl.ppo import PPOConfig, train
        from rc_drift_sim.rl.visual import record_rollout
        model, hist = train(EnvConfig(task="hold", episode_s=0.5), PPOConfig(total_steps=256, num_envs=16, rollout=16,
                                                                              epochs=1, minibatches=2), log=None)
        ep = record_rollout(EnvConfig(task="hold", episode_s=0.5), model.act)
        rl_ok = len(hist) == 1 and ep["steps"] > 0
        print(f"self-test RL: {'ok' if rl_ok else 'FAILED'} (PPO iteration, recorded episode of {ep['steps']} steps)")
        ok = ok and rl_ok
    else:
        print(f"self-test RL: off ({why})")
    print("self-test passed" if ok else "self-test FAILED")
    return 0 if ok else 1


def main() -> int:
    multiprocessing.freeze_support()        # worker processes of the frozen app start here too
    args = sys.argv[1:]
    if "--self-test" in args:
        return self_test()
    from rc_drift_sim.app.__main__ import main as gui
    if "--out" not in args:
        args += ["--out", str(export_dir())]
    return gui(args)


if __name__ == "__main__":
    sys.exit(main())
