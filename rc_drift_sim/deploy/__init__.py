"""Running a trained drift policy outside the trainer: on the car (e.g. a Jetson Orin Nano) or in a
software-in-the-loop test against the simulator.

    from rc_drift_sim.deploy import PolicyRuntime
    rt = PolicyRuntime("rl_runs/<run>/final")             # folder written by rl.export
    cmd = rt.step(sensors)                                # dict of SI sensor readings -> Command

Only NumPy is needed here (no PyTorch, no Gymnasium), so the runtime is light enough for the car.
``safety`` holds the safety filter that training and the car share; ``car_loop`` the 50 Hz loop
(``driftsim-drive``) with the hardware interface to fill in for your sensors and actuators.
"""
from __future__ import annotations

from .safety import SafetyConfig, safety_filter

__all__ = ["SafetyConfig", "safety_filter", "PolicyRuntime", "Command"]


def __getattr__(name: str):
    if name in ("PolicyRuntime", "Command"):
        from . import runtime
        return getattr(runtime, name)
    raise AttributeError(name)
