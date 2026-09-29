"""RC drift simulator: vectorized 4-wheel vehicle dynamics for a 1/10-scale RC drift car.

The most used entry points are importable from the package root:

    from rc_drift_sim import make_vehicle, VehicleBatch, default_params, run_batch, default_spec

    car = make_vehicle("hard_plastic_drift", "epoxy_ptile")
    traj = car.rollout(car.initial_state(), actions)        # actions: (T, 2) steer, throttle in [-1, 1]

Subpackages: ``sim`` (physics), ``control`` (maneuvers, LQR), ``datagen`` (batch data export),
``viz`` (rendering and plots). See README.md and docs/.
"""

__version__ = "0.1.0"

# name -> (module, attribute); resolved on first access so ``import rc_drift_sim`` stays light
_API = {
    "Params": ("rc_drift_sim.sim.params", "Params"),
    "TireCondition": ("rc_drift_sim.sim.params", "TireCondition"),
    "default_params": ("rc_drift_sim.sim.params", "default_params"),
    "load_tires": ("rc_drift_sim.sim.params", "load_tires"),
    "load_surfaces": ("rc_drift_sim.sim.params", "load_surfaces"),
    "Vehicle": ("rc_drift_sim.sim.vehicle", "Vehicle"),
    "VehicleBatch": ("rc_drift_sim.sim.vehicle", "VehicleBatch"),
    "make_vehicle": ("rc_drift_sim.sim.vehicle", "make_vehicle"),
    "Trajectory": ("rc_drift_sim.sim.vehicle", "Trajectory"),
    "default_spec": ("rc_drift_sim.datagen", "default_spec"),
    "validate_spec": ("rc_drift_sim.datagen", "validate_spec"),
    "run_batch": ("rc_drift_sim.datagen", "run_batch"),
}
__all__ = ["__version__", *_API]


def __getattr__(name: str):
    if name in _API:
        import importlib
        mod, attr = _API[name]
        value = getattr(importlib.import_module(mod), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rc_drift_sim' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
