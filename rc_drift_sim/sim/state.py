"""State vector layout shared by the NumPy reference and the JAX port.

The state is a flat float array of length NS. Every entry is integrated by the ODE solver.
Wheel order is FL, FR, RL, RR everywhere.
"""
from __future__ import annotations

import numpy as np

# world pose
X, Y, YAW = 0, 1, 2
# body-frame velocities and yaw rate
VX, VY, R = 3, 4, 5
# wheel angular speeds (rad/s)
OMEGA = slice(6, 10)
# actual steering angle of the virtual center front wheel (rad)
DELTA = 10
# motor current (A)
I_MOTOR = 11
# tire temperatures (degC)
T_TIRE = slice(12, 16)
# lagged load transfer (N): longitudinal (+ = rear loaded), lateral (+ = right side loaded)
DFZ_LONG, DFZ_LAT = 16, 17
# relaxed slip quantities fed to the Magic Formula
KAPPA_LAG = slice(18, 22)
ALPHA_LAG = slice(22, 26)
# tire contamination (dust/dirt on the tread, 0..1), decays with distance rolled
CONTAM = slice(26, 30)

NS = 30

POSE = slice(0, 3)
BODY_VEL = slice(3, 6)

STATE_NAMES = (
    "x", "y", "yaw", "vx", "vy", "r",
    "omega_fl", "omega_fr", "omega_rl", "omega_rr",
    "delta", "i_motor",
    "t_fl", "t_fr", "t_rl", "t_rr",
    "dfz_long", "dfz_lat",
    "kappa_lag_fl", "kappa_lag_fr", "kappa_lag_rl", "kappa_lag_rr",
    "alpha_lag_fl", "alpha_lag_fr", "alpha_lag_rl", "alpha_lag_rr",
    "contam_fl", "contam_fr", "contam_rl", "contam_rr",
)
assert len(STATE_NAMES) == NS

FL, FR, RL, RR = 0, 1, 2, 3


def make_state(x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, r=0.0, omega=None, delta=0.0,
               i_motor=0.0, t_tire=25.0, dfz_long=0.0, dfz_lat=0.0,
               kappa_lag=0.0, alpha_lag=0.0, contam=0.0, dtype=np.float64) -> np.ndarray:
    s = np.zeros(NS, dtype=dtype)
    s[X], s[Y], s[YAW] = x, y, yaw
    s[VX], s[VY], s[R] = vx, vy, r
    s[OMEGA] = 0.0 if omega is None else omega
    s[DELTA] = delta
    s[I_MOTOR] = i_motor
    s[T_TIRE] = t_tire
    s[DFZ_LONG], s[DFZ_LAT] = dfz_long, dfz_lat
    s[KAPPA_LAG] = kappa_lag
    s[ALPHA_LAG] = alpha_lag
    s[CONTAM] = contam
    return s


def state_dict(s: np.ndarray) -> dict:
    """Named view of a single state vector (or of a (T, NS) trajectory: values become arrays)."""
    s = np.asarray(s)
    return {name: s[..., i] for i, name in enumerate(STATE_NAMES)}


def batch_shape(s: np.ndarray) -> tuple:
    """Leading (batch) shape of a state array of shape (..., NS)."""
    return np.shape(s)[:-1]


def speed(s: np.ndarray) -> np.ndarray:
    return np.hypot(s[..., VX], s[..., VY])


def sideslip(s: np.ndarray) -> np.ndarray:
    """Vehicle sideslip angle beta = atan2(vy, vx) in rad."""
    return np.arctan2(s[..., VY], s[..., VX])
