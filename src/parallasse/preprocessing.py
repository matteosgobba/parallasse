"""Training-only categorical vocabulary and deterministic particle coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from parallasse.data import particle_mask

PADDING_TOKEN = 0
UNKNOWN_TOKEN = 1


def fit_pid_vocabulary(training_particles: NDArray) -> list[int]:
    """Caller supplies source training rows, never tuning or target rows."""
    mask = particle_mask(training_particles)
    return np.unique(training_particles[..., 3][mask]).astype(np.int64).tolist()


def encode_pid(particles: NDArray, vocabulary: list[int], include_identity: bool = True) -> NDArray:
    mask = particle_mask(particles)
    if vocabulary != sorted(set(vocabulary)):
        raise ValueError("PID vocabulary must be unique and sorted")
    tokens = np.zeros(mask.shape, dtype=np.int64)
    tokens[mask] = UNKNOWN_TOKEN
    if include_identity:
        for token, pid in enumerate(vocabulary, start=2):
            tokens[mask & (particles[..., 3] == pid)] = token
    return tokens


def kinematics(particles: NDArray) -> tuple[NDArray, NDArray]:
    """Return [z, centered rapidity, wrapped centered azimuth] and a mask.

    Centers are the pt-weighted rapidity mean and circular azimuth mean.
    Accumulation uses float64; outputs use float32. All padded outputs are zero.
    """
    mask = particle_mask(particles)
    values = particles.astype(np.float64, copy=False)
    pt = values[..., 0]
    totals = pt.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(totals)):
        raise ValueError("Non-finite transverse-momentum sum")
    z = pt / totals
    center_y = np.sum(z * values[..., 1], axis=1, keepdims=True)
    sin_sum = np.sum(z * np.sin(values[..., 2]), axis=1, keepdims=True)
    cos_sum = np.sum(z * np.cos(values[..., 2]), axis=1, keepdims=True)
    if np.any(np.hypot(sin_sum, cos_sum) < 1e-12):
        raise ValueError("Ambiguous circular azimuth center")
    center_phi = np.arctan2(sin_sum, cos_sum)
    delta_phi = values[..., 2] - center_phi
    wrapped_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
    encoded = np.stack((z, values[..., 1] - center_y, wrapped_phi), axis=-1)
    encoded[~mask] = 0.0
    if not np.all(np.isfinite(encoded)):
        raise ValueError("Non-finite transformed coordinate")
    return encoded.astype(np.float32), mask
