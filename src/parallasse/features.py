"""The six fixed, PID-blind aggregate features shared by XGB and MLP."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from parallasse.preprocessing import kinematics

FEATURE_NAMES = (
    "multiplicity",
    "pt_dispersion",
    "angularity_beta_0p5",
    "angularity_beta_1",
    "angularity_beta_2",
    "max_pt_fraction",
)
FEATURE_VERSION = "aggregate-v1:phase1-float32-coordinates;float64-reductions;float32-output"


def aggregate_features(particles: NDArray) -> NDArray:
    """Use the fixed particle coordinates, without fitting any statistics.

    Coordinate construction is shared with the particle model. Reductions use
    float64 after the existing float32 coordinate encoding, not a separate
    higher-precision representation unavailable to the PFN.
    """
    encoded, mask = kinematics(particles)
    values = encoded.astype(np.float64)
    z = values[..., 0]
    radius = np.hypot(values[..., 1], values[..., 2])
    output = np.column_stack(
        (
            mask.sum(axis=1),
            np.sqrt(np.sum(z**2, axis=1)),
            np.sum(z * np.sqrt(radius), axis=1),
            np.sum(z * radius, axis=1),
            np.sum(z * radius**2, axis=1),
            z.max(axis=1),
        )
    ).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("Non-finite aggregate feature")
    return output
