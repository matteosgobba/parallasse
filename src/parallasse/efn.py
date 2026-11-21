"""Energy Flow Network with direction-only encoding and energy-weighted pooling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from parallasse.pfn import _network
from parallasse.preprocessing import kinematics
from parallasse.source_neural import neural_scores, pfn_prediction_checks


class EnergyFlowNetwork(nn.Module):
    """Apply Phi to directions, sum z*Phi, then apply the jet head."""

    def __init__(self, particle_widths: list[int], head_widths: list[int]) -> None:
        super().__init__()
        self.particle_encoder = _network(2, particle_widths)
        self.jet_head = _network(particle_widths[-1], head_widths, scalar_output=True)
        # Centered directions need active angular features at initialization.
        for network in (self.particle_encoder, self.jet_head):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.jet_head[-1].weight)

    def forward(self, coordinates, mask, pid_tokens=None):
        # Optional tokens preserve compatibility with existing particle checks.
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("Coordinates must have shape (jets, particles, 3)")
        if mask.shape != coordinates.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("A Boolean mask must match the particle axes")
        jet, constituent = mask.nonzero(as_tuple=True)
        real = coordinates[jet, constituent]
        encoded = self.particle_encoder(real[:, 1:])
        weighted = real[:, :1] * encoded
        pooled = weighted.new_zeros((coordinates.shape[0], weighted.shape[1]))
        pooled = pooled.index_add(0, jet, weighted)
        return self.jet_head(pooled).squeeze(-1)


def efn_prediction_checks(model, particles: np.ndarray, tolerances: dict) -> dict:
    """Check the model and shared physical preprocessing on source probes."""
    raw = particles[:64].copy()
    coordinates, mask = kinematics(raw)
    tokens = torch.zeros(mask.shape, dtype=torch.long)
    result = pfn_prediction_checks(
        model, (torch.from_numpy(coordinates), torch.from_numpy(mask), tokens)
    )

    def scores(values):
        c, m = kinematics(values)
        return neural_scores(model, (torch.from_numpy(c), torch.from_numpy(m)), 64)

    original = scores(raw)
    atol, rtol = tolerances["probability_atol"], tolerances["probability_rtol"]
    first = (raw[..., 0] > 0).argmax(axis=1)
    split = np.pad(raw, ((0, 0), (0, 1), (0, 0)))
    for i, position in enumerate(first):
        split[i, -1] = raw[i, position]
        split[i, position, 0] *= 0.37
        split[i, -1, 0] *= 0.63
    collinear = scores(split)
    np.testing.assert_allclose(original, collinear, atol=atol, rtol=rtol)
    changed = raw.copy()
    changed[..., 3][mask] = 13
    np.testing.assert_array_equal(original, scores(changed))
    scaled = raw.copy()
    scaled[..., 0] *= 7
    np.testing.assert_allclose(original, scores(scaled), atol=atol, rtol=rtol)
    soft = []
    for epsilon in tolerances["soft_relative_pt"]:
        ghost = np.pad(raw, ((0, 0), (0, 1), (0, 0)))
        for i, position in enumerate(first):
            ghost[i, -1] = raw[i, position]
            ghost[i, -1, 0] = epsilon * raw[i, :, 0].sum()
            ghost[i, -1, 1:3] += [0.12, 0.18]
        soft.append(
            {
                "relative_pt": epsilon,
                "max_abs_probability_delta": float(np.max(np.abs(original - scores(ghost)))),
            }
        )
    if soft[-1]["max_abs_probability_delta"] > tolerances["soft_limit_atol"]:
        raise ValueError("EFN soft limit exceeds the declared numerical tolerance")
    result.update(
        collinear_max_abs_probability_delta=float(np.max(np.abs(original - collinear))),
        pid_change_probability_delta=0.0,
        soft_limit=soft,
        soft_limit_atol=tolerances["soft_limit_atol"],
    )
    return result
