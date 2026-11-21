"""Particle Flow Network with matched categorical capacity in both variants."""

from __future__ import annotations

import torch
from torch import nn

from parallasse.preprocessing import PADDING_TOKEN, UNKNOWN_TOKEN


def _network(input_width: int, widths: list[int], scalar_output: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in widths:
        layers.extend((nn.Linear(input_width, width), nn.ReLU()))
        input_width = width
    if scalar_output:
        layers.append(nn.Linear(input_width, 1))
    return nn.Sequential(*layers)


class ParticleFlowNetwork(nn.Module):
    """Encode only real constituents; sum embeddings by jet before classification."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_width: int,
        particle_widths: list[int],
        head_widths: list[int],
        use_pid: bool,
    ) -> None:
        super().__init__()
        self.use_pid = use_pid
        self.identity = nn.Embedding(
            vocabulary_size + 2, embedding_width, padding_idx=PADDING_TOKEN
        )
        self.particle_encoder = _network(3 + embedding_width, particle_widths)
        self.jet_head = _network(particle_widths[-1], head_widths, scalar_output=True)

    def forward(
        self, coordinates: torch.Tensor, mask: torch.Tensor, pid_tokens: torch.Tensor
    ) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("Coordinates must have shape (jets, particles, 3)")
        if mask.shape != coordinates.shape[:2] or pid_tokens.shape != mask.shape:
            raise ValueError("Mask and PID token axes must match the particle axes")
        if mask.dtype != torch.bool or pid_tokens.dtype != torch.long:
            raise ValueError("Expected Boolean mask and long PID tokens")

        # Gather before the biased encoder, so padded values never contribute.
        jet, constituent = mask.nonzero(as_tuple=True)
        tokens = (
            pid_tokens[jet, constituent] if self.use_pid else torch.full_like(jet, UNKNOWN_TOKEN)
        )
        inputs = torch.cat((coordinates[jet, constituent], self.identity(tokens)), dim=1)
        encoded = self.particle_encoder(inputs)
        pooled = encoded.new_zeros((coordinates.shape[0], encoded.shape[1]))
        pooled = pooled.index_add(0, jet, encoded)
        return self.jet_head(pooled).squeeze(-1)
