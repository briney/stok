"""Time embedding modules for masked diffusion language modeling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding: scalar t -> [d_model] vector.

    Generates sinusoidal features from t (same formula as transformer positional
    encodings but applied to a scalar time), then passes through a 2-layer MLP
    (d_model -> 4*d_model -> d_model, SiLU activation).

    Args:
        d_model: Output embedding dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar diffusion times.

        Args:
            t: Diffusion times in [0, 1], shape [B].

        Returns:
            Time embeddings, shape [B, d_model].
        """
        half_dim = self.d_model // 2
        # Sinusoidal frequencies: exp(-log(10000) * i / half_dim)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )  # [half_dim]
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half_dim]
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, d_model]
        # Handle odd d_model by padding
        if self.d_model % 2 == 1:
            embeddings = torch.cat(
                [embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1
            )
        return self.mlp(embeddings)


class AdaptiveLayerNorm(nn.Module):
    """LayerNorm with time-dependent affine parameters (adaLN).

    Applies standard LayerNorm (without learned affine), then modulates with
    time-dependent scale and shift derived from the time embedding.

    Args:
        d_model: Input feature dimension.
        time_embed_dim: Dimension of the time embedding input.
    """

    def __init__(self, d_model: int, time_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear = nn.Linear(time_embed_dim, 2 * d_model)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """Apply time-conditioned layer normalization.

        Args:
            x: Input tensor, shape [B, L, d_model].
            t_embed: Time embedding, shape [B, time_embed_dim].

        Returns:
            Normalized and modulated tensor, shape [B, L, d_model].
        """
        scale, shift = self.linear(t_embed).chunk(2, dim=-1)  # [B, d_model] each
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
