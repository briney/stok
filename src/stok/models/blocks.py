from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiheadAttention
from .mlp import SwiGLU
from .time_embed import AdaptiveLayerNorm


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean subtraction)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * inv_rms.to(dtype=x.dtype)
        return x * self.weight.to(dtype=x.dtype)


class EncoderBlock(nn.Module):
    """Transformer encoder block (pre-layer norm + MHA + SwiGLU)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        resid_dropout: float,
        rope: object,
        norm_type: str = "layernorm",
        ffn_mult: float = 4.0,
        time_conditioning: str | None = None,
        time_embed_dim: int | None = None,
    ):
        """Initialize Pre-LN encoder block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            attn_dropout: Dropout probability for attention outputs.
            resid_dropout: Dropout probability for residual connections.
            rope: Rotary positional embedding instance.
            norm_type: Normalization type ("layernorm" or "rmsnorm").
            ffn_mult: Feedforward multiplier (hidden_dim = d_model * ffn_mult).
            time_conditioning: Time conditioning mode. "adaln" replaces
                LayerNorm with AdaptiveLayerNorm. None uses standard norms.
            time_embed_dim: Dimension of time embedding for adaLN. Defaults
                to d_model if not specified.
        """
        super().__init__()
        self._time_conditioning = time_conditioning

        if time_conditioning == "adaln":
            t_dim = time_embed_dim if time_embed_dim is not None else d_model
            self.norm1 = AdaptiveLayerNorm(d_model, t_dim)
            self.norm2 = AdaptiveLayerNorm(d_model, t_dim)
        else:
            norm_type = norm_type.lower()
            if norm_type == "rmsnorm":
                Norm = RMSNorm
            elif norm_type == "layernorm":
                Norm = nn.LayerNorm
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
            self.norm1 = Norm(d_model)
            self.norm2 = Norm(d_model)

        self.attn = MultiheadAttention(
            d_model=d_model, n_heads=n_heads, dropout=attn_dropout, rope=rope
        )
        self.drop1 = nn.Dropout(resid_dropout)

        self.mlp = SwiGLU(d_model=d_model, expansion=ffn_mult, dropout=resid_dropout)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        t_embed: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder block.

        Args:
            x: Input tensor of shape [B, L, d_model].
            key_padding_mask: Padding mask of shape [B, L] where True marks
                padding positions. Defaults to None.
            attn_mask: Attention mask of shape [B, H, L, S] or [B, 1, L, S].
                Defaults to None.
            output_attentions: If True, also returns attention weights.
                Defaults to False.
            t_embed: Time embedding of shape [B, time_embed_dim]. Required when
                time_conditioning="adaln", ignored otherwise.

        Returns:
            If output_attentions=False: Output tensor of shape [B, L, d_model].
            If output_attentions=True: Tuple of (output, attention_weights) where
                attention_weights has shape [B, H, L, L].
        """
        if self._time_conditioning == "adaln":
            h = self.norm1(x, t_embed)
        else:
            h = self.norm1(x)

        attn_output = self.attn(
            h,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=output_attentions,
        )

        if output_attentions:
            h, attn_weights = attn_output
        else:
            h = attn_output
            attn_weights = None

        x = x + self.drop1(h)

        if self._time_conditioning == "adaln":
            h = self.norm2(x, t_embed)
        else:
            h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.drop2(h)

        if output_attentions:
            return x, attn_weights
        return x
