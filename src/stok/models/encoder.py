import torch
import torch.nn as nn

from .blocks import EncoderBlock, RMSNorm
from .rope import RotaryEmbedding


class Encoder(nn.Module):
    """Stack of encoder blocks with pre-layer norm and shared RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        attn_dropout: float,
        ffn_mult: float = 4.0,
        norm_type: str = "layernorm",
    ):
        """Initialize encoder stack.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers.
            dropout: Dropout probability for residual connections.
            attn_dropout: Dropout probability for attention outputs.
            ffn_mult: Feedforward multiplier (hidden_dim = d_model * ffn_mult).
            norm_type: Normalization type ("layernorm" currently supported).
        """
        super().__init__()
        self.rope = RotaryEmbedding()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attn_dropout=attn_dropout,
                    resid_dropout=dropout,
                    rope=self.rope,
                    norm_type=norm_type,
                    ffn_mult=ffn_mult,
                )
                for _ in range(n_layers)
            ]
        )
        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            Norm = RMSNorm
        elif norm_type == "layernorm":
            Norm = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.final_norm = Norm(d_model)

    def forward(
        self, h: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode [B, L, d_model]; key_padding_mask is True at padding positions."""
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        return self.final_norm(h)
