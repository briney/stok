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
            rope_base: RoPE base frequency.
            rope_dim: RoPE dimensionality (must be even). If None, uses head_dim.
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
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass through encoder stack.

        Args:
            h: Input tensor of shape [B, L, d_model].
            key_padding_mask: Padding mask of shape [B, L] where True marks
                padding positions. Defaults to None.
            attn_mask: Attention mask of shape [B, H, L, S] or [B, 1, L, S].
                Defaults to None.
            output_attentions: If True, also returns attention weights from
                all layers. Defaults to False.
            output_hidden_states: If True, also returns hidden states from
                all layers (including initial embeddings). Defaults to False.

        Returns:
            Depends on output_attentions and output_hidden_states flags:
            - Neither: Output tensor of shape [B, L, d_model].
            - output_attentions only: (output, all_attentions) where
                all_attentions is a tuple of n_layers tensors [B, H, L, L].
            - output_hidden_states only: (output, all_hidden_states) where
                all_hidden_states is a tuple of n_layers+1 tensors [B, L, d_model].
            - Both: (output, all_attentions, all_hidden_states).
        """
        all_attentions: list[torch.Tensor] | None = [] if output_attentions else None
        all_hidden_states: list[torch.Tensor] | None = (
            [] if output_hidden_states else None
        )

        # Collect initial hidden state (before any layers)
        if output_hidden_states:
            all_hidden_states.append(h)

        for layer in self.layers:
            layer_output = layer(
                h,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                h, attn_weights = layer_output
                all_attentions.append(attn_weights)
            else:
                h = layer_output

            # Collect hidden state after each layer (before final norm)
            if output_hidden_states:
                all_hidden_states.append(h)

        h = self.final_norm(h)

        # Build return value based on flags
        if output_attentions and output_hidden_states:
            return h, tuple(all_attentions), tuple(all_hidden_states)
        elif output_attentions:
            return h, tuple(all_attentions)
        elif output_hidden_states:
            return h, tuple(all_hidden_states)
        return h
