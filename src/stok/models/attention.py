import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rope


class MultiheadAttention(nn.Module):
    """Multi-head self-attention using PyTorch SDPA with RoPE.

    Uses `scaled_dot_product_attention` to leverage backend optimizations
    (Flash/Memory-Efficient attention) when available. Falls back to a
    manual implementation when attention weights are requested.
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float, rope: RotaryEmbedding
    ):
        """Initialize multi-head attention with RoPE.

        Args:
            d_model: Model dimension (must be divisible by n_heads).
            n_heads: Number of attention heads.
            dropout: Dropout probability for attention outputs.
            rope: Rotary positional embedding instance.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head attention with RoPE.

        Args:
            x: Input tensor of shape [B, L, d_model].
            key_padding_mask: Padding mask of shape [B, L] where True marks
                padding positions. Defaults to None.
            attn_mask: Attention mask of shape [B, H, L, S] or [B, 1, L, S].
                Additive (negative values) or boolean masks are supported.
                Defaults to None.
            need_weights: If True, returns attention weights along with output.
                Uses a slower manual implementation instead of optimized SDPA.
                Defaults to False.

        Returns:
            If need_weights=False: Output tensor of shape [B, L, d_model].
            If need_weights=True: Tuple of (output, attention_weights) where
                attention_weights has shape [B, H, L, S].
        """
        B, L, _ = x.shape
        qkv = self.qkv(x)  # [B, L, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # RoPE on q,k
        cos, sin = self.rope.get_cos_sin(
            seq_len=L, head_dim=self.head_dim, device=x.device, dtype=x.dtype
        )
        q, k = apply_rope(q, k, cos, sin)

        # Build SDPA masks. key_padding_mask: [B, L] -> [B, 1, 1, S] (True = masked)
        sdpa_mask = attn_mask
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :]  # [B,1,1,S]
            if sdpa_mask is None:
                sdpa_mask = kpm
            else:
                # if an attention mask is provided, combine it with key padding.
                if sdpa_mask.dtype == torch.bool:
                    sdpa_mask = sdpa_mask | kpm
                else:
                    # assume additive mask; add -inf where kpm is True
                    sdpa_mask = sdpa_mask + kpm.to(sdpa_mask.dtype) * float("-inf")

        # convert boolean mask to additive mask expected by SDPA
        if sdpa_mask is not None and sdpa_mask.dtype == torch.bool:
            sdpa_mask = sdpa_mask.float()
            sdpa_mask = sdpa_mask.masked_fill(sdpa_mask > 0, float("-inf"))
            # for DeepSpeed, the dtype of the mask must match the dtype of the query
            sdpa_mask = sdpa_mask.to(dtype=q.dtype)

        if need_weights:
            # Manual attention computation to capture weights
            y, attn_weights = self._attention_with_weights(q, k, v, sdpa_mask)
        else:
            # sdpa expects [B, H, L, D] - use optimized kernels
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )  # [B, H, L, D]
            attn_weights = None

        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.out(y)

        if need_weights:
            return output, attn_weights
        return output

    def _attention_with_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention with explicit weight calculation.

        Args:
            q: Query tensor [B, H, L, D]
            k: Key tensor [B, H, S, D]
            v: Value tensor [B, H, S, D]
            attn_mask: Additive attention mask [B, 1, L, S] or [B, H, L, S]

        Returns:
            Tuple of (output [B, H, L, D], attention_weights [B, H, L, S])
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, S]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout during training
        attn_weights_dropped = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        output = torch.matmul(attn_weights_dropped, v)  # [B, H, L, D]

        # Return pre-dropout weights for interpretability
        return output, attn_weights
