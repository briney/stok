import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rope


class MultiheadAttention(nn.Module):
    """Multi-head self-attention using PyTorch SDPA with RoPE.

    Uses `scaled_dot_product_attention` to leverage backend optimizations
    (Flash/Memory-Efficient attention) when available.
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
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Self-attention over [B, L, d_model]; key_padding_mask is True at padding."""
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

        # Additive SDPA mask [B,1,1,S]; its dtype must match the query (DeepSpeed).
        sdpa_mask = None
        if key_padding_mask is not None:
            sdpa_mask = key_padding_mask[:, None, None, :].float()
            sdpa_mask = sdpa_mask.masked_fill(sdpa_mask > 0, float("-inf"))
            sdpa_mask = sdpa_mask.to(dtype=q.dtype)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, H, L, D]
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(y)
