"""MultiheadAttention matches a manual reference and ignores padded keys."""

import math

import pytest
import torch
import torch.nn.functional as F

from stok.models.attention import MultiheadAttention
from stok.models.encoder import Encoder
from stok.models.rope import RotaryEmbedding, apply_rope


def attention():
    torch.manual_seed(42)
    attn = MultiheadAttention(64, 4, 0.0, RotaryEmbedding(base=10000.0))
    return attn.eval()


def reference(attn, x, padding):
    B, L, _ = x.shape
    q, k, v = (
        t.view(B, L, attn.n_heads, attn.head_dim).transpose(1, 2)
        for t in attn.qkv(x).chunk(3, dim=-1)
    )
    cos, sin = attn.rope.get_cos_sin(
        seq_len=L, head_dim=attn.head_dim, device=x.device, dtype=x.dtype
    )
    q, k = apply_rope(q, k, cos, sin)
    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
    scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
    y = F.softmax(scores, dim=-1) @ v
    return attn.out(y.transpose(1, 2).reshape(B, L, -1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa_matches_manual_attention_with_padding(dtype):
    attn = attention().to(dtype)
    x = torch.randn(2, 16, 64, dtype=dtype)
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[0, -3:] = True
    tol = 1e-5 if dtype == torch.float32 else 0.02
    torch.testing.assert_close(
        attn(x, key_padding_mask=padding),
        reference(attn, x, padding),
        atol=tol,
        rtol=tol,
    )


def test_padded_keys_do_not_change_unpadded_outputs():
    encoder = Encoder(64, 4, 2, 0.0, 0.0).eval()
    x = torch.randn(1, 10, 64)
    padding = torch.zeros(1, 10, dtype=torch.bool)
    padding[:, -4:] = True
    changed = x.clone()
    changed[:, -4:] = torch.randn(1, 4, 64)
    torch.testing.assert_close(
        encoder(x, key_padding_mask=padding)[:, :-4],
        encoder(changed, key_padding_mask=padding)[:, :-4],
    )
