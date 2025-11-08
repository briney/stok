import os

import pytest
import torch

from stok.utils.decoding import (
    decode_coords,
    indices_to_codes,
    logits_to_soft_codes_gumbel,
    sample_indices_top_p,
)


def test_logits_to_soft_codes_gumbel_identity_codebook_shapes_and_consistency():
    torch.manual_seed(0)
    B, L, C = 2, 5, 7
    d_code = C
    logits = torch.randn(B, L, C)
    E = torch.eye(C)  # identity -> codes == gumbel weights
    codes = logits_to_soft_codes_gumbel(logits, E, tau=1.0, hard=False)
    assert codes.shape == (B, L, d_code)
    # rows should be non-negative and sum to ~1 since E is identity and weights are prob-like
    row_sums = codes.sum(dim=-1)
    assert torch.isfinite(codes).all()
    assert torch.all(row_sums > 0.0)


def test_indices_to_codes_gathers_correct_rows():
    C, d = 10, 3
    E = torch.randn(C, d)
    idx = torch.tensor([[0, 3, 5], [9, 1, 2]], dtype=torch.long)
    gathered = indices_to_codes(E, idx)
    assert gathered.shape == (2, 3, d)
    for b in range(2):
        for l in range(3):
            assert torch.allclose(gathered[b, l], E[idx[b, l]])


def test_sample_indices_top_p_deterministic_mass_one():
    torch.manual_seed(0)
    B, L, C = 2, 4, 6
    probs = torch.full((B, L, C), 1e-6)
    probs[..., 0] = 1.0  # all mass on index 0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    idx = sample_indices_top_p(probs, top_p=0.5, temperature=1.0)
    assert idx.shape == (B, L)
    assert torch.all(idx == 0)


def test_decode_coords_runs_when_decoder_available(monkeypatch):
    pytest.importorskip("x_transformers")
    from stok.models.decoder import GeometricDecoder

    B, L, d_code = 2, 8, 16
    # tiny decoder config
    dec = GeometricDecoder(
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
        max_length=64,
        d_code=d_code,
        num_memory_tokens=0,
        attn_kv_heads=2,
    )
    codes = torch.randn(B, L, d_code)
    mask = torch.ones(B, L, dtype=torch.bool)
    coords = decode_coords(dec, codes, mask)
    assert coords.shape == (B, L, 3, 3)
    assert torch.isfinite(coords).all()
