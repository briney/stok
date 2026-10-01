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
        for position in range(3):
            assert torch.allclose(gathered[b, position], E[idx[b, position]])


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


def test_token_aligned_decoder_excludes_boundaries_and_empty_rows():
    from stok.utils.decoding import decode_token_aligned_coords
    from stok.utils.masking import residue_mask_from_tokens

    tokens = torch.tensor([[0, 4, 3, 31, 2, 1], [0, 2, 1, 1, 1, 1]])
    mask = residue_mask_from_tokens(tokens, pad_id=1, bos_id=0, eos_id=2)
    codes = torch.randn(2, 6, 9, requires_grad=True)

    def decoder(x, mask):
        assert x.shape == (1, 4, 9)
        assert mask.tolist() == [[True, True, True, False]]
        return x * 2

    coords = decode_token_aligned_coords(decoder, codes, mask)
    assert torch.isnan(coords[1]).all()
    assert torch.isnan(coords[0, [0, 4, 5]]).all()
    coords[0, 1:4].sum().backward()
    assert torch.isfinite(codes.grad).all()
    assert (codes.grad[0, 1:4] == 2).all()
    assert (codes.grad[1] == 0).all()


def test_missing_indices_are_zero_and_other_invalid_values_are_rejected():
    codebook = torch.arange(12).reshape(4, 3).float()
    indices = torch.tensor([[0, -1, 3]])
    actual = indices_to_codes(codebook, indices, allow_missing=True)
    assert torch.equal(
        actual[0], torch.stack([codebook[0], torch.zeros(3), codebook[3]])
    )
    with pytest.raises(ValueError):
        indices_to_codes(codebook, indices)
    for invalid in (
        torch.tensor([[-2]]),
        torch.tensor([[4]]),
        torch.tensor([[1.0]]),
        torch.tensor([[True]]),
    ):
        with pytest.raises(ValueError):
            indices_to_codes(codebook, invalid, allow_missing=True)


def test_structure_decode_retains_holes_and_skips_all_missing_attention():
    from stok.utils.decoding import decode_structure_tokens

    codebook = torch.randn(4, 9, requires_grad=True)
    indices = torch.tensor([[1, -1, 3, -1], [-1, -1, -1, -1]])
    residues = torch.tensor([[True, True, True, False], [True, True, False, False]])
    calls = []

    def decoder(codes, *, mask):
        calls.append(mask.clone())
        return codes * 2

    result = decode_structure_tokens(decoder, codebook, indices, residue_mask=residues)
    assert len(calls) == 1 and calls[0].tolist() == [[True, False, True, False]]
    assert torch.isnan(result[1]).all() and torch.isnan(result[0, [1, 3]]).all()
    result[0, [0, 2]].sum().backward()
    assert torch.isfinite(codebook.grad).all()
    calls.clear()
    result = decode_structure_tokens(
        decoder, codebook, torch.full_like(indices, -1), residue_mask=residues
    )
    assert torch.isnan(result).all() and not calls


def test_decoder_prefix_override_and_validation_keep_gradients():
    from stok.models.decoder import GeometricDecoder

    decoder = GeometricDecoder(32, 4, 1, 1, 8, 9).eval()
    codes = torch.randn(2, 8, 9, requires_grad=True)
    holes = torch.ones(2, 8, dtype=torch.bool)
    holes[:, 2] = False
    lengths = torch.tensor([6, 4])
    prefix = torch.arange(8)[None] < lengths[:, None]
    actual = decoder(codes, holes, true_lengths=lengths)
    torch.testing.assert_close(actual, decoder(codes, prefix))
    assert not torch.allclose(actual, decoder(codes, holes))
    actual.sum().backward()
    assert torch.isfinite(codes.grad).all()
    for invalid in (
        torch.tensor([6]),
        torch.tensor([9, 4]),
        torch.tensor([-1, 4]),
        torch.tensor([6.0, 4.0]),
    ):
        with pytest.raises(ValueError):
            decoder(codes, holes, true_lengths=invalid)
