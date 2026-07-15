import torch

from experiments.gcp_mdlm.stage1.decode_sanity import decode_sanity_row


class _StubDecoder:
    """Maps each code vector to a deterministic 3-atom backbone; identical codes -> identical coords."""

    def __call__(self, structure_tokens, mask, true_lengths=None):
        # structure_tokens: (B, L, d_code) -> (B, L, 9)
        b, ll, _ = structure_tokens.shape
        base = structure_tokens.sum(dim=-1, keepdim=True)  # (B, L, 1)
        offsets = torch.arange(9, dtype=torch.float32).view(1, 1, 9)
        return base + offsets


def test_identical_tokens_give_perfect_sanity():
    codebook = torch.randn(16, 8)
    tokens = torch.tensor([0, 5, 9, 2, 7])
    row = decode_sanity_row(tokens, tokens, codebook, _StubDecoder())
    assert row["identical_tokens"] is True
    assert row["rmsd"] < 1e-4
    assert row["lddt"] > 0.99
