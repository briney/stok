import torch

from stok.cli.train import _tokenize_and_align
from stok.utils.tokenizer import Tokenizer


def test_tokenize_and_align_alignment_and_ignore_index():
    tokenizer = Tokenizer()
    max_len = 8
    ignore_index = -100
    pad_id = 1

    seq = "LAGVSEK"  # short sequence
    indices = torch.tensor([3, 1, 4, 1], dtype=torch.long)

    tokens, labels = _tokenize_and_align(
        [{"sequence": seq, "structure_tokens": indices}],
        tokenizer,
        max_len=max_len,
        ignore_index=ignore_index,
        pad_id=pad_id,
    )

    assert tokens.shape == (1, max_len)
    assert labels.shape == (1, max_len)

    L = tokens.shape[1]
    copy_len = min(len(indices), max(0, L - 2))

    # BOS position (0) must be ignored
    assert labels[0, 0].item() == ignore_index
    # The supervised span must start at position 1 and match indices (truncated if needed)
    if copy_len > 0:
        assert torch.equal(labels[0, 1 : 1 + copy_len], indices[:copy_len])
    # Any positions beyond the supervised span at least include EOS/PAD which should be ignored
    assert (labels[0, 1 + copy_len :] == ignore_index).any().item()


def test_tokenize_and_align_ignores_negative_indices():
    tokenizer = Tokenizer()
    max_len = 8
    ignore_index = -100
    pad_id = 1

    seq = "LAGVSEK"
    # Include trailing -1 values that should be ignored
    raw_indices = torch.tensor([2, 5, 1, -1, -1, -1], dtype=torch.long)

    tokens, labels = _tokenize_and_align(
        [{"sequence": seq, "structure_tokens": raw_indices}],
        tokenizer,
        max_len=max_len,
        ignore_index=ignore_index,
        pad_id=pad_id,
    )

    assert tokens.shape == (1, max_len)
    assert labels.shape == (1, max_len)

    L = tokens.shape[1]
    valid = raw_indices[raw_indices >= 0]
    copy_len = min(int(valid.numel()), max(0, L - 2))

    # BOS must be ignored
    assert labels[0, 0].item() == ignore_index
    # Only non-negative indices should be copied starting at position 1
    if copy_len > 0:
        assert torch.equal(labels[0, 1 : 1 + copy_len], valid[:copy_len])
    # All positions beyond the supervised span should remain ignored
    assert torch.all(labels[0, 1 + copy_len :] == ignore_index).item()



def test_internal_gap_and_mixed_coordinates_keep_residue_identity():
    from stok.data.collate import mlm_collate
    tokenizer = Tokenizer()
    coords = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
    batch = [{'sequence_id': 'p', 'sequence': 'LAG', 'structure_tokens': torch.tensor([7, -1, 9]), 'coords': coords},
             {'sequence_id': 'q', 'sequence': 'LA', 'structure_tokens': torch.tensor([1, 2])}]
    tokens, labels, aligned = _tokenize_and_align(batch, tokenizer, max_len=6,
        ignore_index=-100, pad_id=1, num_classes=10)
    assert labels[0].tolist() == [-100, 7, -100, 9, -100, -100]
    assert aligned.shape == (2, 6, 3, 3)
    torch.testing.assert_close(aligned[0, 1:4], coords)
    assert torch.isnan(aligned[0, [0, 4, 5]]).all()
    assert torch.isnan(aligned[1]).all()
    _, _, mlm_coords = mlm_collate(batch, tokenizer, max_len=6)
    torch.testing.assert_close(aligned, mlm_coords, equal_nan=True)


def test_alignment_rejects_control_tokens_and_invalid_classes():
    import pytest
    tokenizer = Tokenizer()
    for seq, indices in [('L<mask>G', [1, 2, 3]), ('LAG', [1, 2, 10])]:
        with pytest.raises(ValueError):
            _tokenize_and_align([{'sequence_id': 'bad', 'sequence': seq, 'structure_tokens': torch.tensor(indices)}],
                tokenizer, max_len=6, ignore_index=-100, pad_id=1, num_classes=10)


def test_alignment_truncates_before_eos():
    coords = torch.arange(54, dtype=torch.float32).reshape(6, 3, 3)
    _, labels, aligned = _tokenize_and_align(
        [{'sequence': 'LAGVSE', 'structure_tokens': torch.arange(6), 'coords': coords}],
        Tokenizer(), max_len=5, ignore_index=-100, pad_id=1)
    assert labels.tolist() == [[-100, 0, 1, 2, -100]]
    torch.testing.assert_close(aligned[0, 1:4], coords[:3])
    assert torch.isnan(aligned[0, [0, 4]]).all()
