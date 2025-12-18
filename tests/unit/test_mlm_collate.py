"""Tests for MLM collate function."""

import torch

from stok.data.collate import mlm_collate
from stok.utils.tokenizer import Tokenizer


def test_mlm_collate_returns_correct_shapes():
    """Test that MLM collate returns tensors with expected shapes."""
    tokenizer = Tokenizer()
    batch = [
        {"pid": "test1", "seq": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF"},
        {"pid": "test2", "seq": "MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQLKDF"},
    ]
    max_len = 64

    tokens, labels = mlm_collate(
        batch,
        tokenizer,
        max_len=max_len,
        mask_prob=0.15,
        pad_id=1,
        ignore_index=-100,
    )

    assert tokens.shape == (2, max_len)
    assert labels.shape == (2, max_len)
    assert tokens.dtype == torch.long
    assert labels.dtype == torch.long


def test_mlm_collate_masks_approximately_correct_fraction():
    """Test that MLM collate masks approximately the expected fraction of tokens."""
    tokenizer = Tokenizer()
    # Use longer sequences for more stable statistics
    batch = [
        {"pid": f"test{i}", "seq": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFA" * 3}
        for i in range(8)
    ]
    max_len = 128
    mask_prob = 0.15

    tokens, labels = mlm_collate(
        batch,
        tokenizer,
        max_len=max_len,
        mask_prob=mask_prob,
        pad_id=1,
        ignore_index=-100,
    )

    # Count masked positions (labels != -100)
    masked_count = (labels != -100).sum().item()
    # Count non-padding, non-special positions
    non_pad = (tokens != 1).sum().item()  # exclude pad
    non_special = non_pad - 2 * len(batch)  # subtract CLS and EOS per sequence

    # Check that mask ratio is approximately correct (within reasonable variance)
    actual_ratio = masked_count / max(1, non_special)
    assert 0.05 < actual_ratio < 0.30, f"Mask ratio {actual_ratio:.3f} outside expected range"


def test_mlm_collate_applies_mask_token():
    """Test that MLM collate applies the <mask> token to some positions."""
    tokenizer = Tokenizer()
    batch = [
        {"pid": "test1", "seq": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFADD"},
    ]
    max_len = 64

    # Run multiple times to ensure mask tokens appear (due to randomness)
    mask_token_found = False
    for _ in range(10):
        tokens, labels = mlm_collate(
            batch,
            tokenizer,
            max_len=max_len,
            mask_prob=0.15,
            mask_token_prob=0.8,
            pad_id=1,
            mask_id=31,  # <mask> token
            ignore_index=-100,
        )
        if (tokens == 31).any():
            mask_token_found = True
            break

    assert mask_token_found, "No <mask> tokens found after multiple iterations"


def test_mlm_collate_never_masks_special_tokens():
    """Test that MLM collate never masks special tokens (CLS, PAD, EOS, UNK)."""
    tokenizer = Tokenizer()
    batch = [
        {"pid": "test1", "seq": "LAGVSER"},
    ]
    max_len = 16
    special_token_ids = {0, 1, 2, 3}  # CLS, PAD, EOS, UNK

    # Run multiple times to check
    for _ in range(20):
        tokens, labels = mlm_collate(
            batch,
            tokenizer,
            max_len=max_len,
            mask_prob=0.5,  # High mask prob to stress test
            pad_id=1,
            ignore_index=-100,
            special_token_ids=special_token_ids,
        )

        # Labels at special token positions should always be ignore_index
        for special_id in special_token_ids:
            special_positions = tokens == special_id
            if special_positions.any():
                assert (labels[special_positions] == -100).all(), (
                    f"Special token {special_id} was masked"
                )


def test_mlm_collate_labels_match_original_tokens():
    """Test that labels at masked positions match the original token values."""
    tokenizer = Tokenizer()
    seq = "LAGVSERTIPDKQNFYMHWC"  # 20 AA
    batch = [{"pid": "test1", "seq": seq}]
    max_len = 32

    # Tokenize the sequence to get original tokens
    enc = tokenizer(
        seq,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    original_tokens = enc["input_ids"][0]

    tokens, labels = mlm_collate(
        batch,
        tokenizer,
        max_len=max_len,
        mask_prob=0.15,
        pad_id=1,
        ignore_index=-100,
    )

    # Where labels != -100, they should equal the original token
    masked_positions = labels[0] != -100
    if masked_positions.any():
        assert torch.equal(
            labels[0][masked_positions], original_tokens[masked_positions]
        ), "Labels at masked positions don't match original tokens"


def test_mlm_collate_random_token_replacement():
    """Test that some tokens are replaced with random tokens (not <mask>)."""
    tokenizer = Tokenizer()
    batch = [
        {"pid": f"test{i}", "seq": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFADD" * 2}
        for i in range(4)
    ]
    max_len = 128

    # Encode original sequences
    original_encodings = []
    for item in batch:
        enc = tokenizer(
            item["seq"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        original_encodings.append(enc["input_ids"][0])
    original = torch.stack(original_encodings)

    # Run multiple times to find random replacements
    found_random = False
    for _ in range(20):
        tokens, labels = mlm_collate(
            batch,
            tokenizer,
            max_len=max_len,
            mask_prob=0.15,
            mask_token_prob=0.8,
            random_token_prob=0.1,
            pad_id=1,
            mask_id=31,
            ignore_index=-100,
        )

        # Find positions where labels != -100 (masked) and tokens != 31 (<mask>)
        # and tokens != original (random replacement)
        masked = labels != -100
        not_mask_token = tokens != 31
        changed = tokens != original

        random_replaced = masked & not_mask_token & changed
        if random_replaced.any():
            found_random = True
            break

    assert found_random, "No random token replacements found after multiple iterations"


def test_mlm_collate_returns_coords_when_present():
    """Test that MLM collate returns coordinates when present in batch items."""
    tokenizer = Tokenizer()
    max_len = 16
    seq_len = 10  # Sequence length (coords should be [max_len, 3, 3])

    # Create batch with coordinates
    batch = [
        {
            "pid": "test1",
            "seq": "MVLSPADKTN",
            "coords": torch.randn(max_len, 3, 3),
        },
        {
            "pid": "test2",
            "seq": "LAGVSERQNF",
            "coords": torch.randn(max_len, 3, 3),
        },
    ]

    result = mlm_collate(
        batch,
        tokenizer,
        max_len=max_len,
        mask_prob=0.15,
        pad_id=1,
        ignore_index=-100,
    )

    # Should return 3-tuple (tokens, labels, coords)
    assert len(result) == 3
    tokens, labels, coords = result

    assert tokens.shape == (2, max_len)
    assert labels.shape == (2, max_len)
    assert coords.shape == (2, max_len, 3, 3)


def test_mlm_collate_returns_2tuple_without_coords():
    """Test that MLM collate returns 2-tuple when no coords present."""
    tokenizer = Tokenizer()
    max_len = 16

    # Create batch WITHOUT coordinates
    batch = [
        {"pid": "test1", "seq": "MVLSPADKTN"},
        {"pid": "test2", "seq": "LAGVSERQNF"},
    ]

    result = mlm_collate(
        batch,
        tokenizer,
        max_len=max_len,
        mask_prob=0.15,
        pad_id=1,
        ignore_index=-100,
    )

    # Should return 2-tuple (tokens, labels)
    assert len(result) == 2
    tokens, labels = result

    assert tokens.shape == (2, max_len)
    assert labels.shape == (2, max_len)

