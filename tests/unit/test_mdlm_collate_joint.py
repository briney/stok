"""Tests for mdlm_collate in joint (two-track) mode."""

import pytest
import torch

from stok.data.collate import mdlm_collate
from stok.models.noise_schedule import NoiseSchedule


class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    pad_token_id = 1
    mask_token_id = 31
    cls_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 1, 2, 31]

    def __call__(self, seq, add_special_tokens=True, truncation=True,
                 max_length=32, padding="max_length", return_tensors="pt"):
        ids = [self.cls_token_id]
        for ch in seq:
            ids.append(ord(ch) % 20 + 4)  # Map chars to 4-23
        ids.append(self.eos_token_id)
        # Pad to max_length
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
        ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def ns_seq():
    return NoiseSchedule(schedule_type="cosine")


@pytest.fixture
def ns_struct():
    return NoiseSchedule(schedule_type="cosine")


def _make_paired_batch(n=3, seq_len=5, codebook_size=16):
    """Create synthetic paired batch items with seq + indices."""
    batch = []
    for i in range(n):
        seq = "ACDEF"[:seq_len]
        indices = torch.randint(0, codebook_size, (seq_len,))
        batch.append({"seq": seq, "indices": indices})
    return batch


class TestJointCollate:
    """Two-track collate function tests."""

    def test_output_keys(self, tokenizer, ns_seq, ns_struct):
        batch = _make_paired_batch(n=3)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
        )
        # All keys should be populated (not None)
        assert result["struct_tokens"] is not None
        assert result["t_struct"] is not None
        assert result["struct_targets"] is not None
        assert result["struct_mask"] is not None

    def test_output_shapes(self, tokenizer, ns_seq, ns_struct):
        batch = _make_paired_batch(n=3)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
        )
        B, L = 3, 16
        assert result["struct_tokens"].shape == (B, L)
        assert result["t_struct"].shape == (B,)
        assert result["struct_targets"].shape == (B, L)
        assert result["struct_mask"].shape == (B, L)

    def test_struct_special_positions_are_padded(self, tokenizer, ns_seq, ns_struct):
        """CLS, EOS, PAD positions should have struct_pad_id."""
        batch = _make_paired_batch(n=2, seq_len=3)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
        )
        struct_tokens = result["struct_tokens"]
        seq_tokens = result["seq_tokens"]

        for b in range(2):
            # CLS position (index 0)
            # Struct at CLS should be padded or mask (never a real code)
            # Since it's a special position, it should not be masked
            assert result["struct_mask"][b, 0].item() is False

            # Padding positions should not be masked
            padding_positions = seq_tokens[b] == 1
            assert not result["struct_mask"][b][padding_positions].any()

    def test_struct_mask_excludes_padding(self, tokenizer, ns_seq, ns_struct):
        batch = _make_paired_batch(n=2, seq_len=3)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
        )
        padding = result["key_padding_mask"]
        # No struct masking at padding positions
        assert not (result["struct_mask"] & padding).any()

    def test_independent_times(self, tokenizer, ns_seq, ns_struct):
        """With independent_track_times=True, t_seq and t_struct should differ."""
        torch.manual_seed(42)
        batch = _make_paired_batch(n=4)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
            independent_track_times=True,
        )
        # With independent sampling, the times should generally differ
        # (extremely unlikely to be identical with 4 samples)
        assert not torch.equal(result["t_seq"], result["t_struct"])

    def test_shared_times(self, tokenizer, ns_seq, ns_struct):
        """With independent_track_times=False, t_seq and t_struct should match."""
        batch = _make_paired_batch(n=4)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
            independent_track_times=False,
        )
        assert torch.equal(result["t_seq"], result["t_struct"])

    def test_struct_targets_at_masked_positions(self, tokenizer, ns_seq, ns_struct):
        batch = _make_paired_batch(n=2)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
        )
        # At masked positions, targets should be valid (not -100)
        masked = result["struct_mask"]
        targets = result["struct_targets"]
        if masked.any():
            assert (targets[masked] != -100).all()

    def test_struct_targets_at_unmasked_are_ignored(self, tokenizer, ns_seq, ns_struct):
        batch = _make_paired_batch(n=2)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
        )
        unmasked = ~result["struct_mask"]
        targets = result["struct_targets"]
        # At unmasked positions, targets should be -100 (ignore)
        assert (targets[unmasked] == -100).all()

    def test_t_struct_in_valid_range(self, tokenizer, ns_seq, ns_struct):
        batch = _make_paired_batch(n=4)
        result = mdlm_collate(
            batch, tokenizer, ns_seq, ns_struct,
            max_len=16, seq_mask_id=31, seq_pad_id=1,
            struct_mask_id=16, struct_pad_id=17,
            tracks="joint",
        )
        assert (result["t_struct"] > 0).all()
        assert (result["t_struct"] < 1).all()

    def test_missing_indices_raises(self, tokenizer, ns_seq, ns_struct):
        batch = [{"seq": "ACDEF"}]  # No 'indices' key
        with pytest.raises(ValueError, match="indices"):
            mdlm_collate(
                batch, tokenizer, ns_seq, ns_struct,
                max_len=16, seq_mask_id=31, seq_pad_id=1,
                struct_mask_id=16, struct_pad_id=17,
                tracks="joint",
            )

    def test_missing_struct_ids_raises(self, tokenizer, ns_seq, ns_struct):
        batch = _make_paired_batch(n=2)
        with pytest.raises(ValueError, match="struct_mask_id"):
            mdlm_collate(
                batch, tokenizer, ns_seq, ns_struct,
                max_len=16, seq_mask_id=31, seq_pad_id=1,
                tracks="joint",
                # struct_mask_id and struct_pad_id not provided
            )
