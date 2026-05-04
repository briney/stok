"""Tests for mdlm_collate: MDLM diffusion batch preparation."""

import pytest
import torch

from stok.data.collate import mdlm_collate
from stok.models.noise_schedule import NoiseSchedule
from stok.utils.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()


@pytest.fixture(scope="module")
def noise_schedule():
    return NoiseSchedule(schedule_type="cosine")


def _make_batch(seqs: list[str]) -> list[dict]:
    return [{"seq": s} for s in seqs]


class TestMDLMCollateShapes:
    """Output tensor shapes and keys."""

    def test_output_keys(self, tokenizer, noise_schedule):
        batch = _make_batch(["ACDEFG", "KLMNPQ", "RSTWYV"])
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=16,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            tracks="seq_only",
        )
        expected_keys = {
            "seq_tokens", "t_seq", "seq_targets", "seq_mask",
            "seq_eligible_mask", "key_padding_mask", "struct_tokens",
            "t_struct", "struct_targets", "struct_mask",
            "struct_eligible_mask",
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self, tokenizer, noise_schedule):
        batch = _make_batch(["ACDEFG", "KLMNPQ", "RSTWYV"])
        max_len = 16
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=max_len,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            tracks="seq_only",
        )
        B = 3
        assert result["seq_tokens"].shape == (B, max_len)
        assert result["t_seq"].shape == (B,)
        assert result["seq_targets"].shape == (B, max_len)
        assert result["seq_mask"].shape == (B, max_len)
        assert result["key_padding_mask"].shape == (B, max_len)

    def test_struct_fields_none_in_seq_only(self, tokenizer, noise_schedule):
        batch = _make_batch(["ACDEFG"])
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=16,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            tracks="seq_only",
        )
        assert result["struct_tokens"] is None
        assert result["t_struct"] is None
        assert result["struct_targets"] is None
        assert result["struct_mask"] is None


class TestMDLMCollateMasking:
    """Masking behavior: mask positions, special tokens, padding."""

    def test_mask_not_at_padding(self, tokenizer, noise_schedule):
        # Short sequence -> lots of padding
        batch = _make_batch(["ACD"])
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=20,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            tracks="seq_only",
        )
        pad_mask = result["key_padding_mask"]
        seq_mask = result["seq_mask"]
        # No masked positions at padding
        assert not (seq_mask & pad_mask).any()

    def test_mask_not_at_special_tokens(self, tokenizer, noise_schedule):
        batch = _make_batch(["ACDEFGHIKLMNPQ"])
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=20,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            tracks="seq_only",
        )
        # Special token positions (CLS at 0, EOS after sequence)
        special_ids = set(tokenizer.all_special_ids)
        # Check: wherever original token was special, mask should be False
        # We can verify CLS (position 0) is never masked
        assert not result["seq_mask"][0, 0].item(), "CLS should not be masked"


class TestMDLMCollateTargets:
    """Target construction."""

    def test_targets_at_masked_positions(self, tokenizer, noise_schedule):
        batch = _make_batch(["ACDEFGHIKLMN"])
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=20,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            tracks="seq_only",
        )
        mask = result["seq_mask"]
        targets = result["seq_targets"]
        # At masked positions, targets should be valid token IDs (not -100)
        assert (targets[mask] != -100).all()
        # At unmasked positions, targets should be -100
        assert (targets[~mask] == -100).all()


class TestMDLMCollateTimeValues:
    """Diffusion time sampling."""

    def test_t_in_range(self, tokenizer, noise_schedule):
        batch = _make_batch(["ACDEFG", "KLMNPQ"])
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=16,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            tracks="seq_only",
        )
        t = result["t_seq"]
        assert (t > 0).all()
        assert (t < 1).all()

    def test_antithetic_pairs(self, tokenizer, noise_schedule):
        batch = _make_batch(["AC", "DE", "FG", "HI"])
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=8,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            antithetic_time_sampling=True,
            tracks="seq_only",
        )
        t = result["t_seq"]
        # Pairs should sum to ~1
        assert (t[0] + t[1]).item() == pytest.approx(1.0, abs=1e-4)
        assert (t[2] + t[3]).item() == pytest.approx(1.0, abs=1e-4)

    def test_uniform_sampling(self, tokenizer, noise_schedule):
        batch = _make_batch(["AC", "DE"])
        result = mdlm_collate(
            batch,
            tokenizer,
            noise_schedule_seq=noise_schedule,
            max_len=8,
            seq_mask_id=tokenizer.mask_token_id,
            seq_pad_id=tokenizer.pad_token_id,
            antithetic_time_sampling=False,
            tracks="seq_only",
        )
        t = result["t_seq"]
        assert t.shape == (2,)
        assert (t > 0).all()
        assert (t < 1).all()
