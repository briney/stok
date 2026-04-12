"""Tests for MLM collate edge cases (Phase 0.2)."""

import torch

from stok.data.collate import mlm_collate
from stok.utils.losses import token_ce_loss
from stok.utils.tokenizer import Tokenizer


def test_mlm_collate_force_mask_with_zero_mask_prob():
    """Test that at least one token is masked even with mask_prob=0.0."""
    tokenizer = Tokenizer()
    batch = [{"pid": "test1", "seq": "LAGVSER"}]
    max_len = 16

    tokens, labels = mlm_collate(
        batch,
        tokenizer,
        max_len=max_len,
        mask_prob=0.0,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        ignore_index=-100,
    )

    # At least one token should be masked (labels != -100)
    masked_count = (labels != -100).sum().item()
    assert masked_count >= 1, "Expected at least 1 masked token with mask_prob=0.0"


def test_mlm_collate_force_mask_short_sequence():
    """Test force-mask with a very short sequence (1-2 residues)."""
    tokenizer = Tokenizer()
    # Single residue: CLS + A + EOS + padding
    batch = [{"pid": "test1", "seq": "A"}]
    max_len = 8

    tokens, labels = mlm_collate(
        batch,
        tokenizer,
        max_len=max_len,
        mask_prob=0.0,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        ignore_index=-100,
    )

    # The single non-special token should be force-masked
    masked_count = (labels != -100).sum().item()
    assert masked_count >= 1, "Expected at least 1 masked token for short sequence"


def test_mlm_collate_force_mask_two_residue_sequence():
    """Test force-mask with a two-residue sequence."""
    tokenizer = Tokenizer()
    batch = [{"pid": "test1", "seq": "AG"}]
    max_len = 8

    tokens, labels = mlm_collate(
        batch,
        tokenizer,
        max_len=max_len,
        mask_prob=0.0,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        ignore_index=-100,
    )

    masked_count = (labels != -100).sum().item()
    assert masked_count >= 1


def test_token_ce_loss_all_ignored_returns_zero_not_nan():
    """Test that token_ce_loss returns 0.0, not NaN, when all labels are ignored."""
    B, L, C = 2, 10, 32
    logits = torch.randn(B, L, C)
    labels = torch.full((B, L), -100, dtype=torch.long)

    loss = token_ce_loss(logits, labels, ignore_index=-100)
    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"
    assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"
    assert loss.requires_grad, "Expected loss to require grad for backprop safety"


def test_token_ce_loss_normal_case_still_works():
    """Sanity check: normal case still produces a reasonable loss."""
    B, L, C = 2, 10, 32
    logits = torch.randn(B, L, C)
    labels = torch.randint(0, C, (B, L))

    loss = token_ce_loss(logits, labels, ignore_index=-100)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
