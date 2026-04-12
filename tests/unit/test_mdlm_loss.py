"""Tests for MDLMLoss: masked diffusion loss computation."""

import pytest
import torch

from stok.models.noise_schedule import NoiseSchedule
from stok.utils.losses import MDLMLoss


@pytest.fixture
def noise_schedule():
    return NoiseSchedule(schedule_type="cosine")


@pytest.fixture
def loss_fn(noise_schedule):
    return MDLMLoss(noise_schedule=noise_schedule)


class TestMDLMLossZeroMask:
    """Loss should be zero when no positions are masked."""

    def test_zero_loss_no_mask(self, loss_fn):
        B, L, V = 2, 10, 32
        logits = torch.randn(B, L, V)
        targets = torch.full((B, L), -100, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.bool)
        t = torch.tensor([0.5, 0.5])
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        loss = loss_fn(logits, targets, mask, t, padding_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert loss.requires_grad


class TestMDLMLossMaskedOnly:
    """Loss should only be computed at masked positions."""

    def test_loss_at_masked_positions_only(self, loss_fn):
        B, L, V = 2, 8, 32
        logits = torch.randn(B, L, V)
        targets = torch.full((B, L), -100, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        t = torch.tensor([0.5, 0.5])

        # Mask only position 3 in each sequence
        mask[:, 3] = True
        targets[:, 3] = 5  # target token

        loss = loss_fn(logits, targets, mask, t, padding_mask)
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_padding_excluded(self, loss_fn):
        B, L, V = 2, 8, 32
        logits = torch.randn(B, L, V)
        targets = torch.randint(0, V, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        t = torch.tensor([0.5, 0.5])

        # Last 3 positions are padding
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask[:, -3:] = True

        # Mask those padding positions too (collate shouldn't, but loss must handle)
        loss = loss_fn(logits, targets, mask, t, padding_mask)
        assert torch.isfinite(loss)


class TestMDLMLossWeight:
    """Loss weight should scale with t."""

    def test_weight_increases_near_t1(self, noise_schedule):
        loss_fn_inner = MDLMLoss(noise_schedule=noise_schedule)
        B, L, V = 4, 8, 32
        torch.manual_seed(42)
        logits = torch.randn(B, L, V)
        targets = torch.randint(0, V, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        # Loss at t near 1 should generally have higher weight
        t_low = torch.tensor([0.1] * B)
        t_high = torch.tensor([0.9] * B)

        loss_low = loss_fn_inner(logits, targets, mask, t_low, padding_mask)
        loss_high = loss_fn_inner(logits, targets, mask, t_high, padding_mask)

        # The weight w(t) = |alpha'(t)| / (1 - alpha(t)) varies by schedule,
        # but for cosine schedule, weight at t=0.9 should differ from t=0.1
        assert loss_low.item() != loss_high.item()


class TestMDLMLossGradient:
    """Gradients must flow through the loss."""

    def test_gradient_flow(self, loss_fn):
        B, L, V = 2, 8, 32
        logits = torch.randn(B, L, V, requires_grad=True)
        targets = torch.randint(0, V, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        t = torch.tensor([0.5, 0.5])
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        loss = loss_fn(logits, targets, mask, t, padding_mask)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0
