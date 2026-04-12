"""Tests for MDLM noising and time sampling utilities."""

import pytest
import torch

from stok.models.noise_schedule import NoiseSchedule
from stok.utils.sampling import apply_noise, guarantee_min_mask, sample_t_antithetic


class TestSampleTAntithetic:
    """Antithetic time sampling for variance reduction."""

    def test_correct_size(self):
        t = sample_t_antithetic(8, torch.device("cpu"))
        assert t.shape == (8,)

    def test_odd_batch_size(self):
        t = sample_t_antithetic(7, torch.device("cpu"))
        assert t.shape == (7,)

    def test_values_in_range(self):
        t = sample_t_antithetic(100, torch.device("cpu"))
        assert (t > 0).all()
        assert (t < 1).all()

    def test_antithetic_pairs_sum_to_one(self):
        t = sample_t_antithetic(10, torch.device("cpu"))
        # First 5 pairs should be (t_i, 1-t_i)
        for i in range(5):
            pair_sum = t[2 * i] + t[2 * i + 1]
            assert pair_sum.item() == pytest.approx(1.0, abs=1e-4)

    def test_batch_size_one(self):
        t = sample_t_antithetic(1, torch.device("cpu"))
        assert t.shape == (1,)
        assert 0 < t.item() < 1


class TestGuaranteeMinMask:
    """Ensure minimum masking per sequence."""

    def test_no_change_when_already_sufficient(self):
        mask = torch.tensor([[True, True, False, False]])
        padding_mask = torch.zeros(1, 4, dtype=torch.bool)
        result = guarantee_min_mask(mask, padding_mask, None, min_masked=1)
        assert result.sum().item() >= 1

    def test_force_mask_when_zero(self):
        mask = torch.zeros(2, 6, dtype=torch.bool)
        padding_mask = torch.zeros(2, 6, dtype=torch.bool)
        result = guarantee_min_mask(mask, padding_mask, None, min_masked=1)
        # Each sequence should have at least 1 masked position
        assert result[0].sum().item() >= 1
        assert result[1].sum().item() >= 1

    def test_respects_padding(self):
        mask = torch.zeros(1, 6, dtype=torch.bool)
        padding_mask = torch.tensor([[False, False, True, True, True, True]])
        result = guarantee_min_mask(mask, padding_mask, None, min_masked=1)
        # The forced mask should be in a non-padding position
        assert result[0, :2].any()
        assert not result[0, 2:].any()

    def test_respects_special_tokens(self):
        mask = torch.zeros(1, 6, dtype=torch.bool)
        padding_mask = torch.zeros(1, 6, dtype=torch.bool)
        special = torch.tensor([[True, False, False, False, False, True]])  # CLS, EOS
        result = guarantee_min_mask(mask, padding_mask, special, min_masked=1)
        # Mask should not be at special positions
        assert not result[0, 0].item()
        assert not result[0, 5].item()
        assert result[0, 1:5].any()


class TestApplyNoise:
    """Forward diffusion noising."""

    @pytest.fixture
    def noise_schedule(self):
        return NoiseSchedule(schedule_type="linear")

    def test_t_zero_minimal_masking(self, noise_schedule):
        """At t=0, alpha~1, so almost no tokens should be masked."""
        B, L = 4, 20
        tokens = torch.randint(4, 24, (B, L))
        t = torch.full((B,), 1e-5)  # Near 0
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        noised, mask = apply_noise(tokens, t, 31, noise_schedule, padding_mask)
        # At t~0, very few tokens should be masked (just guarantee_min_mask)
        assert mask.sum().item() <= B * 2  # At most ~1 per sequence from guarantee

    def test_t_one_heavy_masking(self, noise_schedule):
        """At t~1, alpha~0, so most tokens should be masked."""
        B, L = 4, 20
        tokens = torch.randint(4, 24, (B, L))
        t = torch.full((B,), 1.0 - 1e-5)  # Near 1
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        noised, mask = apply_noise(tokens, t, 31, noise_schedule, padding_mask)
        # At t~1, most tokens should be masked
        mask_rate = mask.float().mean().item()
        assert mask_rate > 0.8

    def test_padding_never_masked(self, noise_schedule):
        B, L = 2, 10
        tokens = torch.randint(4, 24, (B, L))
        t = torch.full((B,), 0.5)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask[:, -3:] = True  # Last 3 positions are padding

        noised, mask = apply_noise(tokens, t, 31, noise_schedule, padding_mask)
        assert not mask[:, -3:].any()

    def test_special_tokens_never_masked(self, noise_schedule):
        B, L = 2, 10
        tokens = torch.randint(4, 24, (B, L))
        t = torch.full((B,), 0.9)  # High masking rate
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        special = torch.zeros(B, L, dtype=torch.bool)
        special[:, 0] = True  # CLS
        special[:, -1] = True  # EOS

        noised, mask = apply_noise(
            tokens, t, 31, noise_schedule, padding_mask, special_token_mask=special
        )
        assert not mask[:, 0].any()
        assert not mask[:, -1].any()

    def test_masked_positions_get_mask_token(self, noise_schedule):
        B, L = 2, 10
        mask_token_id = 31
        tokens = torch.randint(4, 24, (B, L))
        t = torch.full((B,), 0.5)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        noised, mask = apply_noise(tokens, t, mask_token_id, noise_schedule, padding_mask)
        assert (noised[mask] == mask_token_id).all()

    def test_unmasked_positions_unchanged(self, noise_schedule):
        B, L = 2, 10
        tokens = torch.randint(4, 24, (B, L))
        t = torch.full((B,), 0.5)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        noised, mask = apply_noise(tokens, t, 31, noise_schedule, padding_mask)
        assert (noised[~mask] == tokens[~mask]).all()

    def test_output_shapes(self, noise_schedule):
        B, L = 3, 12
        tokens = torch.randint(4, 24, (B, L))
        t = torch.full((B,), 0.5)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        noised, mask = apply_noise(tokens, t, 31, noise_schedule, padding_mask)
        assert noised.shape == (B, L)
        assert mask.shape == (B, L)
        assert mask.dtype == torch.bool
