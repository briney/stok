"""Tests for MDLMModel (seq_only mode) and apply_subs."""

import pytest
import torch

from stok.models.mdlm import MDLMModel, apply_subs
from stok.models.noise_schedule import NoiseSchedule


# ---------------------------------------------------------------------------
# apply_subs tests
# ---------------------------------------------------------------------------


class TestApplySubs:
    """SUBS parameterization constraint tests."""

    def test_mask_token_suppressed_at_masked_positions(self):
        B, L, V = 2, 6, 32
        mask_token_id = 31
        logits = torch.randn(B, L, V)
        input_tokens = torch.randint(0, V, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, 2] = True  # Position 2 is masked

        result = apply_subs(logits, input_tokens, mask, mask_token_id)
        # At masked positions, mask_token_id logit should be dtype min
        fmin = torch.finfo(logits.dtype).min
        assert result[0, 2, mask_token_id].item() == pytest.approx(fmin, abs=1.0)
        assert result[1, 2, mask_token_id].item() == pytest.approx(fmin, abs=1.0)

    def test_unmasked_positions_predict_input(self):
        B, L, V = 2, 6, 32
        logits = torch.randn(B, L, V)
        input_tokens = torch.randint(4, 24, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, 2] = True  # Only position 2 is masked

        result = apply_subs(logits, input_tokens, mask, 31)
        # At unmasked positions, argmax should be the input token
        for b in range(B):
            for l in range(L):
                if not mask[b, l]:
                    assert result[b, l].argmax().item() == input_tokens[b, l].item()

    def test_fp16_no_nan(self):
        B, L, V = 2, 6, 32
        logits = torch.randn(B, L, V, dtype=torch.float16)
        input_tokens = torch.randint(4, 24, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        mask[:, 0] = False

        result = apply_subs(logits, input_tokens, mask, 31)
        # No NaNs should appear
        assert not torch.isnan(result).any()

    def test_original_logits_unchanged(self):
        B, L, V = 2, 6, 32
        logits = torch.randn(B, L, V)
        original = logits.clone()
        input_tokens = torch.randint(4, 24, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)

        apply_subs(logits, input_tokens, mask, 31)
        # Original should not be modified (apply_subs clones internally)
        assert torch.equal(logits, original)


# ---------------------------------------------------------------------------
# MDLMModel tests
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    """Build a small MDLMModel for testing."""
    ns = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="seq_only",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.667,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns,
        time_conditioning="adaln",
    )


class TestMDLMModelForward:
    """Basic forward pass tests for seq_only mode."""

    def test_output_keys(self, small_model):
        B, L = 2, 10
        seq_tokens = torch.randint(0, 32, (B, L))
        t_seq = torch.rand(B)
        targets = torch.randint(0, 32, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        out = small_model(
            seq_tokens=seq_tokens,
            t_seq=t_seq,
            seq_targets=targets,
            seq_mask=mask,
            key_padding_mask=padding_mask,
        )
        assert "loss" in out
        assert "loss_seq" in out
        assert "seq_logits" in out

    def test_output_shapes(self, small_model):
        B, L = 3, 8
        seq_tokens = torch.randint(0, 32, (B, L))
        t_seq = torch.rand(B)
        targets = torch.randint(0, 32, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        out = small_model(
            seq_tokens=seq_tokens,
            t_seq=t_seq,
            seq_targets=targets,
            seq_mask=mask,
            key_padding_mask=padding_mask,
        )
        assert out["seq_logits"].shape == (B, L, 32)

    def test_loss_finite_and_requires_grad(self, small_model):
        B, L = 2, 10
        seq_tokens = torch.randint(2, 22, (B, L))
        seq_tokens[:, 0] = 0  # CLS
        t_seq = torch.tensor([0.5, 0.7])
        mask = torch.ones(B, L, dtype=torch.bool)
        mask[:, 0] = False  # Don't mask CLS
        targets = torch.full((B, L), -100, dtype=torch.long)
        targets[mask] = seq_tokens[mask]
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        out = small_model(
            seq_tokens=seq_tokens,
            t_seq=t_seq,
            seq_targets=targets,
            seq_mask=mask,
            key_padding_mask=padding_mask,
        )
        assert torch.isfinite(out["loss"])
        assert out["loss"].requires_grad

    def test_backward_pass(self, small_model):
        B, L = 2, 8
        seq_tokens = torch.randint(2, 22, (B, L))
        t_seq = torch.tensor([0.3, 0.6])
        targets = torch.randint(2, 22, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        out = small_model(
            seq_tokens=seq_tokens,
            t_seq=t_seq,
            seq_targets=targets,
            seq_mask=mask,
            key_padding_mask=padding_mask,
        )
        out["loss"].backward()
        # Check at least one parameter has gradients
        grads = [p.grad for p in small_model.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestMDLMModelEdgeCases:
    """Edge cases and special behavior."""

    def test_no_targets_no_loss(self, small_model):
        B, L = 2, 8
        seq_tokens = torch.randint(0, 32, (B, L))
        t_seq = torch.rand(B)

        out = small_model(seq_tokens=seq_tokens, t_seq=t_seq)
        assert out["loss"] is None

    def test_padding_mask_inferred_from_pad_id(self, small_model):
        B, L = 2, 8
        seq_tokens = torch.randint(2, 22, (B, L))
        seq_tokens[:, -2:] = 1  # pad_id = 1
        t_seq = torch.rand(B)
        targets = torch.randint(2, 22, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)

        # Don't pass key_padding_mask; should be inferred
        out = small_model(
            seq_tokens=seq_tokens,
            t_seq=t_seq,
            seq_targets=targets,
            seq_mask=mask,
        )
        assert out["loss"] is not None

    def test_joint_mode_raises(self):
        ns = NoiseSchedule(schedule_type="cosine")
        with pytest.raises(NotImplementedError, match="Joint mode"):
            MDLMModel(
                tracks="joint",
                noise_schedule_seq=ns,
            )

    def test_invalid_tracks_raises(self):
        ns = NoiseSchedule(schedule_type="cosine")
        with pytest.raises(ValueError, match="Unknown tracks"):
            MDLMModel(
                tracks="invalid",
                noise_schedule_seq=ns,
            )
