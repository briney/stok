"""Tests for MDLM iterative unmasking sampler."""

import pytest
import torch

from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.utils.sampling import sample


@pytest.fixture
def seq_model():
    """Small seq_only model for sampling tests."""
    ns = NoiseSchedule(schedule_type="cosine")
    model = MDLMModel(
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
    model.eval()
    return model


@pytest.fixture
def joint_model():
    """Small joint model for sampling tests."""
    ns_seq = NoiseSchedule(schedule_type="cosine")
    ns_struct = NoiseSchedule(schedule_type="cosine")
    codebook = torch.randn(16, 8)
    model = MDLMModel(
        tracks="joint",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        codebook=codebook,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.667,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns_seq,
        noise_schedule_struct=ns_struct,
        time_conditioning="adaln",
    )
    model.eval()
    return model


class TestUnconditionalSeqOnly:
    """Unconditional seq_only generation."""

    def test_output_shape(self, seq_model):
        result = sample(seq_model, length=10, num_samples=3, num_steps=5)
        assert result["seq_tokens"].shape == (3, 10)

    def test_no_mask_tokens_remaining(self, seq_model):
        result = sample(seq_model, length=8, num_samples=2, num_steps=10)
        mask_id = seq_model.seq_mask_id
        assert (result["seq_tokens"] != mask_id).all()

    def test_valid_token_ids(self, seq_model):
        result = sample(seq_model, length=8, num_samples=2, num_steps=5)
        assert (result["seq_tokens"] >= 0).all()
        assert (result["seq_tokens"] < 32).all()

    def test_single_step(self, seq_model):
        """With num_steps=1, everything should be unmasked in one step."""
        result = sample(seq_model, length=8, num_samples=2, num_steps=1)
        mask_id = seq_model.seq_mask_id
        assert (result["seq_tokens"] != mask_id).all()

    def test_no_struct_in_output(self, seq_model):
        result = sample(seq_model, length=8, num_samples=2, num_steps=3)
        assert "struct_tokens" not in result


class TestConditionalSeqOnly:
    """Conditional generation in seq_only mode."""

    def test_conditioned_positions_unchanged(self, seq_model):
        B, L = 2, 10
        condition = torch.randint(4, 24, (B, L))
        # Only generate positions 3-7
        mask_pos = torch.zeros(B, L, dtype=torch.bool)
        mask_pos[:, 3:8] = True

        result = sample(
            seq_model, length=L, num_samples=B, num_steps=5,
            condition_seq=condition,
            seq_mask_positions=mask_pos,
        )

        # Conditioned positions (not in mask_pos) should be unchanged
        for b in range(B):
            for l in range(L):
                if not mask_pos[b, l]:
                    assert result["seq_tokens"][b, l] == condition[b, l]

    def test_no_mask_tokens_at_generated_positions(self, seq_model):
        B, L = 2, 8
        condition = torch.randint(4, 24, (B, L))
        mask_pos = torch.zeros(B, L, dtype=torch.bool)
        mask_pos[:, 2:6] = True

        result = sample(
            seq_model, length=L, num_samples=B, num_steps=10,
            condition_seq=condition,
            seq_mask_positions=mask_pos,
        )
        mask_id = seq_model.seq_mask_id
        assert (result["seq_tokens"] != mask_id).all()

    def test_full_condition_no_generation(self, seq_model):
        """When condition is fully provided with no mask, output equals input."""
        B, L = 2, 8
        condition = torch.randint(4, 24, (B, L))
        # No mask_positions means nothing to generate
        result = sample(
            seq_model, length=L, num_samples=B, num_steps=5,
            condition_seq=condition,
        )
        assert torch.equal(result["seq_tokens"], condition)


class TestJointSampling:
    """Joint mode generation tests."""

    def test_output_has_both_tracks(self, joint_model):
        result = sample(joint_model, length=8, num_samples=2, num_steps=5)
        assert "seq_tokens" in result
        assert "struct_tokens" in result

    def test_output_shapes(self, joint_model):
        result = sample(joint_model, length=8, num_samples=3, num_steps=5)
        assert result["seq_tokens"].shape == (3, 8)
        assert result["struct_tokens"].shape == (3, 8)

    def test_no_mask_tokens_remaining(self, joint_model):
        result = sample(joint_model, length=8, num_samples=2, num_steps=10)
        assert (result["seq_tokens"] != joint_model.seq_mask_id).all()
        assert (result["struct_tokens"] != joint_model.struct_mask_id).all()

    def test_forward_mode(self, joint_model):
        """Forward: condition on seq, generate struct."""
        B, L = 2, 8
        condition_seq = torch.randint(4, 24, (B, L))
        struct_mask = torch.ones(B, L, dtype=torch.bool)  # Generate all struct

        result = sample(
            joint_model, length=L, num_samples=B, num_steps=5,
            condition_seq=condition_seq,
            struct_mask_positions=struct_mask,
        )
        # Seq should be unchanged
        assert torch.equal(result["seq_tokens"], condition_seq)
        # Struct should have valid indices
        assert (result["struct_tokens"] != joint_model.struct_mask_id).all()

    def test_inverse_mode(self, joint_model):
        """Inverse: condition on struct, generate seq."""
        B, L = 2, 8
        condition_struct = torch.randint(0, 16, (B, L))
        seq_mask = torch.ones(B, L, dtype=torch.bool)  # Generate all seq

        result = sample(
            joint_model, length=L, num_samples=B, num_steps=5,
            condition_struct=condition_struct,
            seq_mask_positions=seq_mask,
        )
        # Struct should be unchanged
        assert torch.equal(result["struct_tokens"], condition_struct)
        # Seq should have no mask tokens
        assert (result["seq_tokens"] != joint_model.seq_mask_id).all()

    def test_temperature(self, joint_model):
        """Verify temperature parameter doesn't crash."""
        result = sample(
            joint_model, length=8, num_samples=2, num_steps=5,
            temperature=0.5,
        )
        assert (result["seq_tokens"] != joint_model.seq_mask_id).all()
