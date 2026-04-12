"""Tests for MDLMModel in joint (two-track) mode."""

import pytest
import torch

from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule


@pytest.fixture
def tiny_codebook():
    """A small codebook for testing: C=16, d_code=8."""
    return torch.randn(16, 8)


@pytest.fixture
def joint_model(tiny_codebook):
    """Build a small joint MDLMModel for testing."""
    ns_seq = NoiseSchedule(schedule_type="cosine")
    ns_struct = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="joint",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        codebook=tiny_codebook,
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


class TestJointModelConstruction:
    """Verify joint model is constructed correctly."""

    def test_has_struct_components(self, joint_model):
        assert hasattr(joint_model, "embed_struct")
        assert hasattr(joint_model, "track_embed")
        assert hasattr(joint_model, "head_struct")
        assert hasattr(joint_model, "loss_fn_struct")

    def test_struct_mask_pad_ids(self, joint_model):
        # C=16, so mask_id=16, pad_id=17
        assert joint_model.struct_mask_id == 16
        assert joint_model.struct_pad_id == 17

    def test_struct_embedding_size(self, joint_model):
        # C + 2 = 18 embeddings
        assert joint_model.embed_struct.num_embeddings == 18

    def test_track_embedding_size(self, joint_model):
        assert joint_model.track_embed.num_embeddings == 2


class TestJointForward:
    """Forward pass tests for joint mode."""

    def _make_batch(self, B=2, L=10, C=16):
        seq_tokens = torch.randint(2, 22, (B, L))
        struct_tokens = torch.randint(0, C, (B, L))
        t_seq = torch.rand(B).clamp(1e-4, 1 - 1e-4)
        t_struct = torch.rand(B).clamp(1e-4, 1 - 1e-4)
        seq_mask = torch.ones(B, L, dtype=torch.bool)
        seq_mask[:, 0] = False  # CLS not masked
        struct_mask = torch.ones(B, L, dtype=torch.bool)
        struct_mask[:, 0] = False
        seq_targets = torch.full((B, L), -100, dtype=torch.long)
        seq_targets[seq_mask] = seq_tokens[seq_mask]
        struct_targets = torch.full((B, L), -100, dtype=torch.long)
        struct_targets[struct_mask] = struct_tokens[struct_mask]
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        return {
            "seq_tokens": seq_tokens,
            "t_seq": t_seq,
            "seq_targets": seq_targets,
            "seq_mask": seq_mask,
            "key_padding_mask": padding_mask,
            "struct_tokens": struct_tokens,
            "t_struct": t_struct,
            "struct_targets": struct_targets,
            "struct_mask": struct_mask,
        }

    def test_output_keys(self, joint_model):
        batch = self._make_batch()
        out = joint_model(**batch)
        assert "loss" in out
        assert "loss_seq" in out
        assert "loss_struct" in out
        assert "seq_logits" in out
        assert "struct_logits" in out

    def test_output_shapes(self, joint_model):
        B, L = 2, 10
        batch = self._make_batch(B=B, L=L)
        out = joint_model(**batch)
        assert out["seq_logits"].shape == (B, L, 32)
        assert out["struct_logits"].shape == (B, L, 16)  # C=16

    def test_loss_is_finite(self, joint_model):
        batch = self._make_batch()
        out = joint_model(**batch)
        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["loss_seq"])
        assert torch.isfinite(out["loss_struct"])

    def test_backward(self, joint_model):
        batch = self._make_batch()
        out = joint_model(**batch)
        out["loss"].backward()
        grads = [p.grad for p in joint_model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_combined_loss_is_sum(self, joint_model):
        batch = self._make_batch()
        out = joint_model(**batch)
        expected = out["loss_seq"] + out["loss_struct"]
        assert torch.isclose(out["loss"], expected, atol=1e-6)

    def test_no_seq_masking_zero_seq_loss(self, joint_model):
        """When t_seq=0 (no masking), seq loss should be 0."""
        B, L = 2, 10
        batch = self._make_batch(B=B, L=L)
        # Set t_seq very close to 0 so alpha ~ 1 (no masking)
        # But we need at least 1 masked position due to guarantee_min_mask
        # So use the mask directly: set all mask to False
        batch["seq_mask"] = torch.zeros(B, L, dtype=torch.bool)
        batch["seq_targets"] = torch.full((B, L), -100, dtype=torch.long)
        out = joint_model(**batch)
        # No valid masked positions -> seq loss should be None or 0
        if out["loss_seq"] is not None:
            assert out["loss_seq"].item() == pytest.approx(0.0, abs=1e-6)

    def test_no_struct_targets_no_struct_loss(self, joint_model):
        B, L = 2, 10
        batch = self._make_batch(B=B, L=L)
        batch["struct_targets"] = None
        out = joint_model(**batch)
        assert out["loss_struct"] is None
        # Total loss should equal seq loss
        assert torch.isclose(out["loss"], out["loss_seq"], atol=1e-6)

    def test_concat_project_time_combine(self, tiny_codebook):
        ns_seq = NoiseSchedule(schedule_type="cosine")
        ns_struct = NoiseSchedule(schedule_type="cosine")
        model = MDLMModel(
            tracks="joint",
            seq_vocab_size=32,
            seq_pad_id=1,
            seq_mask_id=31,
            codebook=tiny_codebook,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
            noise_schedule_seq=ns_seq,
            noise_schedule_struct=ns_struct,
            time_conditioning="adaln",
            time_combine="concat_project",
        )
        batch = self._make_batch()
        out = model(**batch)
        assert torch.isfinite(out["loss"])

    def test_lambda_weighting(self, tiny_codebook):
        ns_seq = NoiseSchedule(schedule_type="cosine")
        ns_struct = NoiseSchedule(schedule_type="cosine")
        model = MDLMModel(
            tracks="joint",
            seq_vocab_size=32,
            seq_pad_id=1,
            seq_mask_id=31,
            codebook=tiny_codebook,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
            noise_schedule_seq=ns_seq,
            noise_schedule_struct=ns_struct,
            lambda_seq=2.0,
            lambda_struct=0.5,
            time_conditioning="adaln",
        )
        batch = self._make_batch()
        out = model(**batch)
        assert torch.isfinite(out["loss"])
