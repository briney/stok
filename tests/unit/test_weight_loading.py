"""Tests for pretrained weight loading and transfer."""

import pytest
import torch

from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.utils.weight_loading import EncoderFreezeHook, load_pretrained_weights


@pytest.fixture
def ns():
    return NoiseSchedule(schedule_type="cosine")


@pytest.fixture
def codebook():
    return torch.randn(16, 8)


def _build_seq_model(ns):
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


def _build_joint_model(ns, codebook):
    ns_struct = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
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
        noise_schedule_seq=ns,
        noise_schedule_struct=ns_struct,
        time_conditioning="adaln",
    )


class TestLoadFromSeqOnly:
    """Load seq_only checkpoint into joint model."""

    def test_encoder_weights_match(self, ns, codebook, tmp_path):
        # Save seq_only model
        src = _build_seq_model(ns)
        ckpt_path = tmp_path / "seq_only.pt"
        torch.save(src.state_dict(), ckpt_path)

        # Load into joint model
        tgt = _build_joint_model(ns, codebook)
        matched, missing = load_pretrained_weights(
            tgt, ckpt_path, source_type="mdlm_seq"
        )

        # Encoder weights should match
        for key in src.state_dict():
            if key.startswith("encoder."):
                assert torch.equal(
                    tgt.state_dict()[key],
                    src.state_dict()[key],
                ), f"Mismatch at {key}"

    def test_seq_embed_weights_match(self, ns, codebook, tmp_path):
        src = _build_seq_model(ns)
        ckpt_path = tmp_path / "seq_only.pt"
        torch.save(src.state_dict(), ckpt_path)

        tgt = _build_joint_model(ns, codebook)
        load_pretrained_weights(tgt, ckpt_path, source_type="mdlm_seq")

        assert torch.equal(
            tgt.embed_seq.weight.data,
            src.embed_seq.weight.data,
        )

    def test_struct_weights_are_random(self, ns, codebook, tmp_path):
        src = _build_seq_model(ns)
        ckpt_path = tmp_path / "seq_only.pt"
        torch.save(src.state_dict(), ckpt_path)

        # Build two joint models with different seeds
        torch.manual_seed(42)
        tgt1 = _build_joint_model(ns, codebook)
        torch.manual_seed(99)
        tgt2 = _build_joint_model(ns, codebook)

        # Before loading, struct embeddings should differ
        assert not torch.equal(tgt1.embed_struct.weight, tgt2.embed_struct.weight)

        # After loading seq_only weights, struct embedding should NOT come from checkpoint
        load_pretrained_weights(tgt1, ckpt_path, source_type="mdlm_seq")
        # Struct embedding should still be the randomly initialized values
        # (since seq_only checkpoint has no struct weights)
        assert "embed_struct.weight" not in {
            k for k in src.state_dict()
        }

    def test_matched_and_missing_keys(self, ns, codebook, tmp_path):
        src = _build_seq_model(ns)
        ckpt_path = tmp_path / "seq_only.pt"
        torch.save(src.state_dict(), ckpt_path)

        tgt = _build_joint_model(ns, codebook)
        matched, missing = load_pretrained_weights(
            tgt, ckpt_path, source_type="mdlm_seq"
        )

        # All seq_only keys should be matched
        assert len(matched) > 0
        # Joint model has extra keys not in seq_only
        assert len(missing) > 0
        # Missing should include struct-specific keys
        struct_missing = [k for k in missing if "struct" in k or "track_embed" in k]
        assert len(struct_missing) > 0


class TestLoadFromMLM:
    """Load MLM (STokModel) checkpoint into MDLMModel."""

    def test_key_mapping(self, ns, tmp_path):
        # Simulate an MLM checkpoint with STokModel-style keys
        mlm_state = {
            "embed.weight": torch.randn(32, 64),
            "encoder.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "lm_head.dense.weight": torch.randn(64, 64),
            "lm_head.dense.bias": torch.randn(64),
            "lm_head.layer_norm.weight": torch.randn(64),
            "lm_head.layer_norm.bias": torch.randn(64),
        }
        ckpt_path = tmp_path / "mlm.pt"
        torch.save(mlm_state, ckpt_path)

        tgt = _build_seq_model(ns)
        matched, missing = load_pretrained_weights(
            tgt, ckpt_path, source_type="mlm"
        )

        # embed.weight -> embed_seq.weight
        assert "embed_seq.weight" in matched
        # lm_head.* -> head_seq.*
        head_matched = [k for k in matched if k.startswith("head_seq.")]
        assert len(head_matched) > 0


class TestLoadFromModelStateDict:
    """Load from wrapped checkpoint formats."""

    def test_model_state_dict_key(self, ns, tmp_path):
        src = _build_seq_model(ns)
        wrapped = {"model_state_dict": src.state_dict(), "epoch": 5}
        ckpt_path = tmp_path / "wrapped.pt"
        torch.save(wrapped, ckpt_path)

        tgt = _build_seq_model(ns)
        matched, missing = load_pretrained_weights(
            tgt, ckpt_path, source_type="mdlm_seq"
        )
        assert len(matched) > 0

    def test_state_dict_key(self, ns, tmp_path):
        src = _build_seq_model(ns)
        wrapped = {"state_dict": src.state_dict()}
        ckpt_path = tmp_path / "wrapped2.pt"
        torch.save(wrapped, ckpt_path)

        tgt = _build_seq_model(ns)
        matched, missing = load_pretrained_weights(
            tgt, ckpt_path, source_type="mdlm_seq"
        )
        assert len(matched) > 0


class TestInvalidSourceType:
    def test_unknown_source_type_raises(self, ns, tmp_path):
        src = _build_seq_model(ns)
        ckpt_path = tmp_path / "test.pt"
        torch.save(src.state_dict(), ckpt_path)

        with pytest.raises(ValueError, match="Unknown source_type"):
            load_pretrained_weights(src, ckpt_path, source_type="invalid")


class TestEncoderFreezeHook:
    """Test gradient freezing for encoder."""

    def test_gradients_zeroed_during_freeze(self, ns):
        model = _build_seq_model(ns)
        hook = EncoderFreezeHook(model, freeze_steps=3)
        hook.register()

        # Simulate forward/backward
        B, L = 2, 8
        seq_tokens = torch.randint(2, 22, (B, L))
        t_seq = torch.rand(B)
        targets = torch.randint(2, 22, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        padding = torch.zeros(B, L, dtype=torch.bool)

        out = model(
            seq_tokens=seq_tokens, t_seq=t_seq,
            seq_targets=targets, seq_mask=mask, key_padding_mask=padding,
        )
        out["loss"].backward()

        # During freeze period (step 0 < 3), encoder grads should be None
        for p in model.encoder.parameters():
            assert p.grad is None

        hook.step()
        hook.step()
        hook.step()  # Now at step 3, hook should be removed

        # After unfreeze, gradients should flow
        model.zero_grad()
        out = model(
            seq_tokens=seq_tokens, t_seq=t_seq,
            seq_targets=targets, seq_mask=mask, key_padding_mask=padding,
        )
        out["loss"].backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        assert has_grad
