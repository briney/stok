"""Integration tests for joint (two-track) MDLM training (Phase 3)."""

import pytest
import torch

from stok.data.collate import mdlm_collate
from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.utils.sampling import sample
from stok.utils.tokenizer import Tokenizer
from stok.utils.weight_loading import load_pretrained_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CODEBOOK_SIZE = 16
CODEBOOK_DIM = 8


def _make_codebook():
    return torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM)


def _build_joint_model(
    codebook: torch.Tensor | None = None,
    schedule_type: str = "cosine",
    time_conditioning: str = "adaln",
    time_combine: str = "sum",
) -> MDLMModel:
    if codebook is None:
        codebook = _make_codebook()
    ns_seq = NoiseSchedule(schedule_type=schedule_type)
    ns_struct = NoiseSchedule(schedule_type=schedule_type)
    return MDLMModel(
        tracks="joint",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        codebook=codebook,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns_seq,
        noise_schedule_struct=ns_struct,
        time_conditioning=time_conditioning,
        time_combine=time_combine,
    )


def _build_seq_model(schedule_type: str = "cosine") -> MDLMModel:
    ns = NoiseSchedule(schedule_type=schedule_type)
    return MDLMModel(
        tracks="seq_only",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns,
        time_conditioning="adaln",
    )


def _make_paired_batch(tokenizer, ns_seq, ns_struct, batch_size=4, max_len=32):
    """Create synthetic paired (seq + struct) batch."""
    seqs = ["ACDEFGHIKLMNPQRSTVWY"[:l] for l in range(6, 6 + batch_size)]
    batch = []
    for s in seqs:
        indices = torch.randint(0, CODEBOOK_SIZE, (len(s),))
        batch.append({"seq": s, "indices": indices})

    return mdlm_collate(
        batch,
        tokenizer,
        noise_schedule_seq=ns_seq,
        noise_schedule_struct=ns_struct,
        max_len=max_len,
        seq_mask_id=tokenizer.mask_token_id,
        seq_pad_id=tokenizer.pad_token_id,
        struct_mask_id=CODEBOOK_SIZE,       # C
        struct_pad_id=CODEBOOK_SIZE + 1,    # C + 1
        tracks="joint",
    )


def _train_joint_steps(model, n_steps, tokenizer, ns_seq, ns_struct, lr=1e-3):
    """Train joint model for n_steps and return list of (loss, loss_seq, loss_struct)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    records = []
    for _ in range(n_steps):
        batch = _make_paired_batch(tokenizer, ns_seq, ns_struct)
        outputs = model(
            seq_tokens=batch["seq_tokens"],
            t_seq=batch["t_seq"],
            seq_targets=batch["seq_targets"],
            seq_mask=batch["seq_mask"],
            key_padding_mask=batch["key_padding_mask"],
            struct_tokens=batch["struct_tokens"],
            t_struct=batch["t_struct"],
            struct_targets=batch["struct_targets"],
            struct_mask=batch["struct_mask"],
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        records.append({
            "loss": loss.item(),
            "loss_seq": outputs["loss_seq"].item(),
            "loss_struct": outputs["loss_struct"].item(),
        })
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJointTrainingSmoke:
    """End-to-end joint training smoke tests."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return Tokenizer()

    @pytest.fixture(scope="class")
    def ns_seq(self):
        return NoiseSchedule(schedule_type="cosine")

    @pytest.fixture(scope="class")
    def ns_struct(self):
        return NoiseSchedule(schedule_type="cosine")

    def test_train_10_steps_finite_loss(self, tokenizer, ns_seq, ns_struct):
        model = _build_joint_model()
        records = _train_joint_steps(model, 10, tokenizer, ns_seq, ns_struct)
        assert len(records) == 10
        for r in records:
            assert torch.isfinite(torch.tensor(r["loss"]))
            assert torch.isfinite(torch.tensor(r["loss_seq"]))
            assert torch.isfinite(torch.tensor(r["loss_struct"]))

    def test_both_losses_reported(self, tokenizer, ns_seq, ns_struct):
        model = _build_joint_model()
        records = _train_joint_steps(model, 5, tokenizer, ns_seq, ns_struct)
        for r in records:
            assert r["loss_seq"] > 0
            assert r["loss_struct"] > 0
            # Combined loss should be sum of individual losses
            expected = r["loss_seq"] + r["loss_struct"]
            assert abs(r["loss"] - expected) < 1e-4

    def test_loss_does_not_explode(self, tokenizer, ns_seq, ns_struct):
        model = _build_joint_model()
        records = _train_joint_steps(model, 10, tokenizer, ns_seq, ns_struct)
        max_loss = max(r["loss"] for r in records)
        assert max_loss < 1000.0, f"Loss exploded: max={max_loss}"


class TestWeightTransfer:
    """Load seq_only weights into joint model and continue training."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return Tokenizer()

    @pytest.fixture(scope="class")
    def ns_seq(self):
        return NoiseSchedule(schedule_type="cosine")

    @pytest.fixture(scope="class")
    def ns_struct(self):
        return NoiseSchedule(schedule_type="cosine")

    def test_seq_to_joint_transfer(self, tmp_path, tokenizer, ns_seq, ns_struct):
        torch.manual_seed(42)

        # Train seq_only for 5 steps
        seq_model = _build_seq_model()
        from stok.data.collate import mdlm_collate

        optimizer = torch.optim.Adam(seq_model.parameters(), lr=1e-3)
        for _ in range(5):
            seqs = ["ACDEFGHIKLMNPQRSTVWY"[:l] for l in range(6, 10)]
            batch_items = [{"seq": s} for s in seqs]
            batch = mdlm_collate(
                batch_items, tokenizer, ns_seq,
                max_len=32, seq_mask_id=tokenizer.mask_token_id,
                seq_pad_id=tokenizer.pad_token_id, tracks="seq_only",
            )
            out = seq_model(
                seq_tokens=batch["seq_tokens"], t_seq=batch["t_seq"],
                seq_targets=batch["seq_targets"], seq_mask=batch["seq_mask"],
                key_padding_mask=batch["key_padding_mask"],
            )
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save checkpoint
        ckpt_path = tmp_path / "seq_only.pt"
        torch.save(seq_model.state_dict(), ckpt_path)

        # Load into joint model
        joint_model = _build_joint_model()
        matched, missing = load_pretrained_weights(
            joint_model, ckpt_path, source_type="mdlm_seq"
        )
        assert len(matched) > 0

        # Continue training joint for 5 steps
        records = _train_joint_steps(joint_model, 5, tokenizer, ns_seq, ns_struct)
        for r in records:
            assert torch.isfinite(torch.tensor(r["loss"]))


class TestConditionalGeneration:
    """Test generation in different conditioning modes."""

    def test_codesign(self):
        model = _build_joint_model()
        result = sample(model, length=8, num_samples=2, num_steps=5)
        assert "seq_tokens" in result
        assert "struct_tokens" in result
        assert (result["seq_tokens"] != model.seq_mask_id).all()
        assert (result["struct_tokens"] != model.struct_mask_id).all()

    def test_forward_mode(self):
        """Condition on seq, generate struct."""
        model = _build_joint_model()
        B, L = 2, 8
        condition_seq = torch.randint(4, 24, (B, L))
        struct_mask = torch.ones(B, L, dtype=torch.bool)

        result = sample(
            model, length=L, num_samples=B, num_steps=5,
            condition_seq=condition_seq,
            struct_mask_positions=struct_mask,
        )
        # Seq unchanged
        assert torch.equal(result["seq_tokens"], condition_seq)
        # Struct should be valid
        assert (result["struct_tokens"] >= 0).all()
        assert (result["struct_tokens"] < CODEBOOK_SIZE + 2).all()

    def test_inverse_mode(self):
        """Condition on struct, generate seq."""
        model = _build_joint_model()
        B, L = 2, 8
        condition_struct = torch.randint(0, CODEBOOK_SIZE, (B, L))
        seq_mask = torch.ones(B, L, dtype=torch.bool)

        result = sample(
            model, length=L, num_samples=B, num_steps=5,
            condition_struct=condition_struct,
            seq_mask_positions=seq_mask,
        )
        # Struct unchanged
        assert torch.equal(result["struct_tokens"], condition_struct)
        # Seq should have valid tokens
        assert (result["seq_tokens"] != model.seq_mask_id).all()

    def test_scaffold_mode(self):
        """Partial conditioning on both tracks."""
        model = _build_joint_model()
        B, L = 2, 10
        condition_seq = torch.randint(4, 24, (B, L))
        condition_struct = torch.randint(0, CODEBOOK_SIZE, (B, L))

        # Only generate positions 3-7
        seq_mask = torch.zeros(B, L, dtype=torch.bool)
        seq_mask[:, 3:8] = True
        struct_mask = torch.zeros(B, L, dtype=torch.bool)
        struct_mask[:, 3:8] = True

        result = sample(
            model, length=L, num_samples=B, num_steps=5,
            condition_seq=condition_seq,
            condition_struct=condition_struct,
            seq_mask_positions=seq_mask,
            struct_mask_positions=struct_mask,
        )

        # Conditioned positions should be unchanged
        for b in range(B):
            for l in range(L):
                if not seq_mask[b, l]:
                    assert result["seq_tokens"][b, l] == condition_seq[b, l]
                if not struct_mask[b, l]:
                    assert result["struct_tokens"][b, l] == condition_struct[b, l]
