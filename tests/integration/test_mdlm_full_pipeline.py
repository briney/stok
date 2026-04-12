"""Full pipeline integration tests for MDLM (Phase 4.2).

Tests the complete pretraining cascade: seq_only -> joint -> generation,
and all generation modes.
"""

import pytest
import torch

from stok.data.collate import mdlm_collate
from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.utils.sampling import sample
from stok.utils.tokenizer import Tokenizer
from stok.utils.weight_loading import load_pretrained_weights


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

CODEBOOK_SIZE = 16
CODEBOOK_DIM = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
MAX_LEN = 32


def _make_codebook() -> torch.Tensor:
    return torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM)


def _build_seq_model(schedule_type: str = "cosine") -> MDLMModel:
    ns = NoiseSchedule(schedule_type=schedule_type)
    return MDLMModel(
        tracks="seq_only",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_mult=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns,
        time_conditioning="adaln",
    )


def _build_joint_model(codebook: torch.Tensor | None = None) -> MDLMModel:
    if codebook is None:
        codebook = _make_codebook()
    ns_seq = NoiseSchedule(schedule_type="cosine")
    ns_struct = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="joint",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        codebook=codebook,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_mult=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns_seq,
        noise_schedule_struct=ns_struct,
        time_conditioning="adaln",
    )


def _make_seq_batch(tokenizer, noise_schedule, batch_size=4):
    seqs = ["ACDEFGHIKLMNPQRSTVWY"[:l] for l in range(6, 6 + batch_size)]
    batch = [{"seq": s} for s in seqs]
    return mdlm_collate(
        batch,
        tokenizer,
        noise_schedule_seq=noise_schedule,
        max_len=MAX_LEN,
        seq_mask_id=tokenizer.mask_token_id,
        seq_pad_id=tokenizer.pad_token_id,
        tracks="seq_only",
    )


def _make_paired_batch(tokenizer, ns_seq, ns_struct, batch_size=4):
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
        max_len=MAX_LEN,
        seq_mask_id=tokenizer.mask_token_id,
        seq_pad_id=tokenizer.pad_token_id,
        struct_mask_id=CODEBOOK_SIZE,
        struct_pad_id=CODEBOOK_SIZE + 1,
        tracks="joint",
    )


def _train_seq_steps(model, n_steps, tokenizer, noise_schedule, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        batch = _make_seq_batch(tokenizer, noise_schedule)
        outputs = model(
            seq_tokens=batch["seq_tokens"],
            t_seq=batch["t_seq"],
            seq_targets=batch["seq_targets"],
            seq_mask=batch["seq_mask"],
            key_padding_mask=batch["key_padding_mask"],
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses


def _train_joint_steps(model, n_steps, tokenizer, ns_seq, ns_struct, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
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
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFullPretrainingCascade:
    """Stage 1 seq_only -> stage 2 joint -> generate -> verify outputs."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return Tokenizer()

    @pytest.fixture(scope="class")
    def codebook(self):
        torch.manual_seed(42)
        return _make_codebook()

    def test_full_pipeline(self, tmp_path, tokenizer, codebook):
        torch.manual_seed(42)

        # -- Stage 1: seq_only pretraining --
        ns_seq = NoiseSchedule(schedule_type="cosine")
        seq_model = _build_seq_model()
        seq_losses = _train_seq_steps(seq_model, 20, tokenizer, ns_seq)
        assert len(seq_losses) == 20
        assert all(torch.isfinite(torch.tensor(l)) for l in seq_losses)

        # Save stage 1 checkpoint
        stage1_ckpt = tmp_path / "stage1.pt"
        torch.save(seq_model.state_dict(), stage1_ckpt)

        # -- Stage 2: load into joint, train --
        ns_struct = NoiseSchedule(schedule_type="cosine")
        joint_model = _build_joint_model(codebook=codebook)
        matched, missing = load_pretrained_weights(
            joint_model, stage1_ckpt, source_type="mdlm_seq"
        )
        assert len(matched) > 0, "No weights transferred from stage 1"

        joint_losses = _train_joint_steps(
            joint_model, 20, tokenizer, ns_seq, ns_struct
        )
        assert len(joint_losses) == 20
        assert all(torch.isfinite(torch.tensor(l)) for l in joint_losses)

        # -- Generate samples in codesign mode --
        result = sample(
            joint_model, length=12, num_samples=5, num_steps=10
        )
        assert result["seq_tokens"].shape == (5, 12)
        assert result["struct_tokens"].shape == (5, 12)
        # No mask tokens should remain
        assert (result["seq_tokens"] != joint_model.seq_mask_id).all()
        assert (result["struct_tokens"] != joint_model.struct_mask_id).all()
        # All outputs should be finite integer IDs
        assert (result["seq_tokens"] >= 0).all()
        assert (result["struct_tokens"] >= 0).all()


class TestAllGenerationModes:
    """Test codesign, forward, inverse, and scaffold on a joint model."""

    @pytest.fixture(scope="class")
    def model(self):
        torch.manual_seed(123)
        return _build_joint_model()

    def test_codesign(self, model):
        """Both tracks fully masked, generate everything."""
        result = sample(model, length=10, num_samples=3, num_steps=5)
        assert "seq_tokens" in result
        assert "struct_tokens" in result
        assert result["seq_tokens"].shape == (3, 10)
        assert result["struct_tokens"].shape == (3, 10)
        assert (result["seq_tokens"] != model.seq_mask_id).all()
        assert (result["struct_tokens"] != model.struct_mask_id).all()

    def test_forward(self, model):
        """Condition on sequence, generate structure."""
        B, L = 3, 10
        condition_seq = torch.randint(4, 24, (B, L))
        struct_mask = torch.ones(B, L, dtype=torch.bool)

        result = sample(
            model, length=L, num_samples=B, num_steps=5,
            condition_seq=condition_seq,
            struct_mask_positions=struct_mask,
        )
        # Sequence should be unchanged
        assert torch.equal(result["seq_tokens"], condition_seq)
        # Structure should be generated (valid indices)
        assert (result["struct_tokens"] >= 0).all()
        assert (result["struct_tokens"] != model.struct_mask_id).all()

    def test_inverse(self, model):
        """Condition on structure, generate sequence."""
        B, L = 3, 10
        condition_struct = torch.randint(0, CODEBOOK_SIZE, (B, L))
        seq_mask = torch.ones(B, L, dtype=torch.bool)

        result = sample(
            model, length=L, num_samples=B, num_steps=5,
            condition_struct=condition_struct,
            seq_mask_positions=seq_mask,
        )
        # Structure should be unchanged
        assert torch.equal(result["struct_tokens"], condition_struct)
        # Sequence should be generated
        assert (result["seq_tokens"] != model.seq_mask_id).all()

    def test_scaffold(self, model):
        """Partial conditioning on both tracks."""
        B, L = 2, 12
        condition_seq = torch.randint(4, 24, (B, L))
        condition_struct = torch.randint(0, CODEBOOK_SIZE, (B, L))

        # Generate positions 4-8, keep the rest fixed
        seq_mask = torch.zeros(B, L, dtype=torch.bool)
        seq_mask[:, 4:9] = True
        struct_mask = torch.zeros(B, L, dtype=torch.bool)
        struct_mask[:, 4:9] = True

        result = sample(
            model, length=L, num_samples=B, num_steps=5,
            condition_seq=condition_seq,
            condition_struct=condition_struct,
            seq_mask_positions=seq_mask,
            struct_mask_positions=struct_mask,
        )

        # Fixed positions should be unchanged
        for b in range(B):
            for l in range(L):
                if not seq_mask[b, l]:
                    assert result["seq_tokens"][b, l] == condition_seq[b, l]
                if not struct_mask[b, l]:
                    assert result["struct_tokens"][b, l] == condition_struct[b, l]

        # Generated positions should not have mask tokens
        for b in range(B):
            for l in range(4, 9):
                assert result["seq_tokens"][b, l] != model.seq_mask_id
                assert result["struct_tokens"][b, l] != model.struct_mask_id

    def test_seq_only_codesign(self):
        """Codesign with a seq_only model produces sequence only."""
        model = _build_seq_model()
        result = sample(model, length=10, num_samples=3, num_steps=5)
        assert "seq_tokens" in result
        assert "struct_tokens" not in result
        assert result["seq_tokens"].shape == (3, 10)
        assert (result["seq_tokens"] != model.seq_mask_id).all()
