"""Integration tests for MDLM seq_only training (Phase 2)."""

import torch
import pytest

from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.data.collate import mdlm_collate
from stok.utils.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_small_model(
    schedule_type: str = "cosine",
    time_conditioning: str = "adaln",
) -> MDLMModel:
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
        time_conditioning=time_conditioning,
    )


def _make_batch(tokenizer, noise_schedule, batch_size=4, max_len=32):
    """Create a synthetic training batch."""
    seqs = ["ACDEFGHIKLMNPQRSTVWY"[:l] for l in range(6, 6 + batch_size)]
    batch = [{"seq": s} for s in seqs]
    return mdlm_collate(
        batch,
        tokenizer,
        noise_schedule_seq=noise_schedule,
        max_len=max_len,
        seq_mask_id=tokenizer.mask_token_id,
        seq_pad_id=tokenizer.pad_token_id,
        tracks="seq_only",
    )


def _train_steps(model, n_steps, tokenizer, noise_schedule, lr=1e-3):
    """Train model for n_steps and return list of losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        batch = _make_batch(tokenizer, noise_schedule)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMDLMSeqOnlySmoke:
    """Smoke test: train for a few steps without crashing."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return Tokenizer()

    @pytest.fixture(scope="class")
    def noise_schedule(self):
        return NoiseSchedule(schedule_type="cosine")

    def test_train_10_steps_finite_loss(self, tokenizer, noise_schedule):
        model = _build_small_model()
        losses = _train_steps(model, 10, tokenizer, noise_schedule)
        assert len(losses) == 10
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    def test_loss_does_not_explode(self, tokenizer, noise_schedule):
        model = _build_small_model()
        losses = _train_steps(model, 10, tokenizer, noise_schedule)
        # Loss should not grow unboundedly. Note: MDLM loss weights can cause
        # spikes at extreme time values, so we use a generous threshold.
        assert max(losses) < 500.0, f"Loss exploded: {losses}"


class TestMDLMCheckpointRoundTrip:
    """Save and load a checkpoint, verify training continues smoothly."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return Tokenizer()

    @pytest.fixture(scope="class")
    def noise_schedule(self):
        return NoiseSchedule(schedule_type="cosine")

    def test_checkpoint_round_trip(self, tmp_path, tokenizer, noise_schedule):
        torch.manual_seed(42)
        model = _build_small_model()

        # Train 5 steps
        losses_before = _train_steps(model, 5, tokenizer, noise_schedule)

        # Save checkpoint
        ckpt_path = tmp_path / "mdlm_test.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Load into fresh model
        model2 = _build_small_model()
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Train 5 more steps
        losses_after = _train_steps(model2, 5, tokenizer, noise_schedule)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses_after)


class TestMDLMTimeConditioning:
    """Test different time conditioning modes."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return Tokenizer()

    @pytest.fixture(scope="class")
    def noise_schedule(self):
        return NoiseSchedule(schedule_type="cosine")

    def test_adaln_produces_finite_loss(self, tokenizer, noise_schedule):
        model = _build_small_model(time_conditioning="adaln")
        losses = _train_steps(model, 5, tokenizer, noise_schedule)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    def test_no_time_conditioning_produces_finite_loss(self, tokenizer, noise_schedule):
        """Without time conditioning, model still works (time info lost but no crash)."""
        ns = NoiseSchedule(schedule_type="cosine")
        model = MDLMModel(
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
            time_conditioning=None,
        )
        losses = _train_steps(model, 5, tokenizer, noise_schedule)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


class TestMDLMNoiseScheduleVariants:
    """Test different noise schedule types all produce finite loss."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return Tokenizer()

    @pytest.mark.parametrize("schedule_type", ["linear", "cosine", "sqrt"])
    def test_schedule_produces_finite_loss(self, tokenizer, schedule_type):
        ns = NoiseSchedule(schedule_type=schedule_type)
        model = _build_small_model(schedule_type=schedule_type)
        losses = _train_steps(model, 3, tokenizer, ns)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


class TestMDLMCLISmoke:
    """Test MDLM training through the CLI interface."""

    def test_cli_mdlm_seq_only_runs(self, tmp_path):
        """Train MDLM seq_only objective through CLI with dummy data."""
        from click.testing import CliRunner
        from stok.cli.cli import cli

        runner = CliRunner()
        overrides = [
            "train.objective=mdlm",
            "train.mdlm.tracks=seq_only",
            "train.mdlm.noise_schedule_seq.type=cosine",
            "train.mdlm.time_conditioning=adaln",
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            "data.max_len=32",
            "data.num_workers=0",
            "data.pin_memory=false",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=100000",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert "Training objective: mdlm" in result.output
        assert "Training complete." in result.output

    def test_cli_mdlm_logs_mask_rate(self, tmp_path):
        """MDLM training should log mask_rate in console output."""
        from click.testing import CliRunner
        from stok.cli.cli import cli

        runner = CliRunner()
        overrides = [
            "train.objective=mdlm",
            "train.mdlm.tracks=seq_only",
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            "data.max_len=32",
            "data.num_workers=0",
            "data.pin_memory=false",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=100000",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert "mask_rate" in result.output
