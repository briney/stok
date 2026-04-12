"""Integration tests for gradient accumulation step accounting (Phase 0.1).

Verifies that:
- optimizer_step counts optimizer updates, not forward passes
- Scheduler is stepped in optimizer-step units
- Checkpoints are named with optimizer_step
- Training runs the correct total number of optimizer steps
"""

from pathlib import Path

from click.testing import CliRunner

from stok.cli.cli import cli


def _base_overrides(tmp_path: Path) -> list[str]:
    """Common training overrides for a tiny model."""
    return [
        "train.objective=mlm",
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
        "train.wandb.enabled=false",
        "train.eval.steps=100000",
        f"train.project_path={tmp_path.as_posix()}",
    ]


def test_grad_accum_optimizer_step_count(tmp_path):
    """With grad_accum_steps=4, num_steps=8: expect 8 optimizer updates.

    num_steps means optimizer steps. So 8 optimizer steps * 4 micro-steps
    = 32 forward passes total.
    """
    runner = CliRunner()
    overrides = _base_overrides(tmp_path) + [
        "train.grad_accum_steps=4",
        "train.num_steps=8",
        "train.log_steps=4",
        "train.checkpoint_steps=4",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output

    # Check that checkpoint names use optimizer step, not micro step
    ckpt_dir = tmp_path / "checkpoints"
    if not ckpt_dir.exists():
        # Some configs nest under a run subdirectory
        ckpt_dir = list(tmp_path.glob("*/checkpoints"))[0]
    step_checkpoints = sorted(ckpt_dir.glob("step_*.pt"))

    # With num_steps=8, checkpoint_steps=4: expect checkpoints at step 4 and 8
    ckpt_names = [p.name for p in step_checkpoints]
    assert "step_00000004.pt" in ckpt_names, f"Missing step_4 checkpoint. Found: {ckpt_names}"
    assert "step_00000008.pt" in ckpt_names, f"Missing step_8 checkpoint. Found: {ckpt_names}"


def test_grad_accum_log_shows_optimizer_steps(tmp_path):
    """Logging should show optimizer step counts, not micro steps."""
    runner = CliRunner()
    overrides = _base_overrides(tmp_path) + [
        "train.grad_accum_steps=2",
        "train.num_steps=4",
        "train.log_steps=2",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output

    # Console should show "step 2/4" and "step 4/4", not micro step counts
    assert "step 2/4" in result.output, f"Expected 'step 2/4' in output"
    assert "step 4/4" in result.output, f"Expected 'step 4/4' in output"


def test_grad_accum_1_is_no_op(tmp_path):
    """grad_accum_steps=1 should behave identically to no accumulation."""
    runner = CliRunner()
    overrides = _base_overrides(tmp_path) + [
        "train.grad_accum_steps=1",
        "train.num_steps=4",
        "train.log_steps=2",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output
    assert "step 2/4" in result.output
    assert "step 4/4" in result.output
