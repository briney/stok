"""Integration tests for MLM training objective."""

import pandas as pd
from click.testing import CliRunner

from stok.cli.cli import cli


def test_cli_train_mlm_smoke_runs_end_to_end_cpu(tmp_path):
    """Test that MLM training runs end-to-end on CPU with dummy data."""
    runner = CliRunner()
    overrides = [
        # MLM objective
        "train.objective=mlm",
        # tiny model for speed
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        # small codebook preset (not used for MLM but config requires it)
        "model.codebook.preset=lite",
        # small data loader
        "data.batch_size=2",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        # fast training
        "train.num_steps=5",
        "train.log_steps=2",
        "train.eval_steps=100000",
        "train.grad_accum_steps=1",
        # disable external logging
        "train.wandb.enabled=false",
        # MLM config
        "train.mlm.mask_prob=0.15",
        "train.mlm.tie_word_embeddings=true",
        # write artifacts to temp dir
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output
    assert "Training objective: mlm" in result.output


def test_cli_train_mlm_logs_mask_accuracy(tmp_path):
    """Test that MLM training logs mask_acc metric."""
    runner = CliRunner()
    overrides = [
        "train.objective=mlm",
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        "model.codebook.preset=lite",
        "data.batch_size=2",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval_steps=100000",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "mask_acc" in result.output


def test_cli_train_mlm_logs_perplexity(tmp_path):
    """Test that MLM training logs perplexity metric."""
    runner = CliRunner()
    overrides = [
        "train.objective=mlm",
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        "model.codebook.preset=lite",
        "data.batch_size=2",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval_steps=100000",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "ppl" in result.output


def test_cli_train_mlm_with_csv_dataset(tmp_path):
    """Test MLM training with a real CSV dataset (without indices column)."""
    # Create CSV with sequences only (no indices)
    train_csv = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "pid": [f"train_{i}" for i in range(20)],
            "protein_sequence": [
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"[: 20 + (i % 10)]
                for i in range(20)
            ],
        }
    )
    df.to_csv(train_csv, index=False)

    runner = CliRunner()
    overrides = [
        "train.objective=mlm",
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        "model.codebook.preset=lite",
        "data.batch_size=4",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        f"data.train={train_csv.as_posix()}",
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval_steps=100000",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output


def test_cli_train_mlm_with_eval_dataset(tmp_path):
    """Test MLM training with train and eval CSV datasets."""
    # Create train CSV
    train_csv = tmp_path / "train.csv"
    df_train = pd.DataFrame(
        {
            "pid": [f"train_{i}" for i in range(20)],
            "protein_sequence": [
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"[: 20 + (i % 10)]
                for i in range(20)
            ],
        }
    )
    df_train.to_csv(train_csv, index=False)

    # Create eval CSV
    eval_csv = tmp_path / "eval.csv"
    df_eval = pd.DataFrame(
        {
            "pid": [f"eval_{i}" for i in range(10)],
            "protein_sequence": [
                "MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQLKDFLSFA"[: 15 + (i % 8)]
                for i in range(10)
            ],
        }
    )
    df_eval.to_csv(eval_csv, index=False)

    runner = CliRunner()
    overrides = [
        "train.objective=mlm",
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        "model.codebook.preset=lite",
        "data.batch_size=4",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        f"data.train={train_csv.as_posix()}",
        f"+data.eval.validation={eval_csv.as_posix()}",
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval_steps=2",  # Trigger eval
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output
    # Check that eval was performed and logged mask_acc
    assert "eval/validation" in result.output


def test_cli_train_mlm_saves_checkpoint(tmp_path):
    """Test that MLM training saves checkpoints correctly."""
    runner = CliRunner()
    overrides = [
        "train.objective=mlm",
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        "model.codebook.preset=lite",
        "data.batch_size=2",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        "train.num_steps=5",
        "train.log_steps=1",
        "train.eval_steps=100000",
        "train.checkpoint_steps=2",  # Save checkpoint every 2 steps
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output

    # Check that final model was saved
    final_checkpoint = tmp_path / "model" / "final.pt"
    assert final_checkpoint.exists(), "Final checkpoint not saved"

    # Check that step checkpoints were saved
    checkpoints_dir = tmp_path / "checkpoints"
    assert checkpoints_dir.exists()
    step_checkpoints = list(checkpoints_dir.glob("step_*.pt"))
    assert len(step_checkpoints) >= 1, "No step checkpoints saved"


def test_cli_train_codebook_still_works(tmp_path):
    """Test that codebook training still works (regression test)."""
    runner = CliRunner()
    overrides = [
        "train.objective=codebook",  # Explicitly set codebook objective
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        "model.codebook.preset=lite",
        "data.batch_size=2",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval_steps=100000",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output
    assert "Training objective: codebook" in result.output
    # Codebook training should log "acc" not "mask_acc"
    assert "acc" in result.output

