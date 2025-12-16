"""Regression tests for the new modular evaluation harness.

These tests ensure the new evaluation system produces results consistent
with the training loop's expectations.
"""

import pandas as pd
from click.testing import CliRunner

from stok.cli.cli import cli


def test_eval_harness_produces_expected_metrics_codebook(tmp_path):
    """Test that eval harness produces expected metrics for codebook objective."""
    # Create minimal train/eval data
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"

    train_data = pd.DataFrame(
        {
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": ["MKTAYIAKQRQISFVKSHFSRQ" for _ in range(10)],
            "indices": ["0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" for _ in range(10)],
        }
    )
    train_data.to_csv(train_csv, index=False)

    eval_data = pd.DataFrame(
        {
            "pid": [f"eval_{i}" for i in range(5)],
            "protein_sequence": ["MKTAYIAKQRQISFVKSHFS" for _ in range(5)],
            "indices": ["0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17" for _ in range(5)],
        }
    )
    eval_data.to_csv(eval_csv, index=False)

    runner = CliRunner()
    overrides = [
        "train.objective=codebook",
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        "model.codebook.preset=lite",
        "data.batch_size=4",
        "data.max_len=32",
        "data.num_workers=0",
        "data.pin_memory=false",
        f"data.train={train_csv.as_posix()}",
        f"+data.eval.validation={eval_csv.as_posix()}",
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval_steps=2",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output

    # Check that expected metrics are logged
    assert "eval/validation" in result.output
    assert "acc" in result.output
    assert "ppl" in result.output
    assert "Training complete." in result.output


def test_eval_harness_produces_expected_metrics_mlm(tmp_path):
    """Test that eval harness produces expected metrics for MLM objective."""
    # Create minimal train/eval data (no indices needed for MLM)
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"

    train_data = pd.DataFrame(
        {
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": ["MKTAYIAKQRQISFVKSHFSRQ" for _ in range(10)],
        }
    )
    train_data.to_csv(train_csv, index=False)

    eval_data = pd.DataFrame(
        {
            "pid": [f"eval_{i}" for i in range(5)],
            "protein_sequence": ["MKTAYIAKQRQISFVKSHFS" for _ in range(5)],
        }
    )
    eval_data.to_csv(eval_csv, index=False)

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
        "data.max_len=32",
        "data.num_workers=0",
        "data.pin_memory=false",
        f"data.train={train_csv.as_posix()}",
        f"+data.eval.validation={eval_csv.as_posix()}",
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval_steps=2",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output

    # Check that expected MLM metrics are logged
    assert "eval/validation" in result.output
    assert "mask_acc" in result.output
    assert "ppl" in result.output
    assert "Training complete." in result.output


def test_eval_harness_multiple_eval_datasets(tmp_path):
    """Test that eval harness works with multiple eval datasets."""
    # Create minimal train/eval data
    train_csv = tmp_path / "train.csv"
    eval_val_csv = tmp_path / "eval_val.csv"
    eval_test_csv = tmp_path / "eval_test.csv"

    train_data = pd.DataFrame(
        {
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": ["MKTAYIAKQRQISFVKSHFSRQ" for _ in range(10)],
            "indices": ["0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" for _ in range(10)],
        }
    )
    train_data.to_csv(train_csv, index=False)

    eval_val_data = pd.DataFrame(
        {
            "pid": [f"val_{i}" for i in range(5)],
            "protein_sequence": ["MKTAYIAKQRQISFVKSHFS" for _ in range(5)],
            "indices": ["0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17" for _ in range(5)],
        }
    )
    eval_val_data.to_csv(eval_val_csv, index=False)

    eval_test_data = pd.DataFrame(
        {
            "pid": [f"test_{i}" for i in range(5)],
            "protein_sequence": ["MKTAYIAKQRQISFVKSHFS" for _ in range(5)],
            "indices": ["0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17" for _ in range(5)],
        }
    )
    eval_test_data.to_csv(eval_test_csv, index=False)

    runner = CliRunner()
    overrides = [
        "train.objective=codebook",
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        "model.codebook.preset=lite",
        "data.batch_size=4",
        "data.max_len=32",
        "data.num_workers=0",
        "data.pin_memory=false",
        f"data.train={train_csv.as_posix()}",
        f"+data.eval.validation={eval_val_csv.as_posix()}",
        f"+data.eval.test={eval_test_csv.as_posix()}",
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval_steps=2",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output

    # Check that both eval datasets are logged
    assert "eval/validation" in result.output
    assert "eval/test" in result.output
    assert "Training complete." in result.output


def test_eval_harness_smoke_dummy_data(tmp_path):
    """Smoke test with dummy data (no data.train specified)."""
    runner = CliRunner()
    overrides = [
        "train.objective=codebook",
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
        "train.eval_steps=100000",  # Don't trigger eval
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output

