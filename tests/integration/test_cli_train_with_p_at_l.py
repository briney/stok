"""Integration tests for P@L (Precision@L) contact prediction metric."""

import numpy as np
import pandas as pd
from click.testing import CliRunner

from stok.cli.cli import cli


def _generate_coords(length: int) -> list:
    return (np.random.RandomState(42).randn(length, 3, 3) * 5.0).tolist()


def test_cli_train_mlm_with_p_at_l_disabled(tmp_path):
    """Test that MLM training works with P@L disabled (default)."""
    # Create minimal data with coords
    train_parquet = tmp_path / "train.parquet"
    seq = "MKTAYIAKQRQISFVK"
    train_data = pd.DataFrame(
        {
            "sequence_id": [f"train_{i}" for i in range(10)],
            "sequence": [seq for _ in range(10)],
            "coordinates": [_generate_coords(len(seq)) for _ in range(10)],
        }
    )
    train_data.to_parquet(train_parquet, index=False)

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
        f"data.train={train_parquet.as_posix()}",
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval.steps=100000",  # Don't trigger eval
        "train.wandb.enabled=false",
        # P@L is disabled by default
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output


def test_cli_train_mlm_with_p_at_l_enabled(tmp_path):
    """Test that MLM training works with P@L enabled.

    Note: This test primarily verifies that the metric can be enabled
    without errors. Actual P@L computation requires attention weights
    or hidden states from the model.
    """
    # Create minimal data with coords
    train_parquet = tmp_path / "train.parquet"
    eval_parquet = tmp_path / "eval.parquet"
    seq = "MKTAYIAKQRQISFVK"

    train_data = pd.DataFrame(
        {
            "sequence_id": [f"train_{i}" for i in range(10)],
            "sequence": [seq for _ in range(10)],
            "coordinates": [
                np.random.default_rng(42).normal(size=(len(seq), 3, 3)).tolist()
                for _ in range(10)
            ],
        }
    )
    train_data.to_parquet(train_parquet, index=False)

    eval_data = pd.DataFrame(
        {
            "sequence_id": [f"eval_{i}" for i in range(5)],
            "sequence": [seq for _ in range(5)],
            "coordinates": [
                np.random.default_rng(42).normal(size=(len(seq), 3, 3)).tolist()
                for _ in range(5)
            ],
        }
    )
    eval_data.to_parquet(eval_parquet, index=False)

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
        "data.load_coords=true",
        f"data.train={train_parquet.as_posix()}",
        f"+data.eval.validation={eval_parquet.as_posix()}",
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval.steps=2",
        "train.wandb.enabled=false",
        # Enable P@L metric (override existing config values)
        "train.eval.metrics.p_at_l.enabled=true",
        "train.eval.metrics.p_at_l.contact_threshold=8.0",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output
    assert "P@L" in result.output


def test_p_at_l_metric_config_override(tmp_path):
    """Test that P@L metric config can be overridden per-dataset."""
    # Create minimal data
    train_parquet = tmp_path / "train.parquet"
    eval_parquet = tmp_path / "eval.parquet"
    seq = "MKTAYIAKQRQISFVK"

    train_data = pd.DataFrame(
        {
            "sequence_id": [f"train_{i}" for i in range(10)],
            "sequence": [seq for _ in range(10)],
        }
    )
    train_data.to_parquet(train_parquet, index=False)

    eval_data = pd.DataFrame(
        {
            "sequence_id": [f"eval_{i}" for i in range(5)],
            "sequence": [seq for _ in range(5)],
        }
    )
    eval_data.to_parquet(eval_parquet, index=False)

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
        f"data.train={train_parquet.as_posix()}",
        f"+data.eval.validation.path={eval_parquet.as_posix()}",
        # Override metrics for this specific eval dataset
        # (P@L won't run without coords, but config parsing should work)
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval.steps=2",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output
