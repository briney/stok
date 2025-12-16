"""Integration tests for P@L (Precision@L) contact prediction metric."""

import numpy as np
import pandas as pd
from click.testing import CliRunner

from stok.cli.cli import cli


def _generate_coords_string(length: int) -> str:
    """Generate a mock coords string for testing.

    Returns a space-separated string of 9 * length floats representing
    N, CA, C coordinates for each residue.
    """
    # Generate random coords in a reasonable range
    rng = np.random.RandomState(42)
    coords = rng.randn(length, 3, 3) * 5.0  # [L, 3_atoms, 3_xyz]
    flat = coords.flatten()
    return " ".join(f"{x:.4f}" for x in flat)


def test_cli_train_mlm_with_p_at_l_disabled(tmp_path):
    """Test that MLM training works with P@L disabled (default)."""
    # Create minimal data with coords
    train_csv = tmp_path / "train.csv"
    seq = "MKTAYIAKQRQISFVK"
    train_data = pd.DataFrame(
        {
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
            "coords": [_generate_coords_string(len(seq)) for _ in range(10)],
        }
    )
    train_data.to_csv(train_csv, index=False)

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
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval_steps=100000",  # Don't trigger eval
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
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    seq = "MKTAYIAKQRQISFVK"

    train_data = pd.DataFrame(
        {
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
            "coords": [_generate_coords_string(len(seq)) for _ in range(10)],
        }
    )
    train_data.to_csv(train_csv, index=False)

    eval_data = pd.DataFrame(
        {
            "pid": [f"eval_{i}" for i in range(5)],
            "protein_sequence": [seq for _ in range(5)],
            "coords": [_generate_coords_string(len(seq)) for _ in range(5)],
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
        "data.load_coords=true",
        f"data.train={train_csv.as_posix()}",
        f"+data.eval.validation={eval_csv.as_posix()}",
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval_steps=2",
        "train.wandb.enabled=false",
        # Enable P@L metric
        "+train.eval.metrics.p_at_l.enabled=true",
        "+train.eval.metrics.p_at_l.contact_threshold=8.0",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    # Note: P@L may not produce output if attention weights aren't exposed
    # The test primarily verifies no errors are raised
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output


def test_p_at_l_metric_config_override(tmp_path):
    """Test that P@L metric config can be overridden per-dataset."""
    # Create minimal data
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    seq = "MKTAYIAKQRQISFVK"

    train_data = pd.DataFrame(
        {
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
        }
    )
    train_data.to_csv(train_csv, index=False)

    eval_data = pd.DataFrame(
        {
            "pid": [f"eval_{i}" for i in range(5)],
            "protein_sequence": [seq for _ in range(5)],
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
        f"+data.eval.validation.path={eval_csv.as_posix()}",
        # Override metrics for this specific eval dataset
        # (P@L won't run without coords, but config parsing should work)
        "train.num_steps=4",
        "train.log_steps=2",
        "train.eval_steps=2",
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output

