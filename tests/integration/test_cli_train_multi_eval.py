import csv
from pathlib import Path

from click.testing import CliRunner

from stok.cli.cli import cli
from tests.utils.synthetic import random_protein_sequence


def _write_csv(path: Path, n_rows: int, seq_min_len: int, seq_max_len: int, indices_len: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "protein_sequence", "indices"])  # header
        indices_str = " ".join(["0"] * indices_len)
        for i in range(n_rows):
            seq = random_protein_sequence(seq_min_len, seq_max_len)
            writer.writerow([f"p{i}", seq, indices_str])


def test_cli_train_with_multiple_eval_datasets(tmp_path):
    runner = CliRunner()

    max_len = 16
    indices_len = max_len - 2  # align with token positions excluding BOS/EOS

    train_csv = tmp_path / "train.csv"
    eval_val_csv = tmp_path / "eval_val.csv"
    eval_test_csv = tmp_path / "eval_test.csv"
    _write_csv(train_csv, n_rows=8, seq_min_len=12, seq_max_len=28, indices_len=indices_len)
    _write_csv(eval_val_csv, n_rows=4, seq_min_len=12, seq_max_len=28, indices_len=indices_len)
    _write_csv(eval_test_csv, n_rows=4, seq_min_len=12, seq_max_len=28, indices_len=indices_len)

    overrides = [
        f"data.train={train_csv.as_posix()}",
        # multiple eval datasets via dict keys
        f"+data.eval.validation={eval_val_csv.as_posix()}",
        f"+data.eval.test={eval_test_csv.as_posix()}",
        # tiny model for speed
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        # small codebook preset
        "model.codebook.preset=lite",
        # small data loader
        "data.batch_size=2",
        f"data.max_len={max_len}",
        "data.num_workers=0",
        "data.pin_memory=false",
        # short run and ensure eval triggers
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval_steps=2",
        # disable external logging
        "train.wandb.enabled=false",
        # write artifacts to temp dir
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    # Expect per-dataset eval logs with step then epoch
    assert "eval/validation | step 2 | epoch" in result.output
    assert "eval/test | step 2 | epoch" in result.output
    assert "Training complete." in result.output


def test_cli_train_with_single_eval_dataset_via_data_eval_equals(tmp_path):
    runner = CliRunner()

    max_len = 16
    indices_len = max_len - 2  # align with token positions excluding BOS/EOS

    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    _write_csv(train_csv, n_rows=8, seq_min_len=12, seq_max_len=28, indices_len=indices_len)
    _write_csv(eval_csv, n_rows=4, seq_min_len=12, seq_max_len=28, indices_len=indices_len)

    overrides = [
        f"data.train={train_csv.as_posix()}",
        # single eval dataset via data.eval=/path syntax
        f"data.eval={eval_csv.as_posix()}",
        # tiny model for speed
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        # small codebook preset
        "model.codebook.preset=lite",
        # small data loader
        "data.batch_size=2",
        f"data.max_len={max_len}",
        "data.num_workers=0",
        "data.pin_memory=false",
        # short run and ensure eval triggers
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval_steps=2",
        # disable external logging
        "train.wandb.enabled=false",
        # write artifacts to temp dir
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    assert "eval/default | step 2 | epoch" in result.output
    assert "Training complete." in result.output

