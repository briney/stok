import csv
from pathlib import Path

from click.testing import CliRunner

from stok.cli.cli import cli
from tests.utils.synthetic import random_protein_sequence


def _write_csv_varlen(path: Path, n_rows: int, seq_min_len: int, seq_max_len: int, max_len_minus_special: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = [0, 3, 5, 9]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "protein_sequence", "indices"])  # header
        for i in range(n_rows):
            seq = random_protein_sequence(seq_min_len, seq_max_len)
            k = lengths[i % len(lengths)]
            k = max(0, min(k, max_len_minus_special))
            indices_str = " ".join(["0"] * k)
            writer.writerow([f"p{i}", seq, indices_str])


def test_cli_train_with_csv_varlen_indices(tmp_path):
    runner = CliRunner()

    max_len = 16
    max_len_minus_special = max_len - 2

    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    _write_csv_varlen(train_csv, n_rows=8, seq_min_len=12, seq_max_len=28, max_len_minus_special=max_len_minus_special)
    _write_csv_varlen(eval_csv, n_rows=4, seq_min_len=12, seq_max_len=28, max_len_minus_special=max_len_minus_special)

    overrides = [
        f"data.train={train_csv.as_posix()}",
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
    assert "Training complete." in result.output


