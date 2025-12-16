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


def test_cli_train_with_multiple_train_datasets_and_fractions(tmp_path):
    runner = CliRunner()

    max_len = 16
    indices_len = max_len - 2

    train_a = tmp_path / "train_a.csv"
    train_b = tmp_path / "train_b.csv"
    _write_csv(train_a, n_rows=8, seq_min_len=12, seq_max_len=28, indices_len=indices_len)
    _write_csv(train_b, n_rows=8, seq_min_len=12, seq_max_len=28, indices_len=indices_len)

    overrides = [
        # multiple train datasets via dict keys + fractions
        f"+data.train.dataset_a.path={train_a.as_posix()}",
        "+data.train.dataset_a.fraction=0.6",
        f"+data.train.dataset_b.path={train_b.as_posix()}",
        "+data.train.dataset_b.fraction=0.4",
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
        # short run
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval.steps=100000",
        # disable external logging
        "train.wandb.enabled=false",
        # write artifacts to temp dir
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output


