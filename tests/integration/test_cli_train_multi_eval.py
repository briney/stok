import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from click.testing import CliRunner

from stok.cli.cli import cli
from tests.utils.synthetic import random_protein_sequence


def _write_parquet(path: Path, n_rows: int, seq_min_len: int, seq_max_len: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_rows):
        seq = random_protein_sequence(seq_min_len, seq_max_len)
        rows.append(
            {
                "sequence_id": f"p{i}",
                "sequence": seq,
                "structure_tokens": [0] * len(seq),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_cli_train_with_multiple_eval_datasets(tmp_path):
    runner = CliRunner()

    max_len = 16

    train_parquet = tmp_path / "train.parquet"
    eval_val_parquet = tmp_path / "eval_val.parquet"
    eval_test_parquet = tmp_path / "eval_test.parquet"
    _write_parquet(train_parquet, n_rows=8, seq_min_len=12, seq_max_len=28)
    _write_parquet(eval_val_parquet, n_rows=4, seq_min_len=12, seq_max_len=28)
    _write_parquet(eval_test_parquet, n_rows=4, seq_min_len=12, seq_max_len=28)

    overrides = [
        f"data.train={train_parquet.as_posix()}",
        # multiple eval datasets via dict keys
        f"+data.eval.validation={eval_val_parquet.as_posix()}",
        f"+data.eval.test={eval_test_parquet.as_posix()}",
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
        "train.batch_size=2",
        f"data.max_len={max_len}",
        "data.num_workers=0",
        "data.pin_memory=false",
        # short run and ensure eval triggers
        "train.max_steps=3",
        "train.log_every=1",
        "train.eval.steps=2",
        # disable external logging
        "train.wandb.enabled=false",
        # write artifacts to temp dir
        f"train.output_dir={tmp_path.as_posix()}",
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

    train_parquet = tmp_path / "train.parquet"
    eval_parquet = tmp_path / "eval.parquet"
    _write_parquet(train_parquet, n_rows=8, seq_min_len=12, seq_max_len=28)
    _write_parquet(eval_parquet, n_rows=4, seq_min_len=12, seq_max_len=28)

    overrides = [
        f"data.train={train_parquet.as_posix()}",
        # single eval dataset via data.eval=/path syntax
        f"data.eval={eval_parquet.as_posix()}",
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
        "train.batch_size=2",
        f"data.max_len={max_len}",
        "data.num_workers=0",
        "data.pin_memory=false",
        # short run and ensure eval triggers
        "train.max_steps=3",
        "train.log_every=1",
        "train.eval.steps=2",
        # disable external logging
        "train.wandb.enabled=false",
        # write artifacts to temp dir
        f"train.output_dir={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    assert "eval/default | step 2 | epoch" in result.output
    assert "Training complete." in result.output
