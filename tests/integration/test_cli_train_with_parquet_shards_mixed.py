import pytest
import pandas as pd
from pathlib import Path

from click.testing import CliRunner

from stok.cli.cli import cli
from tests.utils.synthetic import random_protein_sequence


pytest.importorskip("pyarrow")


def _write_parquet(path: Path, n_rows: int, seq_min_len: int, seq_max_len: int, indices_len: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    indices_val = [0] * indices_len
    for i in range(n_rows):
        seq = random_protein_sequence(seq_min_len, seq_max_len)
        rows.append(
            {
                "pid": f"p{i}",
                "protein_sequence": seq,
                "indices": indices_val,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_cli_train_with_parquet_dir_train_and_file_eval(tmp_path):
    runner = CliRunner()

    max_len = 16
    indices_len = max_len - 2  # align with token positions excluding BOS/EOS

    train_dir = tmp_path / "train_dir"
    shard1 = train_dir / "part-0001.parquet"
    shard2 = train_dir / "part-0002.parquet"
    _write_parquet(shard1, n_rows=6, seq_min_len=12, seq_max_len=28, indices_len=indices_len)
    _write_parquet(shard2, n_rows=6, seq_min_len=12, seq_max_len=28, indices_len=indices_len)

    eval_pq = tmp_path / "eval.parquet"
    _write_parquet(eval_pq, n_rows=4, seq_min_len=12, seq_max_len=28, indices_len=indices_len)

    overrides = [
        f"data.train={train_dir.as_posix()}",
        f"data.eval={eval_pq.as_posix()}",
        # iterable tuning (no effect on single-file eval)
        "data.shuffle_shards=true",
        "data.shuffle_rows=true",
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
        "train.num_steps=4",
        "train.log_steps=1",
        "train.eval.steps=2",
        # disable external logging
        "train.wandb.enabled=false",
        # write artifacts to temp dir
        f"train.project_path={tmp_path.as_posix()}",
    ]

    result = runner.invoke(cli, ["train", *overrides])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output

