import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from click.testing import CliRunner

from stok.cli.cli import cli
from tests.utils.synthetic import random_protein_sequence


pytest.importorskip("pyarrow")


def _make_coords(L: int) -> list[list[list[float]]]:
    # Deterministic tiny coords per residue
    out = []
    for i in range(L):
        out.append([[float(i), 0.0, 0.0], [float(i), 1.0, 0.0], [float(i), 0.0, 1.0]])
    return out


def _write_parquet_with_coords(path: Path, n_rows: int, seq_min_len: int, seq_max_len: int, indices_len: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    indices_val = [0] * indices_len
    for i in range(n_rows):
        seq = random_protein_sequence(seq_min_len, seq_max_len)
        rows.append({
            "pid": f"pc{i}",
            "protein_sequence": seq,
            "indices": indices_val,
            "coordinates": _make_coords(len(seq)),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_cli_train_with_parquet_and_coords_e2e(tmp_path):
    runner = CliRunner()

    max_len = 16
    indices_len = max_len - 2  # align with token positions excluding BOS/EOS

    train_pq = tmp_path / "train.parquet"
    eval_pq = tmp_path / "eval.parquet"
    _write_parquet_with_coords(train_pq, n_rows=6, seq_min_len=12, seq_max_len=24, indices_len=indices_len)
    _write_parquet_with_coords(eval_pq, n_rows=4, seq_min_len=12, seq_max_len=24, indices_len=indices_len)

    overrides = [
        f"data.train={train_pq.as_posix()}",
        f"data.eval={eval_pq.as_posix()}",
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


