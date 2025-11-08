import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

from stok.cli.cli import cli
from stok.utils.codebook import load_codebook


pytest.importorskip("pyarrow")
pytest.importorskip("x_transformers")


def _make_coords(L: int) -> list[list[list[float]]]:
    out = []
    for i in range(L):
        out.append([[float(i), 0.0, 0.0], [float(i), 1.0, 0.0], [float(i), 0.0, 1.0]])
    return out


def _write_parquet_with_coords(path: Path, n_rows: int, seq_min_len: int, seq_max_len: int, indices_len: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    indices_val = [0] * indices_len
    for i in range(n_rows):
        L = np.random.randint(seq_min_len, seq_max_len + 1)
        seq = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=L))
        rows.append({
            "pid": f"pc{i}",
            "protein_sequence": seq,
            "indices": indices_val,
            "coordinates": _make_coords(L),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _make_decoder_ckpt(tmp_path: Path, preset: str = "lite") -> Path:
    from stok.models.decoder import GeometricDecoder
    # match d_code to codebook preset
    codebook = load_codebook(preset=preset)
    d_code = int(codebook.shape[1])
    if preset == "base":
        arch = dict(d_model=1024, ffn_mult=4.0, n_layers=16, n_heads=16, attn_kv_heads=1, num_memory_tokens=0, max_length=1280)
    else:
        arch = dict(d_model=1024, ffn_mult=4.0, n_layers=12, n_heads=8, attn_kv_heads=2, num_memory_tokens=0, max_length=1280)
    model = GeometricDecoder(
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        ffn_mult=arch["ffn_mult"],
        max_length=arch["max_length"],
        d_code=d_code,
        num_memory_tokens=arch["num_memory_tokens"],
        attn_kv_heads=arch["attn_kv_heads"],
    )
    ckpt_path = tmp_path / f"decoder-{preset}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def test_training_with_decoder_and_fape(tmp_path):
    runner = CliRunner()

    max_len = 16
    indices_len = max_len - 2  # align with token positions excluding BOS/EOS

    train_pq = tmp_path / "train.parquet"
    eval_pq = tmp_path / "eval.parquet"
    _write_parquet_with_coords(train_pq, n_rows=4, seq_min_len=12, seq_max_len=18, indices_len=indices_len)
    _write_parquet_with_coords(eval_pq, n_rows=2, seq_min_len=12, seq_max_len=18, indices_len=indices_len)

    ckpt_path = _make_decoder_ckpt(tmp_path, preset="lite")

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
        # enable decoder + fape
        "model.decoder.enabled=true",
        f"model.decoder.path={ckpt_path.as_posix()}",
        "train.fape.enabled=true",
        "train.fape.start_step=0",
        "train.decoding.eval_enabled=true",
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
    # Should still complete
    assert "Training complete." in result.output

