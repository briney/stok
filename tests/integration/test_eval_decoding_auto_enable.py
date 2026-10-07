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


def _write_parquet_with_coords(
    path: Path, n_rows: int, seq_min_len: int, seq_max_len: int, indices_len: int
):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_rows):
        L = np.random.randint(seq_min_len, seq_max_len + 1)
        seq = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=L))
        rows.append(
            {
                "sequence_id": f"pc{i}",
                "sequence": seq,
                "structure_tokens": [0] * len(seq),
                "coordinates": _make_coords(L),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _make_decoder_ckpt(tmp_path: Path, preset: str = "lite") -> Path:
    from stok.models.decoder import GeometricDecoder

    # match d_code to codebook preset
    codebook = load_codebook(preset=preset)
    d_code = int(codebook.shape[1])
    from stok.models.decoder import _DECODER_ARCH

    arch = _DECODER_ARCH[preset]
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


@pytest.mark.parametrize(
    "activation",
    [
        "auto",
        "whitelist",
        "decoder_only",
        "label_free",
        "label_free_mmcif",
        "metrics_example",
    ],
)
def test_eval_decoding_auto_enables_decoder(tmp_path, monkeypatch, activation):
    from stok.eval import Evaluator
    import importlib

    train_module = importlib.import_module("stok.training.engine")
    original_load = train_module.load_pretrained_decoder
    loaded = []

    def load(**kwargs):
        decoder = original_load(**kwargs)
        loaded.append(decoder)
        return decoder

    monkeypatch.setattr(train_module, "load_pretrained_decoder", load)
    evaluated = []
    original_evaluate = Evaluator.evaluate

    def evaluate(self, *args, **kwargs):
        metrics = original_evaluate(self, *args, **kwargs)
        evaluated.append(metrics)
        return metrics

    monkeypatch.setattr(Evaluator, "evaluate", evaluate)
    from stok.models.decoder import _DECODER_ARCH

    monkeypatch.setitem(
        _DECODER_ARCH,
        "lite",
        dict(
            d_model=32,
            ffn_mult=1.0,
            n_layers=1,
            n_heads=2,
            attn_kv_heads=1,
            num_memory_tokens=0,
            max_length=32,
        ),
    )
    runner = CliRunner()

    max_len = 16
    indices_len = max_len - 2  # align with token positions excluding BOS/EOS

    train_pq = tmp_path / "train.parquet"
    eval_pq = tmp_path / "eval.parquet"
    _write_parquet_with_coords(
        train_pq, n_rows=4, seq_min_len=12, seq_max_len=18, indices_len=indices_len
    )
    _write_parquet_with_coords(
        eval_pq, n_rows=2, seq_min_len=12, seq_max_len=18, indices_len=indices_len
    )

    if activation.startswith("label_free"):
        from tests.integration.test_structure_folder_eval import (
            _create_structure_folder,
        )

        eval_pq = _create_structure_folder(tmp_path, n_files=2)
        if activation == "label_free_mmcif":
            from Bio.PDB import PDBParser, MMCIFIO

            for pdb in eval_pq.glob("*.pdb"):
                writer = MMCIFIO()
                writer.set_structure(PDBParser(QUIET=True).get_structure("test", pdb))
                writer.save(str(pdb.with_suffix(".cif")))
                pdb.unlink()
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
        # do NOT enable model.decoder.enabled; ensure auto-enable path is exercised
        f"model.decoder.path={ckpt_path.as_posix()}",
        # eval decoding only
        "train.fape.enabled=false",
        "train.decoding.eval_enabled=true",
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

    if activation == "metrics_example":
        import re
        import shlex

        readme = (Path(__file__).parents[2] / "README.md").read_text()
        recipe = re.search(
            r"# enable decoder but metrics-only \(no FAPE\)\n(.*?)\n\n", readme, re.S
        ).group(1)
        recipe = recipe.replace("\\\n", " ").replace(
            "/abs/path/eval.parquet", str(eval_pq)
        )
        overrides = [
            x
            for x in overrides
            if not x.startswith(("data.eval=", "train.decoding.eval_enabled="))
        ]
        overrides += shlex.split(recipe)[2:]

    if activation == "whitelist":
        overrides = [x for x in overrides if not x.startswith("data.eval=")]
        overrides += [
            "train.decoding.eval_enabled=false",
            f"+data.eval.val.path={eval_pq}",
            "+data.eval.val.metrics.only=[lddt]",
        ]
    elif activation == "decoder_only":
        overrides += ["train.decoding.eval_enabled=false", "model.decoder.enabled=true"]
    result = runner.invoke(cli, ["train", *overrides])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output

    snapshot = (tmp_path / "configs/run.yaml").read_text()
    assert len(loaded) == 1
    assert evaluated
    if activation == "decoder_only":
        assert "lddt" not in evaluated[0]
    else:
        assert "load_coords: true" in snapshot
        assert 0 < evaluated[0]["lddt"] <= 1

    if activation.startswith("label_free"):
        assert "acc" not in evaluated[0] and "ppl" not in evaluated[0]
