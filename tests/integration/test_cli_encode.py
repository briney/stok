"""Integration tests for ``stok encode``.

Covers:
- The CLI command is wired up and parses arguments.
- The ``--weights`` override path builds a ``StructureEncoder`` from a
  locally-synthesized checkpoint and encodes a real CAMEO PDB into a
  Parquet manifest with the expected schema.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

pytest.importorskip("vector_quantize_pytorch")
pytest.importorskip("x_transformers")
pytest.importorskip("graphein")


CAMEO_DIR = Path(__file__).resolve().parent.parent / "test_data" / "cameo"


def _load_conversion_module():
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("convert_gcp_vqvae_weights")


def _build_lite_weights(tmp_path: Path) -> Path:
    """Synthesize a lite-preset checkpoint by reversing the rename rules.

    Equivalent to running the conversion script on a fresh
    (random-init) upstream checkpoint; avoids the 1.7 GB download.
    """
    mod = _load_conversion_module()
    from stok.models.structure_encoder import _ENCODER_ARCH, StructureEncoder

    stok_sd = StructureEncoder(**_ENCODER_ARCH["lite"]).state_dict()
    upstream_sd: dict[str, torch.Tensor] = {}
    for k, v in stok_sd.items():
        if k.startswith("featurizer."):
            continue
        if k.startswith("gcpnet."):
            upstream_sd["encoder." + k[len("gcpnet.") :]] = v
        elif k.startswith(
            ("encoder_tail.", "encoder_blocks.", "encoder_head.", "vector_quantizer.")
        ):
            upstream_sd["vqvae." + k] = v
        else:
            upstream_sd[k] = v

    remapped = mod.remap_state_dict(upstream_sd)
    ckpt_path = tmp_path / "encoder-lite.pt"
    torch.save(remapped, ckpt_path)
    return ckpt_path


class TestEncodeCli:
    def test_encode_help_registers(self):
        from stok.cli.cli import cli

        result = CliRunner().invoke(cli, ["encode", "--help"])
        assert result.exit_code == 0
        assert "structure token" in result.output.lower()
        assert "--preset" in result.output

    def test_encode_with_local_weights_writes_manifest(self, tmp_path: Path):
        from stok.cli.cli import cli

        ckpt_path = _build_lite_weights(tmp_path)

        pdb_paths = sorted(CAMEO_DIR.glob("*.pdb"))[:2]
        if not pdb_paths:
            pytest.skip("No CAMEO PDB fixtures available")

        out_path = tmp_path / "encoded.parquet"
        runner = CliRunner()
        args = [
            "encode",
            "--preset", "lite",
            "--weights", str(ckpt_path),
            "--batch-size", "2",
            "--device", "cpu",
            "--output", str(out_path),
        ]
        for p in pdb_paths:
            args.extend(["--input", str(p)])

        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
        assert out_path.exists(), "manifest not written"

        import pandas as pd

        df = pd.read_parquet(out_path)
        assert set(df.columns) == {
            "sample_id",
            "sequence",
            "struct_tokens",
            "length",
        }
        assert len(df) == len(pdb_paths)
        for i, row in df.iterrows():
            assert len(row["struct_tokens"]) == row["length"]
            assert row["length"] == len(row["sequence"])
            # Indices must be within codebook range.
            assert all(0 <= int(t) < 4096 for t in row["struct_tokens"])
