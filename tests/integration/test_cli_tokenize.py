"""Integration tests for `stok tokenize` (Phase 2)."""

from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from stok.cli.cli import cli

from .conftest import joint_hydra_overrides, seq_only_hydra_overrides


class TestTokenizeJoint:
    def test_tokenize_writes_parquet(self, tmp_path, joint_checkpoint):
        ckpt_path, codebook_path = joint_checkpoint
        seq_file = tmp_path / "seqs.fasta"
        seq_file.write_text(">seq1\nACDEFGHI\n>seq2\nKLMNPQRS\n")

        output = tmp_path / "tokens.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tokenize",
                "--checkpoint", str(ckpt_path),
                "--input-seq-file", str(seq_file),
                "--num-steps", "3",
                "--device", "cpu",
                "--output", str(output),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        assert output.exists()
        df = pd.read_parquet(output)
        assert len(df) == 2
        # Manifest schema: sample_id, sequence, seq_tokens, struct_tokens, length, structure_file.
        expected = {
            "sample_id",
            "sequence",
            "seq_tokens",
            "struct_tokens",
            "length",
            "structure_file",
        }
        assert expected.issubset(set(df.columns))
        # structure_file should be null (no coordinates from tokenize).
        assert df["structure_file"].isna().all()
        assert all(sid.startswith("sample_") for sid in df["sample_id"])


class TestTokenizeErrors:
    def test_seq_only_raises(self, tmp_path, seq_checkpoint):
        seq_file = tmp_path / "seqs.fasta"
        seq_file.write_text(">seq1\nACDE\n")

        output = tmp_path / "tokens.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tokenize",
                "--checkpoint", str(seq_checkpoint),
                "--input-seq-file", str(seq_file),
                "--num-steps", "3",
                "--device", "cpu",
                "--output", str(output),
                *seq_only_hydra_overrides(),
            ],
        )
        assert result.exit_code != 0
        assert "joint" in result.output.lower()
