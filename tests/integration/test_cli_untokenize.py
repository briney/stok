"""Integration tests for `stok untokenize` (Phase 2).

These tests exercise the tokenize → untokenize round-trip through the CLI,
which is the only path users have to get coordinates in Phase 2 (per-sample
file writing arrives in Phase 3).
"""

from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from stok.cli.cli import cli

from .conftest import joint_hydra_overrides


class TestUntokenizeRoundTrip:
    def test_tokenize_then_untokenize(
        self, tmp_path, joint_checkpoint, patch_load_decoder
    ):
        ckpt_path, codebook_path = joint_checkpoint
        seq_file = tmp_path / "seqs.fasta"
        seq_file.write_text(">seq1\nACDEFGHI\n>seq2\nKLMNPQRS\n")

        tokens_path = tmp_path / "tokens.parquet"
        runner = CliRunner()
        tok_result = runner.invoke(
            cli,
            [
                "tokenize",
                "--checkpoint", str(ckpt_path),
                "--input-seq-file", str(seq_file),
                "--num-steps", "3",
                "--device", "cpu",
                "--output", str(tokens_path),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert tok_result.exit_code == 0, f"tokenize failed:\n{tok_result.output}"
        assert tokens_path.exists()

        out_dir = tmp_path / "untokenize_out"
        untok_result = runner.invoke(
            cli,
            [
                "untokenize",
                "--checkpoint", str(ckpt_path),
                "--input-tokens-file", str(tokens_path),
                "--decoder-preset", "base",
                "--device", "cpu",
                "--output-dir", str(out_dir),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert untok_result.exit_code == 0, (
            f"untokenize failed:\n{untok_result.output}"
        )

        manifest = out_dir / "manifest.parquet"
        assert manifest.exists()
        df = pd.read_parquet(manifest)
        assert len(df) == 2
        assert all(sid.startswith("sample_") for sid in df["sample_id"])


class TestUntokenizeInputValidation:
    def test_missing_struct_tokens_column(self, tmp_path, joint_checkpoint):
        ckpt_path, codebook_path = joint_checkpoint
        bad_parquet = tmp_path / "bad.parquet"
        # Write a parquet with the wrong schema — missing `struct_tokens`.
        pd.DataFrame([{"sample_id": "sample_0001", "sequence": "ACDE"}]).to_parquet(
            bad_parquet, index=False
        )

        out_dir = tmp_path / "untokenize_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "untokenize",
                "--checkpoint", str(ckpt_path),
                "--input-tokens-file", str(bad_parquet),
                "--decoder-preset", "base",
                "--device", "cpu",
                "--output-dir", str(out_dir),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert result.exit_code != 0
        assert "struct_tokens" in result.output.lower()
