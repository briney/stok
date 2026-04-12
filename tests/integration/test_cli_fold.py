"""Integration tests for `stok fold` (Phase 2)."""

from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from stok.cli.cli import cli

from .conftest import joint_hydra_overrides, seq_only_hydra_overrides


class TestFoldJoint:
    def test_fold_writes_manifest(
        self, tmp_path, joint_checkpoint, patch_load_decoder
    ):
        ckpt_path, codebook_path = joint_checkpoint
        seq_file = tmp_path / "seqs.fasta"
        seq_file.write_text(">seq1\nACDEFGHI\n>seq2\nKLMNPQRS\n")

        out_dir = tmp_path / "fold_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "fold",
                "--checkpoint", str(ckpt_path),
                "--input-seq-file", str(seq_file),
                "--num-steps", "3",
                "--device", "cpu",
                "--output-dir", str(out_dir),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        manifest = out_dir / "manifest.parquet"
        assert manifest.exists()
        df = pd.read_parquet(manifest)
        assert len(df) == 2
        assert "struct_tokens" in df.columns
        assert all(sid.startswith("sample_") for sid in df["sample_id"])


class TestFoldErrors:
    def test_seq_only_raises_joint_error(self, tmp_path, seq_checkpoint):
        """fold requires a joint model; seq_only should fail with a
        'joint'-mentioning message.
        """
        seq_file = tmp_path / "seqs.fasta"
        seq_file.write_text(">seq1\nACDE\n")

        out_dir = tmp_path / "fold_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "fold",
                "--checkpoint", str(seq_checkpoint),
                "--input-seq-file", str(seq_file),
                "--num-steps", "3",
                "--device", "cpu",
                "--output-dir", str(out_dir),
                *seq_only_hydra_overrides(),
            ],
        )
        assert result.exit_code != 0
        assert "joint" in result.output.lower()

    def test_empty_input_fails_fast(
        self, tmp_path, joint_checkpoint, patch_load_decoder
    ):
        ckpt_path, codebook_path = joint_checkpoint
        seq_file = tmp_path / "empty.fasta"
        seq_file.write_text("")

        out_dir = tmp_path / "fold_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "fold",
                "--checkpoint", str(ckpt_path),
                "--input-seq-file", str(seq_file),
                "--num-steps", "3",
                "--device", "cpu",
                "--output-dir", str(out_dir),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert result.exit_code != 0
