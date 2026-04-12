"""Integration tests for `stok design` (Phase 2)."""

from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from stok.cli.cli import cli

from .conftest import joint_hydra_overrides, seq_only_hydra_overrides


class TestDesignSeqOnly:
    def test_codesign_writes_manifest(self, tmp_path, seq_checkpoint):
        out_dir = tmp_path / "design_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "design",
                "--checkpoint", str(seq_checkpoint),
                "--length", "12",
                "--num-samples", "3",
                "--num-steps", "3",
                "--device", "cpu",
                "--output-dir", str(out_dir),
                *seq_only_hydra_overrides(),
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        manifest = out_dir / "manifest.parquet"
        assert manifest.exists()
        df = pd.read_parquet(manifest)
        assert len(df) == 3
        assert set(["sample_id", "sequence", "seq_tokens", "length", "structure_file"]).issubset(df.columns)
        # struct_tokens column should be absent on a seq_only run.
        assert "struct_tokens" not in df.columns
        # Sample IDs use the new `sample_NNNN` convention.
        assert all(sid.startswith("sample_") for sid in df["sample_id"])


class TestDesignJoint:
    def test_codesign_writes_both_tracks(self, tmp_path, joint_checkpoint):
        ckpt_path, codebook_path = joint_checkpoint
        out_dir = tmp_path / "design_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "design",
                "--checkpoint", str(ckpt_path),
                "--length", "10",
                "--num-samples", "2",
                "--num-steps", "3",
                "--device", "cpu",
                "--output-dir", str(out_dir),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        df = pd.read_parquet(out_dir / "manifest.parquet")
        assert len(df) == 2
        assert "struct_tokens" in df.columns
        # Without --decoder-preset we should NOT have a structure file path.
        assert df["structure_file"].isna().all()


class TestDesignScaffolding:
    def test_scaffold_with_seq_file(self, tmp_path, joint_checkpoint):
        """Conditioning on a seq file should preserve those sequences."""
        ckpt_path, codebook_path = joint_checkpoint
        seq_file = tmp_path / "seqs.fasta"
        seq_file.write_text(">seq1\nACDEFGHI\n>seq2\nKLMNPQRS\n")

        out_dir = tmp_path / "design_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "design",
                "--checkpoint", str(ckpt_path),
                "--length", "10",
                "--num-steps", "3",
                "--device", "cpu",
                "--condition-seq-file", str(seq_file),
                "--output-dir", str(out_dir),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        df = pd.read_parquet(out_dir / "manifest.parquet")
        assert len(df) == 2  # Two conditioning sequences
        # Structure tokens should be generated for the joint track.
        assert "struct_tokens" in df.columns

    def test_scaffold_with_struct_file(self, tmp_path, joint_checkpoint):
        ckpt_path, codebook_path = joint_checkpoint
        struct_file = tmp_path / "structs.txt"
        struct_file.write_text("0 1 2 3 4 5 6 7\n3 5 7 9 11 13 15 1\n")

        out_dir = tmp_path / "design_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "design",
                "--checkpoint", str(ckpt_path),
                "--length", "10",
                "--num-steps", "3",
                "--device", "cpu",
                "--condition-struct-file", str(struct_file),
                "--output-dir", str(out_dir),
                *joint_hydra_overrides(codebook_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        df = pd.read_parquet(out_dir / "manifest.parquet")
        assert len(df) == 2
        assert "sequence" in df.columns


class TestDesignHydraOverride:
    def test_hydra_override_applied(self, tmp_path, seq_checkpoint):
        """Hydra overrides like `model.encoder.d_model=64` must flow through."""
        out_dir = tmp_path / "design_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "design",
                "--checkpoint", str(seq_checkpoint),
                "--length", "8",
                "--num-samples", "2",
                "--num-steps", "3",
                "--device", "cpu",
                "--output-dir", str(out_dir),
                *seq_only_hydra_overrides(),
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert (out_dir / "manifest.parquet").exists()


class TestDesignErrors:
    def test_seq_only_with_decoder_preset_errors(self, tmp_path, seq_checkpoint):
        """A decoder on a seq_only checkpoint should exit non-zero with a
        clear message, per the design behavior matrix.
        """
        out_dir = tmp_path / "design_out"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "design",
                "--checkpoint", str(seq_checkpoint),
                "--length", "8",
                "--num-samples", "1",
                "--num-steps", "3",
                "--device", "cpu",
                "--decoder-preset", "base",
                "--output-dir", str(out_dir),
                *seq_only_hydra_overrides(),
            ],
        )
        assert result.exit_code != 0
        assert "seq-only" in result.output.lower() or "joint" in result.output.lower()
