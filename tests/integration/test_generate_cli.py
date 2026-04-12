"""Integration tests for the stok generate CLI subcommand (Phase 4.1)."""

import pytest
import torch

from click.testing import CliRunner
from stok.cli.cli import cli
from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule


CODEBOOK_SIZE = 16
CODEBOOK_DIM = 8


def _build_seq_model() -> MDLMModel:
    ns = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="seq_only",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns,
        time_conditioning="adaln",
    )


def _build_joint_model(codebook: torch.Tensor | None = None) -> MDLMModel:
    if codebook is None:
        codebook = torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM)
    ns_seq = NoiseSchedule(schedule_type="cosine")
    ns_struct = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="joint",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        codebook=codebook,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns_seq,
        noise_schedule_struct=ns_struct,
        time_conditioning="adaln",
    )


def _save_model_checkpoint(model, path, codebook=None):
    torch.save(model.state_dict(), path)
    if codebook is not None:
        # Save codebook alongside checkpoint for config override
        codebook_path = path.parent / "codebook.pt"
        torch.save(codebook, codebook_path)


class TestGenerateCLISeqOnly:
    """Test generate command with a seq_only model."""

    def test_codesign_generates_output(self, tmp_path):
        model = _build_seq_model()
        ckpt_path = tmp_path / "seq_model.pt"
        _save_model_checkpoint(model, ckpt_path)

        output_path = tmp_path / "output.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate",
                "--checkpoint", str(ckpt_path),
                "--mode", "codesign",
                "--length", "16",
                "--num-samples", "3",
                "--num-steps", "5",
                "--output", str(output_path),
                "train.objective=mdlm",
                "train.mdlm.tracks=seq_only",
                "model.encoder.d_model=64",
                "model.encoder.n_layers=2",
                "model.encoder.n_heads=4",
                "model.encoder.ffn_mult=2.0",
                "model.encoder.dropout=0.0",
                "model.encoder.attn_dropout=0.0",
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert output_path.exists()

        import pandas as pd

        df = pd.read_parquet(output_path)
        assert len(df) == 3
        assert "pid" in df.columns
        assert "sequence" in df.columns
        assert "seq_tokens" in df.columns

    def test_output_sequences_are_valid(self, tmp_path):
        model = _build_seq_model()
        ckpt_path = tmp_path / "seq_model.pt"
        _save_model_checkpoint(model, ckpt_path)

        output_path = tmp_path / "output.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate",
                "--checkpoint", str(ckpt_path),
                "--length", "12",
                "--num-samples", "2",
                "--num-steps", "3",
                "--output", str(output_path),
                "train.objective=mdlm",
                "train.mdlm.tracks=seq_only",
                "model.encoder.d_model=64",
                "model.encoder.n_layers=2",
                "model.encoder.n_heads=4",
                "model.encoder.ffn_mult=2.0",
                "model.encoder.dropout=0.0",
                "model.encoder.attn_dropout=0.0",
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        import pandas as pd

        df = pd.read_parquet(output_path)
        # Sequences should be non-empty strings of amino acids
        for seq in df["sequence"]:
            assert isinstance(seq, str)
            assert len(seq) > 0


class TestGenerateCLIJoint:
    """Test generate command with a joint model."""

    def test_codesign_generates_both_tracks(self, tmp_path):
        codebook = torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM)
        model = _build_joint_model(codebook=codebook)
        ckpt_path = tmp_path / "joint_model.pt"
        _save_model_checkpoint(model, ckpt_path, codebook=codebook)

        output_path = tmp_path / "output.parquet"
        codebook_path = tmp_path / "codebook.pt"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate",
                "--checkpoint", str(ckpt_path),
                "--mode", "codesign",
                "--length", "12",
                "--num-samples", "3",
                "--num-steps", "5",
                "--output", str(output_path),
                "train.objective=mdlm",
                "train.mdlm.tracks=joint",
                "model.encoder.d_model=64",
                "model.encoder.n_layers=2",
                "model.encoder.n_heads=4",
                "model.encoder.ffn_mult=2.0",
                "model.encoder.dropout=0.0",
                "model.encoder.attn_dropout=0.0",
                f"model.codebook.path={codebook_path}",
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        import pandas as pd

        df = pd.read_parquet(output_path)
        assert len(df) == 3
        assert "struct_tokens" in df.columns


class TestGenerateCLIConditioning:
    """Test generate with conditioning files."""

    def test_forward_mode_with_seq_file(self, tmp_path):
        codebook = torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM)
        model = _build_joint_model(codebook=codebook)
        ckpt_path = tmp_path / "joint_model.pt"
        _save_model_checkpoint(model, ckpt_path, codebook=codebook)

        # Write conditioning sequences
        seq_file = tmp_path / "seqs.fasta"
        seq_file.write_text(">seq1\nACDEFGHI\n>seq2\nKLMNPQRS\n")

        output_path = tmp_path / "output.parquet"
        codebook_path = tmp_path / "codebook.pt"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate",
                "--checkpoint", str(ckpt_path),
                "--mode", "forward",
                "--length", "10",
                "--num-steps", "3",
                "--condition-seq-file", str(seq_file),
                "--output", str(output_path),
                "train.objective=mdlm",
                "train.mdlm.tracks=joint",
                "model.encoder.d_model=64",
                "model.encoder.n_layers=2",
                "model.encoder.n_heads=4",
                "model.encoder.ffn_mult=2.0",
                "model.encoder.dropout=0.0",
                "model.encoder.attn_dropout=0.0",
                f"model.codebook.path={codebook_path}",
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        import pandas as pd

        df = pd.read_parquet(output_path)
        assert len(df) == 2  # Two conditioning sequences
        assert "struct_tokens" in df.columns

    def test_inverse_mode_with_struct_file(self, tmp_path):
        codebook = torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM)
        model = _build_joint_model(codebook=codebook)
        ckpt_path = tmp_path / "joint_model.pt"
        _save_model_checkpoint(model, ckpt_path, codebook=codebook)

        # Write conditioning structure indices (space-separated)
        struct_file = tmp_path / "structs.txt"
        struct_file.write_text("0 1 2 3 4 5 6 7\n3 5 7 9 11 13 15 1\n")

        output_path = tmp_path / "output.parquet"
        codebook_path = tmp_path / "codebook.pt"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate",
                "--checkpoint", str(ckpt_path),
                "--mode", "inverse",
                "--length", "10",
                "--num-steps", "3",
                "--condition-struct-file", str(struct_file),
                "--output", str(output_path),
                "train.objective=mdlm",
                "train.mdlm.tracks=joint",
                "model.encoder.d_model=64",
                "model.encoder.n_layers=2",
                "model.encoder.n_heads=4",
                "model.encoder.ffn_mult=2.0",
                "model.encoder.dropout=0.0",
                "model.encoder.attn_dropout=0.0",
                f"model.codebook.path={codebook_path}",
            ],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        import pandas as pd

        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert "sequence" in df.columns
