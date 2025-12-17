"""Integration tests for structure folder evaluation datasets."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pathlib import Path

from stok.cli.cli import cli
from tests.utils.synthetic import random_protein_sequence


# Minimal PDB template for generating test structures
PDB_TEMPLATE = """\
ATOM      1  N   {res1} A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  {res1} A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   {res1} A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   {res2} A   2       1.251   2.520   0.000  1.00  0.00           N
ATOM      5  CA  {res2} A   2       1.700   3.900   0.000  1.00  0.00           C
ATOM      6  C   {res2} A   2       3.200   4.100   0.000  1.00  0.00           C
ATOM      7  N   {res3} A   3       4.000   5.000   0.000  1.00  0.00           N
ATOM      8  CA  {res3} A   3       4.500   6.000   0.000  1.00  0.00           C
ATOM      9  C   {res3} A   3       5.500   6.500   0.000  1.00  0.00           C
ATOM     10  N   {res4} A   4       6.000   7.000   0.000  1.00  0.00           N
ATOM     11  CA  {res4} A   4       6.500   8.000   0.000  1.00  0.00           C
ATOM     12  C   {res4} A   4       7.500   8.500   0.000  1.00  0.00           C
END
"""


def _generate_pdb_content(residues: list[str]) -> str:
    """Generate PDB content with specified residues."""
    return PDB_TEMPLATE.format(
        res1=residues[0] if len(residues) > 0 else "ALA",
        res2=residues[1] if len(residues) > 1 else "GLY",
        res3=residues[2] if len(residues) > 2 else "SER",
        res4=residues[3] if len(residues) > 3 else "VAL",
    )


def _create_structure_folder(tmp_path: Path, n_files: int = 5) -> Path:
    """Create a folder with PDB structure files."""
    folder = tmp_path / "structures"
    folder.mkdir()
    
    residue_sets = [
        ["ALA", "GLY", "SER", "VAL"],
        ["MET", "LYS", "ARG", "ASP"],
        ["LEU", "ILE", "PHE", "TYR"],
        ["PRO", "THR", "CYS", "ASN"],
        ["GLN", "HIS", "TRP", "GLU"],
    ]
    
    for i in range(n_files):
        residues = residue_sets[i % len(residue_sets)]
        pdb_content = _generate_pdb_content(residues)
        (folder / f"protein_{i}.pdb").write_text(pdb_content)
    
    return folder


def _create_train_csv(tmp_path: Path, n_rows: int, seq_len: int, indices_len: int) -> Path:
    """Create a training CSV file with sequences and indices."""
    train_csv = tmp_path / "train.csv"
    rows = []
    for i in range(n_rows):
        seq = random_protein_sequence(seq_len, seq_len)
        indices = " ".join(str(j % 128) for j in range(indices_len))
        rows.append({"pid": f"train_{i}", "protein_sequence": seq, "indices": indices})
    
    df = pd.DataFrame(rows)
    df.to_csv(train_csv, index=False)
    return train_csv


def _create_mlm_train_csv(tmp_path: Path, n_rows: int, seq_len: int) -> Path:
    """Create a training CSV file for MLM (no indices)."""
    train_csv = tmp_path / "train_mlm.csv"
    rows = []
    for i in range(n_rows):
        seq = random_protein_sequence(seq_len, seq_len)
        rows.append({"pid": f"train_{i}", "protein_sequence": seq})
    
    df = pd.DataFrame(rows)
    df.to_csv(train_csv, index=False)
    return train_csv


class TestStructureFolderEvalExplicit:
    """Tests for structure folder eval with explicit format specification."""

    def test_cli_train_with_structure_folder_explicit(self, tmp_path):
        """CLI training with structure folder eval dataset (explicit format)."""
        runner = CliRunner()

        max_len = 16
        indices_len = max_len - 2

        train_csv = _create_train_csv(tmp_path, n_rows=8, seq_len=12, indices_len=indices_len)
        struct_folder = _create_structure_folder(tmp_path, n_files=4)

        overrides = [
            f"data.train={train_csv.as_posix()}",
            # Structure folder eval with explicit format
            f"+data.eval.struct_test.path={struct_folder.as_posix()}",
            "+data.eval.struct_test.format=structure",
            # Tiny model for speed
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            # Small codebook preset
            "model.codebook.preset=lite",
            # Small data loader
            "data.batch_size=2",
            f"data.max_len={max_len}",
            "data.num_workers=0",
            "data.pin_memory=false",
            # Short run with eval
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            # Disable external logging
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, result.output
        assert "Training complete." in result.output


class TestStructureFolderEvalAutoDetect:
    """Tests for structure folder auto-detection."""

    def test_cli_train_with_structure_folder_autodetect(self, tmp_path):
        """Auto-detection of structure folder (no parquet files in dir)."""
        runner = CliRunner()

        max_len = 16
        indices_len = max_len - 2

        train_csv = _create_train_csv(tmp_path, n_rows=8, seq_len=12, indices_len=indices_len)
        struct_folder = _create_structure_folder(tmp_path, n_files=4)

        overrides = [
            f"data.train={train_csv.as_posix()}",
            # Structure folder eval WITHOUT explicit format (should auto-detect)
            f"+data.eval.auto_test.path={struct_folder.as_posix()}",
            # Tiny model
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            f"data.max_len={max_len}",
            "data.num_workers=0",
            "data.pin_memory=false",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, result.output
        assert "Training complete." in result.output


class TestStructureFolderMLM:
    """Tests for structure folder eval with MLM training."""

    def test_mlm_training_with_structure_folder(self, tmp_path):
        """MLM training with structure folder for eval (P@L metric compatible)."""
        runner = CliRunner()

        max_len = 16

        train_csv = _create_mlm_train_csv(tmp_path, n_rows=8, seq_len=12)
        struct_folder = _create_structure_folder(tmp_path, n_files=4)

        overrides = [
            "train.objective=mlm",
            f"data.train={train_csv.as_posix()}",
            # Structure folder eval
            f"+data.eval.struct_eval.path={struct_folder.as_posix()}",
            "+data.eval.struct_eval.format=structure",
            # Tiny model
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            f"data.max_len={max_len}",
            "data.num_workers=0",
            "data.pin_memory=false",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, result.output
        assert "Training objective: mlm" in result.output
        assert "Training complete." in result.output


class TestStructureFolderMetricWhitelist:
    """Tests for per-dataset metric whitelist with structure folders."""

    def test_structure_folder_with_metric_whitelist(self, tmp_path):
        """Per-dataset metric whitelist with structure folder."""
        runner = CliRunner()

        max_len = 16
        indices_len = max_len - 2

        train_csv = _create_train_csv(tmp_path, n_rows=8, seq_len=12, indices_len=indices_len)
        struct_folder = _create_structure_folder(tmp_path, n_files=4)

        overrides = [
            f"data.train={train_csv.as_posix()}",
            # Structure folder eval with metric whitelist
            f"+data.eval.struct_metrics.path={struct_folder.as_posix()}",
            "+data.eval.struct_metrics.format=structure",
            # Enable specific metrics individually (avoid list syntax issues)
            "+data.eval.struct_metrics.metrics.accuracy.enabled=true",
            "+data.eval.struct_metrics.metrics.perplexity.enabled=true",
            # Tiny model
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            f"data.max_len={max_len}",
            "data.num_workers=0",
            "data.pin_memory=false",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, result.output
        assert "Training complete." in result.output


class TestStructureFolderChainId:
    """Tests for chain_id parameter with structure folders."""

    def test_structure_folder_with_chain_id(self, tmp_path):
        """Structure folder eval with specific chain_id."""
        runner = CliRunner()

        max_len = 16
        indices_len = max_len - 2

        train_csv = _create_train_csv(tmp_path, n_rows=8, seq_len=12, indices_len=indices_len)
        struct_folder = _create_structure_folder(tmp_path, n_files=4)

        overrides = [
            f"data.train={train_csv.as_posix()}",
            # Structure folder eval with chain_id
            f"+data.eval.chain_test.path={struct_folder.as_posix()}",
            "+data.eval.chain_test.format=structure",
            "+data.eval.chain_test.chain_id=A",
            # Tiny model
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            f"data.max_len={max_len}",
            "data.num_workers=0",
            "data.pin_memory=false",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, result.output
        assert "Training complete." in result.output
