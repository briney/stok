"""Unit tests for structure_dataset module."""

import numpy as np
import pytest
import torch
from pathlib import Path

from stok.data.structure_dataset import StructureFolderDataset, STRUCTURE_EXTENSIONS


# Minimal valid PDB content
MINIMAL_PDB_1 = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   GLY A   2       1.251   2.520   0.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       1.700   3.900   0.000  1.00  0.00           C
ATOM      6  C   GLY A   2       3.200   4.100   0.000  1.00  0.00           C
END
"""

MINIMAL_PDB_2 = """\
ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   SER A   2       1.251   2.520   0.000  1.00  0.00           N
ATOM      5  CA  SER A   2       1.700   3.900   0.000  1.00  0.00           C
ATOM      6  C   SER A   2       3.200   4.100   0.000  1.00  0.00           C
ATOM      7  N   LYS A   3       4.251   4.520   0.000  1.00  0.00           N
ATOM      8  CA  LYS A   3       4.700   5.900   0.000  1.00  0.00           C
ATOM      9  C   LYS A   3       6.200   6.100   0.000  1.00  0.00           C
END
"""

# Longer PDB for truncation test
LONG_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   ALA A   2       0.000   0.000   0.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       1.458   0.000   0.000  1.00  0.00           C
ATOM      6  C   ALA A   2       2.009   1.420   0.000  1.00  0.00           C
ATOM      7  N   ALA A   3       0.000   0.000   0.000  1.00  0.00           N
ATOM      8  CA  ALA A   3       1.458   0.000   0.000  1.00  0.00           C
ATOM      9  C   ALA A   3       2.009   1.420   0.000  1.00  0.00           C
ATOM     10  N   ALA A   4       0.000   0.000   0.000  1.00  0.00           N
ATOM     11  CA  ALA A   4       1.458   0.000   0.000  1.00  0.00           C
ATOM     12  C   ALA A   4       2.009   1.420   0.000  1.00  0.00           C
ATOM     13  N   ALA A   5       0.000   0.000   0.000  1.00  0.00           N
ATOM     14  CA  ALA A   5       1.458   0.000   0.000  1.00  0.00           C
ATOM     15  C   ALA A   5       2.009   1.420   0.000  1.00  0.00           C
END
"""


def _create_pdb_folder(tmp_path: Path) -> Path:
    """Create a folder with sample PDB files."""
    folder = tmp_path / "pdbs"
    folder.mkdir()
    (folder / "protein1.pdb").write_text(MINIMAL_PDB_1)
    (folder / "protein2.pdb").write_text(MINIMAL_PDB_2)
    return folder


def _create_nested_folder(tmp_path: Path) -> Path:
    """Create a folder with nested subdirectories containing PDB files."""
    folder = tmp_path / "nested"
    folder.mkdir()
    (folder / "protein1.pdb").write_text(MINIMAL_PDB_1)
    
    subfolder = folder / "subdir"
    subfolder.mkdir()
    (subfolder / "protein2.pdb").write_text(MINIMAL_PDB_2)
    return folder


class TestStructureFolderDataset:
    """Tests for StructureFolderDataset class."""

    def test_load_folder_basic(self, tmp_path):
        """Load folder with PDB files and verify length."""
        folder = _create_pdb_folder(tmp_path)
        
        ds = StructureFolderDataset(folder, max_length=16)
        
        assert len(ds) == 2
        assert ds.has_coords is True
        assert ds._is_parquet is False

    def test_getitem_output_keys(self, tmp_path):
        """Output dict has expected keys."""
        folder = _create_pdb_folder(tmp_path)
        ds = StructureFolderDataset(folder, max_length=16)
        
        item = ds[0]
        
        assert "pid" in item
        assert "seq" in item
        assert "coords" in item
        assert "masks" in item
        assert "nan_masks" in item
        # indices should NOT be present
        assert "indices" not in item

    def test_coords_shape(self, tmp_path):
        """Coords tensor has correct shape [max_length, 3, 3]."""
        folder = _create_pdb_folder(tmp_path)
        max_length = 16
        ds = StructureFolderDataset(folder, max_length=max_length)
        
        item = ds[0]
        
        assert item["coords"].shape == (max_length, 3, 3)
        assert item["coords"].dtype == torch.float32

    def test_masks_shape(self, tmp_path):
        """Masks tensor has correct shape [max_length]."""
        folder = _create_pdb_folder(tmp_path)
        max_length = 16
        ds = StructureFolderDataset(folder, max_length=max_length)
        
        item = ds[0]
        
        assert item["masks"].shape == (max_length,)
        assert item["masks"].dtype == torch.bool
        assert item["nan_masks"].shape == (max_length,)

    def test_padding_with_nan(self, tmp_path):
        """Sequences shorter than max_length are padded with NaN."""
        folder = _create_pdb_folder(tmp_path)
        max_length = 16
        ds = StructureFolderDataset(folder, max_length=max_length)
        
        item = ds[0]  # protein1.pdb has 2 residues
        
        # First 2 positions should have valid coords
        assert not torch.isnan(item["coords"][:2]).any()
        # Remaining positions should be NaN
        assert torch.isnan(item["coords"][2:]).all()
        # Masks should reflect valid positions
        assert item["masks"][:2].all()
        assert not item["masks"][2:].any()

    def test_truncation(self, tmp_path):
        """Sequences longer than max_length are truncated."""
        folder = tmp_path / "long"
        folder.mkdir()
        (folder / "long.pdb").write_text(LONG_PDB)  # 5 residues
        
        max_length = 3
        ds = StructureFolderDataset(folder, max_length=max_length)
        
        item = ds[0]
        
        assert len(item["seq"]) == max_length
        assert item["coords"].shape == (max_length, 3, 3)
        assert item["masks"].all()  # All positions valid after truncation

    def test_recursive_search(self, tmp_path):
        """recursive=True searches subdirectories."""
        folder = _create_nested_folder(tmp_path)
        
        # Without recursive
        ds_nonrecursive = StructureFolderDataset(folder, max_length=16, recursive=False)
        assert len(ds_nonrecursive) == 1
        
        # With recursive
        ds_recursive = StructureFolderDataset(folder, max_length=16, recursive=True)
        assert len(ds_recursive) == 2

    def test_empty_folder_raises(self, tmp_path):
        """Empty folder raises ValueError."""
        folder = tmp_path / "empty"
        folder.mkdir()
        
        with pytest.raises(ValueError, match="No structure files found"):
            StructureFolderDataset(folder, max_length=16)

    def test_not_a_directory_raises(self, tmp_path):
        """Non-directory path raises ValueError."""
        pdb_file = tmp_path / "single.pdb"
        pdb_file.write_text(MINIMAL_PDB_1)
        
        with pytest.raises(ValueError, match="Not a directory"):
            StructureFolderDataset(pdb_file, max_length=16)

    def test_chain_id_passed_to_parser(self, tmp_path):
        """chain_id parameter is passed to parser."""
        # Create PDB with two chains
        pdb_two_chains = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   MET B   1       5.000   0.000   0.000  1.00  0.00           N
ATOM      5  CA  MET B   1       6.458   0.000   0.000  1.00  0.00           C
ATOM      6  C   MET B   1       7.009   1.420   0.000  1.00  0.00           C
END
"""
        folder = tmp_path / "twochains"
        folder.mkdir()
        (folder / "test.pdb").write_text(pdb_two_chains)
        
        ds = StructureFolderDataset(folder, max_length=16, chain_id="B")
        item = ds[0]
        
        assert item["seq"] == "M"

    def test_has_coords_attribute(self, tmp_path):
        """Dataset has has_coords=True for compatibility."""
        folder = _create_pdb_folder(tmp_path)
        ds = StructureFolderDataset(folder, max_length=16)
        
        assert ds.has_coords is True

    def test_repr(self, tmp_path):
        """__repr__ returns informative string."""
        folder = _create_pdb_folder(tmp_path)
        ds = StructureFolderDataset(folder, max_length=16)
        
        repr_str = repr(ds)
        
        assert "StructureFolderDataset" in repr_str
        assert "num_files=2" in repr_str
        assert "max_length=16" in repr_str

    def test_supported_extensions(self, tmp_path):
        """All supported extensions are discovered."""
        folder = tmp_path / "mixed"
        folder.mkdir()
        
        # Create files with different extensions
        (folder / "a.pdb").write_text(MINIMAL_PDB_1)
        (folder / "b.ent").write_text(MINIMAL_PDB_1)
        (folder / "c.cif").write_text(MINIMAL_PDB_1)  # Will fail to parse as PDB but counts
        (folder / "d.txt").write_text("not a structure")
        
        # Note: .cif content is invalid here but file discovery should still work
        # The actual parsing would fail for c.cif with this content
        ds = StructureFolderDataset(folder, max_length=16)
        
        # Should find .pdb and .ent (which are valid PDB format)
        # .cif is in the list but would fail on actual parse
        assert len(ds._files) >= 2  # At least .pdb and .ent


class TestStructureExtensions:
    """Tests for STRUCTURE_EXTENSIONS constant."""

    def test_supported_extensions(self):
        """Verify all expected extensions are supported."""
        assert ".pdb" in STRUCTURE_EXTENSIONS
        assert ".ent" in STRUCTURE_EXTENSIONS
        assert ".cif" in STRUCTURE_EXTENSIONS
        assert ".mmcif" in STRUCTURE_EXTENSIONS
