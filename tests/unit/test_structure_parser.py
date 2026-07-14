"""Unit tests for structure_parser module."""

import numpy as np
import pytest

from stok.utils.structure_parser import (
    AA3TO1,
    StructureData,
    _get_one_letter_code,
    parse_structure,
)

# Minimal valid PDB content with two residues (ALA, GLY)
MINIMAL_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   GLY A   2       1.251   2.520   0.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       1.700   3.900   0.000  1.00  0.00           C
ATOM      6  C   GLY A   2       3.200   4.100   0.000  1.00  0.00           C
END
"""

# PDB with missing CA atom in second residue
PDB_MISSING_CA = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   GLY A   2       1.251   2.520   0.000  1.00  0.00           N
ATOM      5  C   GLY A   2       3.200   4.100   0.000  1.00  0.00           C
END
"""

# PDB with two chains
PDB_TWO_CHAINS = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   MET B   1       5.000   0.000   0.000  1.00  0.00           N
ATOM      5  CA  MET B   1       6.458   0.000   0.000  1.00  0.00           C
ATOM      6  C   MET B   1       7.009   1.420   0.000  1.00  0.00           C
ATOM      7  N   GLY B   2       6.251   2.520   0.000  1.00  0.00           N
ATOM      8  CA  GLY B   2       6.700   3.900   0.000  1.00  0.00           C
ATOM      9  C   GLY B   2       8.200   4.100   0.000  1.00  0.00           C
END
"""

# Minimal valid mmCIF content (with all required fields for Biopython parser)
MINIMAL_CIF = """\
data_test
_entry.id test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 0.000 0.000 0.000 1.00 0.00 ? 1 ALA A N 1
ATOM 2 C CA . ALA A 1 1 ? 1.458 0.000 0.000 1.00 0.00 ? 1 ALA A CA 1
ATOM 3 C C . ALA A 1 1 ? 2.009 1.420 0.000 1.00 0.00 ? 1 ALA A C 1
ATOM 4 N N . SER A 1 2 ? 1.251 2.520 0.000 1.00 0.00 ? 2 SER A N 1
ATOM 5 C CA . SER A 1 2 ? 1.700 3.900 0.000 1.00 0.00 ? 2 SER A CA 1
ATOM 6 C C . SER A 1 2 ? 3.200 4.100 0.000 1.00 0.00 ? 2 SER A C 1
#
"""

# PDB with non-standard amino acid (MSE = selenomethionine)
PDB_NONSTANDARD = """\
ATOM      1  N   MSE A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MSE A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   MSE A   1       2.009   1.420   0.000  1.00  0.00           C
END
"""


class TestParseStructure:
    """Tests for parse_structure function."""

    def test_parse_valid_pdb(self, tmp_path):
        """Parse a valid PDB file and verify sequence and coords."""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(MINIMAL_PDB)

        result = parse_structure(pdb_file)

        assert isinstance(result, StructureData)
        assert result.pid == "test"
        assert result.protein_sequence == "AG"
        assert result.coords.shape == (2, 3, 3)
        assert result.chain_id == "A"
        # Verify first residue N atom coords
        assert np.allclose(result.coords[0, 0], [0.0, 0.0, 0.0])
        # Verify first residue CA atom coords
        assert np.allclose(result.coords[0, 1], [1.458, 0.0, 0.0])

    def test_parse_valid_mmcif(self, tmp_path):
        """Parse a valid mmCIF file and verify sequence and coords."""
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(MINIMAL_CIF)

        result = parse_structure(cif_file)

        assert isinstance(result, StructureData)
        assert result.protein_sequence == "AS"
        assert result.coords.shape == (2, 3, 3)

    def test_missing_backbone_atoms_nonstrict(self, tmp_path):
        """Missing backbone atoms fill with NaN when strict=False."""
        pdb_file = tmp_path / "missing.pdb"
        pdb_file.write_text(PDB_MISSING_CA)

        result = parse_structure(pdb_file, strict=False)

        assert result.protein_sequence == "AG"
        assert result.coords.shape == (2, 3, 3)
        # First residue should be fine
        assert not np.isnan(result.coords[0]).any()
        # Second residue should have NaN for all atoms (missing CA)
        assert np.isnan(result.coords[1]).all()

    def test_missing_backbone_atoms_strict(self, tmp_path):
        """Missing backbone atoms raise ValueError when strict=True."""
        pdb_file = tmp_path / "missing.pdb"
        pdb_file.write_text(PDB_MISSING_CA)

        with pytest.raises(ValueError, match="Missing backbone atom"):
            parse_structure(pdb_file, strict=True)

    def test_chain_selection(self, tmp_path):
        """Select specific chain by chain_id."""
        pdb_file = tmp_path / "twochains.pdb"
        pdb_file.write_text(PDB_TWO_CHAINS)

        # Select chain B
        result = parse_structure(pdb_file, chain_id="B")

        assert result.protein_sequence == "MG"
        assert result.chain_id == "B"
        assert result.coords.shape == (2, 3, 3)

    def test_chain_selection_invalid(self, tmp_path):
        """Invalid chain_id raises ValueError."""
        pdb_file = tmp_path / "twochains.pdb"
        pdb_file.write_text(PDB_TWO_CHAINS)

        with pytest.raises(ValueError, match="Chain 'X' not found"):
            parse_structure(pdb_file, chain_id="X")

    def test_first_chain_fallback(self, tmp_path):
        """Uses first polymer chain when chain_id=None."""
        pdb_file = tmp_path / "twochains.pdb"
        pdb_file.write_text(PDB_TWO_CHAINS)

        result = parse_structure(pdb_file, chain_id=None)

        # Should get chain A (first)
        assert result.chain_id == "A"
        assert result.protein_sequence == "A"

    def test_nonstandard_amino_acid(self, tmp_path):
        """Non-standard amino acids are mapped correctly."""
        pdb_file = tmp_path / "nonstandard.pdb"
        pdb_file.write_text(PDB_NONSTANDARD)

        result = parse_structure(pdb_file)

        # MSE (selenomethionine) should map to M
        assert result.protein_sequence == "M"

    def test_file_not_found(self, tmp_path):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_structure(tmp_path / "nonexistent.pdb")

    def test_empty_structure(self, tmp_path):
        """Structure with no amino acids raises ValueError."""
        pdb_file = tmp_path / "empty.pdb"
        pdb_file.write_text("END\n")

        with pytest.raises(ValueError, match="(No protein chain found|No models found)"):
            parse_structure(pdb_file)


class TestAA3TO1Mapping:
    """Tests for amino acid mapping dictionary."""

    def test_standard_amino_acids(self):
        """All 20 standard amino acids are mapped."""
        standard = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }
        for three, one in standard.items():
            assert AA3TO1.get(three) == one

    def test_nonstandard_amino_acids(self):
        """Common non-standard amino acids are mapped."""
        assert AA3TO1.get("MSE") == "M"  # Selenomethionine
        assert AA3TO1.get("UNK") == "X"  # Unknown


def test_unknown_residue_does_not_import_removed_biopython_api():
    assert _get_one_letter_code("FME") == "X"
