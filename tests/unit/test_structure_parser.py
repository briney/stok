"""Unit tests for structure_parser module."""

import numpy as np
import pytest
from pathlib import Path

from stok.utils.structure_parser import parse_structure, StructureData, AA3TO1


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

        with pytest.raises(
            ValueError, match="(No protein chain found|No models found)"
        ):
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

    def test_unknown_residue_code(self):
        from stok.utils.structure_parser import _get_one_letter_code

        assert _get_one_letter_code("ZZZ") == "X"


POLYMER_FIXTURES = Path(__file__).parents[1] / "test_data/gcp_vqvae/polymer"


@pytest.mark.parametrize("suffix", ["pdb", "cif"])
def test_polymer_correspondence_retains_unresolved_positions_and_atom_observations(
    suffix,
):
    from stok.utils.structure_parser import parse_polymer_structure

    result = parse_polymer_structure(
        POLYMER_FIXTURES / f"mapped.{suffix}", chain_id="A"
    )
    assert result.sequence == "MAGSK"
    assert result.coordinates.shape == (5, 4, 3)
    assert result.atom_mask.tolist() == [
        [False] * 4,
        [True, True, True, False],
        [False] * 4,
        [True, True, True, False],
        [False] * 4,
    ]
    assert np.isnan(result.coordinates[[0, 2, 4]]).all()
    assert [row["polymer_position"] for row in result.residue_map] == list(range(5))
    assert [row["monomer_id"] for row in result.residue_map] == [
        "MET",
        "ALA",
        "GLY",
        "SER",
        "LYS",
    ]
    assert [row["observed_monomer_id"] for row in result.residue_map] == [
        None,
        "ALA",
        None,
        "SER",
        None,
    ]
    assert result.residue_map[1]["author_residue_id"] == -5
    assert result.residue_map[3]["author_residue_id"] == 100
    assert result.residue_map[3]["insertion_code"] == "A"
    assert result.source["sequence_source"] == (
        "seqres" if suffix == "pdb" else "entity_poly_seq"
    )
    assert (
        not result.coordinates.flags.writeable and not result.atom_mask.flags.writeable
    )
    if suffix == "cif":
        assert result.source["label_chain_id"] == "L"
        assert result.residue_map[3]["label_seq_id"] == 4


def test_polymer_metadata_required_or_explicit_observed_fallback(tmp_path):
    from stok.utils.structure_parser import parse_polymer_structure

    path = tmp_path / "observed.pdb"
    path.write_text(MINIMAL_PDB)
    with pytest.raises(ValueError, match="sequence_metadata_missing"):
        parse_polymer_structure(path)
    result = parse_polymer_structure(path, allow_observed_sequence=True)
    assert result.sequence == "AG" and result.source["sequence_source"] == "observed"
    supplied = parse_polymer_structure(path, sequence="MAGSK")
    assert supplied.sequence == "MAGSK"
    assert supplied.residue_map[0]["monomer_id"] is None
    assert [row["observed_monomer_id"] for row in supplied.residue_map] == [
        None,
        "ALA",
        "GLY",
        None,
        None,
    ]


def test_polymer_repeated_sequence_alignment_and_disagreement_are_rejected(tmp_path):
    from stok.utils.structure_parser import parse_polymer_structure

    path = tmp_path / "ambiguous.pdb"
    path.write_text(
        "SEQRES   1 A    3  ALA ALA ALA\n"
        + MINIMAL_PDB.split("ATOM      4")[0]
        + "END\n"
    )
    with pytest.raises(ValueError, match="mapping_ambiguous"):
        parse_polymer_structure(path)
    with pytest.raises(ValueError, match="sequence_conflict"):
        parse_polymer_structure(POLYMER_FIXTURES / "mapped.pdb", sequence="MAGTK")


def test_polymer_modified_monomer_and_multiple_models_preserve_identity(tmp_path):
    from stok.utils.structure_parser import parse_polymer_structure

    path = tmp_path / "modified.pdb"
    atoms = PDB_NONSTANDARD.removesuffix("END\n")
    path.write_text(
        "SEQRES   1 A    1  MSE\nMODEL        1\n"
        + atoms
        + "ENDMDL\nMODEL        2\n"
        + atoms.replace("1.458", "9.458")
        + "ENDMDL\nEND\n"
    )
    first = parse_polymer_structure(path)
    second = parse_polymer_structure(path, model_index=1)
    assert first.sequence == second.sequence == "M"
    assert first.residue_map[0]["monomer_id"] == "MSE"
    assert first.coordinates[0, 1, 0] == pytest.approx(1.458)
    assert second.coordinates[0, 1, 0] == pytest.approx(9.458)
    with pytest.raises(ValueError, match="model_not_found"):
        parse_polymer_structure(path, model_index=2)


def test_polymer_requires_explicit_chain_and_namespace(tmp_path):
    from stok.utils.structure_parser import parse_polymer_structure

    path = tmp_path / "chains.pdb"
    path.write_text(PDB_TWO_CHAINS)
    with pytest.raises(ValueError, match="chain_ambiguous"):
        parse_polymer_structure(path, allow_observed_sequence=True)
    result = parse_polymer_structure(
        POLYMER_FIXTURES / "mapped.cif", chain_id="L", chain_namespace="label"
    )
    assert result.source["author_chain_id"] == "A"
    with pytest.raises(ValueError, match="chain_not_found"):
        parse_polymer_structure(
            POLYMER_FIXTURES / "mapped.cif", chain_id="A", chain_namespace="label"
        )


def test_polymer_conformer_selection_uses_one_occupancy_ranked_altloc(tmp_path):
    from stok.utils.structure_parser import parse_polymer_structure

    path = tmp_path / "altloc.pdb"
    n, ca, c = MINIMAL_PDB.splitlines()[:3]
    ca_a = ca[:16] + "A" + ca[17:54] + "  0.60" + ca[60:]
    ca_b = ca[:16] + "B" + ca[17:30] + "   9.458" + ca[38:54] + "  0.40" + ca[60:]
    c_a = c[:16] + "A" + c[17:54] + "  0.10" + c[60:]
    c_b = c[:16] + "B" + c[17:30] + "   9.009" + c[38:54] + "  0.90" + c[60:]
    path.write_text(
        "SEQRES   1 A    1  ALA\n" + "\n".join([n, ca_a, ca_b, c_a, c_b, "END"]) + "\n"
    )
    result = parse_polymer_structure(path)
    assert result.residue_map[0]["selected_altloc"] == "B"
    assert result.coordinates[0, 1, 0] == pytest.approx(9.458)
    assert result.coordinates[0, 2, 0] == pytest.approx(9.009)
    assert result.atom_mask[0].tolist() == [True, True, True, False]
    path.write_text(
        path.read_text().replace("  0.10", "  0.40").replace("  0.90", "  0.60")
    )
    tied = parse_polymer_structure(path)
    assert tied.residue_map[0]["selected_altloc"] == "A"
    assert tied.coordinates[0, 2, 0] == pytest.approx(2.009)


def test_mmcif_author_label_collisions_and_unspecified_chain_are_rejected(tmp_path):
    from stok.utils.structure_parser import parse_polymer_structure

    text = (
        (POLYMER_FIXTURES / "mapped.cif")
        .read_text()
        .replace("L 1\nloop_\n_entity_poly_seq", "L 1\nA 1\nloop_\n_entity_poly_seq")
    )
    extra = []
    for line in text.splitlines():
        if line.startswith("ATOM"):
            columns = line.split()
            columns[1] = str(int(columns[1]) + 100)
            columns[6] = "A"
            columns[17] = "L"
            columns[10] = str(float(columns[10]) + 20)
            extra.append(" ".join(columns))
    path = tmp_path / "collisions.cif"
    path.write_text(text.removesuffix("#\n") + "\n".join(extra) + "\n#\n")
    author = parse_polymer_structure(path, chain_id="A", chain_namespace="author")
    label = parse_polymer_structure(path, chain_id="A", chain_namespace="label")
    assert author.source["label_chain_id"] == "L"
    assert label.source["author_chain_id"] == "L"
    assert label.coordinates[1, 0, 0] - author.coordinates[1, 0, 0] == 20
    with pytest.raises(ValueError, match="chain_ambiguous"):
        parse_polymer_structure(path)
    path.write_text(
        path.read_text()
        .replace(" L N 1", " A N 1")
        .replace(" L CA 1", " A CA 1")
        .replace(" L C 1", " A C 1")
    )
    with pytest.raises(ValueError, match="chain_ambiguous"):
        parse_polymer_structure(path, chain_id="A", chain_namespace="author")


def test_mmcif_scheme_can_supply_polymer_without_entity_sequence(tmp_path):
    from stok.utils.structure_parser import parse_polymer_structure

    text = (POLYMER_FIXTURES / "mapped.cif").read_text()
    start = text.index("loop_\n_entity_poly_seq.")
    end = text.index("loop_\n_pdbx_poly_seq_scheme.")
    path = tmp_path / "scheme.cif"
    path.write_text(text[:start] + text[end:])
    result = parse_polymer_structure(path, chain_id="A")
    assert result.sequence == "MAGSK"
    assert result.source["sequence_source"] == "poly_seq_scheme"
    assert result.atom_mask[:, 0].tolist() == [False, True, False, True, False]


@pytest.mark.parametrize(
    "fault", ["monomer", "scheme", "missing_label", "duplicate_author", "unknown_model"]
)
def test_mmcif_inconsistent_polymer_or_coordinate_identity_fails_closed(
    tmp_path, fault
):
    from stok.utils.structure_parser import (
        StructureMappingError,
        parse_polymer_structure,
    )

    text = (POLYMER_FIXTURES / "mapped.cif").read_text()
    if fault == "monomer":
        text = text.replace("1 2 ALA n", "1 2 ALA n\n1 2 GLY y")
    elif fault == "scheme":
        text = text.replace("L 1 2 ALA -5", "L 1 2 GLY -5")
    elif fault == "missing_label":
        text = text.replace("ALA L 1 2 ?", "ALA L 1 ? ?")
    elif fault == "duplicate_author":
        text = (
            text.replace("L 1 4 SER 100 A", "L 1 4 SER -5 .")
            .replace("SER L 1 4 A", "SER L 1 4 ?")
            .replace("100 SER A", "-5 SER A")
        )
    path = tmp_path / "invalid.cif"
    path.write_text(text)
    with pytest.raises(StructureMappingError):
        parse_polymer_structure(
            path, chain_id="A", model_index=1 if fault == "unknown_model" else 0
        )


def test_deposited_parent_mapping_and_antibody_insertion_ids(tmp_path):
    from stok.utils.structure_parser import parse_polymer_structure

    path = tmp_path / "parents.cif"
    text = (POLYMER_FIXTURES / "mapped.cif").read_text().replace("LYS", "ZZZ")
    path.write_text(
        text.replace(
            "data_polymer\n",
            "data_polymer\nloop_\n_chem_comp.id\n_chem_comp.mon_nstd_parent_comp_id\nZZZ LYS\n",
        )
    )
    result = parse_polymer_structure(path, chain_id="A")
    assert result.sequence == "MAGSK" and result.residue_map[4]["monomer_id"] == "ZZZ"
    path = tmp_path / "parents.pdb"
    path.write_text(
        "SEQRES   1 A    1  ZZZ\nMODRES test ZZZ A    1  LYS\n"
        + PDB_NONSTANDARD.replace("MSE", "ZZZ")
    )
    modified = parse_polymer_structure(path)
    assert modified.sequence == "K"
    assert modified.residue_map[0]["observed_one_letter"] == "K"
    path = tmp_path / "insertions.pdb"
    lines = ["SEQRES   1 A    5  MET ALA GLY SER LYS"]
    for line in MINIMAL_PDB.splitlines():
        if line.startswith("ATOM"):
            code = "A" if line[17:20] == "ALA" else "B"
            lines.append(line[:22] + "   5" + code + line[27:])
    path.write_text("\n".join([*lines, "END"]) + "\n")
    result = parse_polymer_structure(path)
    assert result.sequence == "MAGSK"
    assert [result.residue_map[i]["author_residue_id"] for i in (1, 2)] == [5, 5]
    assert [result.residue_map[i]["insertion_code"] for i in (1, 2)] == ["A", "B"]
