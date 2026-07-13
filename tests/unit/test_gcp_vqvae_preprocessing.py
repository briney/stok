"""Tests for upstream-compatible GCP-VQVAE structure parsing."""

from pathlib import Path

import numpy as np

from stok.utils.gcp_vqvae_preprocessing import (
    GCPVQVAEStructureSample,
    estimate_missing_from_distance,
    evaluate_missing_content,
    parse_gcp_vqvae_samples,
    propagate_nan_residues,
    sequence_similarity,
)

THREE_LETTER_CODE = {
    "A": "ALA",
    "G": "GLY",
    "S": "SER",
}


def make_test_pdb(
    chains: dict[str, str],
    *,
    residue_ids: dict[str, list[int]] | None = None,
    omitted_atoms: dict[tuple[str, int], set[str]] | None = None,
) -> str:
    """Build peptide-connected chains with N/CA/C/O coordinates."""
    lines: list[str] = []
    serial = 1
    omitted_atoms = omitted_atoms or {}

    for chain_id, sequence in chains.items():
        chain_residue_ids = (
            residue_ids[chain_id]
            if residue_ids is not None and chain_id in residue_ids
            else list(range(1, len(sequence) + 1))
        )
        for index, (one_letter, residue_id) in enumerate(
            zip(sequence, chain_residue_ids, strict=True)
        ):
            base_x = index * 3.8
            atoms = {
                "N": (base_x, 0.0, 0.0),
                "CA": (base_x + 1.45, 0.0, 0.0),
                "C": (base_x + 2.50, 0.0, 0.0),
                "O": (base_x + 2.95, 0.9, 0.0),
            }
            residue_name = THREE_LETTER_CODE[one_letter]
            for atom_name, (x, y, z) in atoms.items():
                if atom_name in omitted_atoms.get((chain_id, index), set()):
                    continue
                element = atom_name[0]
                lines.append(
                    f"ATOM  {serial:5d}  {atom_name:<3s} {residue_name:>3s} "
                    f"{chain_id:1s}{residue_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
                )
                serial += 1
        lines.append("TER")

    lines.append("END")
    return "\n".join(lines) + "\n"


def make_gapped_test_pdb(*, length: int, gap_after: int, numeric_gap: int) -> str:
    residue_ids = list(range(1, length + 1))
    for index in range(gap_after + 1, length):
        residue_ids[index] += numeric_gap
    return make_test_pdb({"A": "A" * length}, residue_ids={"A": residue_ids})


def test_selects_distinct_chains_and_deduplicates_similar_chains(tmp_path: Path):
    path = tmp_path / "complex.pdb"
    path.write_text(make_test_pdb({"A": "A" * 25, "B": "A" * 26, "C": "G" * 25}))

    samples = parse_gcp_vqvae_samples(path, file_index=7, max_length=1280)

    assert [(sample.pid, sample.chain_id, sample.sequence) for sample in samples] == [
        ("7_complex_chain_id_B", "B", "A" * 26),
        ("7_complex_chain_id_C", "C", "G" * 25),
    ]
    assert all(isinstance(sample, GCPVQVAEStructureSample) for sample in samples)
    assert all(sample.coords.dtype == np.float32 for sample in samples)
    assert all(sample.coords.shape == (len(sample.sequence), 4, 3) for sample in samples)
    assert all(sample.source_path == str(path) for sample in samples)


def test_single_selected_chain_identifier_omits_chain_suffix(tmp_path: Path):
    path = tmp_path / "single.pdb"
    path.write_text(make_test_pdb({"Q": "S" * 25}))

    [sample] = parse_gcp_vqvae_samples(path, file_index=3, max_length=1280)

    assert sample.pid == "3_single"
    assert sample.chain_id == "Q"


def test_missing_content_limits_match_upstream():
    coords = np.zeros((25, 4, 3), dtype=np.float32)
    coords[:5] = np.nan
    assert evaluate_missing_content(coords) == (True, "")

    coords[5] = np.nan
    assert evaluate_missing_content(coords) == (False, "missing_ratio_exceeded")
    assert evaluate_missing_content(np.empty((0, 4, 3), dtype=np.float32)) == (
        False,
        "missing_ratio_exceeded",
    )


def test_missing_content_allows_fifteen_but_not_sixteen_consecutive_residues():
    coords = np.zeros((100, 4, 3), dtype=np.float32)
    coords[:15] = np.nan
    assert evaluate_missing_content(coords) == (True, "")

    coords[15] = np.nan
    assert evaluate_missing_content(coords) == (False, "missing_block_exceeded")


def test_numbering_gap_inserts_unknown_residues(tmp_path: Path):
    path = tmp_path / "gap.pdb"
    path.write_text(make_gapped_test_pdb(length=25, gap_after=12, numeric_gap=2))

    [sample] = parse_gcp_vqvae_samples(path, file_index=0, max_length=1280)

    assert sample.sequence[13:15] == "XX"
    assert np.isnan(sample.coords[13:15]).all()


def test_large_numbering_gap_is_limited_by_distance_estimate(tmp_path: Path):
    path = tmp_path / "estimated-gap.pdb"
    path.write_text(make_gapped_test_pdb(length=25, gap_after=12, numeric_gap=10))

    [sample] = parse_gcp_vqvae_samples(path, file_index=0, max_length=1280)

    assert sample.sequence == "A" * 25
    assert sample.coords.shape == (25, 4, 3)


def test_partial_residue_coordinates_propagate_to_entire_residue(tmp_path: Path):
    path = tmp_path / "partial.pdb"
    path.write_text(make_test_pdb({"A": "A" * 25}, omitted_atoms={("A", 10): {"O"}}))

    [sample] = parse_gcp_vqvae_samples(path, file_index=0, max_length=1280)

    assert np.isnan(sample.coords[10]).all()
    assert not np.isnan(sample.coords[:10]).any()
    assert not np.isnan(sample.coords[11:]).any()


def test_rejects_short_and_over_maximum_length_chains(tmp_path: Path):
    short_path = tmp_path / "short.pdb"
    short_path.write_text(make_test_pdb({"A": "A" * 24}))
    long_path = tmp_path / "long.pdb"
    long_path.write_text(make_test_pdb({"A": "A" * 25}))

    assert parse_gcp_vqvae_samples(short_path, file_index=0, max_length=1280) == []
    assert parse_gcp_vqvae_samples(long_path, file_index=0, max_length=24) == []


def test_upstream_helper_semantics():
    assert sequence_similarity("AAAA", "AAAAA") == 1.0
    assert sequence_similarity("AAAA", "GGGG") == 0.0
    assert estimate_missing_from_distance([0.0, 0.0, 0.0], [7.6, 0.0, 0.0]) == 1
    assert estimate_missing_from_distance([np.nan, 0.0, 0.0], [7.6, 0.0, 0.0]) is None

    coords = np.zeros((2, 4, 3), dtype=np.float32)
    coords[0, 3] = np.nan
    coords[1] = np.nan
    assert propagate_nan_residues(coords) == 1
    assert np.isnan(coords[0]).all()
    assert np.isnan(coords[1]).all()
