"""Upstream-compatible structure selection for GCP-VQVAE preprocessing.

This module reproduces the public behavior of the pinned GCP-VQVAE structure
parser without importing private upstream modules at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import PPBuilder

if TYPE_CHECKING:
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Structure import Structure

PREPROCESS_MIN_LEN = 25
PREPROCESS_MAX_MISSING_RATIO = 0.2
PREPROCESS_MAX_CONSECUTIVE_MISSING = 15
PREPROCESS_USE_GAP_ESTIMATION = True
PREPROCESS_GAP_THRESHOLD = 5
PREPROCESS_SIMILARITY_THRESHOLD = 0.90

UPSTREAM_AA_MAP = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "ASX": "B",
    "GLX": "Z",
    "PYL": "O",
    "SEC": "U",
}

_GLOBAL_IDENTITY_ALIGNER = PairwiseAligner(
    mode="global",
    match_score=1.0,
    mismatch_score=0.0,
    open_gap_score=0.0,
    extend_gap_score=0.0,
)

__all__ = [
    "GCPVQVAEStructureSample",
    "PREPROCESS_GAP_THRESHOLD",
    "PREPROCESS_MAX_CONSECUTIVE_MISSING",
    "PREPROCESS_MAX_MISSING_RATIO",
    "PREPROCESS_MIN_LEN",
    "PREPROCESS_SIMILARITY_THRESHOLD",
    "PREPROCESS_USE_GAP_ESTIMATION",
    "UPSTREAM_AA_MAP",
    "estimate_missing_from_distance",
    "evaluate_missing_content",
    "parse_gcp_vqvae_samples",
    "propagate_nan_residues",
    "sequence_similarity",
]


@dataclass(frozen=True)
class GCPVQVAEStructureSample:
    """One structure chain accepted by the upstream preprocessing policy."""

    pid: str
    sequence: str
    coords: np.ndarray
    chain_id: str
    source_path: str


def sequence_similarity(seq1: str, seq2: str) -> float:
    """Return upstream global identity normalized by the shorter sequence."""
    score = _GLOBAL_IDENTITY_ALIGNER.score(seq1, seq2)
    return float(score) / min(len(seq1), len(seq2))


def estimate_missing_from_distance(
    prev_ca_coord: object,
    next_ca_coord: object,
    ideal_ca_ca: float = 3.8,
) -> int | None:
    """Estimate missing residues between two C-alpha coordinates."""
    try:
        x1, y1, z1 = prev_ca_coord  # type: ignore[misc]
        x2, y2, z2 = next_ca_coord  # type: ignore[misc]
        if any(math.isnan(value) for value in (x1, y1, z1, x2, y2, z2)):
            return None
    except Exception:
        return None

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return max(0, int(math.floor((distance / ideal_ca_ca) * 1.2) - 1))


def evaluate_missing_content(
    coords: object,
    max_missing_ratio: float = PREPROCESS_MAX_MISSING_RATIO,
    max_consecutive_missing: int = PREPROCESS_MAX_CONSECUTIVE_MISSING,
) -> tuple[bool, str]:
    """Apply the upstream C-alpha missing-content limits."""
    total = len(coords)  # type: ignore[arg-type]
    if total == 0:
        return False, "missing_ratio_exceeded"

    missing_flags: list[bool] = []
    for residue in coords:  # type: ignore[union-attr]
        ca_coords = residue[1] if len(residue) > 1 else []
        if len(ca_coords) != 3:
            missing_flags.append(True)
            continue
        missing_flags.append(any(math.isnan(value) for value in ca_coords))

    missing_count = sum(missing_flags)
    if missing_count / total > max_missing_ratio:
        return False, "missing_ratio_exceeded"

    longest_run = 0
    current_run = 0
    for is_missing in missing_flags:
        if is_missing:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    if longest_run > max_consecutive_missing:
        return False, "missing_block_exceeded"
    return True, ""


def propagate_nan_residues(coords: object) -> int:
    """Replace partially missing residues with fully NaN four-atom rows."""
    updated_count = 0
    for index, residue_coords in enumerate(coords):  # type: ignore[union-attr]
        is_fully_nan = True
        has_any_missing = False
        for atom_coords in residue_coords:
            if len(atom_coords) != 3:
                has_any_missing = True
                continue
            any_nan = any(math.isnan(value) for value in atom_coords)
            if any_nan:
                has_any_missing = True
            else:
                is_fully_nan = False
        if has_any_missing and not is_fully_nan:
            coords[index] = [[math.nan, math.nan, math.nan] for _ in range(4)]  # type: ignore[index]
            updated_count += 1
    return updated_count


def _select_parser(path: str):
    if path.lower().endswith((".cif", ".mmcif")):
        return MMCIFParser(QUIET=True, auth_chains=False)
    return PDBParser(QUIET=True)


def _parse_structure(path: str) -> Structure:
    parser = _select_parser(path)
    try:
        return parser.get_structure("protein", path)
    except Exception:
        fallback = (
            MMCIFParser(QUIET=True, auth_chains=False)
            if isinstance(parser, PDBParser)
            else PDBParser(QUIET=True)
        )
        return fallback.get_structure("protein", path)


def _find_chain_sequences(structure: Structure) -> dict[str, str]:
    peptide_builder = PPBuilder()
    chains = [chain for model in structure for chain in model]
    sequences: dict[str, str] = {}
    for chain in chains:
        sequence = "".join(
            str(peptide.get_sequence()) for peptide in peptide_builder.build_peptides(chain)
        )
        if len(sequence) >= PREPROCESS_MIN_LEN:
            sequences[chain.id] = sequence
    return sequences


def _filter_best_chains(
    chain_sequences: dict[str, str],
    structure: Structure,
) -> dict[str, tuple[str, str]]:
    sequence_to_chain: dict[str, tuple[str, int]] = {}

    for chain_id, sequence in chain_sequences.items():
        chain = structure[0][chain_id]
        ca_count = sum(1 for residue in chain if "CA" in residue)

        is_similar = False
        for existing_sequence in sequence_to_chain:
            if sequence_similarity(sequence, existing_sequence) > PREPROCESS_SIMILARITY_THRESHOLD:
                is_similar = True
                _, existing_ca_count = sequence_to_chain[existing_sequence]
                if ca_count > existing_ca_count:
                    sequence_to_chain[existing_sequence] = (chain_id, ca_count)
                break

        if not is_similar:
            sequence_to_chain[sequence] = (chain_id, ca_count)

    return {
        chain_id: (sequence, chain_id)
        for sequence, (chain_id, _ca_count) in sequence_to_chain.items()
    }


def _extract_chain(
    chain: Chain,
    *,
    use_gap_estimation: bool,
    gap_threshold: int,
) -> tuple[str, list[list[list[float]]]] | None:
    residues = [residue for residue in chain if residue.id[0] == " "]
    if not residues:
        return None

    sequence = ""
    coords: list[list[list[float]]] = []
    for residue in residues:
        sequence += UPSTREAM_AA_MAP.get(residue.resname, "X")
        residue_coords: list[list[float]] = []
        for atom_name in ("N", "CA", "C", "O"):
            if atom_name in residue:
                residue_coords.append(list(residue[atom_name].coord))
            else:
                residue_coords.append([math.nan, math.nan, math.nan])
        coords.append(residue_coords)

    for index in range(len(residues) - 1, 0, -1):
        current_residue_id = residues[index].id
        previous_residue_id = residues[index - 1].id
        if current_residue_id[1] <= previous_residue_id[1] + 1:
            continue

        numeric_gap_size = current_residue_id[1] - previous_residue_id[1] - 1
        insert_count = numeric_gap_size
        if use_gap_estimation and numeric_gap_size > gap_threshold:
            estimated_missing = estimate_missing_from_distance(
                coords[index - 1][1], coords[index][1]
            )
            if estimated_missing is not None:
                insert_count = min(numeric_gap_size, estimated_missing)
            else:
                insert_count = gap_threshold

        if insert_count <= 0:
            continue

        sequence = sequence[:index] + ("X" * insert_count) + sequence[index:]
        nan_residue = [[math.nan, math.nan, math.nan] for _ in range(4)]
        coords[index:index] = [nan_residue] * insert_count

    propagate_nan_residues(coords)
    return sequence, coords


def parse_gcp_vqvae_samples(
    path: str | Path,
    *,
    file_index: int,
    max_length: int,
) -> list[GCPVQVAEStructureSample]:
    """Parse every chain accepted by the pinned upstream selection policy."""
    source_path = str(path)
    structure = _parse_structure(source_path)
    chain_sequences = _find_chain_sequences(structure)
    best_chains = _filter_best_chains(chain_sequences, structure)

    samples: list[GCPVQVAEStructureSample] = []
    for chain_id in best_chains:
        extracted = _extract_chain(
            structure[0][chain_id],
            use_gap_estimation=PREPROCESS_USE_GAP_ESTIMATION,
            gap_threshold=PREPROCESS_GAP_THRESHOLD,
        )
        if extracted is None:
            continue
        sequence, coords = extracted

        if len(sequence) < PREPROCESS_MIN_LEN or len(sequence) > max_length:
            continue
        is_valid, _reason = evaluate_missing_content(coords)
        if not is_valid:
            continue

        basename = Path(source_path).stem
        pid = f"{basename}_chain_id_{chain_id}" if len(best_chains) > 1 else basename
        pid = f"{file_index}_{pid}"
        samples.append(
            GCPVQVAEStructureSample(
                pid=pid,
                sequence=sequence,
                coords=np.asarray(coords, dtype=np.float32),
                chain_id=chain_id,
                source_path=source_path,
            )
        )

    return samples
