"""Corpus inclusion filters: mean pLDDT from the CIF B-factor + threshold classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser


def mean_plddt_from_cif(cif_path: str | Path) -> float:
    """Mean CA-atom B-factor of the first model (AlphaFold stores per-residue pLDDT there)."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_path))
    model = next(iter(structure))
    values = [
        residue["CA"].get_bfactor() for chain in model for residue in chain if "CA" in residue
    ]
    if not values:
        raise ValueError(f"no CA atoms in {cif_path}")
    return float(np.mean(values))


@dataclass(frozen=True)
class CorpusFilters:
    """Corpus inclusion thresholds (length bounds are enforced by the parity parser)."""

    min_mean_plddt: float = 70.0


def classify(*, mean_plddt: float, filters: CorpusFilters) -> str:
    """Return ``"accepted"`` or ``"rejected_plddt"`` for a structure's mean pLDDT."""
    if mean_plddt < filters.min_mean_plddt:
        return "rejected_plddt"
    return "accepted"
