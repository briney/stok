"""Parser for PDB and mmCIF structure files.

Provides utilities for extracting amino acid sequences and backbone coordinates
from protein structure files using Biopython.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = ["parse_structure", "StructureData"]


class StructureData(NamedTuple):
    """Parsed structure data.

    Attributes:
        pid: Structure identifier (filename stem or structure ID).
        protein_sequence: One-letter amino acid sequence.
        coords: Backbone coordinates [L, 3, 3] for N, CA, C atoms.
        chain_id: Chain identifier used for extraction.
    """

    pid: str
    protein_sequence: str
    coords: np.ndarray
    chain_id: str | None


# Standard 3-letter to 1-letter amino acid mapping
AA3TO1 = {
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
    # Non-standard / modified residues
    "MSE": "M",  # Selenomethionine
    "SEC": "C",  # Selenocysteine (sometimes)
    "PYL": "K",  # Pyrrolysine
    "HYP": "P",  # Hydroxyproline
    "SEP": "S",  # Phosphoserine
    "TPO": "T",  # Phosphothreonine
    "PTR": "Y",  # Phosphotyrosine
    "CSO": "C",  # S-hydroxycysteine
    "CME": "C",  # S,S-(2-hydroxyethyl)thiocysteine
    "MLY": "K",  # N-dimethyl-lysine
    "UNK": "X",  # Unknown
}


def parse_structure(
    path: str | Path,
    *,
    chain_id: str | None = None,
    strict: bool = False,
) -> StructureData:
    """Parse a PDB or mmCIF file and extract sequence and backbone coordinates.

    Args:
        path: Path to .pdb, .ent, .cif, or .mmcif file.
        chain_id: Specific chain to extract. If None, uses first polymer chain.
        strict: If True, raise on missing backbone atoms; else fill with NaN.

    Returns:
        StructureData with pid, sequence, and coords [L, 3, 3].

    Raises:
        ValueError: If structure cannot be parsed or has no valid residues.
        FileNotFoundError: If the file does not exist.
    """
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = path.suffix.lower()

    # Select parser based on file extension
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:  # .pdb, .ent, or unknown
        parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(path.stem, str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse structure file {path}: {e}") from e

    models = list(structure.get_models())
    if len(models) == 0:
        raise ValueError(f"No models found in {path}")

    model = models[0]

    # Find target chain
    chain = None
    if chain_id is not None:
        for ch in model:
            if ch.id == chain_id:
                chain = ch
                break
        if chain is None:
            raise ValueError(f"Chain '{chain_id}' not found in {path}")
    else:
        # Find first chain with amino acid residues
        for ch in model:
            residues = [r for r in ch if is_aa(r, standard=False)]
            if residues:
                chain = ch
                break
        if chain is None:
            raise ValueError(f"No protein chain found in {path}")

    used_chain_id = chain.id

    seq_chars: list[str] = []
    coords_list: list[list[list[float]]] = []

    for residue in chain:
        if not is_aa(residue, standard=False):
            continue

        # Get one-letter code
        res_name = residue.resname.upper().strip()
        aa = _get_one_letter_code(res_name)
        seq_chars.append(aa)

        # Extract N, CA, C coordinates
        try:
            n_coord = residue["N"].coord.tolist()
            ca_coord = residue["CA"].coord.tolist()
            c_coord = residue["C"].coord.tolist()
            coords_list.append([n_coord, ca_coord, c_coord])
        except KeyError as e:
            if strict:
                raise ValueError(
                    f"Missing backbone atom {e} in residue {residue.id} of {path}"
                ) from e
            # Fill with NaN for missing atoms
            coords_list.append([[np.nan] * 3, [np.nan] * 3, [np.nan] * 3])

    if len(seq_chars) == 0:
        raise ValueError(f"No amino acid residues extracted from {path}")

    return StructureData(
        pid=path.stem,
        protein_sequence="".join(seq_chars),
        coords=np.array(coords_list, dtype=np.float32),
        chain_id=used_chain_id,
    )


def _get_one_letter_code(res_name: str) -> str:
    """Convert 3-letter amino acid code to 1-letter code.

    Args:
        res_name: 3-letter residue name (uppercase).

    Returns:
        1-letter amino acid code, or 'X' for unknown residues.
    """
    # First check our mapping
    if res_name in AA3TO1:
        return AA3TO1[res_name]

    # Try Biopython's three_to_one for standard residues
    try:
        from Bio.PDB.Polypeptide import three_to_one

        return three_to_one(res_name)
    except KeyError:
        pass

    # Unknown residue
    return "X"
