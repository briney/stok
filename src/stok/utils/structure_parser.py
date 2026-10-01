"""Parser for PDB and mmCIF structure files.

Provides utilities for extracting amino acid sequences and backbone coordinates
from protein structure files using Biopython.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

import numpy as np

__all__ = [
    "parse_structure",
    "StructureData",
    "parse_polymer_structure",
    "PolymerStructure",
    "StructureMappingError",
]


class StructureMappingError(ValueError):
    """Categorized source/mapping failure, never an inference failure."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class PolymerStructure:
    sequence_id: str
    sequence: str
    coordinates: np.ndarray
    atom_mask: np.ndarray
    residue_map: tuple[Mapping[str, Any], ...]
    source: Mapping[str, Any]

    def __post_init__(self):
        length = len(self.sequence)
        coordinates = np.array(self.coordinates, dtype=np.float32, copy=True)
        atoms = np.array(self.atom_mask, copy=True)
        if (
            not length
            or coordinates.shape != (length, 4, 3)
            or atoms.shape != (length, 4)
            or atoms.dtype != bool
        ):
            raise ValueError(
                "Polymer observations must be float [L,4,3] and boolean [L,4]"
            )
        if not np.array_equal(atoms, np.isfinite(coordinates).all(axis=-1)):
            raise ValueError("Atom mask must describe original finite observations")
        if len(self.residue_map) != length or [
            row["polymer_position"] for row in self.residue_map
        ] != list(range(length)):
            raise ValueError(
                "Residue map must contain each polymer position exactly once"
            )
        coordinates.setflags(write=False)
        atoms.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "atom_mask", atoms)
        object.__setattr__(
            self,
            "residue_map",
            tuple(MappingProxyType(dict(row)) for row in self.residue_map),
        )
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))


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
    from Bio.PDB.Polypeptide import is_aa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    structure = _read_structure(path)

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
    return AA3TO1.get(res_name, "X")


def _read_structure(
    path: Path, *, label_ids: bool = False, strict_source: bool = False
):
    from Bio.PDB import MMCIFParser, PDBParser

    if not path.is_file():
        raise FileNotFoundError(f"Structure file not found: {path}")
    parser = (
        MMCIFParser(QUIET=True, auth_chains=not label_ids, auth_residues=not label_ids)
        if path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True, PERMISSIVE=not strict_source)
    )
    try:
        return parser.get_structure(path.stem, str(path))
    except Exception as error:
        raise StructureMappingError(
            "parse_failed", f"Failed to parse structure file {path}: {error}"
        ) from error


def _monomer_letter(name: str | None, parents: Mapping[str, str]) -> str:
    from Bio.Data.PDBData import protein_letters_3to1_extended

    if name is None:
        return "X"
    name = parents.get(name, name)
    letter = AA3TO1.get(name, protein_letters_3to1_extended.get(name, "X"))
    return letter if letter in "ACDEFGHIKLMNPQRSTVWYX" else "X"


def _align_observations(polymer: str, observed: str, context: str) -> list[int]:
    from Bio.Align import PairwiseAligner

    if not observed:
        return []
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = float("-inf")
    aligner.gap_score = 0
    # Coordinate residues may not be dropped. Polymer gaps/termini may be unresolved.
    aligner.target_gap_score = float("-inf")
    aligner.wildcard = "X"
    alignments = aligner.align(polymer, observed)
    if not math.isfinite(alignments.score):
        raise StructureMappingError("mapping_conflict", context)
    iterator = iter(alignments)
    first = next(iterator).indices
    mapping = first[0, first[1] >= 0].tolist()
    if len(mapping) != len(observed) or any(position < 0 for position in mapping):
        raise StructureMappingError("mapping_conflict", context)
    # Inspect only the second alignment: repetitive constructs can have enormous counts.
    if next(iterator, None) is not None:
        raise StructureMappingError("mapping_ambiguous", context)
    return mapping


def _residue_observations(residue, context: str) -> tuple[np.ndarray, str]:
    if residue.is_disordered() == 2:
        raise StructureMappingError(
            "monomer_conflict", f"{context}, residue {residue.id}"
        )
    atoms = [
        atom
        for atom in residue.get_unpacked_list()
        if atom.name in {"N", "CA", "C", "O"}
    ]
    alternatives = sorted(
        {atom.altloc.strip() for atom in atoms if atom.altloc.strip()}
    )
    selected = (
        min(
            alternatives,
            key=lambda alt: (
                -sum(
                    (atom.occupancy or 0)
                    for atom in atoms
                    if atom.altloc.strip() == alt
                ),
                alt,
            ),
        )
        if alternatives
        else ""
    )
    coordinates = np.full((4, 3), np.nan, dtype=np.float32)
    for i, name in enumerate(("N", "CA", "C", "O")):
        choices = [
            atom
            for atom in atoms
            if atom.name == name and atom.altloc.strip() in {"", selected}
        ]
        chosen = next(
            (atom for atom in choices if atom.altloc.strip() == selected),
            choices[0] if choices else None,
        )
        if chosen is not None and np.isfinite(chosen.coord).all():
            coordinates[i] = chosen.coord
    return coordinates, selected


def _cif_rows(
    cif: Mapping[str, Any], category: str, *fields: str
) -> list[dict[str, str]]:
    count = len(cif.get(f"_{category}.{fields[0]}", []))
    columns = [cif.get(f"_{category}.{field}", ["?"] * count) for field in fields]
    if any(len(column) != count for column in columns):
        raise StructureMappingError(
            "metadata_conflict", f"Inconsistent {category} columns"
        )
    return [dict(zip(fields, values)) for values in zip(*columns)]


def _optional_int(value: str, context: str) -> int | None:
    if value in {"?", ".", ""}:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise StructureMappingError(
            "metadata_conflict", f"{context}: invalid residue ID {value}"
        ) from error


def _pdb_metadata(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    sequences: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    parents: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("SEQRES"):
            chain = line[11]
            count = _optional_int(line[13:17].strip(), f"{path}, SEQRES chain {chain}")
            if count is None or count < 1:
                raise StructureMappingError(
                    "metadata_conflict", f"{path}, SEQRES chain {chain}"
                )
            if counts.setdefault(chain, count) != count:
                raise StructureMappingError(
                    "metadata_conflict", f"{path}, SEQRES chain {chain}"
                )
            sequences.setdefault(chain, []).extend(line[19:70].split())
        elif line.startswith("MODRES"):
            monomer, parent = line[12:15].strip(), line[24:27].strip()
            if parents.setdefault(monomer, parent) != parent:
                raise StructureMappingError(
                    "monomer_conflict", f"{path}, MODRES {monomer}"
                )
    if any(len(sequences[chain]) != count for chain, count in counts.items()):
        raise StructureMappingError("metadata_conflict", f"{path}, incomplete SEQRES")
    return sequences, parents


def parse_polymer_structure(
    path: str | Path,
    *,
    chain_id: str | None = None,
    chain_namespace: Literal["author", "label"] = "author",
    model_index: int = 0,
    sequence: str | None = None,
    allow_observed_sequence: bool = False,
) -> PolymerStructure:
    """Map observations to deposited polymer slots, rejecting ambiguous correspondence.

    Select shared blank-altloc atoms plus the backbone conformer with greatest
    summed occupancy; occupancy ties use lexical altloc order.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.PDB.Polypeptide import is_aa

    path = Path(path)
    context = f"{path}, {chain_namespace} chain {chain_id}, model {model_index}"
    if (
        chain_namespace not in {"author", "label"}
        or type(model_index) is not int
        or model_index < 0
    ):
        raise ValueError("Invalid chain namespace or zero-based model index")
    if sequence is not None and (
        not sequence
        or any(letter not in "ACDEFGHIKLMNPQRSTVWYX" for letter in sequence)
    ):
        raise ValueError("Supplied sequence must contain uppercase amino acids or X")
    is_cif = path.suffix.lower() in {".cif", ".mmcif"}
    structure = _read_structure(path, label_ids=is_cif, strict_source=True)
    models = list(structure)
    if model_index >= len(models):
        raise StructureMappingError("model_not_found", context)
    model = models[model_index]
    parents: dict[str, str] = {}
    atom_metadata: dict[int, dict[str, Any]] = {}
    scheme: dict[int, dict[str, Any]] = {}
    label_chain: str | None = None
    entity_id: str | None = None
    author_chain: str | None = None
    monomers: list[str | None]
    if is_cif:
        cif = MMCIF2Dict(str(path))
        entity_monomers: dict[str, dict[int, str]] = {}
        entity_types = {
            row["entity_id"]: row["type"]
            for row in _cif_rows(cif, "entity_poly", "entity_id", "type")
        }
        for row in _cif_rows(cif, "chem_comp", "id", "mon_nstd_parent_comp_id"):
            parent = row["mon_nstd_parent_comp_id"]
            if parent not in {"?", "."}:
                if "," in parent:
                    raise StructureMappingError(
                        "monomer_conflict",
                        f"{context}, multiple parents for {row['id']}",
                    )
                parents[row["id"]] = parent
        for row in _cif_rows(cif, "entity_poly_seq", "entity_id", "num", "mon_id"):
            if row["entity_id"] in entity_types and not entity_types[
                row["entity_id"]
            ].startswith("polypeptide"):
                continue
            position = _optional_int(row["num"], context)
            if position is None or row["mon_id"] in {"?", "."}:
                raise StructureMappingError("metadata_conflict", context)
            deposited = entity_monomers.setdefault(row["entity_id"], {})
            if deposited.setdefault(position, row["mon_id"]) != row["mon_id"]:
                raise StructureMappingError(
                    "monomer_conflict", f"{context}, label position {position}"
                )
        chain_entities = {
            row["id"]: row["entity_id"]
            for row in _cif_rows(cif, "struct_asym", "id", "entity_id")
        }
        atom_rows = _cif_rows(
            cif,
            "atom_site",
            "label_asym_id",
            "auth_asym_id",
            "label_entity_id",
            "label_seq_id",
            "auth_seq_id",
            "pdbx_PDB_ins_code",
            "label_comp_id",
            "pdbx_PDB_model_num",
        )
        schemes = _cif_rows(
            cif,
            "pdbx_poly_seq_scheme",
            "asym_id",
            "entity_id",
            "seq_id",
            "mon_id",
            "auth_seq_num",
            "pdb_ins_code",
            "pdb_strand_id",
        )
        aliases: dict[str, set[str]] = {}
        for row in atom_rows:
            label = row["label_asym_id"]
            entity = row["label_entity_id"]
            if chain_entities.setdefault(label, entity) != entity:
                raise StructureMappingError(
                    "metadata_conflict", f"{context}, entity of chain {label}"
                )
            aliases.setdefault(label, set()).add(row["auth_asym_id"])
        for row in schemes:
            if row["pdb_strand_id"] not in {"?", "."}:
                aliases.setdefault(row["asym_id"], set()).add(row["pdb_strand_id"])
        primary_entities = set(entity_monomers)
        for row in schemes:
            entity = chain_entities.get(row["asym_id"], row["entity_id"])
            if row["entity_id"] not in {"?", ".", entity}:
                raise StructureMappingError(
                    "metadata_conflict",
                    f"{context}, scheme entity for {row['asym_id']}",
                )
            if entity in primary_entities or (
                entity in entity_types
                and not entity_types[entity].startswith("polypeptide")
            ):
                continue
            position = _optional_int(row["seq_id"], context)
            if position is None or row["mon_id"] in {"?", "."}:
                raise StructureMappingError("metadata_conflict", context)
            chain_entities.setdefault(row["asym_id"], entity)
            deposited = entity_monomers.setdefault(entity, {})
            if deposited.setdefault(position, row["mon_id"]) != row["mon_id"]:
                raise StructureMappingError(
                    "monomer_conflict", f"{context}, scheme position {position}"
                )
        candidates = [
            label
            for label, entity in chain_entities.items()
            if entity in entity_monomers
            or (
                not entity_monomers
                and label in model
                and any(is_aa(residue, standard=False) for residue in model[label])
            )
        ]
        selected = [
            label
            for label in candidates
            if chain_id is None
            or (
                label == chain_id
                if chain_namespace == "label"
                else chain_id in aliases.get(label, set())
            )
        ]
        if len(selected) != 1:
            raise StructureMappingError(
                "chain_ambiguous" if selected else "chain_not_found", context
            )
        label_chain = selected[0]
        entity_id = chain_entities[label_chain]
        if len(aliases.get(label_chain, set())) > 1:
            raise StructureMappingError("chain_ambiguous", context)
        author_chain = next(iter(aliases.get(label_chain, set())), None)
        deposited = entity_monomers.get(entity_id, {})
        if deposited and sorted(deposited) != list(range(1, len(deposited) + 1)):
            raise StructureMappingError(
                "metadata_conflict",
                f"{context}, noncontiguous deposited label positions",
            )
        monomers = [deposited[position] for position in sorted(deposited)]
        residues = list(model[label_chain]) if label_chain in model else []
        for row in schemes:
            if row["asym_id"] != label_chain:
                continue
            position = _optional_int(row["seq_id"], context)
            if position not in deposited or row["mon_id"] != deposited[position]:
                raise StructureMappingError(
                    "metadata_conflict",
                    f"{context}, polymer scheme position {position}",
                )
            identity = {
                "author_residue_id": _optional_int(row["auth_seq_num"], context),
                "insertion_code": None
                if row["pdb_ins_code"] == "?"
                else ""
                if row["pdb_ins_code"] == "."
                else row["pdb_ins_code"],
            }
            if position in scheme and scheme[position] != identity:
                raise StructureMappingError("mapping_conflict", context)
            scheme[position] = identity
        for row in atom_rows:
            if row["label_asym_id"] != label_chain or row["pdbx_PDB_model_num"] != str(
                model.serial_num
            ):
                continue
            position = _optional_int(row["label_seq_id"], context)
            if position is None:
                raise StructureMappingError(
                    "mapping_conflict",
                    f"{context}, coordinate polymer residue lacks label_seq_id",
                )
            identity = {
                "author_residue_id": _optional_int(row["auth_seq_id"], context),
                "insertion_code": row["pdbx_PDB_ins_code"]
                if row["pdbx_PDB_ins_code"] not in {"?", "."}
                else "",
                "observed_monomer_id": row["label_comp_id"],
            }
            if position in atom_metadata and atom_metadata[position] != identity:
                raise StructureMappingError(
                    "monomer_conflict", f"{context}, label position {position}"
                )
            atom_metadata[position] = identity
        sequence_source = (
            "entity_poly_seq" if entity_id in primary_entities else "poly_seq_scheme"
        )
    else:
        deposited_sequences, parents = _pdb_metadata(path)
        candidates = [
            chain.id
            for chain in model
            if chain.id in deposited_sequences
            or any(
                is_aa(residue, standard=False) or residue.id[0] == " "
                for residue in chain
            )
        ]
        selected = [
            candidate
            for candidate in candidates
            if chain_id is None or candidate == chain_id
        ]
        if len(selected) != 1:
            raise StructureMappingError(
                "chain_ambiguous" if selected else "chain_not_found", context
            )
        author_chain = selected[0]
        monomers = [monomer for monomer in deposited_sequences.get(author_chain, [])]
        residues = [
            residue
            for residue in model[author_chain]
            if is_aa(residue, standard=False)
            or residue.id[0] == " "
            or residue.resname in parents
        ]
        sequence_source = "seqres"
    observed = "".join(
        _monomer_letter(residue.resname, parents) for residue in residues
    )
    if monomers:
        polymer = "".join(_monomer_letter(monomer, parents) for monomer in monomers)
        if sequence is not None and sequence != polymer:
            raise StructureMappingError("sequence_conflict", context)
    elif sequence is not None:
        polymer = sequence
        # A supplied one-letter construct does not establish deposited monomer IDs.
        monomers = [None] * len(polymer)
        sequence_source = "supplied"
    elif allow_observed_sequence and observed:
        polymer = observed
        monomers = [residue.resname for residue in residues]
        sequence_source = "observed"
    else:
        raise StructureMappingError("sequence_metadata_missing", context)
    deposited_labels = is_cif and sequence_source in {
        "entity_poly_seq",
        "poly_seq_scheme",
    }
    if deposited_labels:
        positions = [residue.id[1] - 1 for residue in residues]
    else:
        positions = (
            list(range(len(residues)))
            if sequence_source == "observed"
            else _align_observations(polymer, observed, context)
        )
    coordinates = np.full((len(polymer), 4, 3), np.nan, dtype=np.float32)
    residue_map: list[dict[str, Any]] = [
        {
            "polymer_position": i,
            "monomer_id": monomer,
            "observed_monomer_id": None,
            "observed_one_letter": None,
            "label_seq_id": i + 1 if deposited_labels else None,
            "author_residue_id": None,
            "insertion_code": None,
            "selected_altloc": None,
        }
        for i, monomer in enumerate(monomers)
    ]
    for label_position, metadata in scheme.items():
        residue_map[label_position - 1].update(metadata)
    seen = set()
    for identity_letter, residue, position in zip(observed, residues, positions):
        if position in seen or position < 0 or position >= len(polymer):
            raise StructureMappingError(
                "mapping_conflict", f"{context}, residue {residue.id}"
            )
        seen.add(position)
        if identity_letter != polymer[position] and "X" not in {
            identity_letter,
            polymer[position],
        }:
            raise StructureMappingError(
                "monomer_conflict",
                f"{context}, residue {residue.id}, polymer position {position}",
            )
        coordinates[position], alternate = _residue_observations(residue, context)
        metadata = (
            atom_metadata.get(residue.id[1], {})
            if is_cif
            else {
                "author_residue_id": residue.id[1],
                "insertion_code": residue.id[2].strip(),
            }
        )
        if is_cif and not metadata:
            raise StructureMappingError(
                "mapping_conflict", f"{context}, residue {residue.id}"
            )
        if (
            is_cif
            and residue.id[1] in scheme
            and any(
                scheme[residue.id[1]][key] not in {None, metadata[key]}
                for key in ("author_residue_id", "insertion_code")
            )
        ):
            raise StructureMappingError(
                "mapping_conflict",
                f"{context}, author identity for label position {residue.id[1]}",
            )
        residue_map[position].update(
            metadata,
            observed_monomer_id=residue.resname,
            observed_one_letter=identity_letter,
            selected_altloc=alternate,
        )
    author_positions = [
        (row["author_residue_id"], row["insertion_code"])
        for row in residue_map
        if row["author_residue_id"] is not None and row["insertion_code"] is not None
    ]
    if len(set(author_positions)) != len(author_positions):
        raise StructureMappingError(
            "mapping_ambiguous", f"{context}, repeated author residue identity"
        )
    return PolymerStructure(
        sequence_id=path.stem,
        sequence=polymer,
        coordinates=coordinates,
        atom_mask=np.isfinite(coordinates).all(axis=-1),
        residue_map=tuple(residue_map),
        source={
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "label_chain_id": label_chain,
            "author_chain_id": author_chain,
            "entity_id": entity_id,
            "model_index": model_index,
            "model_serial_id": model.serial_num,
            "sequence_source": sequence_source,
        },
    )
