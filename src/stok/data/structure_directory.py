"""Discover structure selections; callers supply explicit source identities."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa

from .structure_dataset import STRUCTURE_EXTENSIONS
from ..utils.structure_parser import _cif_rows, _read_structure


def iter_structure_directory(
    directory: str | Path, *, recursive: bool = False
) -> Iterator[dict[str, Any]]:
    """Yield deterministic manifest rows, one protein chain per file, first model.

    IDs include the relative filename (with extension) and escaped chain ID.
    Copies are retained. CIF chains use label IDs, avoiding author-ID collisions.
    Deposited CIF chains without observations are retained for writer exclusions.
    Malformed structures fail discovery; mapping/coverage exclusions are handled
    by the dataset writer. This does not infer biological assemblies.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    if type(recursive) is not bool:
        raise ValueError("recursive must be boolean")
    files = directory.rglob("*") if recursive else directory.iterdir()
    count = 0
    # ponytail: sort paths in memory; shard input directories for huge collections.
    for path in sorted(files):
        if not path.is_file() or path.suffix.lower() not in STRUCTURE_EXTENSIONS:
            continue
        is_cif = path.suffix.lower() in {".cif", ".mmcif"}
        structure = _read_structure(path, label_ids=is_cif, strict_source=True)
        model = next(iter(structure))
        chains = {
            chain.id
            for chain in model
            if any(is_aa(residue, standard=False) for residue in chain)
        }
        if is_cif:
            cif = MMCIF2Dict(str(path))
            types = {
                row["entity_id"]: row["type"]
                for row in _cif_rows(cif, "entity_poly", "entity_id", "type")
            }
            proteins = {
                row["entity_id"]
                for row in _cif_rows(cif, "entity_poly_seq", "entity_id", "mon_id")
                if is_aa(row["mon_id"], standard=False)
            }
            for row in _cif_rows(cif, "struct_asym", "id", "entity_id"):
                entity = row["entity_id"]
                if types.get(entity, "").startswith("polypeptide") or (
                    entity not in types and entity in proteins
                ):
                    chains.add(row["id"])
                elif entity in types:
                    chains.discard(row["id"])
        relative = quote(path.relative_to(directory).as_posix(), safe="/")
        for chain in sorted(chains):
            count += 1
            yield {
                "sequence_id": f"{relative}:{quote(chain, safe='')}",
                "path": str(path),
                "chain_id": chain,
                "chain_namespace": "label" if is_cif else "author",
                "model_index": 0,
            }
    if not count:
        raise ValueError(f"No protein chains found in directory: {directory}")
