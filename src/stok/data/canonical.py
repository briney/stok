"""Frozen original polymer observations, independent of their representations."""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np

from .structure_encoding import iter_structure_manifest
from ..utils.pretrained import file_sha256, json_sha256
from ..utils.structure_parser import (
    PolymerStructure,
    StructureMappingError,
    parse_polymer_structure,
)


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def canonical_record(
    structure: PolymerStructure,
    *,
    source_namespace: str,
    source_accession: str,
    parent_ids: Sequence[str] = (),
    record_kind: Literal["structure", "sequence"] = "structure",
) -> dict[str, Any]:
    source = dict(structure.source)
    label_chain = source.get("label_chain_id")
    identity = {
        "canonical_identity_version": 1,
        "record_kind": record_kind,
        "source_namespace": source_namespace,
        "source_accession": source_accession,
        "source_revision_sha256": source.get("sha256"),
        "model_index": source.get("model_index")
        if record_kind == "structure"
        else None,
        "model_serial_id": source.get("model_serial_id")
        if record_kind == "structure"
        else None,
        "chain_namespace": ("label" if label_chain is not None else "author")
        if record_kind == "structure"
        else None,
        "chain_id": (
            label_chain if label_chain is not None else source.get("author_chain_id")
        )
        if record_kind == "structure"
        else None,
    }
    coordinates = [
        [
            values.tolist() if available else [None, None, None]
            for values, available in zip(residue, mask)
        ]
        for residue, mask in zip(structure.coordinates, structure.atom_mask)
    ]
    content = {
        "sequence": structure.sequence,
        "coordinates": coordinates,
        "atom_mask": structure.atom_mask.tolist(),
        "residue_map": [dict(row) for row in structure.residue_map],
    }
    record = {
        "canonical_id": json_sha256(identity),
        "identity": identity,
        **content,
        "content_sha256": json_sha256(content),
        "residue_map_sha256": json_sha256(content["residue_map"]),
        "parent_ids": sorted(parent_ids),
        "provenance": {"sequence_id": structure.sequence_id, "source": source},
    }
    return record


def record_to_polymer(record: Mapping[str, Any]) -> PolymerStructure:
    return PolymerStructure(
        sequence_id=record["provenance"]["sequence_id"],
        sequence=record["sequence"],
        coordinates=np.asarray(record["coordinates"], dtype=np.float32),
        atom_mask=np.asarray(record["atom_mask"], dtype=bool),
        residue_map=tuple(record["residue_map"]),
        source=record["provenance"]["source"],
    )


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected JSON object")
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed inventory row"
                ) from error
            yield row


def _resolved_selection(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        identity[key]
        for key in (
            "source_revision_sha256",
            "model_index",
            "model_serial_id",
            "chain_namespace",
            "chain_id",
        )
    )


def _population(records: Sequence[Mapping[str, Any]]) -> str:
    return json_sha256(
        sorted(
            (
                record["canonical_id"],
                record["content_sha256"],
                record["residue_map_sha256"],
                record["parent_ids"],
            )
            for record in records
        )
    )


def validate_canonical_dataset(directory: str | Path) -> dict[str, Any]:
    """Check completion and file integrity of an inventory this module wrote."""
    directory = Path(directory)
    summary = json.loads((directory / "manifest.json").read_text())
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != 1
        or summary.get("status") != "complete"
    ):
        raise ValueError("Canonical inventory has no valid completion manifest")
    for name in ("records", "rejections"):
        if file_sha256(directory / (name + ".jsonl")) != summary.get(name + "_sha256"):
            raise ValueError(f"Corrupt canonical {name} inventory")
    return summary


def iter_canonical_records(directory: str | Path) -> Iterator[dict[str, Any]]:
    yield from _jsonl(Path(directory) / "records.jsonl")


def prepare_canonical_dataset(
    manifest: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Parse originals once, before any representation-specific admission policy."""
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed run leaves only this hidden staging directory behind.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    members, selections, exclusions = [], set(), Counter()
    requested = 0
    with (
        (staging / "records.jsonl").open("w") as records,
        (staging / "rejections.jsonl").open("w") as rejections,
    ):
        for entry in iter_structure_manifest(Path(manifest).resolve()):
            requested += 1
            try:
                structure = replace(
                    parse_polymer_structure(
                        entry["path"],
                        chain_id=entry.get("chain_id"),
                        chain_namespace=entry["chain_namespace"],
                        model_index=entry["model_index"],
                        sequence=entry.get("sequence"),
                    ),
                    sequence_id=entry["sequence_id"],
                )
            except StructureMappingError as error:
                exclusions[error.reason] += 1
                rejections.write(
                    json.dumps(
                        {
                            "sequence_id": entry["sequence_id"],
                            "path": str(entry["path"]),
                            "reason": error.reason,
                            "detail": str(error),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                continue
            record = canonical_record(
                structure,
                source_namespace=entry["source_namespace"],
                source_accession=entry["source_accession"],
                parent_ids=entry.get("parent_ids", ()),
            )
            selection = _resolved_selection(record["identity"])
            if selection in selections:
                raise ValueError(
                    "Duplicate resolved raw-revision/model/chain selection"
                )
            selections.add(selection)
            members.append(
                {
                    key: record[key]
                    for key in (
                        "canonical_id",
                        "content_sha256",
                        "residue_map_sha256",
                        "parent_ids",
                    )
                }
            )
            records.write(json.dumps(record, allow_nan=False) + "\n")
    summary = {
        "schema_version": 1,
        "status": "complete",
        **{
            name + "_sha256": file_sha256(staging / (name + ".jsonl"))
            for name in ("records", "rejections")
        },
        "population_sha256": _population(members),
        "requested_input_count": requested,
        "canonical_record_count": len(members),
        "parser_rejection_count": sum(exclusions.values()),
        "exclusions": dict(exclusions),
    }
    (staging / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.rename(staging, destination)
    return summary
