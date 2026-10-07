"""Frozen original polymer observations, independent of their representations."""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
import ctypes
from dataclasses import replace
import json
import math
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

IDENTITY_FIELDS = (
    "canonical_identity_version",
    "record_kind",
    "source_namespace",
    "source_accession",
    "source_revision_sha256",
    "model_index",
    "model_serial_id",
    "chain_namespace",
    "chain_id",
)


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def publish_directory(staging: Path, destination: Path) -> None:
    """Publish without replacing even an empty directory created concurrently."""
    if os.name == "nt":
        os.rename(staging, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-replace directory publication requires renameat2")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


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
    validate_canonical_record(record)
    return record


def validate_canonical_record(record: Mapping[str, Any]) -> None:
    """Reject contradictory identity, original observations, mapping or lineage."""
    try:
        if set(record) != {
            "canonical_id",
            "identity",
            "sequence",
            "coordinates",
            "atom_mask",
            "residue_map",
            "content_sha256",
            "residue_map_sha256",
            "parent_ids",
            "provenance",
        }:
            raise ValueError("Invalid canonical record fields")
        identity = record["identity"]
        if not isinstance(identity, dict) or set(identity) != set(IDENTITY_FIELDS):
            raise ValueError("Invalid canonical identity fields")
        if (
            type(identity["canonical_identity_version"]) is not int
            or identity["canonical_identity_version"] != 1
            or identity["record_kind"] not in {"structure", "sequence"}
        ):
            raise ValueError("Unsupported canonical identity version/kind")
        if not all(
            _identifier(identity[key])
            for key in ("source_namespace", "source_accession")
        ) or not _digest(identity["source_revision_sha256"]):
            raise ValueError("Invalid explicit source identity/revision")
        structural = ("model_index", "model_serial_id", "chain_namespace", "chain_id")
        if identity["record_kind"] == "sequence":
            if any(identity[key] is not None for key in structural):
                raise ValueError("Sequence identity must have absent structural fields")
        elif (
            type(identity["model_index"]) is not int
            or identity["model_index"] < 0
            or type(identity["model_serial_id"]) is not int
            or identity["chain_namespace"] not in {"label", "author"}
            or not _identifier(identity["chain_id"])
        ):
            raise ValueError("Invalid resolved model/chain identity")
        if not _digest(record["canonical_id"]) or record["canonical_id"] != json_sha256(
            identity
        ):
            raise ValueError("Canonical ID mismatch")
        sequence = record["sequence"]
        if (
            not isinstance(sequence, str)
            or not sequence
            or any(aa not in "ACDEFGHIKLMNPQRSTVWYX" for aa in sequence)
        ):
            raise ValueError("Invalid canonical sequence")
        length = len(sequence)
        for key in ("coordinates", "atom_mask", "residue_map"):
            if not isinstance(record[key], list) or len(record[key]) != length:
                raise ValueError(
                    "Canonical observations/map must match sequence length"
                )
        for residue, mask in zip(record["coordinates"], record["atom_mask"]):
            if (
                not isinstance(residue, list)
                or len(residue) != 4
                or not isinstance(mask, list)
                or len(mask) != 4
                or any(type(flag) is not bool for flag in mask)
            ):
                raise ValueError(
                    "Canonical observations require four atoms and boolean masks"
                )
            for values, present in zip(residue, mask):
                if not isinstance(values, list) or len(values) != 3:
                    raise ValueError("Canonical atoms require three coordinates")
                if present:
                    if identity["record_kind"] == "sequence" or any(
                        type(value) not in {int, float}
                        or not math.isfinite(value)
                        or not np.isfinite(np.float32(value))
                        or float(np.float32(value)) != value
                        for value in values
                    ):
                        raise ValueError(
                            "Canonical observations must be original finite float32 values"
                        )
                elif values != [None, None, None]:
                    raise ValueError(
                        "Absent observations must be null and agree with atom mask"
                    )
        for position, entry in enumerate(record["residue_map"]):
            if not isinstance(entry, dict) or set(entry) != {
                "polymer_position",
                "monomer_id",
                "observed_monomer_id",
                "observed_one_letter",
                "label_seq_id",
                "author_residue_id",
                "insertion_code",
                "selected_altloc",
            }:
                raise ValueError(
                    "Canonical residue map requires complete parser correspondence"
                )
            if (
                not isinstance(entry, dict)
                or type(entry.get("polymer_position")) is not int
                or entry["polymer_position"] != position
            ):
                raise ValueError(
                    "Canonical residue map must contain every polymer position"
                )
            for name in ("label_seq_id", "author_residue_id"):
                if entry.get(name) is not None and type(entry[name]) is not int:
                    raise ValueError("Residue identifiers must be integers or null")
            for name in (
                "monomer_id",
                "observed_monomer_id",
                "observed_one_letter",
                "insertion_code",
                "selected_altloc",
            ):
                if entry.get(name) is not None and not isinstance(entry[name], str):
                    raise ValueError("Residue annotations must be strings or null")
        parents = record["parent_ids"]
        if (
            not isinstance(parents, list)
            or any(not _digest(parent) for parent in parents)
            or parents != sorted(set(parents))
            or record["canonical_id"] in parents
        ):
            raise ValueError("Parent IDs must be unique sorted explicit canonical IDs")
        for key in ("content_sha256", "residue_map_sha256"):
            if not _digest(record[key]):
                raise ValueError("Malformed canonical digest")
        if record["residue_map_sha256"] != json_sha256(record["residue_map"]) or record[
            "content_sha256"
        ] != json_sha256(
            {
                key: record[key]
                for key in ("sequence", "coordinates", "atom_mask", "residue_map")
            }
        ):
            raise ValueError("Canonical content/map digest mismatch")
        provenance = record["provenance"]
        source = provenance["source"]
        if (
            not _identifier(provenance["sequence_id"])
            or not isinstance(source["path"], str)
            or not source["path"]
            or source["sha256"] != identity["source_revision_sha256"]
            or source.get("sequence_source")
            not in {
                "entity_poly_seq",
                "poly_seq_scheme",
                "seqres",
                "supplied",
                "observed",
            }
        ):
            raise ValueError("Canonical source declarations disagree")
        if identity["record_kind"] == "structure":
            namespace = (
                "label" if source.get("label_chain_id") is not None else "author"
            )
            if (
                namespace != identity["chain_namespace"]
                or source.get(namespace + "_chain_id") != identity["chain_id"]
                or any(
                    source.get(key) != identity[key] or type(source.get(key)) is not int
                    for key in ("model_index", "model_serial_id")
                )
            ):
                raise ValueError(
                    "Parser correspondence disagrees with canonical identity"
                )
        json.dumps(record, allow_nan=False)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed canonical record") from error


def record_to_polymer(record: Mapping[str, Any]) -> PolymerStructure:
    validate_canonical_record(record)
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
    directory = Path(directory)
    summary = json.loads((directory / "manifest.json").read_text())
    if (
        not isinstance(summary, dict)
        or type(summary.get("schema_version")) is not int
        or summary.get("schema_version") != 1
        or summary.get("status") != "complete"
    ):
        raise ValueError("Canonical inventory has no valid completion manifest")
    for name in ("records", "inputs", "rejections"):
        if file_sha256(directory / (name + ".jsonl")) != summary.get(name + "_sha256"):
            raise ValueError(f"Corrupt canonical {name} inventory")
    # ponytail: inventory metadata stays in RAM; index JSONL if corpus size requires it.
    records = []
    ids, selections = set(), set()
    for record in _jsonl(directory / "records.jsonl"):
        validate_canonical_record(record)
        if record["canonical_id"] in ids:
            raise ValueError("Duplicate or conflicting canonical record")
        ids.add(record["canonical_id"])
        if record["identity"]["record_kind"] == "structure":
            selection = _resolved_selection(record["identity"])
            if selection in selections:
                raise ValueError(
                    "Duplicate resolved raw-revision/model/chain selection"
                )
            selections.add(selection)
        records.append(
            {
                key: record[key]
                for key in (
                    "canonical_id",
                    "identity",
                    "content_sha256",
                    "residue_map_sha256",
                    "parent_ids",
                    "provenance",
                )
            }
        )
    inputs = list(_jsonl(directory / "inputs.jsonl"))
    rejections = list(_jsonl(directory / "rejections.jsonl"))
    requests = [row.get("request_id") for row in inputs]
    if any(not _digest(request) for request in requests) or len(set(requests)) != len(
        requests
    ):
        raise ValueError("Invalid canonical request inventory")
    admitted = {
        record["provenance"].get("request_id"): record["canonical_id"]
        for record in records
    }
    rejected = {row.get("request_id") for row in rejections}
    if (
        len(admitted) != len(records)
        or len(rejected) != len(rejections)
        or admitted.keys() & rejected
        or set(requests) != admitted.keys() | rejected
    ):
        raise ValueError("Canonical parser requests do not partition inventory")
    by_id = {record["canonical_id"]: record for record in records}
    for ordinal, row in enumerate(inputs):
        if any(
            not _identifier(row.get(key))
            for key in ("sequence_id", "path", "source_namespace", "source_accession")
        ):
            raise ValueError(
                "Canonical request requires explicit string source declarations"
            )
        if "canonical_id" in row and not _digest(row["canonical_id"]):
            raise ValueError("Invalid canonical request record ID")
        sequence_only = (
            row.get("canonical_id") in by_id
            and by_id[row["canonical_id"]]["identity"]["record_kind"] == "sequence"
        )
        if sequence_only:
            if (
                row.get("model_index") is not None
                or row.get("chain_namespace") is not None
                or row.get("chain_id") is not None
            ):
                raise ValueError(
                    "Sequence-only request must have absent structural selection"
                )
        elif (
            type(row.get("model_index")) is not int
            or row["model_index"] < 0
            or not isinstance(row.get("chain_namespace"), str)
            or row["chain_namespace"] not in {"author", "label"}
        ):
            raise ValueError("Canonical request has invalid model/chain selection")
        if row.get("chain_id") is not None and not _identifier(row["chain_id"]):
            raise ValueError("Canonical request chain_id must be a string or null")
        if row.get("sequence") is not None and (
            not isinstance(row["sequence"], str)
            or not row["sequence"]
            or any(aa not in "ACDEFGHIKLMNPQRSTVWYX" for aa in row["sequence"])
        ):
            raise ValueError(
                "Canonical request sequence must contain uppercase amino acids or X"
            )
        parents = row.get("parent_ids")
        if (
            not isinstance(parents, list)
            or any(not _digest(parent) for parent in parents)
            or parents != sorted(set(parents))
        ):
            raise ValueError(
                "Canonical request parent IDs must be unique sorted digests"
            )
        if not _digest(row.get("source_revision_sha256")):
            raise ValueError("Invalid canonical request source revision")
        request = {
            key: value
            for key, value in row.items()
            if key not in {"request_id", "canonical_id"}
        }
        if row["request_id"] != json_sha256(
            {"request_version": 1, "ordinal": ordinal, "request": request}
        ):
            raise ValueError("Canonical request identity mismatch")
        if row.get("canonical_id") != admitted.get(row["request_id"]):
            raise ValueError("Canonical input/record correspondence disagrees")
        if row.get("canonical_id") is not None:
            record = by_id[row["canonical_id"]]
            identity = record["identity"]
            source = record["provenance"]["source"]
            if (
                any(
                    row.get(key) != identity[key]
                    for key in (
                        "source_namespace",
                        "source_accession",
                        "source_revision_sha256",
                        "model_index",
                    )
                )
                or row.get("parent_ids") != record["parent_ids"]
                or row.get("path") != source["path"]
                or row.get("sequence_id") != record["provenance"]["sequence_id"]
            ):
                raise ValueError("Canonical request source identity disagrees")
            if row.get("chain_id") is not None and row["chain_id"] != source.get(
                row["chain_namespace"] + "_chain_id"
            ):
                raise ValueError("Canonical request chain correspondence disagrees")
    if any(
        "canonical_id" in row or not _identifier(row.get("reason"))
        for row in rejections
    ):
        raise ValueError("Invalid parser rejection audit")
    expected = {
        "requested_input_count": len(inputs),
        "canonical_record_count": len(records),
        "parser_rejection_count": len(rejections),
    }
    if (
        any(
            type(summary.get(key)) is not int or summary[key] != value
            for key, value in expected.items()
        )
        or summary.get("population_sha256") != _population(records)
        or summary.get("exclusions")
        != dict(Counter(row["reason"] for row in rejections))
    ):
        raise ValueError("Canonical counts/population digest disagree")
    return summary


def iter_canonical_records(directory: str | Path) -> Iterator[dict[str, Any]]:
    validate_canonical_dataset(directory)
    for record in _jsonl(Path(directory) / "records.jsonl"):
        validate_canonical_record(record)
        yield record


def prepare_canonical_dataset(
    manifest: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Parse originals once, before any representation-specific admission policy."""
    manifest = Path(manifest).resolve()
    manifest_hash = file_sha256(manifest)
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    (staging / "INCOMPLETE.json").write_text(
        json.dumps({"input_manifest": str(manifest)}) + "\n"
    )
    members, selections, exclusions = [], set(), Counter()
    requested = 0
    try:
        with (
            (staging / "records.jsonl").open("w") as records,
            (staging / "inputs.jsonl").open("w") as inputs,
            (staging / "rejections.jsonl").open("w") as rejections,
        ):
            for entry in iter_structure_manifest(manifest):
                before = file_sha256(entry["path"])
                entry["source_revision_sha256"] = before
                requested += 1
                request_id = json_sha256(
                    {"request_version": 1, "ordinal": requested - 1, "request": entry}
                )
                request = {**entry, "request_id": request_id}
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
                    if file_sha256(entry["path"]) != before:
                        raise RuntimeError(
                            "Input structure changed during parsing"
                        ) from error
                    exclusions[error.reason] += 1
                    rejections.write(
                        json.dumps(
                            {
                                "request_id": request_id,
                                "reason": error.reason,
                                "detail": str(error),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    inputs.write(json.dumps(request, allow_nan=False) + "\n")
                    continue
                if (
                    before != structure.source["sha256"]
                    or file_sha256(entry["path"]) != before
                ):
                    raise RuntimeError("Input structure changed during parsing")
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
                record["provenance"]["request_id"] = request_id
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
                inputs.write(
                    json.dumps(
                        {**request, "canonical_id": record["canonical_id"]},
                        allow_nan=False,
                    )
                    + "\n"
                )
        if file_sha256(manifest) != manifest_hash:
            raise RuntimeError("Manifest changed during preparation")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "input_manifest_sha256": manifest_hash,
            **{
                name + "_sha256": file_sha256(staging / (name + ".jsonl"))
                for name in ("records", "inputs", "rejections")
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
        validate_canonical_dataset(staging)
        (staging / "INCOMPLETE.json").unlink()
        publish_directory(staging, destination)
        return summary
    except BaseException as error:
        (staging / "manifest.json").unlink(missing_ok=True)
        (staging / "FAILED.json").write_text(
            json.dumps({"error_type": type(error).__name__, "error": str(error)}) + "\n"
        )
        raise
