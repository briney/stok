"""Bounded, atomically published polymer-aligned structure-token datasets."""

from collections import Counter
from collections.abc import Mapping
import json
import resource
from pathlib import Path
import tempfile
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .canonical import (
    iter_canonical_records,
    record_to_polymer,
    validate_canonical_dataset,
    publish_directory,
)
from .dataset import TokenizedDataset, _structure_provenance, parquet_shards
from .structure_encoding import (
    StructureExclusion,
    prepare_structure,
    tokenize_structures,
)
from ..models.gcp_vqvae import GCPVQTokenizer
from ..utils.pretrained import (
    file_sha256,
    inference_metadata,
    json_sha256,
    state_sha256,
)


def structure_export_schema(
    *, include_coordinates: bool, metadata: Mapping[bytes, bytes]
) -> pa.Schema:
    residue = pa.struct(
        [
            (
                name,
                pa.int64()
                if name in {"polymer_position", "label_seq_id", "author_residue_id"}
                else pa.string(),
            )
            for name in (
                "polymer_position",
                "monomer_id",
                "observed_monomer_id",
                "observed_one_letter",
                "label_seq_id",
                "author_residue_id",
                "insertion_code",
                "selected_altloc",
            )
        ]
    )
    source = pa.struct(
        [
            (
                name,
                pa.int64()
                if name in {"model_index", "model_serial_id"}
                else pa.string(),
            )
            for name in (
                "path",
                "sha256",
                "label_chain_id",
                "author_chain_id",
                "entity_id",
                "model_index",
                "model_serial_id",
                "sequence_source",
            )
        ]
    )
    identity = pa.struct(
        [
            (
                name,
                pa.int64()
                if name
                in {"canonical_identity_version", "model_index", "model_serial_id"}
                else pa.string(),
            )
            for name in (
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
        ]
    )
    fields = [
        pa.field("canonical_id", pa.string(), nullable=False),
        pa.field("canonical_content_sha256", pa.string(), nullable=False),
        pa.field("residue_map_sha256", pa.string(), nullable=False),
        pa.field("canonical_identity", identity, nullable=False),
        pa.field("parent_ids", pa.list_(pa.string()), nullable=False),
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
        pa.field("structure_tokens", pa.list_(pa.int64()), nullable=False),
        pa.field("residue_map", pa.list_(residue), nullable=False),
        pa.field("source", source, nullable=False),
    ]
    if include_coordinates:
        fields.append(
            pa.field(
                "coordinates",
                pa.list_(pa.list_(pa.list_(pa.float32()))),
                nullable=False,
            )
        )
    return pa.schema(fields, metadata=metadata)


def _tokenizer_identity(tokenizer):
    encoder_config = {
        key: tokenizer.config[key]
        for key in ("features", "gcp", "encoder", "max_length")
    }
    return {
        "encoder_state_sha256": state_sha256(tokenizer.encoder.state_dict()),
        "quantizer_state_sha256": state_sha256(tokenizer.quantizer.state_dict()),
        "encoder_config": encoder_config,
        "encoder_config_sha256": json_sha256(encoder_config),
        "quantizer_config": tokenizer.config["quantizer"],
        "quantizer_config_sha256": json_sha256(tokenizer.config["quantizer"]),
        "codebook_sha256": state_sha256({"codebook": tokenizer.quantizer.codebook}),
        "codebook_size": tokenizer.quantizer.codebook.size(0),
        "source_revision": tokenizer.config["source"]["commit"],
    }


def representation_sha256(provenance: Mapping[str, Any]) -> str:
    """Logical representation identity excludes paths, layout and runtime audit."""
    return json_sha256(
        {
            "schema_version": 2,
            "canonical_population_sha256": provenance["canonical_population_sha256"],
            "tokenizer": provenance["tokenizer"],
            "policy": provenance["policy"],
            "numerical": {
                key: value
                for key, value in provenance["execution"].items()
                if key not in {"stok_revision", "source_files"}
            },
        }
    )


def _validate_shard(path, expected, provenance):
    parquet = pq.ParquetFile(path)
    if (
        parquet.metadata.num_rows != expected["row_count"]
        or file_sha256(path) != expected["sha256"]
    ):
        raise ValueError(f"Corrupt shard: {path}")
    metadata = parquet.schema_arrow.metadata or {}
    if metadata.get(b"stok.provenance") != provenance:
        raise ValueError(f"Incompatible shard provenance: {path}")
    if not parquet.schema_arrow.equals(
        structure_export_schema(
            include_coordinates="coordinates" in parquet.schema_arrow.names,
            metadata=metadata,
        ),
        check_metadata=False,
    ):
        raise ValueError(f"Invalid schema-2 representation fields: {path}")
    _structure_provenance(path, parquet.schema_arrow)
    reader = TokenizedDataset(str(path), max_length=1280)
    for i in range(len(reader)):
        reader[i]  # Existing source schema, row shape and nullable-label checks.
    rows = parquet.read(
        columns=["sequence", "structure_tokens", "residue_map"]
    ).to_pylist()
    codebook_size = json.loads(provenance)["tokenizer"]["codebook_size"]
    residues, nulls = 0, 0
    for row in rows:
        if any(
            index is not None and index >= codebook_size
            for index in row["structure_tokens"]
        ):
            raise ValueError(f"Out-of-codebook token: {path}")
        length = len(row["sequence"])
        if len(row["residue_map"]) != length or [
            entry["polymer_position"] for entry in row["residue_map"]
        ] != list(range(length)):
            raise ValueError(f"Misaligned residue map: {path}")
        residues += length
        nulls += row["structure_tokens"].count(None)
    if (residues, nulls) != (expected["residue_count"], expected["null_count"]):
        raise ValueError(f"Shard counts disagree: {path}")


def validate_structure_dataset(directory: str | Path) -> dict[str, Any]:
    """Verify a completed dataset's hashes, metadata and existing reader contract."""
    directory = Path(directory)
    summary = json.loads((directory / "manifest.json").read_text())
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "complete"
        or type(summary.get("schema_version")) is not int
        or summary.get("schema_version") != 2
        or not summary.get("shards")
    ):
        raise ValueError("Dataset has no valid completion manifest")
    provenance = summary["provenance"]
    if (
        json_sha256(provenance["tokenizer"]) != summary["tokenizer_sha256"]
        or json_sha256(provenance["policy"]) != summary["policy_sha256"]
    ):
        raise ValueError("Dataset provenance digest mismatch")
    encoded = json.dumps(provenance, sort_keys=True, allow_nan=False).encode()
    names = [shard["path"] for shard in summary["shards"]]
    if len(names) != len(set(names)) or any(
        Path(name).name != name or not name.endswith(".parquet") for name in names
    ):
        raise ValueError("Invalid shard inventory")
    if set(names) != {path.name for path in parquet_shards(directory)}:
        raise ValueError("Dataset shard inventory disagrees")
    for shard in summary["shards"]:
        _validate_shard(directory / shard["path"], shard, encoded)
    for name in ("row_count", "residue_count", "null_count"):
        if (
            type(summary.get(name)) is not int
            or any(
                type(shard.get(name)) is not int or shard[name] < 0
                for shard in summary["shards"]
            )
            or summary[name] != sum(shard[name] for shard in summary["shards"])
        ):
            raise ValueError("Dataset counts disagree")
    if file_sha256(directory / "rejections.jsonl") != summary["rejections_sha256"]:
        raise ValueError("Corrupt rejection report")
    canonical_dir = Path(summary["canonical_directory"])
    if not canonical_dir.is_absolute():
        canonical_dir = directory / canonical_dir
    canonical_summary = validate_canonical_dataset(canonical_dir)
    if (
        file_sha256(canonical_dir / "manifest.json")
        != summary["canonical_manifest_sha256"]
        or canonical_summary["population_sha256"]
        != summary["canonical_population_sha256"]
        or provenance["canonical_population_sha256"]
        != summary["canonical_population_sha256"]
    ):
        raise ValueError("Canonical inventory reference disagrees")
    if representation_sha256(provenance) != summary["representation_sha256"]:
        raise ValueError("Representation digest mismatch")
    # ponytail: compact inventory metadata stays in RAM; index JSONL for larger corpora.
    records: dict[str, dict[str, Any]] = {
        record["canonical_id"]: {
            **{
                key: record[key]
                for key in (
                    "identity",
                    "sequence",
                    "content_sha256",
                    "residue_map_sha256",
                    "parent_ids",
                    "provenance",
                )
            },
            "coordinates_sha256": json_sha256(
                [residue[:3] for residue in record["coordinates"]]
            ),
            "token_mask_sha256": json_sha256(
                [all(mask) for mask in record["atom_mask"]]
            ),
        }
        for record in iter_canonical_records(canonical_dir)
    }
    admitted = set()
    for shard in summary["shards"]:
        for row in pq.read_table(directory / shard["path"]).to_pylist():
            canonical_id = row["canonical_id"]
            if canonical_id not in records or canonical_id in admitted:
                raise ValueError("Unknown or duplicate admitted canonical ID")
            admitted.add(canonical_id)
            if (
                provenance["policy"].get("allow_observed_sequence") is False
                and row["source"]["sequence_source"] == "observed"
            ):
                raise ValueError(
                    "Observed sequence source is forbidden by representation policy"
                )
            record = records[canonical_id]
            expected = {
                "canonical_content_sha256": record["content_sha256"],
                "residue_map_sha256": record["residue_map_sha256"],
                "canonical_identity": record["identity"],
                "parent_ids": record["parent_ids"],
                "sequence": record["sequence"],
                "source": record["provenance"]["source"],
                "sequence_id": record["provenance"]["sequence_id"],
            }
            if any(row[key] != value for key, value in expected.items()):
                raise ValueError("Export row disagrees with canonical originals")
            if json_sha256(row["residue_map"]) != record["residue_map_sha256"]:
                raise ValueError(
                    "Export residue map disagrees with canonical originals"
                )
            if (
                "coordinates" in row
                and json_sha256(row["coordinates"]) != record["coordinates_sha256"]
            ):
                raise ValueError("Export coordinates disagree with canonical originals")
            if (
                json_sha256([value is not None for value in row["structure_tokens"]])
                != record["token_mask_sha256"]
            ):
                raise ValueError(
                    "Export null tokens disagree with original observations"
                )
    rejections = [
        json.loads(line)
        for line in (directory / "rejections.jsonl").read_text().splitlines()
    ]
    rejected = {row.get("canonical_id") for row in rejections}
    if (
        len(rejected) != len(rejections)
        or any(
            not isinstance(row.get("reason"), str) or not row["reason"]
            for row in rejections
        )
        or admitted & rejected
        or admitted | rejected != records.keys()
    ):
        raise ValueError("Representation admission/rejection correspondence disagrees")
    for key in (
        "requested_input_count",
        "canonical_record_count",
        "parser_rejection_count",
    ):
        if type(summary.get(key)) is not int or summary[key] != canonical_summary[key]:
            raise ValueError("Canonical request counts disagree")
    if (
        any(
            type(summary.get(key)) is not int
            for key in ("representation_rejection_count", "rejection_count")
        )
        or summary["row_count"] != len(admitted)
        or summary["representation_rejection_count"] != len(rejections)
        or summary["rejection_count"] != len(rejections)
        or summary["exclusions"] != dict(Counter(row["reason"] for row in rejections))
    ):
        raise ValueError("Representation counts disagree")
    return summary


@torch.inference_mode()
def write_structure_dataset(
    canonical_dir: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: GCPVQTokenizer,
    batch_size: int = 1,
    rows_per_shard: int = 1000,
    include_coordinates: bool = False,
) -> dict[str, Any]:
    """Export full chains with the fixed training-native-reference policy."""
    for name, value in (("batch_size", batch_size), ("rows_per_shard", rows_per_shard)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(include_coordinates) is not bool:
        raise ValueError("include_coordinates must be boolean")
    canonical_dir = Path(canonical_dir).resolve()
    canonical_summary = validate_canonical_dataset(canonical_dir)
    if not canonical_summary["canonical_record_count"]:
        raise ValueError("Cannot export an empty canonical population")
    device = next(tokenizer.parameters()).device
    policy: dict[str, Any] = {
        "schema_version": 1,
        "name": "training-native-reference",
        "sequence_source": "deposited_or_supplied",
        "allow_observed_sequence": False,
        "sequence_mode": "native",
        "required_atoms": ["N", "CA", "C", "O"],
        "imputation": "reference",
        "graph_context": "independent_singleton",
        "context_scope": "full_chain",
        "cropping": "none",
        "dtype": "float32",
        "min_length": 25,
        "max_length": 1280,
        "max_missing_ratio": None,
        "max_missing_block": None,
    }
    if (
        tokenizer.training
        or tokenizer.encoder.max_length != 1280
        or any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in tokenizer.state_dict().values()
        )
    ):
        raise ValueError(
            "Dataset production requires an eval-mode FP32 tokenizer with max_length=1280"
        )
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    environment = inference_metadata(device)
    identity = _tokenizer_identity(tokenizer)
    provenance = {
        "schema_version": 2,
        "tokenizer": identity,
        "policy": policy,
        "execution": environment,
        "canonical_population_sha256": canonical_summary["population_sha256"],
    }
    metadata = {
        b"stok.provenance": json.dumps(
            provenance, sort_keys=True, allow_nan=False
        ).encode(),
        b"stok.tokenizer_sha256": json_sha256(identity).encode(),
        b"stok.policy_sha256": json_sha256(policy).encode(),
    }
    schema = structure_export_schema(
        include_coordinates=include_coordinates, metadata=metadata
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    (staging / "INCOMPLETE.json").write_text(
        json.dumps(
            {"destination": str(destination), "canonical_directory": str(canonical_dir)}
        )
        + "\n"
    )
    rows, pending, shards = [], [], []
    exclusions = Counter()
    pending_records = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def flush_shard():
        if not rows:
            return
        name = f"part-{len(shards):06d}.parquet"
        temporary = staging / (name + ".partial")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
        )
        temporary.rename(staging / name)
        shard = {
            "path": name,
            "sha256": file_sha256(staging / name),
            "row_count": len(rows),
            "residue_count": sum(len(row["sequence"]) for row in rows),
            "null_count": sum(row["structure_tokens"].count(None) for row in rows),
        }
        _validate_shard(staging / name, shard, metadata[b"stok.provenance"])
        shards.append(shard)
        rows.clear()

    def flush_batch():
        if not pending:
            return
        ids = tokenize_structures(
            tokenizer,
            pending,
            sequence_mode=policy["sequence_mode"],
            imputation=policy["imputation"],
            min_length=25,
            max_length=1280,
            max_missing_ratio=None,
            max_missing_block=None,
        )
        if len(ids) != len(pending):
            raise RuntimeError("Tokenizer omitted input chains")
        for structure, record, indices in zip(pending, pending_records, ids):
            length = len(structure.sequence)
            if (
                indices.shape != (length,)
                or indices.dtype != torch.int64
                or (indices < -1).any()
                or (indices >= identity["codebook_size"]).any()
                or not torch.equal(
                    indices >= 0, torch.tensor(structure.atom_mask.all(-1))
                )
            ):
                raise ValueError(
                    f"{structure.sequence_id}: invalid model token IDs or availability"
                )
            row = {
                "canonical_id": record["canonical_id"],
                "canonical_content_sha256": record["content_sha256"],
                "residue_map_sha256": record["residue_map_sha256"],
                "canonical_identity": record["identity"],
                "parent_ids": record["parent_ids"],
                "sequence_id": structure.sequence_id,
                "sequence": structure.sequence,
                "structure_tokens": [
                    None if index == -1 else index for index in indices.tolist()
                ],
                "residue_map": [dict(entry) for entry in structure.residue_map],
                "source": dict(structure.source),
            }
            if include_coordinates:
                row["coordinates"] = [residue[:3] for residue in record["coordinates"]]
            rows.append(row)
            if len(rows) == rows_per_shard:
                flush_shard()
        pending.clear()
        pending_records.clear()

    try:
        with (staging / "rejections.jsonl").open("w") as rejected:
            for record in iter_canonical_records(canonical_dir):
                structure = record_to_polymer(record)
                try:
                    if structure.source["sequence_source"] == "observed":
                        raise StructureExclusion(
                            "sequence_metadata_missing", structure.sequence_id
                        )
                    prepare_structure(
                        structure,
                        sequence_mode=policy["sequence_mode"],
                        imputation=policy["imputation"],
                        min_length=25,
                        max_length=1280,
                        max_missing_ratio=None,
                        max_missing_block=None,
                    )
                except StructureExclusion as error:
                    exclusions[error.reason] += 1
                    rejected.write(
                        json.dumps(
                            {
                                "canonical_id": record["canonical_id"],
                                "reason": error.reason,
                                "detail": str(error),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    continue
                pending.append(structure)
                pending_records.append(record)
                if len(pending) == batch_size:
                    flush_batch()
            flush_batch()
            flush_shard()
        if not shards:
            raise ValueError("All input structures were rejected; no dataset produced")
        if validate_canonical_dataset(
            canonical_dir
        ) != canonical_summary or identity != _tokenizer_identity(tokenizer):
            raise RuntimeError(
                "Canonical inventory or tokenizer state changed during generation"
            )
        summary = {
            "schema_version": 2,
            "status": "complete",
            "provenance": provenance,
            "tokenizer_sha256": json_sha256(identity),
            "policy_sha256": json_sha256(policy),
            "canonical_directory": str(canonical_dir),
            "canonical_manifest_sha256": file_sha256(canonical_dir / "manifest.json"),
            "canonical_population_sha256": canonical_summary["population_sha256"],
            "requested_input_count": canonical_summary["requested_input_count"],
            "canonical_record_count": canonical_summary["canonical_record_count"],
            "parser_rejection_count": canonical_summary["parser_rejection_count"],
            "representation_rejection_count": sum(exclusions.values()),
            "representation_sha256": representation_sha256(provenance),
            "row_order": "accepted_canonical_record_order",
            "shards": shards,
            **{
                name: sum(shard[name] for shard in shards)
                for name in ("row_count", "residue_count", "null_count")
            },
            "rejection_count": sum(exclusions.values()),
            "exclusions": dict(exclusions),
            "rejections_sha256": file_sha256(staging / "rejections.jsonl"),
            "runtime_seconds": time.perf_counter() - start,
            "batch_size": batch_size,
            "rows_per_shard": rows_per_shard,
            "include_coordinates": include_coordinates,
            "grouping_context": "independent_singleton",
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "peak_device_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
        (staging / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        validate_structure_dataset(staging)
        (staging / "INCOMPLETE.json").unlink()
        publish_directory(staging, destination)
        return summary
    except BaseException as error:
        for partial in staging.glob("*.partial"):
            partial.unlink(missing_ok=True)
        (staging / "manifest.json").unlink(missing_ok=True)
        (staging / "FAILED.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "destination": str(destination),
                    "completed_shards": len(shards),
                }
            )
            + "\n"
        )
        raise
