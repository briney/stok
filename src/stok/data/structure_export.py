"""Bounded polymer-aligned structure-token datasets, published by rename."""

from collections import Counter
from collections.abc import Mapping
import json
import os
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
)
from .dataset import parquet_shards
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
            # Only settings that can change token values; not device or GPU model.
            "numerical": {
                key: provenance["execution"][key]
                for key in (
                    "dtype",
                    "matmul_precision",
                    "cuda_matmul_allow_tf32",
                    "cudnn_allow_tf32",
                )
            },
        }
    )


def validate_structure_dataset(directory: str | Path) -> dict[str, Any]:
    """Check completion and shard inventory of a dataset this module wrote."""
    directory = Path(directory)
    summary = json.loads((directory / "manifest.json").read_text())
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != 2
        or summary.get("status") != "complete"
    ):
        raise ValueError("Dataset has no valid completion manifest")
    shards = {shard["path"]: shard for shard in summary["shards"]}
    if not shards or shards.keys() != {p.name for p in parquet_shards(directory)}:
        raise ValueError("Dataset shard inventory disagrees")
    for name, shard in shards.items():
        if pq.ParquetFile(directory / name).metadata.num_rows != shard["row_count"]:
            raise ValueError(f"Shard row count disagrees: {name}")
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
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            staging / name,
            compression="zstd",
        )
        shards.append(
            {
                "path": name,
                "sha256": file_sha256(staging / name),
                "row_count": len(rows),
                "residue_count": sum(len(row["sequence"]) for row in rows),
                "null_count": sum(row["structure_tokens"].count(None) for row in rows),
            }
        )
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

    # A failed run leaves only the hidden staging directory behind.
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
    os.rename(staging, destination)
    return summary
