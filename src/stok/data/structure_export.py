"""Bounded, atomically published polymer-aligned structure-token datasets."""

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import ctypes
from dataclasses import replace
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

from .dataset import TokenizedDataset, _structure_provenance
from .structure_encoding import (
    StructureExclusion,
    iter_structure_manifest,
    prepare_structure,
    tokenize_structures,
    validate_structure_limits,
)
from ..models.gcp_vqvae import GCPVQTokenizer
from ..utils.pretrained import (
    file_sha256,
    inference_metadata,
    json_sha256,
    state_sha256,
)
from ..utils.structure_parser import StructureMappingError, parse_polymer_structure


_QUALIFIED_EXECUTION_FIELDS = (
    "python",
    "dependencies",
    "device",
    "dtype",
    "torch_cuda",
    "torch_hip",
    "accelerator",
    "attention",
    "matmul_precision",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "sdp_backends_enabled",
)

_FILTER_FIELDS = ("min_length", "max_length", "max_missing_ratio", "max_missing_block")


def validate_structure_policy(
    policy: Mapping[str, Any], *, device: torch.device
) -> dict[str, Any]:
    """Validate preparation contracts and configurable length/coverage admission."""
    if not isinstance(policy, Mapping):
        raise ValueError("Structure policy must be a JSON object")
    policy = deepcopy(dict(policy))
    fixed = {
        "schema_version": 1,
        "sequence_source": "deposited_or_supplied",
        "required_atoms": ["N", "CA", "C", "O"],
        "graph_context": "independent_singleton",
        "context_scope": "full_chain",
        "cropping": "none",
        "dtype": "float32",
    }
    fields = (
        set(fixed)
        | set(_FILTER_FIELDS)
        | {
            "sequence_mode",
            "imputation",
            "allow_observed_sequence",
            "device",
            "stok_revision",
        }
    )
    if fields - policy.keys() or policy.keys() - fields - {
        "implementation_sha256",
        "qualification",
    }:
        raise ValueError("Unsupported or incomplete structure policy fields")
    if any(
        policy[key] != value or type(policy[key]) is not type(value)
        for key, value in fixed.items()
    ):
        raise ValueError("Unsupported structure policy settings")
    validate_structure_limits(**{key: policy[key] for key in _FILTER_FIELDS})
    if (
        type(policy["allow_observed_sequence"]) is not bool
        or not isinstance(policy["sequence_mode"], str)
        or policy["sequence_mode"] not in {"native", "unknown", "polymer"}
        or not isinstance(policy["imputation"], str)
        or policy["imputation"] not in {"reference", "linear", "observed_only"}
        or not isinstance(policy["stok_revision"], str)
        or not policy["stok_revision"]
    ):
        raise ValueError("Invalid structure policy settings")
    try:
        if not isinstance(policy["device"], str):
            raise ValueError("Policy device must be a string")
        requested = torch.device(policy["device"])
    except (TypeError, RuntimeError) as error:
        raise ValueError("Invalid policy device") from error
    if requested.type != device.type or requested.index not in {None, device.index}:
        raise ValueError("Policy device does not match tokenizer device")
    digest = policy.get("implementation_sha256")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid policy implementation digest")
    if "qualification" in policy:
        qualification = policy["qualification"]
        if (
            not isinstance(qualification, dict)
            or qualification.keys() != {"tokenizer_sha256", "execution"}
            or not isinstance(qualification["tokenizer_sha256"], str)
            or len(qualification["tokenizer_sha256"]) != 64
            or any(
                c not in "0123456789abcdef" for c in qualification["tokenizer_sha256"]
            )
            or not isinstance(qualification["execution"], dict)
            or qualification["execution"].keys() != set(_QUALIFIED_EXECUTION_FIELDS)
        ):
            raise ValueError("Invalid policy qualification constraints")
    json_sha256(policy)
    return policy


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
    fields = [
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


def _publish_directory(staging: Path, destination: Path) -> None:
    """Publish without replacing even an empty directory created concurrently."""
    if os.name == "nt":
        os.rename(staging, destination)  # Windows rename refuses an existing target.
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
        or summary.get("schema_version") != 1
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
    if set(names) != {path.name for path in directory.glob("*.parquet")}:
        raise ValueError("Dataset shard inventory disagrees")
    for shard in summary["shards"]:
        _validate_shard(directory / shard["path"], shard, encoded)
    for name in ("row_count", "residue_count", "null_count"):
        if summary[name] != sum(shard[name] for shard in summary["shards"]):
            raise ValueError("Dataset counts disagree")
    if file_sha256(directory / "rejections.jsonl") != summary["rejections_sha256"]:
        raise ValueError("Corrupt rejection report")
    if "inputs_sha256" in summary and (
        file_sha256(directory / "inputs.jsonl") != summary["inputs_sha256"]
    ):
        raise ValueError("Corrupt input inventory")
    return summary


@torch.inference_mode()
def write_structure_dataset(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: GCPVQTokenizer,
    policy: Mapping[str, Any],
    batch_size: int = 1,
    rows_per_shard: int = 1000,
    include_coordinates: bool = True,
) -> dict[str, Any]:
    for name, value in (("batch_size", batch_size), ("rows_per_shard", rows_per_shard)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(include_coordinates) is not bool:
        raise ValueError("include_coordinates must be boolean")
    device = next(tokenizer.parameters()).device
    policy = validate_structure_policy(policy, device=device)
    limits = {key: policy[key] for key in _FILTER_FIELDS}
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
    manifest = Path(manifest).resolve()
    manifest_hash = file_sha256(manifest)
    environment = inference_metadata(device)
    if (
        policy.get("implementation_sha256", environment["implementation_sha256"])
        != environment["implementation_sha256"]
    ):
        raise ValueError(
            "Policy implementation digest does not match this tokenizer pipeline"
        )
    identity = _tokenizer_identity(tokenizer)
    if "qualification" in policy:
        qualification = policy["qualification"]
        if qualification["tokenizer_sha256"] != json_sha256(identity):
            raise ValueError("Policy qualification tokenizer does not match this model")
        if json_sha256(qualification["execution"]) != json_sha256(
            {key: environment[key] for key in _QUALIFIED_EXECUTION_FIELDS}
        ):
            raise ValueError(
                "Policy qualification execution does not match this runtime"
            )
    provenance = {
        "schema_version": 1,
        "tokenizer": identity,
        "policy": policy,
        "execution": environment,
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
        json.dumps({"destination": str(destination), "input_manifest": str(manifest)})
        + "\n"
    )
    rows, pending, shards = [], [], []
    exclusions = Counter()
    input_count = 0
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
            **limits,
        )
        if len(ids) != len(pending):
            raise RuntimeError("Tokenizer omitted input chains")
        for structure, indices in zip(pending, ids):
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
                "sequence_id": structure.sequence_id,
                "sequence": structure.sequence,
                "structure_tokens": [
                    None if index == -1 else index for index in indices.tolist()
                ],
                "residue_map": [dict(entry) for entry in structure.residue_map],
                "source": dict(structure.source),
            }
            if include_coordinates:
                row["coordinates"] = structure.coordinates[:, :3].tolist()
            rows.append(row)
            if len(rows) == rows_per_shard:
                flush_shard()
        pending.clear()

    try:
        with (
            (staging / "rejections.jsonl").open("w") as rejected,
            (staging / "inputs.jsonl").open("w") as inputs,
        ):
            for entry in iter_structure_manifest(manifest):
                input_count += 1
                inputs.write(json.dumps(entry, allow_nan=False) + "\n")
                try:
                    before = file_sha256(entry["path"])
                    structure = replace(
                        parse_polymer_structure(
                            **{
                                key: value
                                for key, value in entry.items()
                                if key != "sequence_id"
                            },
                            allow_observed_sequence=policy["allow_observed_sequence"],
                        ),
                        sequence_id=entry["sequence_id"],
                    )
                    if before != structure.source["sha256"]:
                        raise RuntimeError("Input structure changed during parsing")
                    prepare_structure(
                        structure,
                        sequence_mode=policy["sequence_mode"],
                        imputation=policy["imputation"],
                        **limits,
                    )
                except (StructureMappingError, StructureExclusion) as error:
                    exclusions[error.reason] += 1
                    rejected.write(
                        json.dumps(
                            {**entry, "reason": error.reason, "detail": str(error)},
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    continue
                pending.append(structure)
                if len(pending) == batch_size:
                    flush_batch()
            flush_batch()
            flush_shard()
        if not shards:
            raise ValueError("All input structures were rejected; no dataset produced")
        if file_sha256(manifest) != manifest_hash or identity != _tokenizer_identity(
            tokenizer
        ):
            raise RuntimeError("Manifest or tokenizer state changed during generation")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "provenance": provenance,
            "tokenizer_sha256": json_sha256(identity),
            "policy_sha256": json_sha256(policy),
            "input_manifest_sha256": manifest_hash,
            "inputs_sha256": file_sha256(staging / "inputs.jsonl"),
            "input_count": input_count,
            "row_order": "accepted_input_manifest_order",
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
        _publish_directory(staging, destination)
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
