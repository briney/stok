from __future__ import annotations

import hashlib
import json
import re
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from gcp_vqvae import GCPVQVAE

from stok.models.structure_encoder import load_pretrained_encoder
from stok.utils.structure_loader import load_structures
from stok.utils.structure_parser import parse_structure

from .common import (
    atomic_write_json,
    capture_environment,
    configure_determinism,
    discover_inputs,
    require_cuda,
    select_shard,
)
from .metrics import compare_floats, compare_indices
from .preprocessing import build_preprocessing_audit
from .report import RunSummary, render_markdown, summarize_run
from .stages import (
    CANONICAL_STAGES,
    StageCapture,
    capture_reference_stages,
    capture_stok_stages,
    compare_stage_captures,
)
from .types import TensorComparison
from .weights import audit_weight_parity, convert_checkpoint, resolve_hf_revision

RECORD_SCHEMA_VERSION = 1
PUBLIC_STAGES = (
    "public.preprocessing",
    "public.indices",
    "public.embeddings",
)


@dataclass(frozen=True)
class QualificationConfig:
    input_dir: Path
    output_dir: Path
    preset: str = "base"
    hf_repo_id: str = "Mahdip72/gcp-vqvae-large"
    hf_revision: str | None = None
    device: str = "cuda:0"
    batch_size: int = 1
    seed: int = 0
    shard_index: int = 0
    num_shards: int = 1
    max_samples: int | None = None
    rtol: float = 1e-5
    atol: float = 1e-6
    resume: bool = True


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _failure(name: str) -> TensorComparison:
    return TensorComparison(
        name=name,
        shape=(),
        compared=0,
        mismatched=1,
        exact=False,
        passed=False,
        within_tolerance=False,
    )


def _failure_comparisons(names: Sequence[str]) -> list[TensorComparison]:
    return [_failure(name) for name in names]


def _success(name: str) -> TensorComparison:
    return TensorComparison(
        name=name,
        shape=(),
        compared=1,
        mismatched=0,
        exact=True,
        passed=True,
        within_tolerance=True,
    )


def _comparison_from_dict(payload: dict[str, Any]) -> TensorComparison:
    normalized = dict(payload)
    normalized["shape"] = tuple(normalized["shape"])
    return TensorComparison(**normalized)


def _has_stage_set(payloads: Any, expected: Sequence[str]) -> bool:
    if not isinstance(payloads, list):
        return False
    try:
        comparisons = [_comparison_from_dict(item) for item in payloads]
    except (KeyError, TypeError, ValueError):
        return False
    return tuple(item.name for item in comparisons) == tuple(expected)


def _is_usable_resume_record(
    payload: Any,
    *,
    expected_context: dict[str, Any],
) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("error_type") is None
        and payload.get("context") == expected_context
        and _has_stage_set(payload.get("core"), CANONICAL_STAGES)
        and _has_stage_set(payload.get("public"), PUBLIC_STAGES)
    )


def _move_reference_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["graph"] = batch["graph"].to(device)
    for key in ("masks", "nan_masks"):
        moved[key] = batch[key].to(device)
    return moved


def _compare_public_stok_path(
    *,
    reference: StageCapture,
    reference_sequence: str,
    source_path: Path,
    encoder: torch.nn.Module,
    device: torch.device,
) -> list[TensorComparison]:
    loaded = load_structures(
        source_path,
        max_length=encoder.max_length,
        device=device,
    )
    if loaded.sequences != [reference_sequence]:
        return _failure_comparisons(PUBLIC_STAGES)
    with torch.inference_mode():
        output = encoder(loaded.graph, loaded.mask, loaded.nan_mask)
    valid_ref = reference.tensors["valid"]
    valid_stok = output["valid"].detach().cpu()
    indices = compare_indices(
        "public.indices",
        reference.tensors["indices"],
        output["indices"].detach().cpu(),
        valid_ref,
        valid_stok,
    )
    shared = valid_ref & valid_stok
    embeddings = compare_floats(
        "public.embeddings",
        reference.tensors["embeddings"][shared],
        output["embeddings"].detach().cpu()[shared],
        rtol=0.0,
        atol=0.0,
    )
    return [_success("public.preprocessing"), indices, embeddings]


def _validate_config(config: QualificationConfig) -> None:
    if config.preset == "lite" and config.hf_repo_id == "Mahdip72/gcp-vqvae-large":
        raise ValueError("The lite preset requires --hf-repo-id Mahdip72/gcp-vqvae-lite")
    if config.batch_size != 1:
        raise ValueError("The primary qualification requires batch_size=1")
    if config.seed != 0:
        raise ValueError("The primary qualification requires seed=0")
    if config.num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if config.shard_index < 0 or config.shard_index >= config.num_shards:
        raise ValueError(f"shard_index must be in [0, {config.num_shards})")
    if config.max_samples is not None and config.max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    if config.rtol < 0 or config.atol < 0:
        raise ValueError("rtol and atol must be nonnegative")


def _require_float32(module: torch.nn.Module, *, name: str) -> None:
    tensors = [*module.parameters(), *module.buffers()]
    wrong = sorted(
        {
            str(tensor.dtype)
            for tensor in tensors
            if tensor.is_floating_point() and tensor.dtype != torch.float32
        }
    )
    if wrong:
        raise RuntimeError(f"{name} must use float32 floating tensors, found {wrong}")


def _manifest_fingerprint(records: Sequence[Any]) -> str:
    serialized = json.dumps(
        [record.to_dict() for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _record_context(
    *,
    config: QualificationConfig,
    environment: dict[str, Any],
    resolved_revision: str,
    manifest_fingerprint: str,
    weight_audit: Any,
    source_path: str,
    source_sha256: str,
    pid: str,
    sequence: str | None,
    sample_ordinal: int | None,
) -> dict[str, Any]:
    sequence_sha256 = None
    if sequence is not None:
        sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "git_commit": environment.get("git_commit"),
        "manifest_fingerprint": manifest_fingerprint,
        "preset": config.preset,
        "hf_repo_id": config.hf_repo_id,
        "resolved_revision": resolved_revision,
        "device": config.device,
        "dtype": "float32",
        "batch_size": config.batch_size,
        "seed": config.seed,
        "shard_index": config.shard_index,
        "num_shards": config.num_shards,
        "max_samples": config.max_samples,
        "rtol": config.rtol,
        "atol": config.atol,
        "upstream_sha256": weight_audit.upstream_sha256,
        "stok_sha256": weight_audit.stok_sha256,
        "required_core_stages": list(CANONICAL_STAGES),
        "required_public_stages": list(PUBLIC_STAGES),
        "source_path": source_path,
        "source_sha256": source_sha256,
        "pid": pid,
        "sequence_sha256": sequence_sha256,
        "sample_ordinal": sample_ordinal,
    }


def _record_path(
    samples_dir: Path,
    *,
    pid: str,
    source_sha256: str,
    sample_ordinal: int | None,
) -> Path:
    identity = f"{source_sha256}\0{pid}\0{sample_ordinal}"
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    name = _safe_name(pid).strip("._")[:80] or "sample"
    return samples_dir / f"{name}-{suffix}.json"


def _load_resume_record(path: Path, context: dict[str, Any]) -> dict[str, Any] | None:
    try:
        candidate = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if _is_usable_resume_record(candidate, expected_context=context):
        return candidate
    return None


def _error_payload(
    *,
    pid: str,
    source_path: str,
    context: dict[str, Any],
    error_type: str,
    error_message: str,
    traceback_text: str | None,
    core: Sequence[TensorComparison] | None = None,
) -> dict[str, Any]:
    core_comparisons = list(core) if core is not None else _failure_comparisons(CANONICAL_STAGES)
    return {
        "pid": pid,
        "source_path": source_path,
        "context": context,
        "error_type": error_type,
        "error_message": error_message,
        "traceback": traceback_text,
        "core": [item.to_dict() for item in core_comparisons],
        "public": [item.to_dict() for item in _failure_comparisons(PUBLIC_STAGES)],
    }


def _append_metrics(
    *,
    payload: dict[str, Any],
    pid: str,
    source_path: str,
    core_comparisons: list[TensorComparison],
    public_comparisons: list[TensorComparison],
    metric_rows: list[dict[str, Any]],
) -> None:
    sample_core = [_comparison_from_dict(item) for item in payload["core"]]
    sample_public = [_comparison_from_dict(item) for item in payload["public"]]
    core_comparisons.extend(sample_core)
    public_comparisons.extend(sample_public)
    for category, comparisons in (("core", sample_core), ("public", sample_public)):
        for comparison in comparisons:
            metric_rows.append(
                {
                    "pid": pid,
                    "source_path": source_path,
                    "category": category,
                    **comparison.to_dict(),
                }
            )


def run_qualification(config: QualificationConfig) -> RunSummary:
    _validate_config(config)
    configure_determinism(config.seed)
    device = require_cuda(config.device)
    output_dir = config.output_dir.resolve()
    samples_dir = output_dir / "samples"
    failures_dir = output_dir / "failures"
    checkpoints_dir = output_dir / "checkpoints"
    for directory in (samples_dir, failures_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_inputs = discover_inputs(config.input_dir.resolve())
    selected_inputs = select_shard(
        all_inputs,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
    )
    selected_paths = {record.path for record in selected_inputs}
    input_rows = [
        {
            **record.to_dict(),
            "selected": record.path in selected_paths,
            "shard_index": config.shard_index if record.path in selected_paths else None,
            "num_shards": config.num_shards,
        }
        for record in all_inputs
    ]
    pd.DataFrame(input_rows).to_parquet(output_dir / "input_manifest.parquet", index=False)

    environment = capture_environment(device)
    atomic_write_json(output_dir / "environment.json", environment)
    resolved_revision = resolve_hf_revision(config.hf_repo_id, config.hf_revision)
    atomic_write_json(
        output_dir / "reference.json",
        {
            "repo_id": config.hf_repo_id,
            "requested_revision": config.hf_revision,
            "resolved_revision": resolved_revision,
            "preset": config.preset,
        },
    )
    wrapper = GCPVQVAE(
        mode="embed",
        hf_model_id=config.hf_repo_id,
        hf_revision=resolved_revision,
        device=str(device),
        mixed_precision="no",
        deterministic=True,
        seed=config.seed,
    )
    if not isinstance(wrapper.model, torch.nn.Module):
        raise RuntimeError("Reference wrapper has no loaded torch model")
    _require_float32(wrapper.model, name="reference model")

    upstream_path = Path(wrapper.checkpoint_path)
    converted_path = checkpoints_dir / f"encoder-{config.preset}.pt"
    if not converted_path.exists():
        convert_checkpoint(
            preset=config.preset,
            upstream_path=upstream_path,
            output_path=converted_path,
        )
    weight_audit = audit_weight_parity(upstream_path, converted_path)
    atomic_write_json(output_dir / "weights.json", weight_audit.to_dict())
    if not weight_audit.passed:
        raise RuntimeError(
            "Converted STōk checkpoint is not bit-identical to retained upstream tensors"
        )

    dataset, collate_fn = wrapper._build_dataset(
        pdb_dir=str(config.input_dir.resolve()),
        max_task_samples=None,
        progress=True,
    )
    preprocessing = build_preprocessing_audit(
        all_inputs,
        dataset.samples,
        stok_parser=parse_structure,
    )
    preprocessing_rows = []
    for record in preprocessing:
        row = record.to_dict()
        row["reference_samples"] = json.dumps(row["reference_samples"], sort_keys=True)
        preprocessing_rows.append(row)
    pd.DataFrame(preprocessing_rows).to_parquet(
        output_dir / "preprocessing_comparison.parquet", index=False
    )

    sample_indices_by_source = {record.path: [] for record in selected_inputs}
    for dataset_index, sample in enumerate(dataset.samples):
        try:
            source = str(Path(sample["source_path"]).resolve())
        except (KeyError, TypeError, ValueError):
            continue
        if source in sample_indices_by_source:
            sample_indices_by_source[source].append(dataset_index)
    eligible_indices = [
        dataset_index
        for record in selected_inputs
        for dataset_index in sample_indices_by_source[record.path]
    ]
    if config.max_samples is not None:
        eligible_indices = eligible_indices[: config.max_samples]
    enabled_indices = set(eligible_indices)

    encoder = load_pretrained_encoder(
        preset=config.preset,
        path=str(converted_path),
        device=device,
        freeze=True,
    )
    _require_float32(encoder, name="STōk encoder")

    core_comparisons: list[TensorComparison] = []
    public_comparisons: list[TensorComparison] = []
    metric_rows: list[dict[str, Any]] = []
    sample_manifest_rows: list[dict[str, Any]] = []
    completed_sources: set[str] = set()
    completed_indices_by_source = {record.path: set() for record in selected_inputs}
    manifest_fingerprint = _manifest_fingerprint(all_inputs)

    for input_record in selected_inputs:
        source_path_text = input_record.path
        source_path = Path(source_path_text)
        source_indices = sample_indices_by_source[source_path_text]
        if not source_indices:
            pid = f"missing-reference:{input_record.name}"
            context = _record_context(
                config=config,
                environment=environment,
                resolved_revision=resolved_revision,
                manifest_fingerprint=manifest_fingerprint,
                weight_audit=weight_audit,
                source_path=source_path_text,
                source_sha256=input_record.sha256,
                pid=pid,
                sequence=None,
                sample_ordinal=None,
            )
            sample_path = _record_path(
                samples_dir,
                pid=pid,
                source_sha256=input_record.sha256,
                sample_ordinal=None,
            )
            payload = _error_payload(
                pid=pid,
                source_path=source_path_text,
                context=context,
                error_type="ReferenceSampleMissing",
                error_message="The reference dataset produced no sample for this manifest input",
                traceback_text=None,
            )
            atomic_write_json(sample_path, payload)
            _append_metrics(
                payload=payload,
                pid=pid,
                source_path=source_path_text,
                core_comparisons=core_comparisons,
                public_comparisons=public_comparisons,
                metric_rows=metric_rows,
            )
            sample_manifest_rows.append(
                {
                    "pid": pid,
                    "source_path": source_path_text,
                    "record_path": str(sample_path),
                    "reused": False,
                    "error_type": payload["error_type"],
                }
            )
            completed_sources.add(source_path_text)
            continue

        for sample_ordinal, dataset_index in enumerate(source_indices):
            if dataset_index not in enabled_indices:
                continue
            sample = dataset.samples[dataset_index]
            pid = str(sample["pid"])
            sequence = str(sample["seq"])
            context = _record_context(
                config=config,
                environment=environment,
                resolved_revision=resolved_revision,
                manifest_fingerprint=manifest_fingerprint,
                weight_audit=weight_audit,
                source_path=source_path_text,
                source_sha256=input_record.sha256,
                pid=pid,
                sequence=sequence,
                sample_ordinal=sample_ordinal,
            )
            sample_path = _record_path(
                samples_dir,
                pid=pid,
                source_sha256=input_record.sha256,
                sample_ordinal=sample_ordinal,
            )
            payload = None
            reused = False
            if config.resume and sample_path.exists():
                payload = _load_resume_record(sample_path, context)
                reused = payload is not None

            if payload is None:
                reference_capture = None
                stok_capture = None
                try:
                    batch = collate_fn([dataset[dataset_index]])
                    batch = _move_reference_batch(batch, device)
                    reference_capture = capture_reference_stages(wrapper, batch)
                    stok_capture = capture_stok_stages(encoder, batch)
                    core = compare_stage_captures(
                        reference_capture,
                        stok_capture,
                        rtol=config.rtol,
                        atol=config.atol,
                    )
                    if tuple(item.name for item in core) != CANONICAL_STAGES:
                        raise RuntimeError(
                            "Core comparison did not account for every required stage"
                        )
                except Exception as exc:
                    payload = _error_payload(
                        pid=pid,
                        source_path=source_path_text,
                        context=context,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                else:
                    try:
                        public = _compare_public_stok_path(
                            reference=reference_capture,
                            reference_sequence=str(batch["seq"][0]),
                            source_path=source_path,
                            encoder=encoder,
                            device=device,
                        )
                        if tuple(item.name for item in public) != PUBLIC_STAGES:
                            raise RuntimeError(
                                "Public comparison did not account for every required stage"
                            )
                    except Exception as exc:
                        payload = _error_payload(
                            pid=pid,
                            source_path=source_path_text,
                            context=context,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            traceback_text=traceback.format_exc(),
                            core=core,
                        )
                    else:
                        payload = {
                            "pid": pid,
                            "source_path": source_path_text,
                            "context": context,
                            "error_type": None,
                            "error_message": None,
                            "traceback": None,
                            "core": [item.to_dict() for item in core],
                            "public": [item.to_dict() for item in public],
                        }

                comparisons = [
                    *[_comparison_from_dict(item) for item in payload["core"]],
                    *[_comparison_from_dict(item) for item in payload["public"]],
                ]
                if not all(item.passed for item in comparisons):
                    diagnostic_path = failures_dir / sample_path.with_suffix(".pt").name
                    if reference_capture is not None and stok_capture is not None:
                        try:
                            torch.save(
                                {
                                    "reference": reference_capture.tensors,
                                    "stok": stok_capture.tensors,
                                },
                                diagnostic_path,
                            )
                        except Exception as exc:
                            payload["failure_artifact_error_type"] = type(exc).__name__
                            payload["failure_artifact_error_message"] = str(exc)
                atomic_write_json(sample_path, payload)

            _append_metrics(
                payload=payload,
                pid=pid,
                source_path=source_path_text,
                core_comparisons=core_comparisons,
                public_comparisons=public_comparisons,
                metric_rows=metric_rows,
            )
            sample_manifest_rows.append(
                {
                    "pid": pid,
                    "source_path": source_path_text,
                    "record_path": str(sample_path),
                    "reused": reused,
                    "error_type": payload["error_type"],
                }
            )
            completed_indices_by_source[source_path_text].add(dataset_index)

        if completed_indices_by_source[source_path_text] == set(source_indices):
            completed_sources.add(source_path_text)

    metric_columns = [
        "pid",
        "source_path",
        "category",
        "name",
        "shape",
        "compared",
        "mismatched",
        "exact",
        "passed",
        "mask_equal",
        "finite_pattern_equal",
        "within_tolerance",
        "max_abs",
        "max_rel",
        "p50_abs",
        "p95_abs",
        "p99_abs",
    ]
    pd.DataFrame(metric_rows, columns=metric_columns).to_parquet(
        output_dir / "metrics.parquet", index=False
    )
    sample_manifest_columns = [
        "pid",
        "source_path",
        "record_path",
        "reused",
        "error_type",
    ]
    pd.DataFrame(sample_manifest_rows, columns=sample_manifest_columns).to_parquet(
        output_dir / "sample_manifest.parquet", index=False
    )
    qualification_rows = []
    for record in all_inputs:
        selected = record.path in selected_paths
        source_indices = sample_indices_by_source.get(record.path, [])
        complete = record.path in completed_sources
        qualification_rows.append(
            {
                **record.to_dict(),
                "selected": selected,
                "reference_sample_count": len(source_indices),
                "completed": complete,
                "status": (
                    "not_selected" if not selected else "completed" if complete else "incomplete"
                ),
            }
        )
    pd.DataFrame(qualification_rows).to_parquet(
        output_dir / "qualification_manifest.parquet", index=False
    )

    summary = summarize_run(
        weight_pass=weight_audit.passed,
        core_comparisons=core_comparisons,
        public_comparisons=public_comparisons,
        input_count=len(selected_inputs),
        completed_count=len(completed_sources),
    )
    atomic_write_json(output_dir / "summary.json", summary.to_dict())
    (output_dir / "report.md").write_text(render_markdown(summary))
    return summary
