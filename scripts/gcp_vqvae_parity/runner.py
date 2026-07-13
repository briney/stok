from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import traceback
import uuid
from collections.abc import Mapping, Sequence
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
from .types import MAX_STORAGE_INTEGER, TensorComparison
from .weights import audit_weight_parity, convert_checkpoint, resolve_hf_revision

RECORD_SCHEMA_VERSION = 3
PUBLIC_STAGES = (
    "public.preprocessing",
    "public.indices",
    "public.embeddings",
)
_EXACT_CORE_STAGES = frozenset({"indices", "embeddings", "valid"})
_EXACT_PUBLIC_STAGES = frozenset(PUBLIC_STAGES)
_INDEX_STAGES = frozenset({"indices", "public.indices"})
_RECORD_KEYS = frozenset(
    {
        "pid",
        "source_path",
        "context",
        "error_type",
        "error_message",
        "traceback",
        "failure_artifact_error_type",
        "failure_artifact_error_message",
        "core",
        "public",
    }
)
_COMPARISON_KEYS = frozenset(
    {
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
    }
)
_PUBLIC_ARTIFACTS = (
    "summary.json",
    "report.md",
    "completion.json",
    "input_manifest.parquet",
    "environment.json",
    "source_state.json",
    "run_context.json",
    "reference.json",
    "weights.json",
    "preprocessing_comparison.parquet",
    "metrics.parquet",
    "sample_manifest.parquet",
    "qualification_manifest.parquet",
    "samples",
    "failures",
    "checkpoints",
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


@dataclass(frozen=True)
class _ReferenceEntry:
    dataset_index: int
    source_path: str
    source_sha256: str
    pid: str
    sequence: str | None
    error_message: str | None


@dataclass(frozen=True)
class _RunTransaction:
    root: Path
    staging: Path
    cache: Path
    run_id: str

    def bind_context(self, context_sha256: str) -> _RunTransaction:
        run_id = f"run-{context_sha256[:16]}-{uuid.uuid4().hex}"
        staging = self.root / ".runs" / f".{run_id}.staging"
        self.staging.replace(staging)
        return _RunTransaction(
            root=self.root,
            staging=staging,
            cache=self.cache,
            run_id=run_id,
        )

    def publish(self) -> None:
        destination = self.root / ".runs" / self.run_id
        self.staging.replace(destination)
        temporary_link = self.root / ".current.tmp"
        if temporary_link.is_symlink() or temporary_link.exists():
            temporary_link.unlink()
        temporary_link.symlink_to(Path(".runs") / self.run_id, target_is_directory=True)
        os.replace(temporary_link, self.root / "current")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _begin_transaction(root: Path) -> _RunTransaction:
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ".cache"
    for directory in (cache / "samples", cache / "failures", cache / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    current = root / "current"
    previous_run = None
    if current.is_symlink() or current.exists():
        try:
            previous_run = current.resolve(strict=True)
        except OSError:
            previous_run = None
        _remove_path(current)
    if previous_run is not None:
        for published_record in (previous_run / "samples").glob("*.json"):
            shutil.copy2(published_record, cache / "samples" / published_record.name)
    for name in _PUBLIC_ARTIFACTS:
        public_path = root / name
        expected_target = Path("current") / name
        if public_path.is_symlink() and Path(os.readlink(public_path)) == expected_target:
            continue
        if public_path.is_symlink() or public_path.exists():
            _remove_path(public_path)
        public_path.symlink_to(expected_target)
    run_id = f"run-{uuid.uuid4().hex}"
    staging = root / ".runs" / f".{run_id}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    return _RunTransaction(root=root, staging=staging, cache=cache, run_id=run_id)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _failure(name: str) -> TensorComparison:
    return TensorComparison(
        name=name,
        shape=(),
        compared=1,
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
    if type(payload) is not dict or set(payload) != _COMPARISON_KEYS:
        raise ValueError("Comparison record keys are incomplete or unexpected")
    name = payload["name"]
    shape = payload["shape"]
    if type(name) is not str or not name:
        raise ValueError("Comparison name must be a nonempty string")
    if type(shape) not in (list, tuple) or any(
        type(dimension) is not int or not 0 <= dimension <= MAX_STORAGE_INTEGER
        for dimension in shape
    ):
        raise ValueError(
            "Comparison shape must contain storage-safe signed-int64 nonnegative integers"
        )
    for key in ("compared", "mismatched"):
        if type(payload[key]) is not int or not 0 <= payload[key] <= MAX_STORAGE_INTEGER:
            raise ValueError(
                f"Comparison {key} must be a storage-safe signed-int64 nonnegative integer"
            )
    if payload["compared"] > math.prod(shape):
        raise ValueError("Comparison count cannot exceed the tensor shape capacity")
    if payload["mismatched"] > payload["compared"]:
        raise ValueError("Comparison mismatched count cannot exceed compared count")
    boolean_keys = (
        "exact",
        "passed",
        "mask_equal",
        "finite_pattern_equal",
        "within_tolerance",
    )
    if any(type(payload[key]) is not bool for key in boolean_keys):
        raise ValueError("Comparison flags must be booleans")
    metric_keys = ("max_abs", "max_rel", "p50_abs", "p95_abs", "p99_abs")
    if any(
        type(payload[key]) not in (int, float)
        or not math.isfinite(payload[key])
        or payload[key] < 0
        for key in metric_keys
    ):
        raise ValueError("Comparison metrics must be finite nonnegative numbers")
    if not (payload["p50_abs"] <= payload["p95_abs"] <= payload["p99_abs"] <= payload["max_abs"]):
        raise ValueError("Comparison absolute-error quantiles are inconsistent")
    if payload["mismatched"] == 0 and payload["max_abs"] != 0:
        raise ValueError("A zero-mismatch comparison cannot have absolute error")
    if payload["max_abs"] == 0 and payload["max_rel"] != 0:
        raise ValueError("A zero-absolute-error comparison cannot have relative error")
    if payload["passed"] != payload["within_tolerance"]:
        raise ValueError("Comparison pass and tolerance flags are inconsistent")
    if payload["passed"] and (not payload["mask_equal"] or not payload["finite_pattern_equal"]):
        raise ValueError("A passing comparison must have matching masks and finite patterns")
    if payload["exact"] and (
        payload["mismatched"] != 0
        or not payload["passed"]
        or payload["max_abs"] != 0
        or payload["max_rel"] != 0
    ):
        raise ValueError("Exact comparison invariants are inconsistent")
    normalized = dict(payload)
    normalized["shape"] = tuple(normalized["shape"])
    return TensorComparison(**normalized)


def _has_stage_set(
    payloads: Any,
    expected: Sequence[str],
    *,
    exact_required: frozenset[str],
) -> bool:
    if not isinstance(payloads, list):
        return False
    try:
        comparisons = [_comparison_from_dict(item) for item in payloads]
    except (KeyError, OverflowError, TypeError, ValueError):
        return False
    if tuple(item.name for item in comparisons) != tuple(expected):
        return False
    for comparison in comparisons:
        if comparison.name in exact_required and not (
            comparison.exact
            and comparison.mismatched == 0
            and comparison.passed
            and comparison.within_tolerance
            and comparison.mask_equal
            and comparison.finite_pattern_equal
        ):
            return False
        if (
            comparison.name not in _INDEX_STAGES
            and comparison.name != "public.preprocessing"
            and comparison.mismatched > 0
            and comparison.max_abs == 0
        ):
            return False
    return True


def _is_usable_resume_record(
    payload: Any,
    *,
    expected_context: dict[str, Any],
) -> bool:
    if type(payload) is not dict or set(payload) != _RECORD_KEYS:
        return False
    if type(payload["pid"]) is not str or not payload["pid"]:
        return False
    if type(payload["source_path"]) is not str or not payload["source_path"]:
        return False
    if type(payload["context"]) is not dict or payload["context"] != expected_context:
        return False
    if payload["pid"] != expected_context.get("pid"):
        return False
    if payload["source_path"] != expected_context.get("source_path"):
        return False
    if (
        payload["error_type"] is not None
        or payload["error_message"] is not None
        or payload["traceback"] is not None
    ):
        return False
    artifact_error_type = payload["failure_artifact_error_type"]
    artifact_error_message = payload["failure_artifact_error_message"]
    if (artifact_error_type is None) != (artifact_error_message is None):
        return False
    if artifact_error_type is not None and (
        type(artifact_error_type) is not str
        or not artifact_error_type
        or type(artifact_error_message) is not str
        or not artifact_error_message
    ):
        return False
    return _has_stage_set(
        payload["core"], CANONICAL_STAGES, exact_required=_EXACT_CORE_STAGES
    ) and _has_stage_set(payload["public"], PUBLIC_STAGES, exact_required=_EXACT_PUBLIC_STAGES)


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
    return _canonical_fingerprint([record.to_dict() for record in records])


def _canonical_fingerprint(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _classify_reference_samples(
    samples: Sequence[Any],
    inputs: Sequence[Any],
) -> tuple[dict[str, list[_ReferenceEntry]], list[_ReferenceEntry], list[dict[str, str]]]:
    inputs_by_path = {record.path: record for record in inputs}
    entries_by_source = {record.path: [] for record in inputs}
    unassigned: list[_ReferenceEntry] = []
    preprocessing_samples: list[dict[str, str]] = []
    for dataset_index, sample in enumerate(samples):
        errors: list[str] = []
        mapping = sample if isinstance(sample, Mapping) else {}
        if not isinstance(sample, Mapping):
            errors.append(f"sample must be a mapping, got {type(sample).__name__}")

        raw_source = mapping.get("source_path")
        resolved_source = None
        if not isinstance(raw_source, (str, Path)) or not str(raw_source):
            errors.append("source_path must be a nonempty string or Path")
        else:
            try:
                resolved_source = str(Path(raw_source).resolve())
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"source_path cannot be resolved: {type(exc).__name__}: {exc}")
        input_record = inputs_by_path.get(resolved_source)
        if resolved_source is not None and input_record is None:
            errors.append("source_path is not present in the input manifest")

        raw_pid = mapping.get("pid")
        if type(raw_pid) is not str or not raw_pid:
            errors.append("pid must be a nonempty string")
        raw_sequence = mapping.get("seq")
        if type(raw_sequence) is not str or not raw_sequence:
            errors.append("seq must be a nonempty string")

        fallback_pid = f"invalid-reference:{dataset_index:06d}"
        fallback_source = resolved_source or f"<invalid-source:{dataset_index:06d}>"
        source_sha256 = (
            input_record.sha256
            if input_record is not None
            else hashlib.sha256(
                f"reference-anomaly:{dataset_index}:{fallback_source}".encode()
            ).hexdigest()
        )
        entry = _ReferenceEntry(
            dataset_index=dataset_index,
            source_path=fallback_source,
            source_sha256=source_sha256,
            pid=fallback_pid if errors else raw_pid,
            sequence=raw_sequence if type(raw_sequence) is str else None,
            error_message="; ".join(errors) if errors else None,
        )
        if input_record is None:
            unassigned.append(entry)
        else:
            entries_by_source[input_record.path].append(entry)
        if not errors:
            preprocessing_samples.append(
                {
                    "source_path": input_record.path,
                    "pid": raw_pid,
                    "seq": raw_sequence,
                }
            )
    return entries_by_source, unassigned, preprocessing_samples


def _relevant_source_paths(project_root: Path) -> list[Path]:
    source_roots = (
        project_root / "scripts" / "gcp_vqvae_parity",
        project_root / "src" / "stok",
    )
    source_paths = {
        path for source_root in source_roots for path in source_root.rglob("*.py") if path.is_file()
    }
    converter = project_root / "scripts" / "convert_gcp_vqvae_weights.py"
    if converter.is_file():
        source_paths.add(converter)
    return sorted(source_paths)


def _capture_source_state() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    source_paths = _relevant_source_paths(project_root)
    digest = hashlib.sha256()
    for path in source_paths:
        relative = path.relative_to(project_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "scripts/gcp_vqvae_parity",
            "scripts/convert_gcp_vqvae_weights.py",
            "src/stok",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "code_sha256": digest.hexdigest(),
        "file_count": len(source_paths),
    }


def _record_context(
    *,
    config: QualificationConfig,
    environment: dict[str, Any],
    source_state: dict[str, Any],
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
        "environment_sha256": _canonical_fingerprint(environment),
        "source_state_sha256": _canonical_fingerprint(source_state),
        "source_git_commit": source_state["git_commit"],
        "source_dirty": source_state["dirty"],
        "source_status_sha256": source_state["status_sha256"],
        "source_code_sha256": source_state["code_sha256"],
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


def _failure_bundle_path(
    failures_dir: Path,
    *,
    sample_path: Path,
    context: dict[str, Any],
) -> Path:
    context_suffix = _canonical_fingerprint(context)[:16]
    return failures_dir / f"{sample_path.stem}-{context_suffix}.pt"


def _load_resume_record(path: Path, context: dict[str, Any]) -> dict[str, Any] | None:
    try:
        candidate = json.loads(path.read_text())
    except (OSError, RecursionError, UnicodeDecodeError, ValueError):
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
        "failure_artifact_error_type": None,
        "failure_artifact_error_message": None,
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
    transaction = _begin_transaction(config.output_dir.resolve())
    configure_determinism(config.seed)
    device = require_cuda(config.device)
    output_dir = transaction.staging
    samples_dir = transaction.cache / "samples"
    failures_dir = transaction.cache / "failures"
    checkpoints_dir = transaction.cache / "checkpoints"

    all_inputs = discover_inputs(config.input_dir.resolve())
    selected_inputs = select_shard(
        all_inputs,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
    )
    manifest_fingerprint = _manifest_fingerprint(all_inputs)
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
    source_state = _capture_source_state()
    atomic_write_json(output_dir / "source_state.json", source_state)
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
    run_context = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "environment_sha256": _canonical_fingerprint(environment),
        "source_state_sha256": _canonical_fingerprint(source_state),
        "manifest_fingerprint": manifest_fingerprint,
        "preset": config.preset,
        "hf_repo_id": config.hf_repo_id,
        "requested_revision": config.hf_revision,
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
    }
    run_context_sha256 = _canonical_fingerprint(run_context)
    run_context["context_sha256"] = run_context_sha256
    transaction = transaction.bind_context(run_context_sha256)
    output_dir = transaction.staging
    atomic_write_json(output_dir / "run_context.json", run_context)

    dataset, collate_fn = wrapper._build_dataset(
        pdb_dir=str(config.input_dir.resolve()),
        max_task_samples=None,
        progress=True,
    )
    reference_entries_by_source, unassigned_entries, preprocessing_samples = (
        _classify_reference_samples(dataset.samples, all_inputs)
    )
    preprocessing = build_preprocessing_audit(
        all_inputs,
        preprocessing_samples,
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

    sample_entries_by_source = {
        record.path: reference_entries_by_source[record.path] for record in selected_inputs
    }
    eligible_indices = [
        entry.dataset_index
        for record in selected_inputs
        for entry in sample_entries_by_source[record.path]
        if entry.error_message is None
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

    def record_metadata_anomaly(entry: _ReferenceEntry) -> None:
        context = _record_context(
            config=config,
            environment=environment,
            source_state=source_state,
            resolved_revision=resolved_revision,
            manifest_fingerprint=manifest_fingerprint,
            weight_audit=weight_audit,
            source_path=entry.source_path,
            source_sha256=entry.source_sha256,
            pid=entry.pid,
            sequence=entry.sequence,
            sample_ordinal=entry.dataset_index,
        )
        sample_path = _record_path(
            samples_dir,
            pid=entry.pid,
            source_sha256=entry.source_sha256,
            sample_ordinal=entry.dataset_index,
        )
        payload = _error_payload(
            pid=entry.pid,
            source_path=entry.source_path,
            context=context,
            error_type="ReferenceSampleMetadataError",
            error_message=entry.error_message or "Invalid reference sample metadata",
            traceback_text=None,
        )
        atomic_write_json(sample_path, payload)
        _append_metrics(
            payload=payload,
            pid=entry.pid,
            source_path=entry.source_path,
            core_comparisons=core_comparisons,
            public_comparisons=public_comparisons,
            metric_rows=metric_rows,
        )
        sample_manifest_rows.append(
            {
                "pid": entry.pid,
                "source_path": entry.source_path,
                "record_path": str(transaction.root / "samples" / sample_path.name),
                "cache_path": str(sample_path),
                "reused": False,
                "error_type": payload["error_type"],
            }
        )

    for input_record in selected_inputs:
        source_path_text = input_record.path
        source_path = Path(source_path_text)
        source_entries = sample_entries_by_source[source_path_text]
        if not source_entries:
            pid = f"missing-reference:{input_record.name}"
            context = _record_context(
                config=config,
                environment=environment,
                source_state=source_state,
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
                    "record_path": str(transaction.root / "samples" / sample_path.name),
                    "cache_path": str(sample_path),
                    "reused": False,
                    "error_type": payload["error_type"],
                }
            )
            completed_sources.add(source_path_text)
            continue

        for sample_ordinal, entry in enumerate(source_entries):
            dataset_index = entry.dataset_index
            if entry.error_message is not None:
                record_metadata_anomaly(entry)
                completed_indices_by_source[source_path_text].add(dataset_index)
                continue
            if dataset_index not in enabled_indices:
                continue
            pid = entry.pid
            sequence = entry.sequence
            if sequence is None:
                raise RuntimeError("Validated reference entry has no sequence")
            context = _record_context(
                config=config,
                environment=environment,
                source_state=source_state,
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
            failure_cache_path = None
            diagnostic_path = _failure_bundle_path(
                failures_dir,
                sample_path=sample_path,
                context=context,
            )
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
                            "failure_artifact_error_type": None,
                            "failure_artifact_error_message": None,
                            "core": [item.to_dict() for item in core],
                            "public": [item.to_dict() for item in public],
                        }

                comparisons = [
                    *[_comparison_from_dict(item) for item in payload["core"]],
                    *[_comparison_from_dict(item) for item in payload["public"]],
                ]
                if not all(item.passed for item in comparisons):
                    if reference_capture is not None and stok_capture is not None:
                        try:
                            torch.save(
                                {
                                    "reference": reference_capture.tensors,
                                    "stok": stok_capture.tensors,
                                },
                                diagnostic_path,
                            )
                            failure_cache_path = str(diagnostic_path)
                        except Exception as exc:
                            payload["failure_artifact_error_type"] = type(exc).__name__
                            payload["failure_artifact_error_message"] = str(exc)
                atomic_write_json(sample_path, payload)

            if reused:
                comparisons = [
                    *[_comparison_from_dict(item) for item in payload["core"]],
                    *[_comparison_from_dict(item) for item in payload["public"]],
                ]
                if not all(item.passed for item in comparisons) and diagnostic_path.exists():
                    failure_cache_path = str(diagnostic_path)

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
                    "record_path": str(transaction.root / "samples" / sample_path.name),
                    "cache_path": str(sample_path),
                    "failure_cache_path": failure_cache_path,
                    "reused": reused,
                    "error_type": payload["error_type"],
                }
            )
            completed_indices_by_source[source_path_text].add(dataset_index)

        if completed_indices_by_source[source_path_text] == {
            entry.dataset_index for entry in source_entries
        }:
            completed_sources.add(source_path_text)

    for entry in unassigned_entries:
        if entry.dataset_index % config.num_shards == config.shard_index:
            record_metadata_anomaly(entry)

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
        source_entries = reference_entries_by_source.get(record.path, [])
        complete = record.path in completed_sources
        qualification_rows.append(
            {
                **record.to_dict(),
                "selected": selected,
                "reference_sample_count": len(source_entries),
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
        diagnostic=config.max_samples is not None,
    )
    atomic_write_json(output_dir / "summary.json", summary.to_dict())
    (output_dir / "report.md").write_text(render_markdown(summary))
    published_samples = output_dir / "samples"
    published_failures = output_dir / "failures"
    published_checkpoints = output_dir / "checkpoints"
    for directory in (published_samples, published_failures, published_checkpoints):
        directory.mkdir(parents=True, exist_ok=True)
    for row in sample_manifest_rows:
        cache_path = Path(row["cache_path"])
        shutil.copy2(cache_path, published_samples / cache_path.name)
        failure_path_value = row.get("failure_cache_path")
        if failure_path_value is not None:
            failure_path = Path(failure_path_value)
            shutil.copy2(failure_path, published_failures / failure_path.name)
    shutil.copy2(converted_path, published_checkpoints / converted_path.name)
    atomic_write_json(
        output_dir / "completion.json",
        {
            "complete": True,
            "context_sha256": run_context_sha256,
            "diagnostic": summary.diagnostic,
            "qualification_complete": summary.complete,
            "run_id": transaction.run_id,
            "status": summary.status.value,
        },
    )
    transaction.publish()
    return summary
