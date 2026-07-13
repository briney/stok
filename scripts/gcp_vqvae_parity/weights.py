from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi

from scripts.convert_gcp_vqvae_weights import convert

from .common import sha256_file

_HF_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")
_CHECKPOINT_ENVELOPES = ("model_state_dict", "state_dict", "model")
_DROP_PREFIXES = (
    "vqvae.decoder.",
    "vqvae.ntp_",
    "vqvae.markov_",
    "vqvae.ti_tok_",
    "protein_encoder.",
)
_DROP_EXACT = {"vqvae.vector_quantizer.zero"}
_RENAME_RULES = (
    ("vqvae.encoder_tail.", "encoder_tail."),
    ("vqvae.encoder_blocks.", "encoder_blocks."),
    ("vqvae.encoder_head.", "encoder_head."),
    ("vqvae.vector_quantizer.", "vector_quantizer."),
    ("encoder.", "gcpnet."),
)


@dataclass(frozen=True)
class WeightAudit:
    upstream_sha256: str
    stok_sha256: str
    compared: int
    missing: list[str]
    unexpected: list[str]
    different: list[str]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_hf_revision(repo_id: str, requested_revision: str | None) -> str:
    info = HfApi().model_info(repo_id, revision=requested_revision)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or _HF_COMMIT_SHA.fullmatch(sha) is None:
        raise RuntimeError(
            f"Hugging Face did not return a full 40-character hexadecimal commit SHA "
            f"for {repo_id}"
        )
    return sha


def convert_checkpoint(*, preset: str, upstream_path: Path, output_path: Path) -> Path:
    convert(preset, input_path=upstream_path, output_path=output_path)
    return output_path


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: checkpoint state dict must be a dict")
    state = raw
    for envelope in _CHECKPOINT_ENVELOPES:
        if envelope not in raw:
            continue
        state = raw[envelope]
        if not isinstance(state, dict):
            raise TypeError(f"{path}: checkpoint state dict envelope {envelope!r} must be a dict")
        break
    if not state:
        raise ValueError(f"{path}: checkpoint state dict is empty")
    if any(not isinstance(key, str) for key in state):
        raise TypeError(f"{path}: checkpoint state dict keys must be strings")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"{path}: checkpoint state dict values must be tensors")
    return state


def _source_destination(source: str) -> str | None:
    if source in _DROP_EXACT or any(source.startswith(prefix) for prefix in _DROP_PREFIXES):
        return None
    for source_prefix, destination_prefix in _RENAME_RULES:
        if source.startswith(source_prefix):
            return destination_prefix + source.removeprefix(source_prefix)
    if source.startswith("vqvae."):
        raise ValueError(f"Uncategorized upstream key: {source}")
    return source


def _build_source_mapping(state: dict[str, torch.Tensor]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    dropped: set[str] = set()
    source_by_destination: dict[str, str] = {}
    for source in state:
        destination = _source_destination(source)
        if destination is None:
            dropped.add(source)
            continue
        if destination in source_by_destination:
            other = source_by_destination[destination]
            raise ValueError(
                f"Duplicate destination target: {destination} ({other!r} and {source!r})"
            )
        mapping[source] = destination
        source_by_destination[destination] = source
    unaccounted = set(state) - set(mapping) - dropped
    if unaccounted:
        raise ValueError(f"Unaccounted upstream keys: {sorted(unaccounted)}")
    return mapping


def audit_weight_parity(upstream_path: Path, stok_path: Path) -> WeightAudit:
    upstream = _load_state(upstream_path)
    actual = _load_state(stok_path)
    source_mapping = _build_source_mapping(upstream)
    source_by_destination = {
        destination: source for source, destination in source_mapping.items()
    }
    expected_keys = set(source_by_destination)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    different = sorted(
        key
        for key in expected_keys & actual_keys
        if upstream[source_by_destination[key]].dtype != actual[key].dtype
        or upstream[source_by_destination[key]].shape != actual[key].shape
        or not torch.equal(upstream[source_by_destination[key]], actual[key])
    )
    return WeightAudit(
        upstream_sha256=sha256_file(upstream_path),
        stok_sha256=sha256_file(stok_path),
        compared=len(expected_keys & actual_keys),
        missing=missing,
        unexpected=unexpected,
        different=different,
        passed=not missing and not unexpected and not different,
    )
