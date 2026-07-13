from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi

from scripts.convert_gcp_vqvae_weights import (
    _unwrap_state_dict,
    convert,
    remap_state_dict,
)

from .common import sha256_file


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
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}")
    return info.sha


def convert_checkpoint(*, preset: str, upstream_path: Path, output_path: Path) -> Path:
    convert(preset, input_path=upstream_path, output_path=output_path)
    return output_path


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return _unwrap_state_dict(raw)


def audit_weight_parity(upstream_path: Path, stok_path: Path) -> WeightAudit:
    expected = remap_state_dict(_load_state(upstream_path))
    actual = _load_state(stok_path)
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    different = sorted(
        key
        for key in expected_keys & actual_keys
        if expected[key].dtype != actual[key].dtype
        or expected[key].shape != actual[key].shape
        or not torch.equal(expected[key], actual[key])
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
