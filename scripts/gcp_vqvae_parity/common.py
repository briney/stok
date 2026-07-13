from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class InputFile:
    path: str
    name: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_inputs(root: Path) -> list[InputFile]:
    suffixes = {".cif", ".mmcif", ".pdb", ".ent"}
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not paths:
        raise ValueError(f"No structure inputs found under {root}")
    return [
        InputFile(
            path=str(path.resolve()),
            name=path.name,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in paths
    ]


def select_shard(records: Sequence[T], *, shard_index: int, num_shards: int) -> list[T]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    return list(records[shard_index::num_shards])


def configure_determinism(seed: int = 0) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def require_cuda(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"Qualification device must be CUDA, got {resolved}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command outside the managed sandbox")
    torch.empty(1, device=resolved)
    return resolved


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(device: torch.device) -> dict[str, Any]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "packages": {
            name: _version(name)
            for name in (
                "gcp-vqvae",
                "graphein",
                "torch-geometric",
                "torch-cluster",
                "torch-scatter",
                "x-transformers",
                "vector-quantize-pytorch",
                "biopython",
            )
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
