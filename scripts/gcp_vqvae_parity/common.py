from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import importlib.util
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

_PACKAGE_MODULES = (
    ("gcp-vqvae", "gcp_vqvae"),
    ("graphein", "graphein"),
    ("torch-geometric", "torch_geometric"),
    ("torch-cluster", "torch_cluster"),
    ("torch-scatter", "torch_scatter"),
    ("x-transformers", "x_transformers"),
    ("vector-quantize-pytorch", "vector_quantize_pytorch"),
    ("biopython", "Bio"),
)


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
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
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
    requested = torch.device(device)
    if requested.type != "cuda" or requested.index not in (None, 0):
        raise ValueError(f"Qualification device must be CUDA device 0, got {requested}")
    resolved = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command outside the managed sandbox")
    torch.empty(1, device=resolved)
    return resolved


def _empty_source() -> dict[str, str | None]:
    return {
        "url": None,
        "subdirectory": None,
        "vcs": None,
        "requested_revision": None,
        "commit_id": None,
    }


def _source_provenance(distribution: importlib.metadata.Distribution | None) -> dict[str, Any]:
    source = _empty_source()
    if distribution is None:
        return source
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return source
    try:
        direct_url = json.loads(direct_url_text)
    except (json.JSONDecodeError, TypeError):
        return source
    if not isinstance(direct_url, dict):
        return source
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        vcs_info = {}
    values = {
        "url": direct_url.get("url"),
        "subdirectory": direct_url.get("subdirectory"),
        "vcs": vcs_info.get("vcs"),
        "requested_revision": vcs_info.get("requested_revision"),
        "commit_id": vcs_info.get("commit_id"),
    }
    return {key: value if isinstance(value, str) else None for key, value in values.items()}


def _package_provenance(distribution_name: str, module_name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    origin = getattr(spec, "origin", None)
    return {
        "metadata_present": distribution is not None,
        "importable": spec is not None,
        "module": module_name,
        "origin": origin if isinstance(origin, str) else None,
        "version": distribution.version if distribution is not None else None,
        "source": _source_provenance(distribution),
    }


def _nvidia_driver_version(device_index: int) -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _cuda_runtime_version() -> str | None:
    try:
        discovered = ctypes.util.find_library("cudart")
    except OSError:
        discovered = None
    candidates = [name for name in (discovered, "libcudart.so") if name]
    runtime = None
    for library_name in dict.fromkeys(candidates):
        try:
            runtime = ctypes.CDLL(library_name)
        except OSError:
            continue
        break
    if runtime is None:
        return None
    try:
        get_version = runtime.cudaRuntimeGetVersion
    except AttributeError:
        return None
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    raw_version = ctypes.c_int()
    try:
        status = get_version(ctypes.byref(raw_version))
    except (OSError, TypeError):
        return None
    if status != 0:
        return None
    version = raw_version.value
    if version <= 0:
        return None
    major = version // 1000
    minor = (version % 1000) // 10
    patch = version % 10
    return f"{major}.{minor}.{patch}" if patch else f"{major}.{minor}"


def capture_environment(device: torch.device) -> dict[str, Any]:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"Environment capture requires CUDA, got {resolved}")
    device_index = resolved.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    device_name = torch.cuda.get_device_name(device_index)
    capability = list(torch.cuda.get_device_capability(device_index))
    package_provenance = {
        distribution_name: _package_provenance(distribution_name, module_name)
        for distribution_name, module_name in _PACKAGE_MODULES
    }
    upstream = package_provenance["gcp-vqvae"]
    upstream_source = upstream["source"]
    return {
        "git_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": device_name,
        "cuda_capability": capability,
        "cuda": {
            "device": {
                "index": device_index,
                "name": device_name,
                "capability": capability,
            },
            "versions": {
                "nvidia_driver": _nvidia_driver_version(device_index),
                "cuda_runtime": _cuda_runtime_version(),
                "torch_build_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
        },
        "packages": {
            name: provenance["version"] for name, provenance in package_provenance.items()
        },
        "package_provenance": package_provenance,
        "upstream_vq_encoder_decoder": {
            "repository": "vq_encoder_decoder",
            "distribution": "gcp-vqvae",
            "distribution_version": upstream["version"],
            "module": upstream["module"],
            "module_origin": upstream["origin"],
            "source_url": upstream_source["url"],
            "source_revision": upstream_source["commit_id"],
            "requested_revision": upstream_source["requested_revision"],
            "subdirectory": upstream_source["subdirectory"],
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
