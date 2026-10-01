"""Safe, strict adaptation of the two pinned GCP-VQVAE releases."""

from collections.abc import Mapping
import hashlib
import importlib.resources as resources
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import tempfile
import platform
import subprocess
from typing import Literal
import zipfile

import numpy as np
from numpy.core import multiarray
import torch
import yaml

SOURCE_COMMIT = "68c4c284fe204de27fdf61db27fcc01136ea9f28"
ARTIFACTS = {
    "lite": {
        "repository": "Mahdip72/gcp-vqvae-lite",
        "revision": "d67ca3dfc1ecd6c6a1c3841f584e82f2f9803447",
        "filename": "checkpoints/best_valid.pth",
        "sha256": "0a212868f83ba1cc426404ec5d09c0d83012c8e34715483a7107091ffcf16e30",
    },
    "large": {
        "repository": "Mahdip72/gcp-vqvae-large",
        "revision": "64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1",
        "filename": "checkpoints/best_valid.pth",
        "sha256": "7d1d43950a29834e7f702409bf957e9ffb75eb3cb3952074ba4c545bb4130eaf",
    },
}
_PREFIXES = {
    "encoder": {
        "encoder.featuriser.": "featuriser.",
        "encoder.encoder.": "encoder.",
        "vqvae.encoder_tail.": "encoder_tail.",
        "vqvae.encoder_blocks.": "encoder_blocks.",
        "vqvae.encoder_head.": "encoder_head.",
        "encoder.encoder_tail.": "encoder_tail.",
        "encoder.encoder_blocks.": "encoder_blocks.",
        "encoder.encoder_head.": "encoder_head.",
    },
    "quantizer": {"vqvae.vector_quantizer.": "", "quantizer.": ""},
    "decoder": {"vqvae.decoder.": "", "decoder.": ""},
}
_UNUSED_HEAD = {
    "pairwise_classification_head." + name
    for name in (
        "downproject.weight",
        "linear1.weight",
        "norm.weight",
        "norm.bias",
        "linear2.weight",
    )
}


def canonical_preset(preset: str) -> str:
    preset = "large" if preset == "base" else preset
    if preset not in ARTIFACTS:
        raise ValueError(f"Unsupported GCP-VQVAE preset: {preset}")
    return preset


def read_gcp_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """Read model tensors only; arbitrary pickle metadata is never enabled.

    Raw codebook matrices are returned under ``codebook`` for standalone lookup;
    they cannot replace a complete tokenizer's quantizer state.
    """
    path = Path(path)
    # Only the scalar/dtype globals present in the released archives are trusted.
    with torch.serialization.safe_globals(
        [
            getattr(multiarray, "scalar"),
            np.dtype,
            type(np.dtype("float64")),
        ]
    ):
        payload = torch.load(
            path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path)
        )
    if isinstance(payload, torch.Tensor):
        payload = {"codebook": payload}
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Checkpoint must contain a nonempty tensor state dictionary")
    state = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Model state must contain string keys and tensor values")
        normalized = ".".join(part for part in key.split(".") if part != "_orig_mod")
        if normalized in state:
            raise ValueError(f"Checkpoint key normalization collision: {key}")
        state[normalized] = value
    return state


def extract_gcp_component(
    state: Mapping[str, torch.Tensor],
    component: Literal["encoder", "quantizer", "decoder"],
) -> dict[str, torch.Tensor]:
    """Extract an explicit component; its caller must load with strict=True."""
    if component not in _PREFIXES:
        raise ValueError(f"Unknown GCP component: {component}")
    full = any(
        key.startswith(
            ("vqvae.", "encoder.featuriser.", "encoder.encoder.", "quantizer.")
        )
        for key in state
    )
    result = {}
    for key, tensor in state.items():
        if full:
            matches = [
                (owner, key[len(prefix) :], replacement)
                for owner, prefixes in _PREFIXES.items()
                for prefix, replacement in prefixes.items()
                if key.startswith(prefix)
            ]
            if len(matches) != 1:
                raise ValueError(f"Unexpected or ambiguous component key: {key}")
            owner, suffix, replacement = matches[0]
            target = replacement + suffix
        else:
            owner, target = component, key
        if owner == "decoder" and target.startswith("pairwise_classification_head."):
            if target not in _UNUSED_HEAD:
                raise ValueError(f"Unexpected auxiliary tensor: {key}")
            projection = next(
                (
                    state[k]
                    for k in (
                        "vqvae.decoder.projector_in.weight",
                        "decoder.projector_in.weight",
                        "projector_in.weight",
                    )
                    if k in state
                ),
                None,
            )
            if projection is None or projection.ndim != 2 or projection.shape[1] != 256:
                raise ValueError(
                    "Only the verified Large decoder auxiliary tensors may be excluded"
                )
            logging.getLogger(__name__).info(
                "Excluded unused Large decoder tensor: %s", key
            )
            continue
        if owner == component:
            if target in result:
                raise ValueError(f"Component key mapping collision: {key}")
            result[target] = tensor
    if not result:
        raise ValueError(f"Missing GCP-VQVAE component: {component}")
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash complete tensor state, including dtype, shape, names and buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(
            json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inference_metadata(device: torch.device) -> dict:
    """Record actual execution and source bytes even for an uncommitted checkout."""
    package = Path(__file__).parents[1]
    source = {
        str(path.relative_to(package)): file_sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=package,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    dependencies = {}
    for name in (
        "torch",
        "numpy",
        "biopython",
        "graphein",
        "torch-geometric",
        "x-transformers",
        "vector-quantize-pytorch",
        "pyarrow",
    ):
        dependencies[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(),
        "dependencies": dependencies,
        "device": str(device),
        "dtype": "float32",
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "accelerator": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "attention": "SDPA",
        "stok_revision": revision,
        "implementation_sha256": json_sha256(source),
        "source_files": source,
    }


def resolve_gcp_artifact(
    preset: str,
    *,
    cache_dir: str | Path | None = None,
    progress: bool = True,
) -> Path:
    preset = canonical_preset(preset)
    artifact = ARTIFACTS[preset]
    if cache_dir is None:
        cache_dir = (
            os.environ.get("STOK_GCP_CACHE")
            or Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "stok"
            / "gcp_vqvae"
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"gcp-{preset}-{artifact['sha256']}.pth"

    def verify(path: Path) -> None:
        if file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"Checkpoint digest mismatch: {path}")

    if destination.is_file():
        verify(destination)
        return destination
    url = f"https://huggingface.co/{artifact['repository']}/resolve/{artifact['revision']}/{artifact['filename']}"
    with tempfile.NamedTemporaryFile(
        dir=cache_dir, prefix=f".{preset}-", suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.hub.download_url_to_file(url, str(temporary), progress=progress)
        verify(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_gcp_config(preset: str) -> dict:
    filename = canonical_preset(preset) + ".yaml"
    path = resources.files("stok") / "configs" / "gcp_vqvae" / filename
    with path.open("r") as handle:
        return yaml.safe_load(handle)
