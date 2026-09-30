"""Utilities for loading codebook tensors."""

import importlib.resources as r
from pathlib import Path

import torch


def load_codebook(
    preset: str | None = None,
    path: str | None = None,
) -> torch.Tensor:
    """Load a codebook tensor from a custom path or built-in preset.

    Args:
        preset: Built-in preset name ("base" or "lite"). Defaults to "base" when not provided.
        path: Custom codebook file path. When set, this OVERRIDES any preset.

    Returns:
        Codebook tensor of shape [C, d_code]

    Raises:
        ValueError: If preset is not one of the supported values.
        FileNotFoundError: If path is provided but file doesn't exist.
    """
    # codebook path has precedence over preset
    if path is not None:
        codebook_path = Path(path)
        if not codebook_path.exists():
            raise FileNotFoundError(f"Codebook file not found: {path}")
        codebook = torch.load(codebook_path, map_location="cpu")
        return codebook

    # otherwise, use preset (default to "base")
    effective_preset = preset or "base"
    codebook_file = f"{effective_preset}.pt"
    pkg_path = r.files("stok") / "checkpoints" / "codebook" / codebook_file
    if not pkg_path.is_file():
        raise FileNotFoundError(f"Codebook file not found: {pkg_path}")
    with r.as_file(pkg_path) as path_obj:
        return torch.load(path_obj, map_location="cpu")
