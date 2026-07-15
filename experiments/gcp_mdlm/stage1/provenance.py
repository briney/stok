"""Content hashes for reproducibility manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array's dtype, shape, and raw bytes."""
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()
