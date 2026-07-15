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


def build_run_manifest(
    *,
    corpus_path: str | Path,
    codebook_path: str | Path,
    backbone_checkpoint: str | Path,
    decoder_checkpoint: str | Path,
) -> dict:
    """Assemble the four provenance hashes required in a Stage 1 run manifest.

    Hashes the training corpus, the GCP codebook, the frozen backbone checkpoint,
    and the frozen decoder checkpoint so a feature-cache manifest can be traced back
    to the exact inputs that produced it.

    Args:
        corpus_path: Path to the paired-record corpus (parquet) used to build the cache.
        codebook_path: Path to the GCP-VQVAE codebook file.
        backbone_checkpoint: Path to the frozen seq-only backbone checkpoint.
        decoder_checkpoint: Path to the frozen GCP decoder checkpoint.

    Returns:
        A dict with keys ``corpus_sha256``, ``codebook_sha256``,
        ``backbone_checkpoint_sha256``, and ``decoder_checkpoint_sha256``, intended
        to be passed as ``manifest_extra=`` to ``write_feature_cache``.
    """
    return {
        "corpus_sha256": sha256_file(corpus_path),
        "codebook_sha256": sha256_file(codebook_path),
        "backbone_checkpoint_sha256": sha256_file(backbone_checkpoint),
        "decoder_checkpoint_sha256": sha256_file(decoder_checkpoint),
    }
