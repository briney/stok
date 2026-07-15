"""Reproducibility manifest for the corpus build."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_corpus_manifest(
    *,
    encoder_checkpoint: str | Path | None,
    codebook_checkpoint: str | Path | None,
    preset: str,
    min_mean_plddt: float,
    max_length: int,
    mmseqs_params: dict,
    split_seed: int,
    val_size: int,
    test_size: int,
) -> dict:
    """Assemble a provenance manifest; hash provided checkpoints (``"<none>"`` if absent)."""

    def _hash(p: str | Path | None) -> str:
        return sha256_file(p) if p is not None else "<none>"

    return {
        "preset": preset,
        "encoder_checkpoint_sha256": _hash(encoder_checkpoint),
        "codebook_checkpoint_sha256": _hash(codebook_checkpoint),
        "min_mean_plddt": min_mean_plddt,
        "max_length": max_length,
        "mmseqs_params": dict(mmseqs_params),
        "split": {"seed": split_seed, "val_size": val_size, "test_size": test_size},
    }
