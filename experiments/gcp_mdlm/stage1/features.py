"""Memory-bounded, two-pass feature cache for the Stage 1 frozen backbone.

Pass 1 counts valid residues (cheap, no model) to size a float16 memmap; pass 2
runs the frozen backbone in minibatches and writes only valid-residue features.
Storage is flattened across proteins with a protein index for per-protein grouping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stok.data.paired_records import PairedRecord

from . import provenance

_FEATURES = "features.npy"
_TOKENS = "token_ids.npy"
_INDEX = "protein_index.json"
_MANIFEST = "manifest.json"


def write_feature_cache(
    cache_dir: str | Path,
    records: list[PairedRecord],
    encode_fn,
    tokenizer,
    *,
    d_model: int,
    max_train_proteins: int | None = None,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    manifest_extra: dict | None = None,
) -> dict:
    """Write a feature cache for ``records`` and return the manifest dict."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if max_train_proteins is not None:
        records = records[:max_train_proteins]

    # Pass 1: size the cache.
    valid_counts = [int(r.valid_residue_mask.sum()) for r in records]
    n_res = int(sum(valid_counts))

    features = np.lib.format.open_memmap(
        cache_dir / _FEATURES, mode="w+", dtype=np.float16, shape=(n_res, d_model)
    )
    token_ids = np.empty(n_res, dtype=np.int64)
    protein_index: list[tuple[str, int, int]] = []

    # Pass 2: fill in minibatches.
    cursor = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        seqs = [torch.tensor(tokenizer.encode(r.sequence, add_special_tokens=False)) for r in batch]
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        kpm = torch.ones(len(batch), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = s
            kpm[i, : len(s)] = False
        padded = padded.to(device)
        kpm = kpm.to(device)
        with torch.no_grad():
            feats = encode_fn(padded, kpm).float().cpu().numpy()  # (B, L, d_model)
        for i, rec in enumerate(batch):
            valid = rec.valid_residue_mask
            n_valid = int(valid.sum())
            protein_index.append((rec.sequence_id, cursor, n_valid))
            features[cursor : cursor + n_valid] = feats[i, : len(valid)][valid].astype(np.float16)
            token_ids[cursor : cursor + n_valid] = rec.structure_tokens[valid]
            cursor += n_valid
    features.flush()
    np.save(cache_dir / _TOKENS, token_ids)
    (cache_dir / _INDEX).write_text(json.dumps(protein_index))

    manifest = {
        "n_residues": n_res,
        "n_proteins": len(records),
        "d_model": d_model,
        "features_sha256": provenance.sha256_array(np.asarray(features)),
        "token_ids_sha256": provenance.sha256_array(token_ids),
        **(manifest_extra or {}),
    }
    (cache_dir / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


@dataclass
class CachedFeatures:
    """Loaded feature cache with per-protein grouping."""

    features: np.ndarray  # float16 memmap (N_res, d_model)
    token_ids: np.ndarray  # int64 (N_res,)
    protein_ranges: list[tuple[str, int, int]]
    manifest: dict

    @classmethod
    def load(cls, cache_dir: str | Path) -> CachedFeatures:
        cache_dir = Path(cache_dir)
        features = np.load(cache_dir / _FEATURES, mmap_mode="r")
        token_ids = np.load(cache_dir / _TOKENS)
        ranges = [tuple(x) for x in json.loads((cache_dir / _INDEX).read_text())]
        manifest = json.loads((cache_dir / _MANIFEST).read_text())
        return cls(features, token_ids, ranges, manifest)
