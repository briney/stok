"""Phase 1 entrypoint: resumable, sharded structure tokenization."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import pandas as pd

from .filters import CorpusFilters
from .manifest import build_corpus_manifest
from .tokenize import Row, StructureOutcome, tokenize_paths


def shard_paths(paths: list[str], shard_size: int) -> list[list[str]]:
    """Split a pre-sorted path list into contiguous shards of ``shard_size``."""
    return [paths[i : i + shard_size] for i in range(0, len(paths), shard_size)]


def write_shard(
    rows: list[Row], outcomes: list[StructureOutcome], staging_dir: Path, shard_index: int
) -> None:
    """Atomically write one shard's parquet rows and its outcome sidecar."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "sequence_id": [r.sequence_id for r in rows],
            "sequence": [r.sequence for r in rows],
            "structure_tokens": [r.structure_tokens for r in rows],
            "length": [r.length for r in rows],
            "mean_plddt": [r.mean_plddt for r in rows],
        }
    )
    # Write outcomes atomically first (write-commit ordering for resume safety)
    outcomes_path = staging_dir / f"shard_{shard_index:05d}.outcomes.json"
    outcomes_tmp = outcomes_path.with_suffix(".outcomes.json.tmp")
    outcomes_tmp.write_text(json.dumps([dataclasses.asdict(o) for o in outcomes]))
    outcomes_tmp.replace(outcomes_path)
    # Then write parquet atomically as final completion signal for resume checks
    pq = staging_dir / f"shard_{shard_index:05d}.parquet"
    tmp = pq.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(pq)


def run(
    dataset_dir: str | Path,
    staging_dir: str | Path,
    *,
    preset: str = "base",
    batch_size: int = 32,
    num_workers: int = 16,
    shard_size: int = 2000,
    limit: int | None = None,
    batch_forward: bool = True,
    device: str = "cuda",
) -> dict:
    """Tokenize all ``*.cif.gz`` under ``dataset_dir`` into resumable staging shards."""
    from stok.models.structure_encoder import load_pretrained_encoder

    dataset_dir, staging_dir = Path(dataset_dir), Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(str(p) for p in dataset_dir.glob("*.cif.gz"))
    if limit is not None:
        paths = paths[:limit]
    shards = shard_paths(paths, shard_size)
    encoder = load_pretrained_encoder(
        preset, path=os.environ.get("STOK_ENCODER_CHECKPOINT"), device=device, freeze=True
    )
    filters = CorpusFilters()
    summary = {"shards": len(shards), "written": 0, "skipped": 0, "rows": 0}
    for i, shard in enumerate(shards):
        if (staging_dir / f"shard_{i:05d}.parquet").exists():
            summary["skipped"] += 1
            continue
        rows, outcomes = tokenize_paths(
            shard,
            encoder,
            filters,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            batch_forward=batch_forward,
        )
        write_shard(rows, outcomes, staging_dir, i)
        summary["written"] += 1
        summary["rows"] += len(rows)
        print(f"shard {i}/{len(shards)}: {len(rows)} rows, {len(outcomes)} rejected")
    manifest = build_corpus_manifest(
        encoder_checkpoint=os.environ.get("STOK_ENCODER_CHECKPOINT"),
        codebook_checkpoint=None,
        preset=preset,
        min_mean_plddt=filters.min_mean_plddt,
        max_length=encoder.max_length,
        mmseqs_params={},  # filled by Phase 2 (cluster_split) into its own _split/ manifest
        split_seed=-1,
        val_size=-1,
        test_size=-1,
    )
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_dir")
    p.add_argument("staging_dir")
    p.add_argument("--preset", default="base")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-batch-forward", dest="batch_forward", action="store_false")
    args = p.parse_args()
    summary = run(
        args.dataset_dir,
        args.staging_dir,
        preset=args.preset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
        limit=args.limit,
        batch_forward=args.batch_forward,
        device=args.device,
    )
    print(summary)


if __name__ == "__main__":
    main()
