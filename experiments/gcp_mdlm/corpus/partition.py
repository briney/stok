"""Phase 3: partition staging shards into per-split sharded parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_COLUMNS = ["sequence_id", "sequence", "structure_tokens", "length", "mean_plddt"]


def partition(
    staging_dir: str | Path, split_map: dict[str, str], out_dir: str | Path,
    *, rows_per_shard: int = 5000,
) -> dict[str, int]:
    """Route staging rows to ``out_dir/{split}/part_*.parquet`` by ``sequence_id``."""
    staging_dir, out_dir = Path(staging_dir), Path(out_dir)
    buffers: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    part_idx = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0, "dropped": 0}

    def flush(split: str, force: bool = False) -> None:
        rows = sum(len(b) for b in buffers[split])
        if rows and (force or rows >= rows_per_shard):
            df = pd.concat(buffers[split], ignore_index=True)[_COLUMNS]
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(split_dir / f"part_{part_idx[split]:05d}.parquet", index=False)
            part_idx[split] += 1
            buffers[split] = []

    for pq in sorted(staging_dir.glob("shard_*.parquet")):
        df = pd.read_parquet(pq)
        df["_split"] = df["sequence_id"].map(split_map)
        counts["dropped"] += int(df["_split"].isna().sum())
        for split in ("train", "val", "test"):
            part = df[df["_split"] == split]
            if len(part):
                buffers[split].append(part[_COLUMNS])
                counts[split] += len(part)
                flush(split)
    for split in ("train", "val", "test"):
        flush(split, force=True)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("staging_dir")
    p.add_argument("splits_json")
    p.add_argument("out_dir")
    p.add_argument("--rows-per-shard", type=int, default=5000)
    args = p.parse_args()
    split_map = json.loads(Path(args.splits_json).read_text())
    counts = partition(args.staging_dir, split_map, args.out_dir, rows_per_shard=args.rows_per_shard)
    print(counts)


if __name__ == "__main__":
    main()
