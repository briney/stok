"""Phase 2: mmseqs 30%-identity clustering + whole-cluster train/val/test assignment."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def write_fasta(staging_dir: str | Path, fasta_path: str | Path) -> int:
    """Write one FASTA record per accepted sequence across all staging shards."""
    staging_dir, fasta_path = Path(staging_dir), Path(fasta_path)
    count = 0
    with open(fasta_path, "w") as handle:
        for pq in sorted(staging_dir.glob("shard_*.parquet")):
            df = pd.read_parquet(pq, columns=["sequence_id", "sequence"])
            for sid, seq in zip(df["sequence_id"], df["sequence"]):
                handle.write(f">{sid}\n{seq}\n")
                count += 1
    return count


def run_mmseqs_cluster(
    fasta_path: str | Path,
    out_prefix: str | Path,
    tmp_dir: str | Path,
    *,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
) -> Path:
    """Run ``mmseqs easy-cluster``; return the ``*_cluster.tsv`` path (representative, member)."""
    out_prefix = Path(out_prefix)
    subprocess.run(
        [
            "mmseqs",
            "easy-cluster",
            str(fasta_path),
            str(out_prefix),
            str(tmp_dir),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
        ],
        check=True,
    )
    return Path(f"{out_prefix}_cluster.tsv")


def assign_splits(
    cluster_tsv: str | Path, *, val_size: int, test_size: int, seed: int = 0
) -> dict[str, str]:
    """Assign whole clusters to train/val/test so no cluster crosses a split."""
    members: dict[str, list[str]] = defaultdict(list)
    for line in Path(cluster_tsv).read_text().splitlines():
        rep, mem = line.split("\t")
        members[rep].append(mem)
    reps = sorted(members)
    rng = np.random.default_rng(seed)
    rng.shuffle(reps)
    split: dict[str, str] = {}
    val_n = test_n = 0
    for rep in reps:
        cluster = members[rep]
        if val_n < val_size:
            target = "val"
            val_n += len(cluster)
        elif test_n < test_size:
            target = "test"
            test_n += len(cluster)
        else:
            target = "train"
        for mem in cluster:
            split[mem] = target
    return split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("staging_dir")
    p.add_argument("work_dir", help="dir for fasta/mmseqs outputs/splits.json")
    p.add_argument("--min-seq-id", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--test-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "sequences.fasta"
    n = write_fasta(args.staging_dir, fasta)
    tsv = run_mmseqs_cluster(
        fasta,
        work / "clust",
        work / "tmp",
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
    )
    split = assign_splits(tsv, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    (work / "splits.json").write_text(json.dumps(split))
    counts = {s: sum(1 for v in split.values() if v == s) for s in ("train", "val", "test")}
    print(f"{n} sequences; splits: {counts}")


if __name__ == "__main__":
    main()
