"""Materialize a sequence-only parquet in the schema the existing CLI expects.

The corpus uses (sequence_id, sequence, structure_tokens); seq-only pretraining
via ``stok train`` reads (pid, protein_sequence). This drops the tokens and renames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def prepare_pretrain_parquet(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    id_column: str = "sequence_id",
    seq_column: str = "sequence",
) -> int:
    """Write ``dst_path`` with columns ``pid``/``protein_sequence``; return row count."""
    frame = pd.read_parquet(Path(src_path), columns=[id_column, seq_column])
    out = frame.rename(columns={id_column: "pid", seq_column: "protein_sequence"})
    out = out[["pid", "protein_sequence"]]
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(dst_path), index=False)
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="corpus parquet (sequence_id, sequence, structure_tokens)")
    parser.add_argument("dst", help="output parquet (pid, protein_sequence)")
    args = parser.parse_args()
    n = prepare_pretrain_parquet(args.src, args.dst)
    print(f"wrote {n} rows to {args.dst}")


if __name__ == "__main__":
    main()
