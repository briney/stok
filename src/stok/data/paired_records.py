"""Aligned (sequence, GCP structure-token) records for the Stage 1 corpus.

Enforces the audit §5.3 alignment contract: sequence length must equal the number
of structure tokens, per residue, with no silent truncation or filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedRecord:
    """One monomer: residue-aligned sequence and structure tokens."""

    sequence_id: str
    sequence: str
    structure_tokens: np.ndarray  # int64, shape (L,)
    valid_residue_mask: np.ndarray  # bool, shape (L,)


def _parse_tokens(value: object) -> np.ndarray:
    """Coerce a cell into an int64 token vector (list/ndarray or space-delimited str)."""
    if isinstance(value, str):
        return np.array([int(x) for x in value.split()], dtype=np.int64)
    return np.asarray(list(value), dtype=np.int64)


def load_paired_records(
    path: str | Path,
    *,
    id_column: str = "sequence_id",
    seq_column: str = "sequence",
    token_column: str = "structure_tokens",
    pad_sentinel: int = -1,
) -> list[PairedRecord]:
    """Load and validate aligned records from the three-column corpus parquet.

    Args:
        path: Parquet file with ``id_column``, ``seq_column``, ``token_column``.
        pad_sentinel: Token value marking an invalid/unresolved residue.

    Returns:
        One ``PairedRecord`` per row, in file order.

    Raises:
        ValueError: If a row's sequence length differs from its token count.
    """
    frame = pd.read_parquet(Path(path))
    missing = {id_column, seq_column, token_column} - set(frame.columns)
    if missing:
        raise ValueError(f"corpus missing columns: {sorted(missing)}")

    records: list[PairedRecord] = []
    for row in frame.itertuples(index=False):
        sample_id = str(getattr(row, id_column))
        sequence = str(getattr(row, seq_column))
        tokens = _parse_tokens(getattr(row, token_column))
        if len(sequence) != len(tokens):
            raise ValueError(
                f"length mismatch for {sample_id!r}: "
                f"{len(sequence)} residues vs {len(tokens)} tokens"
            )
        valid = tokens != pad_sentinel
        records.append(PairedRecord(sample_id, sequence, tokens, valid))
    return records
