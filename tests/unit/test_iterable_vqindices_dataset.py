import pandas as pd
import pytest
from pathlib import Path

from stok.data.dataset import IterableTokenizedDataset


pytest.importorskip("pyarrow")


def _write_shard(dir_path: Path, name: str, rows: int, indices_len: int = 5):
    dir_path.mkdir(parents=True, exist_ok=True)
    data = []
    for i in range(rows):
        seq = "ACDEFGHIKLMNPQRSTVWY"[: (6 + (i % 10))]
        data.append(
            {
                "pid": f"{name}_{i}",
                "protein_sequence": seq,
                "indices": list(range(indices_len)),
            }
        )
    df = pd.DataFrame(data)
    df.to_parquet(dir_path / f"{name}.parquet", index=False)


def test_iterable_len_equals_total_rows_single_rank(tmp_path):
    d = tmp_path / "shards"
    _write_shard(d, "a", rows=5)
    _write_shard(d, "b", rows=7)
    ds = IterableTokenizedDataset(d.as_posix(), max_length=16, shuffle_shards=False, shuffle_rows=False, seed=123)
    # world_size=1 -> __len__ equals total rows
    assert len(ds) == 12
    # exhaust iterator to ensure it yields len(ds) items
    got = 0
    for _ in ds:
        got += 1
    assert got == len(ds)


def test_iterable_epoch_shuffle_changes_order(tmp_path):
    d = tmp_path / "shards2"
    _write_shard(d, "x", rows=4)
    _write_shard(d, "y", rows=4)
    ds = IterableTokenizedDataset(d.as_posix(), max_length=16, shuffle_shards=True, shuffle_rows=True, seed=0)
    # collect pids for two epochs and ensure order differs
    epoch1 = [item["pid"] for item in ds]
    epoch2 = [item["pid"] for item in ds]
    assert len(epoch1) == len(epoch2) == len(ds)
    assert epoch1 != epoch2

