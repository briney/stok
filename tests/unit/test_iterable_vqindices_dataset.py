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
                "sequence_id": f"{name}_{i}",
                "sequence": seq,
                "structure_tokens": list(range(len(seq))),
            }
        )
    df = pd.DataFrame(data)
    df.to_parquet(dir_path / f"{name}.parquet", index=False)


def test_iterable_len_equals_total_rows_single_rank(tmp_path):
    d = tmp_path / "shards"
    _write_shard(d, "a", rows=5)
    _write_shard(d, "b", rows=7)
    ds = IterableTokenizedDataset(
        d.as_posix(), max_length=16, shuffle_shards=False, shuffle_rows=False, seed=123
    )
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
    ds = IterableTokenizedDataset(
        d.as_posix(), max_length=16, shuffle_shards=True, shuffle_rows=True, seed=0
    )
    # collect pids for two epochs and ensure order differs
    epoch1 = [item["sequence_id"] for item in ds]
    epoch2 = [item["sequence_id"] for item in ds]
    assert len(epoch1) == len(epoch2) == len(ds)
    assert epoch1 != epoch2


def test_mixed_optional_columns_and_required_schema(tmp_path):
    from stok.data.mdlm import prepare_mdlm_batch
    from stok.utils.tokenizer import Tokenizer
    import torch

    _write_shard(tmp_path, "a", 1)
    df = pd.DataFrame(
        [
            {
                "sequence_id": "b",
                "sequence": "LA",
                "structure_tokens": [1, 2],
                "coordinates": [[[0.0, 0.0, 0.0]] * 3] * 2,
            }
        ]
    )
    df.to_parquet(tmp_path / "b.parquet", index=False)
    ds = IterableTokenizedDataset(
        str(tmp_path),
        max_length=None,
        shuffle_shards=False,
        shuffle_rows=False,
        load_coords=True,
    )
    batch = list(ds)
    for row in batch:
        row["dataset"] = "mixed-columns"
    coords = prepare_mdlm_batch(
        batch, Tokenizer(), max_len=8, codebook_size=32, crop="center", seeds=[0, 0]
    )["coords"]
    assert coords.shape == (2, 8, 3, 3)
    assert torch.isnan(coords[0]).all()
    assert torch.isfinite(coords[1, 1:3]).all()
    df.drop(columns="structure_tokens").to_parquet(
        tmp_path / "bad.parquet", index=False
    )
    with pytest.raises(ValueError, match="bad.parquet.*structure_tokens"):
        IterableTokenizedDataset(str(tmp_path), max_length=8)
