import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pathlib import Path

from stok.data.dataset import InterleavedIterableDataset, IterableTokenizedDataset


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
    pq.write_table(pa.Table.from_pylist(data), dir_path / f"{name}.parquet")


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
    written = {
        "sequence_id": "b",
        "sequence": "LA",
        "structure_tokens": [1, 2],
        "coordinates": [[[0.0, 0.0, 0.0]] * 3] * 2,
    }
    pq.write_table(pa.Table.from_pylist([written]), tmp_path / "b.parquet")
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
    del written["structure_tokens"]
    pq.write_table(pa.Table.from_pylist([written]), tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="bad.parquet.*structure_tokens"):
        IterableTokenizedDataset(str(tmp_path), max_length=8)


def _three_shard_source(root, name, offset):
    root.mkdir()
    for shard in range(3):
        rows = [
            {
                "sequence_id": f"{name}{offset + shard * 10 + i}",
                "sequence": "ACD",
                "structure_tokens": [0, 1, 2],
            }
            for i in range(4)
        ]
        pq.write_table(pa.Table.from_pylist(rows), root / f"{shard}.parquet")
    return IterableTokenizedDataset(
        str(root), max_length=None, seed=7, dataset_name=name
    )


def test_rows_are_built_only_by_their_owner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from stok.data import dataset as module

    built = []
    original = module._build_output_from_row
    monkeypatch.setattr(
        module,
        "_build_output_from_row",
        lambda row, **kw: built.append(row["sequence_id"]) or original(row, **kw),
    )
    ds = _three_shard_source(tmp_path / "a", "a", 0)
    seen = []
    for rank in range(2):
        for worker in range(2):
            ds.rank, ds.world_size = rank, 2
            ds.set_epoch(0)
            monkeypatch.setattr(
                module,
                "get_worker_info",
                lambda w=worker: SimpleNamespace(id=w, num_workers=2),
            )
            seen.extend(row["sequence_id"] for row in ds)
    assert sorted(built) == sorted(seen) and len(seen) == 12


def test_stream_order_unchanged_by_lazy_rows(tmp_path):
    a = _three_shard_source(tmp_path / "a", "a", 0)
    b = _three_shard_source(tmp_path / "b", "b", 100)
    # Recorded on main before rows became lazy references.
    assert [row["sequence_id"] for row in a] == [
        "a1", "a2", "a3", "a0", "a12", "a13", "a10", "a11", "a22", "a20", "a23", "a21"
    ]  # fmt: skip
    mixture = InterleavedIterableDataset([a, b], [0.5, 0.5], seed=3)
    assert [row["sequence_id"] for row in mixture] == [
        "b101", "b102", "b103", "b100", "b112", "a1", "a2", "a3", "a0", "b113",
        "a12", "b110", "b111", "a13", "a10", "b122", "a11", "b120", "b123", "b121",
        "a22", "b123", "b122", "b121",
    ]  # fmt: skip
