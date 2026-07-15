import pandas as pd

from experiments.gcp_mdlm.corpus.partition import partition


def _staging(tmp_path):
    d = tmp_path / "_staging"
    d.mkdir()
    pd.DataFrame(
        {
            "sequence_id": ["A", "B", "C"],
            "sequence": ["MK", "AA", "GG"],
            "structure_tokens": [[1, 2], [3, 4], [5, 6]],
            "length": [2, 2, 2],
            "mean_plddt": [80.0, 85.0, 90.0],
        }
    ).to_parquet(d / "shard_00000.parquet")
    return d


def test_partition_routes_rows(tmp_path):
    d = _staging(tmp_path)
    split = {"A": "train", "B": "val", "C": "test"}
    out = tmp_path / "out"
    counts = partition(d, split, out, rows_per_shard=5000)
    assert counts == {"train": 1, "val": 1, "test": 1, "dropped": 0}
    train = pd.read_parquet(out / "train")
    assert train["sequence_id"].tolist() == ["A"]
    assert list(train.columns) == [
        "sequence_id",
        "sequence",
        "structure_tokens",
        "length",
        "mean_plddt",
    ]


def test_partition_flush_shards_and_drops(tmp_path):
    d = tmp_path / "_staging"
    d.mkdir()
    # Shard 0: 3 train rows (triggers a mid-stream size flush on its own),
    # 1 val row (buffered, below threshold), 1 row absent from split_map (dropped).
    pd.DataFrame(
        {
            "sequence_id": ["TR1", "TR2", "TR3", "VA1", "ZZ1"],
            "sequence": ["MK", "AA", "GG", "CC", "TT"],
            "structure_tokens": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            "length": [2, 2, 2, 2, 2],
            "mean_plddt": [80.0, 81.0, 82.0, 83.0, 84.0],
        }
    ).to_parquet(d / "shard_00000.parquet")
    # Shard 1: 2 more train rows (own size flush), 1 more val row (completes the
    # cross-shard val buffer and triggers its flush), 1 test row (only written via
    # the final forced flush since it never reaches rows_per_shard alone).
    pd.DataFrame(
        {
            "sequence_id": ["TR4", "TR5", "VA2", "TE1"],
            "sequence": ["LL", "PP", "QQ", "RR"],
            "structure_tokens": [[11, 12], [13, 14], [15, 16], [17, 18]],
            "length": [2, 2, 2, 2],
            "mean_plddt": [85.0, 86.0, 87.0, 88.0],
        }
    ).to_parquet(d / "shard_00001.parquet")

    split_map = {
        "TR1": "train",
        "TR2": "train",
        "TR3": "train",
        "TR4": "train",
        "TR5": "train",
        "VA1": "val",
        "VA2": "val",
        "TE1": "test",
        # "ZZ1" intentionally absent -> dropped.
    }
    out = tmp_path / "out"
    counts = partition(d, split_map, out, rows_per_shard=2)

    assert counts == {"train": 5, "val": 2, "test": 1, "dropped": 1}
    total_input_rows = 5 + 4
    assert counts["train"] + counts["val"] + counts["test"] + counts["dropped"] == total_input_rows

    train_dir = out / "train"
    assert (train_dir / "part_00000.parquet").exists()
    assert (train_dir / "part_00001.parquet").exists()

    for split, expected_ids in (
        ("train", {"TR1", "TR2", "TR3", "TR4", "TR5"}),
        ("val", {"VA1", "VA2"}),
        ("test", {"TE1"}),
    ):
        part_files = sorted((out / split).glob("part_*.parquet"))
        assert part_files
        combined = pd.concat((pd.read_parquet(p) for p in part_files), ignore_index=True)
        assert set(combined["sequence_id"]) == expected_ids
        assert len(combined) == len(expected_ids)
        assert list(combined.columns) == [
            "sequence_id",
            "sequence",
            "structure_tokens",
            "length",
            "mean_plddt",
        ]
