import pandas as pd

from experiments.gcp_mdlm.corpus.partition import partition


def _staging(tmp_path):
    d = tmp_path / "_staging"
    d.mkdir()
    pd.DataFrame({
        "sequence_id": ["A", "B", "C"], "sequence": ["MK", "AA", "GG"],
        "structure_tokens": [[1, 2], [3, 4], [5, 6]], "length": [2, 2, 2],
        "mean_plddt": [80.0, 85.0, 90.0],
    }).to_parquet(d / "shard_00000.parquet")
    return d


def test_partition_routes_rows(tmp_path):
    d = _staging(tmp_path)
    split = {"A": "train", "B": "val", "C": "test"}
    out = tmp_path / "out"
    counts = partition(d, split, out, rows_per_shard=5000)
    assert counts == {"train": 1, "val": 1, "test": 1, "dropped": 0}
    train = pd.read_parquet(out / "train")
    assert train["sequence_id"].tolist() == ["A"]
    assert list(train.columns) == ["sequence_id", "sequence", "structure_tokens", "length", "mean_plddt"]
