import json

import pandas as pd

from experiments.gcp_mdlm.corpus.run_tokenize import shard_paths, write_shard
from experiments.gcp_mdlm.corpus.tokenize import Row, StructureOutcome


def test_shard_paths_deterministic_contiguous():
    paths = [f"p{i}" for i in range(5)]
    assert shard_paths(paths, 2) == [["p0", "p1"], ["p2", "p3"], ["p4"]]


def test_write_shard_schema_and_outcomes(tmp_path):
    rows = [Row("A", "MKV", [1, 2, 3], 3, 88.0), Row("B", "AA", [4, 5], 2, 91.0)]
    outcomes = [StructureOutcome("C", "rejected_plddt")]
    write_shard(rows, outcomes, tmp_path, 7)
    pq = tmp_path / "shard_00007.parquet"
    df = pd.read_parquet(pq)
    assert list(df.columns) == [
        "sequence_id",
        "sequence",
        "structure_tokens",
        "length",
        "mean_plddt",
    ]
    assert df["sequence_id"].tolist() == ["A", "B"]
    assert [len(t) for t in df["structure_tokens"]] == df["length"].tolist()
    oc = json.loads((tmp_path / "shard_00007.outcomes.json").read_text())
    assert oc == [{"sequence_id": "C", "status": "rejected_plddt"}]
