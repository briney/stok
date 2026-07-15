import pandas as pd

from experiments.gcp_mdlm.stage1.prepare_pretrain_parquet import prepare_pretrain_parquet


def test_prepare_writes_cli_schema(tmp_path):
    src = tmp_path / "corpus.parquet"
    pd.DataFrame(
        [
            {"sequence_id": "p1", "sequence": "MKV", "structure_tokens": [1, 2, 3]},
            {"sequence_id": "p2", "sequence": "AA", "structure_tokens": [4, 5]},
        ]
    ).to_parquet(src)
    dst = tmp_path / "pretrain.parquet"

    n = prepare_pretrain_parquet(src, dst)

    assert n == 2
    out = pd.read_parquet(dst)
    assert list(out.columns) == ["pid", "protein_sequence"]
    assert out["pid"].tolist() == ["p1", "p2"]
    assert out["protein_sequence"].tolist() == ["MKV", "AA"]
