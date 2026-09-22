import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import torch

from stok.data.dataset import TokenizedDataset


pytest.importorskip("pyarrow")


def _make_coords(L: int) -> list[list[list[float]]]:
    # Simple deterministic coords: residue i -> [[i,0,0],[i,1,0],[i,0,1]]
    out = []
    for i in range(L):
        out.append([[float(i), 0.0, 0.0], [float(i), 1.0, 0.0], [float(i), 0.0, 1.0]])
    return out


def test_dataset_parquet_with_coords_returns_coords_tensor(tmp_path):
    pq = tmp_path / "with_coords.parquet"
    rows = []
    for i in range(3):
        seq = "ACDEFGHIKLM"[: 6 + i]  # variable lengths
        L = len(seq)
        rows.append(
            {
                "pid": f"p{i}",
                "protein_sequence": seq,
                "indices": list(range(L)),  # arbitrary
                "coordinates": _make_coords(L),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(pq, index=False)

    max_len = 8
    ds = TokenizedDataset(str(pq), max_length=max_len)
    item = ds[0]
    assert "coords" in item
    coords = item["coords"]
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (max_len, 3, 3)
    # NaNs in padded tail
    L0 = len(rows[0]["coordinates"])
    pad_after = coords[L0:]
    assert torch.isnan(pad_after).all()
    # Head matches the first residue pattern
    np.testing.assert_allclose(coords[0].numpy(), np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))


def test_dataset_parquet_without_coords_omits_key(tmp_path):
    pq = tmp_path / "no_coords.parquet"
    rows = []
    for i in range(2):
        seq = "ACDEFGHIKLM"[: 6 + i]
        rows.append(
            {
                "pid": f"q{i}",
                "protein_sequence": seq,
                "indices": list(range(len(seq))),
            }
        )
    pd.DataFrame(rows).to_parquet(pq, index=False)

    ds = TokenizedDataset(str(pq), max_length=8)
    item = ds[0]
    assert "coords" not in item


def test_dataset_csv_never_includes_coords(tmp_path):
    csv = tmp_path / "data.csv"
    rows = []
    for i in range(2):
        seq = "ACDEFGHIKLM"[: 6 + i]
        indices_str = " ".join(str(x) for x in range(len(seq)))
        rows.append(
            {
                "pid": f"c{i}",
                "protein_sequence": seq,
                "indices": indices_str,
            }
        )
    pd.DataFrame(rows).to_csv(csv, index=False)

    ds = TokenizedDataset(str(csv), max_length=8)
    item = ds[0]
    assert "coords" not in item




def test_parser_preserves_missing_positions_and_rejects_bad_lengths(tmp_path):
    from stok.data.dataset import BaseTokenizedDataset
    row = pd.Series({'pid': 'p', 'protein_sequence': 'LAG', 'indices': [7, None, 9]})
    item = BaseTokenizedDataset._build_output_from_row(row, max_length=6, has_coords=False)
    assert item['indices'].tolist() == [7, -1, 9, -1, -1, -1]
    row['indices'] = [7, 9]
    with pytest.raises(ValueError, match='p.*length'):
        BaseTokenizedDataset._build_output_from_row(row, max_length=6, has_coords=False)
