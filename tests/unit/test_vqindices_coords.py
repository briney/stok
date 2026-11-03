import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import torch

from stok.data.dataset import IterableTokenizedDataset


def first_item(directory, max_length):
    dataset = IterableTokenizedDataset(
        str(directory), max_length=max_length, shuffle_shards=False, shuffle_rows=False
    )
    return next(iter(dataset))


def _make_coords(L: int) -> list[list[list[float]]]:
    # Simple deterministic coords: residue i -> [[i,0,0],[i,1,0],[i,0,1]]
    out = []
    for i in range(L):
        out.append([[float(i), 0.0, 0.0], [float(i), 1.0, 0.0], [float(i), 0.0, 1.0]])
    return out


def test_dataset_parquet_with_coords_returns_coords_tensor(tmp_path):
    rows = []
    for i in range(3):
        seq = "ACDEFGHIKLM"[: 6 + i]  # variable lengths
        L = len(seq)
        rows.append(
            {
                "sequence_id": f"p{i}",
                "sequence": seq,
                "structure_tokens": list(range(L)),  # arbitrary
                "coordinates": _make_coords(L),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "with_coords.parquet")

    max_len = 8
    item = first_item(tmp_path, max_len)
    assert "coords" in item
    coords = item["coords"]
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (max_len, 3, 3)
    # NaNs in padded tail
    L0 = len(rows[0]["coordinates"])
    pad_after = coords[L0:]
    assert torch.isnan(pad_after).all()
    # Head matches the first residue pattern
    np.testing.assert_allclose(
        coords[0].numpy(),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_dataset_parquet_without_coords_omits_key(tmp_path):
    rows = []
    for i in range(2):
        seq = "ACDEFGHIKLM"[: 6 + i]
        rows.append(
            {
                "sequence_id": f"q{i}",
                "sequence": seq,
                "structure_tokens": list(range(len(seq))),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "no_coords.parquet")

    item = first_item(tmp_path, 8)
    assert "coords" not in item


def test_parser_preserves_missing_positions_and_rejects_bad_lengths(tmp_path):
    from stok.data.dataset import _build_output_from_row

    row = {"sequence_id": "p", "sequence": "LAG", "structure_tokens": [7, None, 9]}
    item = _build_output_from_row(row, max_length=6, has_coords=False)
    assert item["structure_tokens"].tolist() == [7, -1, 9]
    row["structure_tokens"] = [7, 9]
    with pytest.raises(ValueError, match="p.*length"):
        _build_output_from_row(row, max_length=6, has_coords=False)
