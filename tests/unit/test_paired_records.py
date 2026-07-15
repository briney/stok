import numpy as np
import pandas as pd
import pytest

from stok.data.paired_records import PairedRecord, load_paired_records


def _write_parquet(tmp_path, rows):
    path = tmp_path / "corpus.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_loads_aligned_records(tmp_path):
    path = _write_parquet(
        tmp_path,
        [
            {"sequence_id": "p1", "sequence": "MKV", "structure_tokens": [10, 11, 12]},
            {"sequence_id": "p2", "sequence": "AA", "structure_tokens": [3, -1]},
        ],
    )
    records = load_paired_records(path)
    assert [r.sequence_id for r in records] == ["p1", "p2"]
    assert isinstance(records[0], PairedRecord)
    np.testing.assert_array_equal(records[0].structure_tokens, np.array([10, 11, 12]))
    np.testing.assert_array_equal(records[0].valid_residue_mask, np.array([True, True, True]))
    # pad_sentinel (-1) marks an invalid residue
    np.testing.assert_array_equal(records[1].valid_residue_mask, np.array([True, False]))


def test_accepts_space_delimited_tokens(tmp_path):
    path = _write_parquet(
        tmp_path, [{"sequence_id": "p1", "sequence": "MK", "structure_tokens": "7 8"}]
    )
    records = load_paired_records(path)
    np.testing.assert_array_equal(records[0].structure_tokens, np.array([7, 8]))


def test_rejects_length_mismatch(tmp_path):
    path = _write_parquet(
        tmp_path, [{"sequence_id": "bad", "sequence": "MKV", "structure_tokens": [1, 2]}]
    )
    with pytest.raises(ValueError, match="length mismatch.*bad"):
        load_paired_records(path)
