import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from stok.data.mdlm import prepare_mdlm_batch
from stok.data.dataset import IterableTokenizedDataset
from stok.utils.tokenizer import Tokenizer


def ordered(directory, **kwargs):
    return IterableTokenizedDataset(
        str(directory), shuffle_shards=False, shuffle_rows=False, **kwargs
    )


def write_parquet(path, tokens=(4, None, 7, None), token_type=None):
    if token_type is None:
        token_type = pa.int32()
    table = pa.table(
        {
            "sequence_id": ["p1"],
            "sequence": ["ACDE"],
            "structure_tokens": pa.array([tokens], type=pa.list_(token_type)),
        }
    )
    pq.write_table(table, path)


@pytest.mark.parametrize("token_type", [pa.int32(), pa.int64()])
def test_null_tokens_preserve_residue_alignment(tmp_path, token_type):
    write_parquet(tmp_path / "data.parquet", token_type=token_type)
    item = next(iter(ordered(tmp_path, max_length=8)))

    assert item["sequence_id"] == "p1"
    assert item["sequence"] == "ACDE"
    assert item["structure_tokens"].tolist() == [4, -1, 7, -1]
    item["dataset"] = "alignment-fixture"
    item["canonical_id"] = "a" * 64
    batch = prepare_mdlm_batch(
        [item], Tokenizer(), max_len=8, codebook_size=8, crop="center", seeds=[0]
    )
    assert batch["structure_tokens"].tolist() == [[8, 4, 10, 7, 10, 8, 8, 8]]
    assert batch["structure_valid"].tolist() == [
        [False, True, False, True, False, False, False, False]
    ]
    truncated = prepare_mdlm_batch(
        [item], Tokenizer(), max_len=4, codebook_size=8, crop="center", seeds=[0]
    )
    assert truncated["structure_tokens"].tolist() == [[8, 10, 7, 8]]


def test_sequence_only_parquet_for_diagnostics(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"sequence_id": ["p1"], "sequence": ["ACDE"]}), path)
    item = next(iter(ordered(tmp_path, max_length=8, require_structure_tokens=False)))
    assert item == {"sequence_id": "p1", "sequence": "ACDE"}


@pytest.mark.parametrize(
    "tokens,token_type",
    [
        ("4 5 7 8", pa.string()),
        ([4.0, 5.0, 7.0, 8.0], pa.list_(pa.float64())),
    ],
)
def test_rejects_untyped_structure_tokens(tmp_path, tokens, token_type):
    path = tmp_path / "bad.parquet"
    table = pa.table(
        {
            "sequence_id": ["p1"],
            "sequence": ["ACDE"],
            "structure_tokens": pa.array([tokens], type=token_type),
        }
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="structure_tokens.*integer"):
        ordered(tmp_path, max_length=8)


@pytest.mark.parametrize(
    "tokens,match",
    [
        ([1, 2], "length"),
        ([1, -1, 2, 3], "negative"),
        (None, "null"),
    ],
)
def test_rejects_invalid_token_rows(tmp_path, tokens, match):
    write_parquet(tmp_path / "bad.parquet", tokens=tokens)
    with pytest.raises(ValueError, match=match):
        next(iter(ordered(tmp_path, max_length=8)))


def test_checks_required_columns_in_every_shard(tmp_path):
    write_parquet(tmp_path / "a.parquet")
    pq.write_table(
        pa.table({"sequence_id": ["p2"], "sequence": ["ACDE"]}), tmp_path / "b.parquet"
    )
    with pytest.raises(ValueError, match="b.parquet.*structure_tokens"):
        IterableTokenizedDataset(str(tmp_path), max_length=8)


def test_rejects_legacy_column_names(tmp_path):
    path = tmp_path / "old.parquet"
    pq.write_table(
        pa.table(
            {"pid": ["p1"], "protein_sequence": ["ACDE"], "indices": [[1, 2, 3, 4]]}
        ),
        path,
    )
    with pytest.raises(ValueError, match="sequence"):
        ordered(tmp_path, max_length=8)


@pytest.mark.parametrize("load_coords", [False, True])
def test_shards_with_missing_optional_coordinates(tmp_path, load_coords):
    write_parquet(tmp_path / "a.parquet")
    table = pa.table(
        {
            "sequence_id": ["p2"],
            "sequence": ["ACDE"],
            "structure_tokens": [[1, 2, 3, 4]],
            "coordinates": [[[[1.0, 2.0, 3.0]] * 3] * 4],
        }
    )
    pq.write_table(table, tmp_path / "b.parquet")
    ds = IterableTokenizedDataset(
        str(tmp_path),
        max_length=8,
        load_coords=load_coords,
        shuffle_shards=False,
        shuffle_rows=False,
    )
    first, second = list(ds)
    if load_coords:
        assert torch.isnan(first["coords"]).all()
        assert second["coords"].shape == (8, 3, 3)
        assert second["coords"][0, 0].tolist() == [1.0, 2.0, 3.0]
        assert torch.isnan(second["coords"][4:]).all()
    else:
        assert "coords" not in first
        assert "coords" not in second


@pytest.mark.parametrize("load_coords", [False, True])
@pytest.mark.parametrize(
    "coordinates,coordinate_type",
    [
        ([[["1", "2", "3"]] * 3] * 4, pa.list_(pa.list_(pa.list_(pa.string())))),
        ([[["x", "y", "z"]] * 3] * 4, pa.list_(pa.list_(pa.list_(pa.large_string())))),
        ([[[True, False, True]] * 3] * 4, pa.list_(pa.list_(pa.list_(pa.bool_())))),
        (1.0, pa.float64()),
        ([1.0, 2.0, 3.0], pa.list_(pa.float64())),
        ([[1.0, 2.0, 3.0]] * 4, pa.list_(pa.list_(pa.float64()))),
        (
            [[[[1.0, 2.0, 3.0]]] * 3] * 4,
            pa.list_(pa.list_(pa.list_(pa.list_(pa.float64())))),
        ),
    ],
    ids=[
        "numeric-strings",
        "strings",
        "booleans",
        "scalar",
        "one-list",
        "two-lists",
        "four-lists",
    ],
)
def test_coordinate_schema_checked_only_when_loaded(
    tmp_path, load_coords, coordinates, coordinate_type
):
    write_parquet(tmp_path / "a.parquet")
    path = tmp_path / "b.parquet"
    write_parquet(path)
    table = pq.read_table(path).append_column(
        "coordinates", pa.array([coordinates], type=coordinate_type)
    )
    pq.write_table(table, path)
    if load_coords:
        with pytest.raises(ValueError, match="coordinates") as exc:
            ordered(tmp_path, max_length=8, load_coords=True)
        assert str(path) in str(exc.value)
    else:
        items = list(ordered(tmp_path, max_length=8, load_coords=False))
        assert len(items) == 2
        assert all("coords" not in item for item in items)


@pytest.mark.parametrize(
    "coordinate_type",
    [
        pa.list_(pa.list_(pa.list_(pa.int32()))),
        pa.list_(pa.list_(pa.list_(pa.int64()))),
        pa.list_(pa.list_(pa.list_(pa.float32()))),
        pa.list_(pa.list_(pa.list_(pa.float64()))),
        pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
        pa.list_(pa.large_list(pa.list_(pa.float64()))),
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 3)),
    ],
)
def test_numeric_coordinate_schema_preserves_values_and_missing_rows(
    tmp_path, coordinate_type
):
    path = tmp_path / "data.parquet"
    table = pa.table(
        {
            "sequence_id": ["p1", "missing"],
            "sequence": ["ACDE", "ACDE"],
            "structure_tokens": [[1, 2, 3, 4], [1, 2, 3, 4]],
            "coordinates": pa.array(
                [[[[1, 2, 3]] * 3] * 4, None], type=coordinate_type
            ),
        }
    )
    pq.write_table(table, path)
    present, missing = list(ordered(tmp_path, max_length=8))
    torch.testing.assert_close(
        present["coords"][:4], torch.tensor([[[1.0, 2.0, 3.0]] * 3] * 4)
    )
    assert torch.isnan(present["coords"][4:]).all()
    assert torch.isnan(missing["coords"]).all()


def test_numeric_coordinate_rows_still_require_backbone_shape(tmp_path):
    path = tmp_path / "data.parquet"
    write_parquet(path)
    table = pq.read_table(path).append_column(
        "coordinates", pa.array([[[[1.0, 2.0]] * 3] * 4])
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="p1: coordinates.*shape"):
        next(iter(ordered(tmp_path, max_length=8)))


@pytest.mark.parametrize("max_length", [None, 3, 8])
def test_raw_coordinate_length_mode_preserves_integer_semantics(tmp_path, max_length):
    coords = torch.arange(45, dtype=torch.float32).reshape(5, 3, 3)
    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": ["observed", "missing"],
                "sequence": ["ACDEF", "ACDEF"],
                "structure_tokens": [[1, 2, 3, 4, 5]] * 2,
                "coordinates": [coords.tolist(), None],
            }
        ),
        path,
    )
    rows = list(ordered(tmp_path, max_length=max_length, dataset_name="corpus"))
    rows.sort(key=lambda row: row["sequence_id"])
    missing, observed = rows
    length = 5 if max_length is None else max_length
    assert observed["coords"].shape == missing["coords"].shape == (length, 3, 3)
    torch.testing.assert_close(
        observed["coords"][: min(length, 5)], coords[: min(length, 5)]
    )
    assert torch.isnan(observed["coords"][5:]).all()
    assert torch.isnan(missing["coords"]).all()
    assert observed["dataset"] == "corpus"


@pytest.mark.parametrize("suffix", [".parq", ".pq", ".PARQUET", ".PARQ", ".PQ"])
def test_standalone_alias_shards_remain_readable(tmp_path, suffix):
    from tests.utils.synthetic import make_mdlm_rows

    path = tmp_path / ("rows" + suffix)
    pq.write_table(pa.Table.from_pylist(make_mdlm_rows()), path)
    assert len(list(IterableTokenizedDataset(str(tmp_path), max_length=None))) == 2


def test_schema_two_rows_retain_canonical_identity(tmp_path):
    from tests.utils.synthetic import make_mdlm_rows, write_dataset

    originals = make_mdlm_rows()
    directory = write_dataset(tmp_path / "current", originals)
    item = next(iter(ordered(directory, max_length=None)))
    for key in (
        "canonical_id",
        "canonical_content_sha256",
        "residue_map_sha256",
        "canonical_identity",
        "parent_ids",
    ):
        assert item[key] == originals[0][key]


def test_schema_one_representation_provenance_is_rejected(tmp_path):
    import json
    from tests.utils.synthetic import make_mdlm_rows, write_dataset

    directory = write_dataset(tmp_path / "old", make_mdlm_rows())
    shard = directory / "part-000000.parquet"
    table = pq.read_table(shard)
    metadata = dict(table.schema.metadata)
    provenance = json.loads(metadata[b"stok.provenance"])
    provenance["schema_version"] = 1
    metadata[b"stok.provenance"] = json.dumps(provenance).encode()
    pq.write_table(table.replace_schema_metadata(metadata), shard)
    with pytest.raises(ValueError, match="provenance"):
        ordered(directory, max_length=None)
