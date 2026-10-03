import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

from stok.cli.train import _build_dataloaders, _tokenize_and_align
from stok.data.dataset import IterableTokenizedDataset, TokenizedDataset
from stok.utils.losses import token_ce_loss
from stok.utils.tokenizer import Tokenizer


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


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("token_type", [pa.int32(), pa.int64()])
def test_null_tokens_preserve_residue_alignment(tmp_path, sharded, token_type):
    path = tmp_path / "data.parquet"
    write_parquet(path, token_type=token_type)
    if sharded:
        ds = IterableTokenizedDataset(str(tmp_path), max_length=8)
        item = next(iter(ds))
    else:
        ds = TokenizedDataset(str(path), max_length=8)
        item = ds[0]

    assert item["sequence_id"] == "p1"
    assert item["sequence"] == "ACDE"
    assert item["structure_tokens"].tolist() == [4, -1, 7, -1]
    _, labels = _tokenize_and_align(
        [item],
        Tokenizer(),
        max_len=8,
        ignore_index=-100,
        pad_id=1,
    )
    assert labels.tolist() == [[-100, 4, -100, 7, -100, -100, -100, -100]]

    _, truncated = _tokenize_and_align(
        [item],
        Tokenizer(),
        max_len=4,
        ignore_index=-100,
        pad_id=1,
    )
    assert truncated.tolist() == [[-100, 4, -100, -100]]


@pytest.mark.parametrize("sharded", [False, True])
def test_sequence_only_parquet_for_mlm(tmp_path, sharded):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"sequence_id": ["p1"], "sequence": ["ACDE"]}), path)
    cls = IterableTokenizedDataset if sharded else TokenizedDataset
    ds = cls(
        str(tmp_path if sharded else path), max_length=8, require_structure_tokens=False
    )
    item = next(iter(ds)) if sharded else ds[0]
    assert item == {"sequence_id": "p1", "sequence": "ACDE"}


@pytest.mark.parametrize(
    "tokens,token_type",
    [
        ("4 5 7 8", pa.string()),
        ([4.0, 5.0, 7.0, 8.0], pa.list_(pa.float64())),
    ],
)
@pytest.mark.parametrize("sharded", [False, True])
def test_rejects_untyped_structure_tokens(tmp_path, tokens, token_type, sharded):
    path = tmp_path / "bad.parquet"
    table = pa.table(
        {
            "sequence_id": ["p1"],
            "sequence": ["ACDE"],
            "structure_tokens": pa.array([tokens], type=token_type),
        }
    )
    pq.write_table(table, path)
    cls = IterableTokenizedDataset if sharded else TokenizedDataset
    with pytest.raises(ValueError, match="structure_tokens.*integer"):
        cls(str(tmp_path if sharded else path), max_length=8)


@pytest.mark.parametrize(
    "tokens,match",
    [
        ([1, 2], "length"),
        ([1, -1, 2, 3], "negative"),
        (None, "null"),
    ],
)
@pytest.mark.parametrize("sharded", [False, True])
def test_rejects_invalid_token_rows(tmp_path, tokens, match, sharded):
    path = tmp_path / "bad.parquet"
    write_parquet(path, tokens=tokens)
    cls = IterableTokenizedDataset if sharded else TokenizedDataset
    ds = cls(str(tmp_path if sharded else path), max_length=8)
    with pytest.raises(ValueError, match=match):
        next(iter(ds)) if sharded else ds[0]


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
        TokenizedDataset(str(path), max_length=8)


def test_rejects_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("sequence_id,sequence,structure_tokens\np1,ACDE,1 2 3 4\n")
    with pytest.raises(ValueError, match="Parquet"):
        TokenizedDataset(str(path), max_length=8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_all_null_tokens_have_zero_loss_and_gradients(tmp_path, dtype):
    path = tmp_path / "data.parquet"
    write_parquet(path, tokens=[None] * 4)
    item = TokenizedDataset(str(path), max_length=8)[0]
    _, labels = _tokenize_and_align(
        [item],
        Tokenizer(),
        max_len=8,
        ignore_index=-100,
        pad_id=1,
    )
    logits = torch.ones(1, 8, 10, dtype=dtype, requires_grad=True)
    loss = token_ce_loss(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("is_mlm", [False, True])
def test_training_rejects_structure_folders(tmp_path, is_mlm):
    (tmp_path / "protein.pdb").write_text("END\n")
    cfg = OmegaConf.create(
        {
            "data": {
                "train": str(tmp_path),
                "max_len": 8,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {"classifier": {"ignore_index": -100}},
            "train": {"batch_size": 1},
        }
    )
    with pytest.raises(ValueError, match="Parquet"):
        _build_dataloaders(cfg, codebook_size=8, pad_id=1, is_mlm=is_mlm)


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


@pytest.mark.parametrize("sharded", [False, True])
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
    tmp_path, sharded, load_coords, coordinates, coordinate_type
):
    if sharded:
        write_parquet(tmp_path / "a.parquet")
    path = tmp_path / "b.parquet"
    write_parquet(path)
    table = pq.read_table(path).append_column(
        "coordinates", pa.array([coordinates], type=coordinate_type)
    )
    pq.write_table(table, path)
    cls = IterableTokenizedDataset if sharded else TokenizedDataset
    source = str(tmp_path if sharded else path)
    if load_coords:
        with pytest.raises(ValueError, match="coordinates") as exc:
            cls(source, max_length=8, load_coords=True)
        assert str(path) in str(exc.value)
    else:
        ds = cls(source, max_length=8, load_coords=False)
        items = list(ds) if sharded else [ds[0]]
        assert len(items) == (2 if sharded else 1)
        assert all("coords" not in item for item in items)


@pytest.mark.parametrize("sharded", [False, True])
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
    tmp_path, sharded, coordinate_type
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
    if sharded:
        ds = IterableTokenizedDataset(
            str(tmp_path), max_length=8, shuffle_shards=False, shuffle_rows=False
        )
        present, missing = list(ds)
    else:
        ds = TokenizedDataset(str(path), max_length=8)
        present, missing = ds[0], ds[1]
    torch.testing.assert_close(
        present["coords"][:4], torch.tensor([[[1.0, 2.0, 3.0]] * 3] * 4)
    )
    assert torch.isnan(present["coords"][4:]).all()
    assert torch.isnan(missing["coords"]).all()


@pytest.mark.parametrize("sharded", [False, True])
def test_numeric_coordinate_rows_still_require_backbone_shape(tmp_path, sharded):
    path = tmp_path / "data.parquet"
    write_parquet(path)
    table = pq.read_table(path).append_column(
        "coordinates", pa.array([[[[1.0, 2.0]] * 3] * 4])
    )
    pq.write_table(table, path)
    cls = IterableTokenizedDataset if sharded else TokenizedDataset
    ds = cls(str(tmp_path if sharded else path), max_length=8)
    with pytest.raises(ValueError, match="p1: coordinates.*shape"):
        next(iter(ds)) if sharded else ds[0]


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("max_length", [None, 3, 8])
def test_raw_coordinate_length_mode_preserves_integer_semantics(
    tmp_path, sharded, max_length
):
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
    cls = IterableTokenizedDataset if sharded else TokenizedDataset
    dataset = cls(
        str(tmp_path if sharded else path), max_length=max_length, dataset_name="corpus"
    )
    rows = list(dataset) if sharded else [dataset[0], dataset[1]]
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
    from stok.data.dataset import IterableTokenizedDataset, TokenizedDataset
    from tests.utils.synthetic import make_mdlm_rows

    path = tmp_path / ("rows" + suffix)
    pq.write_table(pa.Table.from_pylist(make_mdlm_rows()), path)
    assert len(TokenizedDataset(str(path), max_length=None)) == 2
    assert len(list(IterableTokenizedDataset(str(tmp_path), max_length=None))) == 2
