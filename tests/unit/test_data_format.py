import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stok.data.format import TOKENIZER_KEY, read_source, shard_schema

TOKENIZER = {
    "preset": "large",
    "checkpoint_sha256": "a" * 64,
    "codebook_sha256": "b" * 64,
    "codebook_size": 32,
    "stok_version": "1.0",
}


def write_shard(path, rows, *, tokenizer=TOKENIZER, structure=True):
    schema = shard_schema(
        structure=structure,
        coordinates=False,
        extra=None,
        tokenizer=tokenizer if structure else None,
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def rows(n, *, structure=True):
    return [
        {"sequence_id": f"s{i}", "sequence": "ACD"}
        | ({"structure_tokens": [1, None, 3]} if structure else {})
        for i in range(n)
    ]


def test_read_source_footer_summary(tmp_path):
    write_shard(tmp_path / "a.parquet", rows(2))
    write_shard(tmp_path / "b.parquet", rows(3))
    summary = read_source(tmp_path)
    assert [shard["name"] for shard in summary["shards"]] == ["a.parquet", "b.parquet"]
    assert [shard["bytes"] for shard in summary["shards"]] == [
        (tmp_path / name).stat().st_size for name in ("a.parquet", "b.parquet")
    ]
    assert [shard["rows"] for shard in summary["shards"]] == [2, 3]
    assert summary["rows"] == 5
    assert summary["structure"] is True and summary["coordinates"] is False
    assert summary["tokenizer"] == TOKENIZER
    assert pq.read_schema(tmp_path / "a.parquet").field(
        "structure_tokens"
    ).type == pa.list_(pa.int16())


def test_read_source_sequence_only(tmp_path):
    write_shard(tmp_path / "a.parquet", rows(2, structure=False), structure=False)
    summary = read_source(tmp_path)
    assert summary["structure"] is False and summary["tokenizer"] is None


def test_read_source_ignores_subdirectories(tmp_path):
    write_shard(tmp_path / "a.parquet", rows(2))
    (tmp_path / "rejected").mkdir()
    (tmp_path / "rejected" / "a.jsonl").write_text("{}\n")
    (tmp_path / "rejected" / "nested.parquet").write_text("not parquet")
    (tmp_path / "notes.txt").write_text("hello")
    assert [shard["name"] for shard in read_source(tmp_path)["shards"]] == ["a.parquet"]


def test_read_source_rejects_missing_tokenizer_metadata(tmp_path):
    table = pa.Table.from_pylist(rows(1))
    pq.write_table(table, tmp_path / "a.parquet")
    with pytest.raises(ValueError, match="stok.tokenizer"):
        read_source(tmp_path)


def test_read_source_rejects_identity_disagreement(tmp_path):
    write_shard(tmp_path / "a.parquet", rows(1))
    write_shard(
        tmp_path / "b.parquet",
        rows(1),
        tokenizer=TOKENIZER | {"codebook_sha256": "c" * 64},
    )
    with pytest.raises(ValueError, match="codebook_sha256"):
        read_source(tmp_path)


def test_read_source_allows_stok_version_difference(tmp_path):
    write_shard(tmp_path / "a.parquet", rows(1))
    write_shard(
        tmp_path / "b.parquet", rows(1), tokenizer=TOKENIZER | {"stok_version": "2.0"}
    )
    assert read_source(tmp_path)["rows"] == 2


def test_read_source_accepts_zero_row_shard(tmp_path):
    write_shard(tmp_path / "a.parquet", [])
    write_shard(tmp_path / "b.parquet", rows(2))
    assert read_source(tmp_path)["rows"] == 2


def test_read_source_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no Parquet shards"):
        read_source(tmp_path)


def test_tokenizer_metadata_key():
    schema = shard_schema(
        structure=True, coordinates=True, extra=None, tokenizer=TOKENIZER
    )
    assert TOKENIZER_KEY in schema.metadata
    assert schema.field("coordinates").type == pa.list_(
        pa.list_(pa.list_(pa.float32()))
    )


def test_read_source_mixed_optional_coordinates(tmp_path):
    write_shard(tmp_path / "a.parquet", rows(1))
    schema = shard_schema(
        structure=True, coordinates=True, extra=None, tokenizer=TOKENIZER
    )
    row = rows(1)[0] | {"coordinates": [[[0.0] * 3] * 3] * 3}
    pq.write_table(pa.Table.from_pylist([row], schema=schema), tmp_path / "b.parquet")
    summary = read_source(tmp_path)
    assert summary["coordinates"] is True
    assert summary["columns"] == ["sequence_id", "sequence", "structure_tokens"]


def test_read_source_rejects_mixed_structure_presence(tmp_path):
    write_shard(tmp_path / "a.parquet", rows(1))
    write_shard(tmp_path / "b.parquet", rows(1, structure=False), structure=False)
    with pytest.raises(ValueError, match="structure_tokens"):
        read_source(tmp_path)
