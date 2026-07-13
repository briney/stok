import json

import pytest

from scripts.gcp_vqvae_parity.common import (
    atomic_write_json,
    discover_inputs,
    select_shard,
    sha256_file,
)


def test_discover_inputs_is_sorted_and_hashed(tmp_path):
    (tmp_path / "b.cif").write_text("beta")
    (tmp_path / "a.cif").write_text("alpha")
    (tmp_path / "ignored.txt").write_text("ignored")

    records = discover_inputs(tmp_path)

    assert [record.name for record in records] == ["a.cif", "b.cif"]
    assert records[0].sha256 == sha256_file(tmp_path / "a.cif")
    assert len(records[0].sha256) == 64


def test_select_shard_partitions_without_overlap(tmp_path):
    for index in range(7):
        (tmp_path / f"{index}.cif").write_text(str(index))
    records = discover_inputs(tmp_path)
    shards = [select_shard(records, shard_index=i, num_shards=3) for i in range(3)]
    names = [[record.name for record in shard] for shard in shards]
    assert names == [["0.cif", "3.cif", "6.cif"], ["1.cif", "4.cif"], ["2.cif", "5.cif"]]
    assert sorted(name for shard in names for name in shard) == [record.name for record in records]


def test_select_shard_validates_indices(tmp_path):
    (tmp_path / "a.cif").write_text("alpha")
    with pytest.raises(ValueError, match="shard_index"):
        select_shard(discover_inputs(tmp_path), shard_index=2, num_shards=2)


def test_atomic_write_json_replaces_target(tmp_path):
    target = tmp_path / "nested" / "result.json"
    atomic_write_json(target, {"status": "first"})
    atomic_write_json(target, {"status": "second"})
    assert json.loads(target.read_text()) == {"status": "second"}
    assert not target.with_suffix(".json.tmp").exists()
