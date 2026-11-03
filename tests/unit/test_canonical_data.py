from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from stok.data.canonical import (
    canonical_record,
    iter_canonical_records,
    prepare_canonical_dataset,
    record_to_polymer,
    validate_canonical_dataset,
)
from stok.utils.structure_parser import parse_polymer_structure

FIXTURES = Path(__file__).parents[1] / "test_data/gcp_vqvae/polymer"


@pytest.fixture
def record_with_missing_oxygen():
    structure = parse_polymer_structure(FIXTURES / "mapped.pdb")
    coordinates = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    coordinates[2, 3] = np.nan
    structure = replace(
        structure, coordinates=coordinates, atom_mask=np.isfinite(coordinates).all(-1)
    )
    return canonical_record(
        structure, source_namespace="pdb", source_accession="example"
    )


@pytest.fixture
def canonical_examples(record_with_missing_oxygen):
    a = record_with_missing_oxygen
    structure = record_to_polymer(a)
    moved = replace(
        structure,
        sequence_id="alias",
        source={**structure.source, "path": "/gone/moved.pdb"},
    )
    other = replace(
        structure,
        coordinates=structure.coordinates + np.float32(1),
        source={**structure.source, "sha256": "b" * 64},
    )
    return (
        a,
        canonical_record(moved, source_namespace="pdb", source_accession="example"),
        canonical_record(other, source_namespace="pdb", source_accession="example"),
    )


def test_canonical_identity_is_not_path_alias_or_representation(canonical_examples):
    a, relocated, other_observations = canonical_examples
    assert a["canonical_id"] == relocated["canonical_id"]
    assert a["content_sha256"] == relocated["content_sha256"]
    assert a["canonical_id"] != other_observations["canonical_id"]
    assert a["sequence"] == other_observations["sequence"]


def test_original_four_atom_roundtrip_with_missing_oxygen(record_with_missing_oxygen):
    record = record_with_missing_oxygen
    assert record["coordinates"][2][3] == [None, None, None]
    assert record["atom_mask"][2] == [True, True, True, False]
    assert record["residue_map"][2]["polymer_position"] == 2
    restored = record_to_polymer(record)
    assert restored.atom_mask.shape == (len(record["sequence"]), 4)
    assert not restored.atom_mask[2, 3]
    np.testing.assert_array_equal(restored.coordinates[0], np.arange(12).reshape(4, 3))
    assert not restored.coordinates.flags.writeable


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_index", 1),
        ("model_serial_id", 2),
        ("author_chain_id", "B"),
        ("sha256", "e" * 64),
    ],
)
def test_revision_model_chain_change_identity(record_with_missing_oxygen, field, value):
    structure = record_to_polymer(record_with_missing_oxygen)
    changed = replace(structure, source={**structure.source, field: value})
    assert (
        canonical_record(changed, source_namespace="pdb", source_accession="example")[
            "canonical_id"
        ]
        != record_with_missing_oxygen["canonical_id"]
    )


def test_sequence_only_records_have_absent_observations(record_with_missing_oxygen):
    structure = record_to_polymer(record_with_missing_oxygen)
    structure = replace(
        structure,
        coordinates=np.full((5, 4, 3), np.nan),
        atom_mask=np.zeros((5, 4), dtype=bool),
    )
    record = canonical_record(
        structure,
        source_namespace="sequence",
        source_accession="example",
        record_kind="sequence",
    )
    assert record["identity"]["chain_id"] is None
    assert record["identity"]["model_index"] is None
    assert record["coordinates"] == [[[None] * 3] * 4] * 5
    assert record["atom_mask"] == [[False] * 4] * 5


def requests(tmp_path, rows):
    manifest = tmp_path / "requests.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


def request(name="same", **kwargs):
    return {
        "sequence_id": name,
        "path": str(FIXTURES / "mapped.cif"),
        "source_namespace": "pdb",
        "source_accession": "example",
        **kwargs,
    }


def test_preparation_resolves_author_label_alias_duplicates(tmp_path):
    manifest = requests(
        tmp_path,
        [
            request(chain_id="A"),
            request("alias", chain_id="L", chain_namespace="label"),
        ],
    )
    with pytest.raises(ValueError, match="[Dd]uplicate.*selection"):
        prepare_canonical_dataset(manifest, tmp_path / "canonical")
    assert not (tmp_path / "canonical").exists()


def test_duplicate_display_labels_can_belong_to_distinct_ids(tmp_path):
    manifest = requests(
        tmp_path,
        [
            request(),
            request(source_accession="distinct", path=str(FIXTURES / "mapped.pdb")),
        ],
    )
    summary = prepare_canonical_dataset(manifest, tmp_path / "canonical")
    records = list(iter_canonical_records(tmp_path / "canonical"))
    assert summary["canonical_record_count"] == 2
    assert len({row["canonical_id"] for row in records}) == 2
    assert {row["provenance"]["sequence_id"] for row in records} == {"same"}
    assert validate_canonical_dataset(tmp_path / "canonical") == summary


def test_zero_record_inventory_preserves_parser_failure_audit(tmp_path):
    summary = prepare_canonical_dataset(
        requests(tmp_path, [request(chain_id="absent")]), tmp_path / "canonical"
    )
    assert summary["requested_input_count"] == summary["parser_rejection_count"] == 1
    assert summary["canonical_record_count"] == 0
    rejection = json.loads((tmp_path / "canonical/rejections.jsonl").read_text())
    assert rejection["sequence_id"] == "same"
    assert rejection["reason"] == "chain_not_found"
    assert "canonical_id" not in rejection
    assert list(iter_canonical_records(tmp_path / "canonical")) == []


def test_parent_ids_are_sorted_explicit_lineage(record_with_missing_oxygen):
    record = canonical_record(
        record_to_polymer(record_with_missing_oxygen),
        source_namespace="pdb",
        source_accession="example",
        parent_ids=["b" * 64, "a" * 64],
    )
    assert record["parent_ids"] == ["a" * 64, "b" * 64]
    assert record_with_missing_oxygen["parent_ids"] == []


def test_prepare_cli_publishes_canonical_before_any_representation(tmp_path):
    from click.testing import CliRunner
    from stok.cli.cli import cli

    manifest = requests(tmp_path, [request()])
    directory = tmp_path / "canonical"
    result = CliRunner().invoke(
        cli, ["prepare-structures", str(manifest), str(directory)]
    )
    assert result.exit_code == 0, result.output
    assert validate_canonical_dataset(directory)["canonical_record_count"] == 1
    assert not list(directory.glob("*.parquet"))


def test_synthetic_display_alias_preserves_declared_source_identity():
    from tests.utils.synthetic import canonical_fixture, make_mdlm_rows

    row = make_mdlm_rows()[1]
    alias = canonical_fixture({**row, "sequence_id": "display-alias"})
    assert alias["canonical_id"] == row["canonical_id"]
    assert (
        alias["identity"]["source_accession"]
        == row["canonical_identity"]["source_accession"]
    )
