import copy
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
    validate_canonical_record,
)
from stok.utils.pretrained import file_sha256, json_sha256
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


@pytest.mark.parametrize(
    "fault",
    [
        "mask",
        "boolean",
        "id",
        "digest",
        "revision",
        "identity_boolean",
        "position_boolean",
        "coordinate_boolean",
        "coordinate_nan",
        "partial_null",
        "parent_duplicate",
        "whitespace",
    ],
)
def test_rejects_malformed_canonical_records(record_with_missing_oxygen, fault):
    record = copy.deepcopy(record_with_missing_oxygen)
    if fault == "mask":
        record["atom_mask"][2][3] = True
    elif fault == "boolean":
        record["atom_mask"][0][0] = 1
    elif fault == "id":
        record["canonical_id"] = "not-an-id"
    elif fault == "digest":
        record["content_sha256"] = "0" * 64
    elif fault == "revision":
        record["identity"]["source_revision_sha256"] = "G" * 64
    elif fault == "identity_boolean":
        record["identity"]["model_index"] = False
    elif fault == "position_boolean":
        record["residue_map"][0]["polymer_position"] = False
    elif fault == "coordinate_boolean":
        record["coordinates"][0][0][0] = True
    elif fault == "coordinate_nan":
        record["coordinates"][0][0][0] = float("nan")
    elif fault == "partial_null":
        record["coordinates"][0][0][0] = None
    elif fault == "parent_duplicate":
        record["parent_ids"] = ["a" * 64] * 2
    else:
        record["identity"]["source_accession"] = " example "
    with pytest.raises(ValueError):
        validate_canonical_record(record)


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
    validate_canonical_record(record)


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


@pytest.mark.parametrize("change", ["map", "sequence", "coordinates"])
def test_inventory_detects_conflicting_content_even_with_updated_file_hash(
    tmp_path, change
):
    prepare_canonical_dataset(requests(tmp_path, [request()]), tmp_path / "canonical")
    directory = tmp_path / "canonical"
    record = list(iter_canonical_records(directory))[0]
    conflict = copy.deepcopy(record)
    if change == "map":
        conflict["residue_map"][0]["author_residue_id"] = 900
    elif change == "sequence":
        conflict["sequence"] = "A" + conflict["sequence"][1:]
    else:
        next(
            atom
            for residue in conflict["coordinates"]
            for atom in residue
            if atom[0] is not None
        )[0] += 1
    conflict["residue_map_sha256"] = json_sha256(conflict["residue_map"])
    conflict["content_sha256"] = json_sha256(
        {
            key: conflict[key]
            for key in ("sequence", "coordinates", "atom_mask", "residue_map")
        }
    )
    (directory / "records.jsonl").write_text(
        json.dumps(record) + "\n" + json.dumps(conflict) + "\n"
    )
    summary = json.loads((directory / "manifest.json").read_text())
    summary["records_sha256"] = file_sha256(directory / "records.jsonl")
    (directory / "manifest.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="conflicting|duplicate"):
        validate_canonical_dataset(directory)


def test_zero_record_inventory_preserves_parser_failure_audit(tmp_path):
    summary = prepare_canonical_dataset(
        requests(tmp_path, [request(chain_id="absent")]), tmp_path / "canonical"
    )
    assert summary["requested_input_count"] == summary["parser_rejection_count"] == 1
    assert summary["canonical_record_count"] == 0
    rejection = json.loads((tmp_path / "canonical/rejections.jsonl").read_text())
    assert rejection["request_id"] and rejection["reason"] == "chain_not_found"
    assert "canonical_id" not in rejection
    assert list(iter_canonical_records(tmp_path / "canonical")) == []


def test_preparation_verifies_original_after_parser_returns(tmp_path, monkeypatch):
    from stok.data import canonical

    source = tmp_path / "mapped.pdb"
    source.write_bytes((FIXTURES / "mapped.pdb").read_bytes())
    original = canonical.parse_polymer_structure

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_text(source.read_text() + "\n")
        return result

    monkeypatch.setattr(canonical, "parse_polymer_structure", mutate)
    with pytest.raises(RuntimeError, match="changed"):
        prepare_canonical_dataset(
            requests(tmp_path, [request(path=str(source))]), tmp_path / "canonical"
        )


def test_parent_ids_are_sorted_explicit_lineage(record_with_missing_oxygen):
    record = canonical_record(
        record_to_polymer(record_with_missing_oxygen),
        source_namespace="pdb",
        source_accession="example",
        parent_ids=["b" * 64, "a" * 64],
    )
    assert record["parent_ids"] == ["a" * 64, "b" * 64]
    assert record_with_missing_oxygen["parent_ids"] == []


def test_canonical_map_requires_complete_parser_correspondence(
    record_with_missing_oxygen,
):
    record = copy.deepcopy(record_with_missing_oxygen)
    record["residue_map"][0].pop("monomer_id")
    record["residue_map_sha256"] = json_sha256(record["residue_map"])
    record["content_sha256"] = json_sha256(
        {
            key: record[key]
            for key in ("sequence", "coordinates", "atom_mask", "residue_map")
        }
    )
    with pytest.raises(ValueError, match="map|correspondence"):
        validate_canonical_record(record)


def test_inventory_binds_request_source_identity(tmp_path):
    directory = tmp_path / "canonical"
    prepare_canonical_dataset(requests(tmp_path, [request()]), directory)
    row = json.loads((directory / "inputs.jsonl").read_text())
    row["source_accession"] = "forged"
    (directory / "inputs.jsonl").write_text(json.dumps(row) + "\n")
    summary = json.loads((directory / "manifest.json").read_text())
    summary["inputs_sha256"] = file_sha256(directory / "inputs.jsonl")
    (directory / "manifest.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="request|source"):
        validate_canonical_dataset(directory)


def test_validation_population_accumulator_excludes_observation_arrays(
    tmp_path, monkeypatch
):
    from stok.data import canonical

    directory = tmp_path / "canonical"
    prepare_canonical_dataset(requests(tmp_path, [request()]), directory)
    original = canonical._population

    def compact_population(records):
        assert records
        assert all(
            not {"coordinates", "atom_mask", "residue_map"} & record.keys()
            for record in records
        )
        return original(records)

    monkeypatch.setattr(canonical, "_population", compact_population)
    assert validate_canonical_dataset(directory)["canonical_record_count"] == 1


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


def test_parser_rejections_retain_verified_original_revision(tmp_path):
    directory = tmp_path / "canonical"
    prepare_canonical_dataset(
        requests(tmp_path, [request(chain_id="absent")]), directory
    )
    row = json.loads((directory / "inputs.jsonl").read_text())
    assert row["source_revision_sha256"] == file_sha256(FIXTURES / "mapped.cif")


def test_canonical_records_require_sequence_provenance(record_with_missing_oxygen):
    record = copy.deepcopy(record_with_missing_oxygen)
    record["provenance"]["source"].pop("sequence_source")
    with pytest.raises(ValueError, match="source"):
        validate_canonical_record(record)
