"""Shared controls cannot drift with representation, record order or replication."""

import copy
from dataclasses import replace
import json
from pathlib import Path

from click.testing import CliRunner
import pytest
import torch

from stok.data.canonical import canonical_record, record_to_polymer
from stok.data.mdlm import prepare_mdlm_batch
from stok.utils.pretrained import file_sha256, json_sha256
from stok.utils.tokenizer import Tokenizer
from tests.utils.synthetic import canonical_fixture, make_mdlm_rows, write_dataset


def api():
    from stok.eval import cases

    return cases


def splits(tmp_path, records, assignments=None):
    path = tmp_path / "splits.jsonl"
    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                assignments
                if assignments is not None
                else [
                    {
                        "canonical_id": record["canonical_id"],
                        "split": "validation",
                        "cluster_id": f"c{i}",
                    }
                    for i, record in enumerate(records)
                ]
            )
        )
    )
    return path


def recipe(records, **changes):
    return {
        "schema_version": 1,
        "seed": 1729,
        "crop_residues": 6,
        "members": [record["canonical_id"] for record in records],
        "denoising": {
            "golden_token": {
                "regime": "joint_independent",
                "placement": "token",
                "probability": 0.5,
            }
        },
        **changes,
    }


@pytest.fixture
def inventory(tmp_path):
    rows = make_mdlm_rows()
    path = write_dataset(tmp_path / "encoded", rows) / "canonical"
    records = [canonical_fixture(row) for row in rows]
    return path, records, splits(tmp_path, records)


@pytest.fixture
def golden(tmp_path):
    row = {
        **make_mdlm_rows()[1],
        "sequence_id": "golden",
        "sequence": "AXCDEF",
        "structure_tokens": [0, 1, None, 3, 4, 5],
        "coordinates": torch.arange(54, dtype=torch.float32).reshape(6, 3, 3).tolist(),
        "source": {
            "path": "golden.pdb",
            "sha256": "b" * 64,
            "label_chain_id": None,
            "author_chain_id": "A",
        },
        "canonical_identity": {
            "source_namespace": "fixture",
            "source_accession": "golden",
        },
        "residue_map": [
            {"polymer_position": i, "insertion_code": "A" if i == 2 else ""}
            for i in range(6)
        ],
    }
    # Build directly so the literal golden is independent of synthetic helper updates.
    polymer = record_to_polymer(canonical_fixture(row))
    polymer = replace(polymer, source={**polymer.source, "label_chain_id": None})
    record = canonical_record(
        polymer, source_namespace="fixture", source_accession="golden"
    )
    assert (
        record["canonical_id"]
        == "deda4fbe5a708aad9ca6e6ad48a3b1e116776afd5d58bfd094c7209b5de37a9a"
    )
    row["original_atom_mask"] = record["atom_mask"]
    path = write_dataset(tmp_path / "golden-encoded", [row]) / "canonical"
    return path, record, splits(tmp_path, [record]), row


def freeze(inventory, tmp_path, **changes):
    path, records, split = inventory
    return api().freeze_evaluation_cases(
        [path], split, recipe(records, **changes), tmp_path / "cases"
    )


def rewrite(directory, cases):
    (directory / "cases.jsonl").write_text(
        "".join(json.dumps(case) + "\n" for case in cases)
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["cases_sha256"] = file_sha256(directory / "cases.jsonl")
    manifest["shared_cases_sha256"] = json_sha256(
        sorted(cases, key=lambda c: c["ordinal"])
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))


def rehash(case):
    case["case_id"] = json_sha256(
        {key: value for key, value in case.items() if key not in {"case_id", "ordinal"}}
    )


def test_frozen_cases_ignore_representation_and_physical_order(inventory, tmp_path):
    a = freeze(inventory, tmp_path)
    path, records, split = inventory
    other = (
        write_dataset(tmp_path / "reordered", list(reversed(make_mdlm_rows())))
        / "canonical"
    )
    b = api().freeze_evaluation_cases(
        [other], split, recipe(records), tmp_path / "other-cases"
    )
    assert a["shared_cases_sha256"] == b["shared_cases_sha256"]
    assert [(c["case_id"], c["seed"], c["crop"], c["masked"]) for c in a["cases"]] == [
        (c["case_id"], c["seed"], c["crop"], c["masked"]) for c in b["cases"]
    ]
    rewrite(tmp_path / "cases", list(reversed(a["cases"])))
    assert api().read_evaluation_cases(tmp_path / "cases")["cases"] == a["cases"]


def test_replicates_are_cases_not_additional_proteins(inventory, tmp_path):
    result = freeze(
        inventory, tmp_path, members=[inventory[1][0]["canonical_id"]], replicates=2
    )
    cases = result["cases"]
    assert len(cases) == 2
    assert result["unique_sample_count"] == 1
    assert [c["replicate"] for c in cases] == [0, 1]
    assert len({c["seed"] for c in cases}) == len({c["case_id"] for c in cases}) == 2


@pytest.mark.parametrize(
    "placement,seed,groups,masked",
    [
        (
            "token",
            3232611912295380603,
            [[0, 6], [-1, 7], [2, -1], [3, 9], [4, 10], [5, 11]],
            [[1, 1], [0, 1], [0, 0], [1, 1], [0, 1], [1, 0]],
        ),
        (
            "span",
            3831449916570069943,
            [[0, 2], [-1, 2], [1, -1], [1, 2], [1, 2], [1, 3]],
            [[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
        ),
    ],
)
def test_frozen_token_and_span_controls_match_literal_native_goldens(
    golden, tmp_path, placement, seed, groups, masked
):
    path, record, split, _ = golden
    definition = {
        "regime": "joint_independent",
        "placement": placement,
        "probability": 0.5,
        **({"span_mean": 8.0} if placement == "span" else {}),
    }
    rng = torch.get_rng_state().clone()
    result = api().freeze_evaluation_cases(
        [path],
        split,
        recipe([record], denoising={f"golden_{placement}": definition}),
        tmp_path / "cases",
    )
    case = result["cases"][0]
    assert case["seed"] == seed
    assert case["crop"] == [0, 6]
    assert case["positions"] == list(range(6))
    assert case["group_ids"] == groups
    assert case["masked"] == [[bool(a), bool(b)] for a, b in masked]
    assert all(type(flag) is bool for row in case["masked"] for flag in row)
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def projected_batch(rows, cases, max_len=8):
    batch = prepare_mdlm_batch(
        rows,
        Tokenizer(),
        max_len=max_len,
        codebook_size=32,
        crop="center",
        seeds=[0] * len(rows),
    )
    batch["sample_keys"] = [case["canonical_id"] for case in cases]
    return batch


def test_projection_does_not_redraw_missing_representation_codes(golden, tmp_path):
    path, record, split, row = golden
    cases = api().freeze_evaluation_cases(
        [path], split, recipe([record]), tmp_path / "cases"
    )["cases"]
    original = api().project_case_controls(cases, projected_batch([row], cases))
    reduced_row = {**row, "structure_tokens": [None, 1, None, 3, None, 5]}
    reduced = api().project_case_controls(cases, projected_batch([reduced_row], cases))
    for key in ("requested_eligible", "requested_masked", "group_ids"):
        assert torch.equal(original[key], reduced[key])
    assert int(original["eligible"].sum()) == 10
    assert int(reduced["eligible"].sum()) == 8
    assert int(reduced["masked"].sum()) < int(original["masked"].sum())
    assert torch.equal(reduced["weight"], torch.ones(1))
    assert reduced["group_ids"][0, 0].tolist() == [-1, -1]
    assert not reduced["requested_eligible"][:, -1].any()


def test_projection_keeps_replicates_distinct_in_one_batch(golden, tmp_path):
    path, record, split, row = golden
    cases = api().freeze_evaluation_cases(
        [path], split, recipe([record], replicates=2), tmp_path / "cases"
    )["cases"]
    rng = torch.get_rng_state().clone()
    projected = api().project_case_controls(cases, projected_batch([row, row], cases))
    assert projected["case_ids"] == [case["case_id"] for case in cases]
    for i, case in enumerate(cases):
        assert projected["requested_masked"][i, 1:7].tolist() == case["masked"]
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["canonical", "crop", "alignment", "metadata"])
def test_projection_rejects_misalignment(golden, tmp_path, fault):
    path, record, split, row = golden
    cases = api().freeze_evaluation_cases(
        [path], split, recipe([record]), tmp_path / "cases"
    )["cases"]
    batch = projected_batch([row], cases)
    if fault == "canonical":
        batch["sample_keys"] = ["a" * 64]
    if fault == "crop":
        batch["crop_offsets"][0] = 1
    if fault == "alignment":
        batch["residue_mask"][0, 1] = False
    if fault == "metadata":
        batch["sequence_mask_id"] = True
    with pytest.raises(ValueError):
        api().project_case_controls(cases, batch)


def test_generation_uses_full_targets_and_reports_missing_condition(golden, tmp_path):
    path, record, split, row = golden
    cases = api().freeze_evaluation_cases(
        [path],
        split,
        recipe(
            [record],
            denoising={},
            generation={"fold": {"regime": "structure_only", "placement": "token"}},
        ),
        tmp_path / "cases",
    )["cases"]
    assert cases[0]["masked"] == [
        [False, True],
        [False, True],
        [False, False],
        [False, True],
        [False, True],
        [False, True],
    ]
    assert cases[0]["conditioning"][0] == [True, False]
    batch = projected_batch([row], cases)
    batch["sequence_valid"][0, 1] = False
    result = api().project_case_controls(cases, batch)
    assert result["unavailable_conditioning"][0, 1].tolist() == [True, False]
    assert not result["case_available"][0]


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": True},
        {"seed": -1},
        {"schema_version": True},
        {"crop_residues": True},
        {"crop_residues": 0},
        {"replicates": True},
        {"replicates": 0},
        {"unknown": 1},
        {"members": []},
        {"members": ["a" * 64]},
        {"denoising": [], "generation": {}},
        {"denoising": {}, "generation": {}},
        {"denoising": {"x": []}},
        {"denoising": {"x": {"regime": "bad", "probability": 0.5}}},
        {"denoising": {"x": {"regime": "sequence_only", "probability": True}}},
        {"denoising": {"x": {"regime": "sequence_only", "probability": float("nan")}}},
        {
            "denoising": {
                "x": {"regime": "sequence_only", "probability": 0.5, "extra": 1}
            }
        },
        {"generation": {"x": {"regime": "sequence_only", "placement": "bad"}}},
    ],
)
def test_freeze_rejects_invalid_request_before_output(inventory, tmp_path, changes):
    with pytest.raises(ValueError):
        freeze(inventory, tmp_path, **changes)
    assert not (tmp_path / "cases").exists()


def test_freeze_rejects_duplicate_members_and_family_keys(inventory, tmp_path):
    with pytest.raises(ValueError):
        freeze(inventory, tmp_path, members=[inventory[1][0]["canonical_id"]] * 2)
    with pytest.raises(ValueError):
        freeze(
            inventory,
            tmp_path,
            generation={
                "golden_token": {"regime": "sequence_only", "placement": "token"}
            },
        )


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "unassigned",
        "unknown",
        "split",
        "cluster",
        "source",
        "revision",
        "parent",
        "cycle",
        "self",
    ],
)
def test_split_audit_rejects_leakage_and_invalid_lineage(inventory, tmp_path, fault):
    records = copy.deepcopy(inventory[1])
    rows = [
        {
            "canonical_id": r["canonical_id"],
            "split": "validation",
            "cluster_id": f"c{i}",
        }
        for i, r in enumerate(records)
    ]
    if fault == "duplicate":
        records.append(records[0])
    elif fault == "unassigned":
        rows.pop()
    elif fault == "unknown":
        rows.append({"canonical_id": "e" * 64, "split": "test", "cluster_id": "c3"})
    elif fault == "split":
        rows[0]["split"] = "dev"
    elif fault == "cluster":
        rows[1].update(split="train", cluster_id="c0")
    elif fault in {"source", "revision"}:
        polymer = record_to_polymer(records[1])
        source = {
            **polymer.source,
            **(
                {
                    "sha256": records[0]["identity"]["source_revision_sha256"],
                    "label_chain_id": "B",
                    "author_chain_id": "B",
                }
                if fault == "revision"
                else {}
            ),
        }
        records[1] = canonical_record(
            replace(polymer, source=source),
            source_namespace="synthetic",
            source_accession="long" if fault == "source" else "short",
        )
        rows[1].update(canonical_id=records[1]["canonical_id"], split="train")
    elif fault == "parent":
        records[0]["parent_ids"] = ["e" * 64]
    elif fault == "cycle":
        records[0]["parent_ids"] = [records[1]["canonical_id"]]
        records[1]["parent_ids"] = [records[0]["canonical_id"]]
    elif fault == "self":
        records[0]["parent_ids"] = [records[0]["canonical_id"]]
    with pytest.raises(ValueError):
        api().audit_canonical_splits(iter(records), splits(tmp_path, records, rows))


def test_parent_before_or_after_child_has_identical_split_identity(inventory, tmp_path):
    records = copy.deepcopy(inventory[1])
    records[1]["parent_ids"] = [records[0]["canonical_id"]]
    a = api().audit_canonical_splits(iter(records), inventory[2])
    b = api().audit_canonical_splits(iter(reversed(records)), inventory[2])
    assert a["population_sha256"] == b["population_sha256"]
    assert a["split_sha256"] == b["split_sha256"]
    assert a["assignments"] == b["assignments"]
    rows = [
        {
            "canonical_id": r["canonical_id"],
            "split": "validation" if i == 0 else "test",
            "cluster_id": f"c{i}",
        }
        for i, r in enumerate(records)
    ]
    with pytest.raises(ValueError):
        api().audit_canonical_splits(records, splits(tmp_path, records, rows))


def test_split_audit_covers_multiple_inventories_and_rejected_members(
    inventory, tmp_path
):
    rows = make_mdlm_rows()
    a = write_dataset(tmp_path / "train", rows[:1]) / "canonical"
    b = write_dataset(tmp_path / "held", rows[1:]) / "canonical"
    result = api().freeze_evaluation_cases(
        [a, b, a],
        inventory[2],
        recipe(inventory[1], members=[inventory[1][1]["canonical_id"]]),
        tmp_path / "cases",
    )
    assert len(result["canonical_inventories"]) == 2
    assert (
        result["population_sha256"]
        == api().audit_canonical_splits(inventory[1], inventory[2])["population_sha256"]
    )
    with pytest.raises(ValueError):
        api().freeze_evaluation_cases(
            [a], inventory[2], recipe(inventory[1][:1]), tmp_path / "missing"
        )
    duplicate = write_dataset(tmp_path / "dup", rows[:1]) / "canonical"
    with pytest.raises(ValueError):
        api().freeze_evaluation_cases(
            [a, b, duplicate],
            inventory[2],
            recipe(inventory[1]),
            tmp_path / "duplicate",
        )


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_id",
        "repeat",
        "ordinal",
        "content",
        "map",
        "crop",
        "mask",
        "bool",
        "group",
        "span",
        "tied",
        "seed",
        "family",
        "positions",
    ],
)
def test_read_rejects_invalid_frozen_controls(inventory, tmp_path, fault):
    result = freeze(inventory, tmp_path, replicates=2)
    cases = copy.deepcopy(result["cases"])
    case = cases[0]
    if fault == "duplicate_id":
        cases[1]["case_id"] = case["case_id"]
    elif fault == "repeat":
        cases[1] = {**case, "ordinal": 1}
    elif fault == "ordinal":
        cases[1]["ordinal"] = case["ordinal"]
    elif fault == "content":
        case["content_sha256"] = "a" * 64
    elif fault == "map":
        case["residue_map_sha256"] = "a" * 64
    elif fault == "crop":
        case["crop"][1] = 100
    elif fault == "mask":
        case["masked"][0].append(True)
    elif fault == "bool":
        case["masked"][0][0] = 1
    elif fault == "group":
        case["group_ids"][0][0] = -1
    elif fault == "span":
        case["definition"].update(placement="span", span_mean=8.0)
        case["group_ids"][0][0] = 500
    elif fault == "tied":
        case["definition"]["regime"] = "joint_tied"
    elif fault == "seed":
        case["seed"] = True
    elif fault == "family":
        case["definition"]["probability"] = 0.9
    elif fault == "positions":
        case["positions"][0] = True
    if fault not in {"duplicate_id", "repeat", "ordinal"}:
        rehash(case)
    rewrite(tmp_path / "cases", cases)
    with pytest.raises(ValueError):
        api().read_evaluation_cases(tmp_path / "cases")


def test_default_recipe_freezes_all_32_denoising_families(inventory, tmp_path):
    request = recipe(inventory[1], replicates=2)
    del request["denoising"]
    result = api().freeze_evaluation_cases(
        [inventory[0]], inventory[2], request, tmp_path / "cases"
    )
    assert result["denoising_case_count"] == 128
    assert result["generation_case_count"] == 0
    assert result["unique_sample_count"] == 2
    assert len({case["family_key"] for case in result["cases"]}) == 32


def test_generation_writer_does_not_apply_monitor_execution_cap(inventory, tmp_path):
    result = freeze(
        inventory,
        tmp_path,
        replicates=17,
        denoising={},
        generation={"fold": {"regime": "structure_only", "placement": "token"}},
    )
    assert result["generation_case_count"] == 34


def test_existing_and_concurrent_destination_are_preserved(
    inventory, tmp_path, monkeypatch
):
    directory = tmp_path / "cases"
    directory.mkdir()
    (directory / "marker").write_text("untouched")
    with pytest.raises(FileExistsError):
        freeze(inventory, tmp_path)
    assert (directory / "marker").read_text() == "untouched"
    (directory / "marker").unlink()
    directory.rmdir()
    original = api().publish_directory

    def concurrent(staging, destination):
        destination.mkdir()
        return original(staging, destination)

    monkeypatch.setattr(api(), "publish_directory", concurrent)
    with pytest.raises(OSError):
        freeze(inventory, tmp_path)
    assert list(directory.iterdir()) == []
    assert not list(tmp_path.glob(".cases.staging-*"))


def protocol(result, **changes):
    return api().evaluation_protocol(
        {
            "enabled": True,
            "steps": 10,
            "output_dir": "a",
            "denoising": {"evaluator": "native-v1"},
            "generation": {
                "sampling_steps": 64,
                "schedule": {"name": "linear"},
                "decode": True,
            },
            **changes,
        },
        shared_cases=result,
        representation={"representation_sha256": "a" * 64},
        decoder={"state_sha256": "b" * 64},
        environment={"software": "torch-fixture", "dtype": "float32"},
    )


def test_protocol_and_measurement_hashes_are_not_self_referential(inventory, tmp_path):
    cases = freeze(inventory, tmp_path)
    p = protocol(cases)
    assert p["protocol_sha256"] == json_sha256(
        {k: v for k, v in p.items() if k != "protocol_sha256"}
    )
    measurement = api().evaluation_measurement(
        p,
        model_identity={"training_signature": "c" * 64, "global_step": 8},
        coverage={"evaluable": 2},
        metrics={"ce": 0.5},
    )
    assert measurement["measurement_sha256"] == json_sha256(
        {k: v for k, v in measurement.items() if k != "measurement_sha256"}
    )
    assert (
        protocol(cases, steps=100, output_dir="b")["protocol_sha256"]
        == p["protocol_sha256"]
    )
    for bad in (
        {},
        {"global_step": True, "training_signature": "c" * 64},
        {"checkpoint_sha256": "bad"},
    ):
        with pytest.raises(ValueError):
            api().evaluation_measurement(p, model_identity=bad, coverage={}, metrics={})


def test_sampler_decoder_and_environment_change_protocol_not_cases(inventory, tmp_path):
    cases = freeze(inventory, tmp_path)
    base = protocol(cases)
    for field, value in [
        ("settings", {"generation": {"sampling_steps": 32}}),
        ("representation", {"representation_sha256": "d" * 64}),
        ("decoder", {"state_sha256": "d" * 64}),
        ("environment", {"dtype": "float64"}),
    ]:
        args = {
            "settings": {"generation": {"sampling_steps": 64}},
            "representation": {"representation_sha256": "a" * 64},
            "decoder": {"state_sha256": "b" * 64},
            "environment": {"software": "torch-fixture", "dtype": "float32"},
        }
        original = api().evaluation_protocol(shared_cases=cases, **args)
        args[field] = value
        changed = api().evaluation_protocol(shared_cases=cases, **args)
        assert original["protocol_sha256"] != changed["protocol_sha256"]
        assert changed["shared_cases_sha256"] == base["shared_cases_sha256"]


def test_prepare_and_freeze_require_explicit_metadata_before_outputs(tmp_path):
    from stok.cli.cli import cli

    fixture = Path(__file__).parents[1] / "test_data/gcp_vqvae/polymer/mapped.pdb"
    request = tmp_path / "sources.jsonl"
    request.write_text(
        json.dumps({"path": str(fixture), "sequence_id": "mapped"}) + "\n"
    )
    runner = CliRunner()
    failed = runner.invoke(
        cli, ["prepare-structures", str(request), str(tmp_path / "invalid")]
    )
    assert failed.exit_code != 0
    assert not (tmp_path / "invalid").exists()
    request.write_text(
        json.dumps(
            {
                "path": str(fixture),
                "sequence_id": "mapped",
                "source_namespace": "fixture",
                "source_accession": "mapped",
            }
        )
        + "\n"
    )
    prepared = runner.invoke(
        cli, ["prepare-structures", str(request), str(tmp_path / "canonical")]
    )
    assert prepared.exit_code == 0, prepared.output
    from stok.data.canonical import iter_canonical_records

    records = list(iter_canonical_records(tmp_path / "canonical"))
    split = splits(tmp_path, records)
    yaml = tmp_path / "request.yaml"
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create(recipe(records, replicates=2)), yaml)
    missing = runner.invoke(
        cli, ["freeze-eval-cases", str(yaml), str(tmp_path / "missing")]
    )
    assert missing.exit_code != 0
    assert not (tmp_path / "missing").exists()
    frozen = runner.invoke(
        cli,
        [
            "freeze-eval-cases",
            str(yaml),
            str(tmp_path / "cases"),
            "--canonical-dir",
            str(tmp_path / "canonical"),
            "--split-manifest",
            str(split),
        ],
    )
    assert frozen.exit_code == 0, frozen.output
    assert (
        "2 denoising" in frozen.output
        and "0 generation" in frozen.output
        and "1 unique" in frozen.output
    )


def test_multifamily_and_default_artifacts_roundtrip(inventory, tmp_path):
    for name, denoising in [
        ("default", None),
        (
            "named",
            {
                "z": {
                    "regime": "sequence_only",
                    "placement": "token",
                    "probability": 0.5,
                },
                "a": {
                    "regime": "structure_only",
                    "placement": "span",
                    "span_mean": 8.0,
                    "probability": 1.0,
                },
            },
        ),
    ]:
        request = recipe(inventory[1])
        if denoising is None:
            del request["denoising"]
        else:
            request["denoising"] = denoising
        result = api().freeze_evaluation_cases(
            [inventory[0]], inventory[2], request, tmp_path / name
        )
        assert api().read_evaluation_cases(tmp_path / name)["cases"] == result["cases"]


@pytest.mark.parametrize(
    "definition",
    [
        {"regime": "sequence_only", "placement": [], "probability": 0.5},
        {
            "regime": "sequence_only",
            "placement": "span",
            "span_mean": True,
            "probability": 0.5,
        },
    ],
)
def test_malformed_nested_family_is_value_error(inventory, tmp_path, definition):
    with pytest.raises(ValueError):
        freeze(inventory, tmp_path, denoising={"bad": definition})
    assert not (tmp_path / "cases").exists()


def test_malformed_nested_split_is_value_error(inventory, tmp_path):
    records = inventory[1]
    split = splits(
        tmp_path,
        records,
        [
            {"canonical_id": r["canonical_id"], "split": [], "cluster_id": "c"}
            for r in records
        ],
    )
    with pytest.raises(ValueError):
        api().audit_canonical_splits(records, split)


def test_distinct_accessions_cannot_duplicate_one_raw_selection(inventory, tmp_path):
    rows = make_mdlm_rows()
    a = write_dataset(tmp_path / "a", rows[:1]) / "canonical"
    alias = {
        **rows[0],
        "canonical_identity": {
            "source_namespace": "synthetic",
            "source_accession": "raw-alias",
        },
    }
    b = write_dataset(tmp_path / "b", [alias]) / "canonical"
    records = [canonical_fixture(rows[0]), canonical_fixture(alias)]
    with pytest.raises(ValueError):
        api().freeze_evaluation_cases(
            [a, b], splits(tmp_path, records), recipe(records), tmp_path / "cases"
        )


def test_canonical_fixture_keeps_originals_when_arm_code_is_missing(golden, tmp_path):
    _, record, _, row = golden
    reduced = {**row, "structure_tokens": [None, 1, None, 3, 4, 5]}
    assert canonical_fixture(reduced)["content_sha256"] == record["content_sha256"]
    from stok.data.structure_export import validate_structure_dataset

    directory = write_dataset(tmp_path / "unavailable-arm", [reduced])
    with pytest.raises(ValueError):
        validate_structure_dataset(directory)


def test_protocol_hashes_selected_cases_and_actual_evaluator_names(inventory, tmp_path):
    cases = freeze(inventory, tmp_path)
    base = protocol(cases)
    selected = {**cases, "cases": cases["cases"][:1]}
    assert protocol(selected)["protocol_sha256"] != base["protocol_sha256"]
    assert (
        protocol(cases, denoising={"name": "other-evaluator"})["protocol_sha256"]
        != protocol(cases, denoising={"name": "native-evaluator"})["protocol_sha256"]
    )


def test_frozen_reads_do_not_draw_rng(inventory, tmp_path, monkeypatch):
    freeze(inventory, tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("A read must never redraw controls")

    monkeypatch.setattr(torch, "rand", unexpected)
    assert api().read_evaluation_cases(tmp_path / "cases")["case_count"] == 2


def test_read_rejects_impossible_span_jump_without_redrawing(golden, tmp_path):
    path, record, split, _ = golden
    result = api().freeze_evaluation_cases(
        [path],
        split,
        recipe(
            [record],
            denoising={
                "golden_span": {
                    "regime": "joint_independent",
                    "placement": "span",
                    "probability": 0.5,
                    "span_mean": 8.0,
                }
            },
        ),
        tmp_path / "cases",
    )
    result["cases"][0]["group_ids"][5][1] = 10
    rehash(result["cases"][0])
    rewrite(tmp_path / "cases", result["cases"])
    with pytest.raises(ValueError):
        api().read_evaluation_cases(tmp_path / "cases")


def test_protocol_rejects_malformed_selected_cases(inventory, tmp_path):
    result = freeze(inventory, tmp_path)
    with pytest.raises(ValueError):
        protocol({**result, "cases": [None]})


def test_zero_probability_cannot_contain_masked_draw(inventory, tmp_path):
    result = freeze(
        inventory,
        tmp_path,
        denoising={
            "zero": {
                "regime": "joint_independent",
                "placement": "token",
                "probability": 0.0,
            }
        },
    )
    result["cases"][0]["masked"][0][0] = True
    rehash(result["cases"][0])
    rewrite(tmp_path / "cases", result["cases"])
    with pytest.raises(ValueError):
        api().read_evaluation_cases(tmp_path / "cases")


def test_projection_rejects_repeated_member_family_replicate(golden, tmp_path):
    path, record, split, row = golden
    result = api().freeze_evaluation_cases(
        [path], split, recipe([record], replicates=2), tmp_path / "cases"
    )
    cases = result["cases"]
    cases[1]["replicate"] = 0
    rehash(cases[1])
    with pytest.raises(ValueError):
        api().project_case_controls(cases, projected_batch([row, row], cases))


def test_observed_only_rejected_member_remains_in_complete_split_population(
    inventory, tmp_path
):
    from stok.data.canonical import iter_canonical_records
    from stok.data.structure_export import validate_structure_dataset
    from tests.utils.synthetic import declare_synthetic_source

    row = declare_synthetic_source(make_mdlm_rows()[1], source_accession="observed")
    row["source"]["sequence_source"] = "observed"
    directory = write_dataset(
        tmp_path / "observed", [row], policy={"allow_observed_sequence": False}
    )
    with pytest.raises(ValueError):
        validate_structure_dataset(directory)
    records = list(iter_canonical_records(directory / "canonical"))
    assert records[0]["provenance"]["source"]["sequence_source"] == "observed"
    all_records = inventory[1] + records
    split = splits(tmp_path, all_records)
    result = api().freeze_evaluation_cases(
        [inventory[0], directory / "canonical"],
        split,
        recipe(inventory[1]),
        tmp_path / "cases",
    )
    assert (
        result["population_sha256"]
        == api().audit_canonical_splits(all_records, split)["population_sha256"]
    )
    assert result["unique_sample_count"] == 2
    with pytest.raises(ValueError):
        api().freeze_evaluation_cases(
            [inventory[0]], split, recipe(inventory[1]), tmp_path / "omitted"
        )


def test_full_train_validation_test_audit_and_validation_only_cases(
    inventory, tmp_path
):
    from tests.utils.synthetic import declare_synthetic_source

    rows = make_mdlm_rows()
    test_row = declare_synthetic_source(rows[1], source_accession="test")
    test_dir = write_dataset(tmp_path / "test-encoded", [test_row]) / "canonical"
    records = inventory[1] + [canonical_fixture(test_row)]
    assignments = [
        {"canonical_id": record["canonical_id"], "split": split, "cluster_id": f"c{i}"}
        for i, (record, split) in enumerate(
            zip(records, ["train", "validation", "test"])
        )
    ]
    split = splits(tmp_path, records, assignments)
    result = api().freeze_evaluation_cases(
        [inventory[0], test_dir], split, recipe([records[1]]), tmp_path / "cases"
    )
    assert result["unique_sample_count"] == 1
    assert len(result["canonical_inventories"]) == 2
    for record in (records[0], records[2]):
        with pytest.raises(ValueError):
            api().freeze_evaluation_cases(
                [inventory[0], test_dir],
                split,
                recipe([record]),
                tmp_path / "not-validation",
            )


def test_audit_retains_only_compact_metadata(inventory):
    result = api().audit_canonical_splits(iter(inventory[1]), inventory[2])
    assert set(result["members"]) == {record["canonical_id"] for record in inventory[1]}
    for member in result["members"].values():
        assert not set(member) & {"coordinates", "atom_mask", "residue_map", "sequence"}


def test_duplicate_assignment_and_unknown_split_fields_are_rejected(
    inventory, tmp_path
):
    records = inventory[1]
    rows = [
        {
            "canonical_id": record["canonical_id"],
            "split": "validation",
            "cluster_id": f"c{i}",
        }
        for i, record in enumerate(records)
    ]
    for malformed in (rows + [rows[0]], [{**row, "dataset": "legacy"} for row in rows]):
        with pytest.raises(ValueError):
            api().audit_canonical_splits(records, splits(tmp_path, records, malformed))


def test_split_logical_digest_is_independent_of_manifest_record_order(
    inventory, tmp_path
):
    records = inventory[1]
    before = api().audit_canonical_splits(records, inventory[2])
    rows = list(before["assignments"].values())
    after = api().audit_canonical_splits(
        iter(reversed(records)), splits(tmp_path, records, list(reversed(rows)))
    )
    assert before["split_sha256"] == after["split_sha256"]
    assert before["split_manifest_sha256"] != after["split_manifest_sha256"]


def test_case_reader_detects_raw_file_and_inventory_reference_drift(
    inventory, tmp_path
):
    freeze(inventory, tmp_path)
    case_file = tmp_path / "cases/cases.jsonl"
    original = case_file.read_text()
    case_file.write_text(original + "\n")
    with pytest.raises(ValueError):
        api().read_evaluation_cases(tmp_path / "cases")
    case_file.write_text(original)
    manifest_file = inventory[0] / "manifest.json"
    manifest_file.write_text(manifest_file.read_text() + "\n")
    with pytest.raises(ValueError):
        api().read_evaluation_cases(tmp_path / "cases")


def test_read_and_projection_reject_merged_spans_at_deterministic_endpoint(
    tmp_path, monkeypatch
):
    row = {
        **make_mdlm_rows()[1],
        "sequence": "ACD",
        "structure_tokens": [0, 1, 2],
    }
    record = canonical_fixture(row)
    directory = write_dataset(tmp_path / "encoded", [row]) / "canonical"
    request = recipe(
        [record],
        crop_residues=3,
        denoising={
            "endpoint_span": {
                "regime": "joint_independent",
                "placement": "span",
                "span_mean": 1.0,
                "probability": 1.0,
            }
        },
    )
    result = api().freeze_evaluation_cases(
        [directory], splits(tmp_path, [record]), request, tmp_path / "cases"
    )
    case = result["cases"][0]
    assert case["group_ids"] == [[0, 3], [1, 4], [2, 5]]
    batch = projected_batch([row], [case], max_len=5)

    def unexpected_draw(*args, **kwargs):
        raise AssertionError("Endpoint validation must not draw RNG")

    monkeypatch.setattr(torch, "rand", unexpected_draw)
    assert api().read_evaluation_cases(tmp_path / "cases")["cases"] == [case]
    assert api().project_case_controls([case], batch)["group_ids"][0, 1:4].tolist() == [
        [0, 3],
        [1, 4],
        [2, 5],
    ]
    case["group_ids"] = [[0, 1], [0, 1], [0, 1]]
    rehash(case)
    rewrite(tmp_path / "cases", [case])
    with pytest.raises(ValueError):
        api().read_evaluation_cases(tmp_path / "cases")
    with pytest.raises(ValueError):
        api().project_case_controls([case], batch)
