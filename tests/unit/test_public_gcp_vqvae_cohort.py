"""Keep the frozen public selection/evaluation boundary intact."""

import json
import hashlib
from pathlib import Path


def test_public_cohort_freeze_contract():
    root = Path(__file__).resolve().parents[2] / "docs/experiments/gcp-vqvae"
    cohort = json.loads((root / "public-cohort.json").read_text())
    protocol = json.loads((root / "public-protocol.json").read_text())
    records = cohort["chains"]
    assert len(records) == protocol["cohort_size"] == 40
    assert len({row["sequence_id"] for row in records}) == len(records)
    assert len({row["entity_id"] for row in records}) == len(records)
    assert len({row["pdb_id"] for row in records}) == len(records)
    assert len({row["cluster_anchor"] for row in records}) == len(records)
    for split in ("selection", "heldout"):
        selected = [row for row in records if row["split"] == split]
        assert len(selected) == 20
        manifest = [
            json.loads(line)
            for line in (root / f"public-{split}.jsonl").read_text().splitlines()
        ]
        assert [row["sequence_id"] for row in manifest] == [
            row["sequence_id"] for row in selected
        ]
        for entry, record in zip(manifest, selected, strict=True):
            assert entry["chain_namespace"] == "label"
            assert entry["chain_id"] == record["label_chain_id"]
            assert entry["model_index"] == 0
            assert Path(entry["path"]).name == f"{record['pdb_id']}.cif"
        for lower, upper in protocol["length_bins"]:
            group = [row for row in selected if lower <= row["length"] <= upper]
            assert len(group) == 4
            assert sum(row["complete"] for row in group) == 2
    assert protocol["synthetic_gaps"] == [1, 3, 5, 15]
    assert protocol["intended_use"] == "sequence_conditioned_structure_token_training"
    assert protocol["quality_loss_budget"] == {"lddt_ca": 0.02, "tm_kabsch": 0.02}


def test_public_policy_choice_respects_frozen_quality_and_ranking():
    root = Path(__file__).resolve().parents[2] / "docs/experiments/gcp-vqvae"
    protocol = json.loads((root / "public-protocol.json").read_text())
    decision = json.loads((root / "public-selection-decision.json").read_text())
    assert (
        decision["protocol_sha256"]
        == hashlib.sha256((root / "public-protocol.json").read_bytes()).hexdigest()
    )
    assert (
        decision["selection_manifest_sha256"]
        == hashlib.sha256((root / "public-selection.jsonl").read_bytes()).hexdigest()
    )
    for preset in ("lite", "large"):
        result = decision["presets"][preset]
        eligible = []
        for candidate in result["candidates"]:
            assert len(candidate["quality_checks"]) == 20
            assert set(candidate["hard_gate_checks"]) == {
                "padding_zero_changed_ids",
                "grouping_zero_changed_ids",
            }
            assert candidate["ranking_key"][:3] == [
                candidate["rigid_changed_tokens"] / candidate["rigid_compared_tokens"],
                candidate["max_working_observation_displacement_angstrom"],
                -candidate["source_mean_lddt_ca"],
            ]
            passes = [
                check["chain_count"] >= 5
                and check["bootstrap_95_ci"] is not None
                and check["bootstrap_95_ci"][0]
                >= -protocol["quality_loss_budget"][check["metric"]]
                for check in candidate["quality_checks"]
            ]
            assert candidate["eligible"] == (
                all(passes) and all(candidate["hard_gate_checks"].values())
            )
            if candidate["eligible"]:
                eligible.append(candidate)
        if eligible:
            chosen = min(eligible, key=lambda candidate: candidate["ranking_key"])
            assert result["selected_condition"] == chosen["condition"]
        else:
            assert result["selected_condition"] is None


def test_heldout_keeps_frozen_choice_and_packages_only_qualified_policy():
    root = Path(__file__).resolve().parents[2] / "docs/experiments/gcp-vqvae"
    selection = json.loads((root / "public-selection-decision.json").read_text())
    heldout = json.loads((root / "public-heldout-results.json").read_text())
    audit = json.loads((root / "public-integrity-audit.json").read_text())
    assert (
        heldout["selection_decision_sha256"]
        == hashlib.sha256(
            (root / "public-selection-decision.json").read_bytes()
        ).hexdigest()
    )
    for preset in ("lite", "large"):
        result = heldout["presets"][preset]
        condition = selection["presets"][preset]["selected_condition"]
        assert result["selected_condition"] == condition
        for split, data in (("selection", selection), ("heldout", heldout)):
            evidence = audit[f"{split}-{preset}"]
            assert evidence["source_mask_target_checks_passed"]
            assert evidence["report_sha256"] == data["presets"][preset]["report_sha256"]
            for candidate in data["presets"][preset]["candidates"]:
                assert candidate["hard_gate_checks"]["padding_zero_changed_ids"] == (
                    evidence["padding_changed_tokens_by_condition"][
                        candidate["condition"]
                    ]
                    == 0
                )
        expected_qualification = condition is not None and next(
            row["eligible"]
            for row in result["candidates"]
            if row["condition"] == condition
        )
        assert result["qualified"] == expected_qualification
        if result["qualified"]:
            path = Path(__file__).resolve().parents[2] / result["policy_file"]
            policy = json.loads(path.read_text())
            sequence, fill = condition.split("/")
            assert policy["sequence_mode"] == sequence
            assert policy["imputation"] == fill
            assert policy["dtype"] == "float32"
            assert policy["device"] == result["environment"]["device"] == "cuda:0"
            assert not policy["allow_observed_sequence"]
        else:
            assert result["policy_file"] is None
