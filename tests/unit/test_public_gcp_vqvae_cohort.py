"""Keep the frozen public selection/evaluation boundary intact."""

import json
from pathlib import Path


def test_public_cohort_freeze_contract():
    root = Path(__file__).resolve().parents[2] / "docs/experiments/gcp-vqvae"
    cohort = json.loads((root / "public-cohort.json").read_text())
    protocol = json.loads((root / "public-protocol.json").read_text())
    records = cohort["chains"]
    assert len(records) == protocol["cohort_size"] == 40
    assert len({row["sequence_id"] for row in records}) == len(records)
    assert len({row["entity_id"] for row in records}) == len(records)
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
