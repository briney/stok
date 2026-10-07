import json

import numpy as np
import pytest
import torch

from tests.unit.test_gcp_vqvae import tiny_config
from tests.unit.test_structure_encoding import FIXTURES
from stok.utils.structure_parser import parse_polymer_structure


def test_duplicate_synthetic_gaps_rejected_before_output_or_loading(
    tmp_path, monkeypatch
):
    from experiments import gcp_vqvae_policies as experiment

    def unexpected_load(*args, **kwargs):
        pytest.fail("Duplicate synthetic gaps reached checkpoint loading")

    monkeypatch.setattr(experiment, "load_pretrained_tokenizer", unexpected_load)
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="unique"):
        experiment.run_experiments(
            tmp_path / "inputs.jsonl",
            preset="lite",
            checkpoint=tmp_path / "checkpoint.pt",
            output_dir=output,
            synthetic_gaps=(1, 1),
        )
    assert not output.exists()


def test_experiment_matrix_accounting_and_original_targets(tmp_path, monkeypatch):
    from experiments import gcp_vqvae_policies as experiment
    from stok.models.decoder import GeometricDecoder
    from stok.models.gcp_vqvae import GCPVQTokenizer

    config = tiny_config()
    config["max_length"] = 1280
    torch.manual_seed(7)
    model = GCPVQTokenizer(config).eval()
    decoder = GeometricDecoder(
        d_model=32,
        d_code=16,
        n_layers=1,
        n_heads=4,
        ffn_mult=1,
        max_length=1280,
        attn_kv_heads=2,
    ).eval()
    monkeypatch.setattr(experiment, "load_pretrained_tokenizer", lambda *a, **k: model)
    monkeypatch.setattr(experiment, "load_pretrained_decoder", lambda *a, **k: decoder)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"unit test placeholder, not released weights")
    manifest = tmp_path / "inputs.jsonl"
    sequence = parse_polymer_structure(
        FIXTURES / "inputs/complete_pdb.pdb", chain_id="A", allow_observed_sequence=True
    ).sequence
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sequence_id": name,
                    "source_namespace": "test-gcp-experiment",
                    "source_accession": name,
                    "path": str(FIXTURES / "inputs" / f"{name}.pdb"),
                    "chain_id": "A",
                    "sequence": sequence if name == "incomplete_pdb" else None,
                }
            )
            for name in ["incomplete_pdb", "shorter", "too_short"]
        )
        + "\n"
    )
    state = {key: value.clone() for key, value in model.state_dict().items()}
    output = tmp_path / "experiment"
    report = experiment.run_experiments(
        manifest,
        preset="lite",
        checkpoint=checkpoint,
        output_dir=output,
        seed=17,
        allow_observed_sequence=True,
    )
    assert len(report["conditions"]) == 7
    assert report["input_count"] == 3
    assert report["selected_policy"] is None
    for aggregate in report["conditions"].values():
        assert aggregate["accepted"] == 2 and aggregate["rejected"] == 1
        assert aggregate["exclusions"] == {"chains_too_short": 1}
        assert aggregate["oxygen_only_token_omissions"] == 1
        assert aggregate["token_count"] == 70
        assert aggregate["scored_residue_count"] == 70
        assert aggregate["metric_example_counts"]["lddt_ca"] == 2
    rows = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    accepted = [
        row
        for row in rows
        if row["sequence_id"] == "incomplete_pdb" and row["status"] == "accepted"
    ]
    assert len(accepted) == 7
    with np.load(output / accepted[0]["arrays"]) as first:
        expected_truth = first["original_coordinates"].copy()
        mask = first["score_mask"].copy()
    assert np.isnan(expected_truth[4, 3]).all() and np.isnan(expected_truth[11]).all()
    assert mask.sum() == 38
    for row in accepted:
        with np.load(output / row["arrays"]) as arrays:
            np.testing.assert_array_equal(
                arrays["original_coordinates"], expected_truth
            )
            np.testing.assert_array_equal(arrays["score_mask"], mask)
            assert arrays["indices"][[4, 11]].tolist() == [-1, -1]
            assert np.isnan(arrays["decoded_coordinates"][[4, 11]]).all()
        assert row["diagnostics"]["padding"]["compared_tokens"] == 38
        assert row["diagnostics"]["rigid_transform"]["compared_tokens"] == 38
        if row["imputation"] != "reference":
            assert row["max_observed_atom_displacement"] == 0
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in state.items()
    )
    assert json.loads((output / "report.json").read_text()) == report
    assert report["grouping_context"] == "independent_singleton"
    assert report["grouping_check"]["changed_tokens"] == 0
    with pytest.raises(FileExistsError):
        experiment.run_experiments(
            manifest, preset="lite", checkpoint=checkpoint, output_dir=output
        )


def test_provenance_covers_buffers_and_config_snapshot():
    from stok.models.gcp_vqvae import GCPVQTokenizer
    from stok.utils.pretrained import state_sha256

    config = tiny_config()
    model = GCPVQTokenizer(config)
    original = model.config["quantizer"]["dim"]
    config["quantizer"]["dim"] = 123
    assert model.config["quantizer"]["dim"] == original
    first = state_sha256(model.quantizer.state_dict())
    model.quantizer._codebook.cluster_size.add_(1)
    assert state_sha256(model.quantizer.state_dict()) != first
    assert state_sha256({"a": torch.tensor(1), "b": torch.ones(2)}) == state_sha256(
        {"b": torch.ones(2), "a": torch.tensor(1)}
    )


def test_tokenizer_source_provenance_excludes_decoder_and_cli():
    from stok.utils.pretrained import inference_metadata

    files = inference_metadata(torch.device("cpu"))["source_files"]
    assert "models/gcp_vqvae.py" in files
    assert "data/structure_encoding.py" in files
    assert "utils/structure_parser.py" in files
    assert "models/decoder.py" not in files
    assert "cli/train.py" not in files
