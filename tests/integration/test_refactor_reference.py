"""The local test checks harness plumbing, not historical equivalence."""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tests.integration.test_mdlm_resume import snapshot
from tests.integration.test_mdlm_training import mdlm_config, training_fixture


def test_reference_capture_check_and_tamper_rejection(tmp_path):
    from tests.utils.refactor_reference import capture_reference, check_reference

    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(
        tmp_path / "unused",
        source,
        codebook,
        **{
            "train.seed": 1729,
            "model.encoder.dropout": 0.2,
            "train.gradient_accumulation_steps": 2,
            "data.num_workers": 2,
            "train.log_every": 2,
        },
    )
    reference = tmp_path / "reference"
    capture_reference(cfg, reference, source_root=Path(__file__).resolve().parents[2])
    manifest = json.loads((reference / "manifest.json").read_text())
    assert manifest["baseline"]["name"] == "gcp_large_paired_mdlm_tied_v1"
    assert manifest["software_reference"] == "tiny_cpu"
    checkpoint = torch.load(
        reference / "interrupted/checkpoints/step_00000001.pt", weights_only=True
    )
    assert checkpoint["format_version"] == 2 and checkpoint["global_step"] == 1
    assert checkpoint["rank_states"][0]["logging"]["running_updates"] == 1
    final = torch.load(reference / "full/model/final.pt", weights_only=True)
    assert final["global_step"] == 2
    trace = torch.load(reference / "full.yaml.rank0.trace.pt", weights_only=True)
    assert (
        sum(isinstance(row, dict) and row.get("event") == "optimizer" for row in trace)
        == 2
    )
    assert any(isinstance(row, dict) and row.get("event") == "loss" for row in trace)
    before = snapshot(reference)
    check_reference(reference, tmp_path / "candidate")
    assert snapshot(reference) == before
    with pytest.raises(ValueError, match="populated"):
        check_reference(reference, tmp_path / "candidate")
    with pytest.raises(ValueError, match="populated"):
        capture_reference(
            cfg, reference, source_root=Path(__file__).resolve().parents[2]
        )
    final["model"]["sequence_bias"][0] += 1
    torch.save(final, reference / "full/model/final.pt")
    tampered = snapshot(reference)
    with pytest.raises(AssertionError, match="checksum"):
        check_reference(reference, tmp_path / "tampered-check")
    assert snapshot(reference) == tampered
    for key, value, reason in (
        ("train.max_steps", 3, "two updates"),
        ("train.save_every", 2, "cadence"),
        ("train.wandb.enabled", True, "W&B"),
        ("train.eval.mdlm.enabled", True, "benchmark"),
        ("train.eval.mdlm.generation.enabled", True, "benchmark"),
    ):
        invalid = OmegaConf.create(OmegaConf.to_container(cfg))
        OmegaConf.update(invalid, key, value, force_add=True)
        with pytest.raises(ValueError, match=reason):
            capture_reference(
                invalid,
                tmp_path / "invalid",
                source_root=Path(__file__).resolve().parents[2],
            )
        assert not (tmp_path / "invalid").exists()
