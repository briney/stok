"""Programmatic execution freezes choices and records authored/runtime artifacts."""

import pytest
from omegaconf import OmegaConf

from stok.training.engine import run_training
from tests.integration.test_mdlm_training import mdlm_config, training_fixture


def test_readonly_training_preserves_authored_config_and_records_runtime(tmp_path):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook)
    cfg.train.warmup_steps = "${train.max_steps}"
    before = OmegaConf.to_yaml(cfg, resolve=False)
    OmegaConf.set_readonly(cfg, True)
    run_training(cfg)
    assert OmegaConf.is_readonly(cfg)
    assert OmegaConf.to_yaml(cfg, resolve=False) == before
    authored = OmegaConf.load(tmp_path / "run/configs/authored.yaml")
    scientific = OmegaConf.load(tmp_path / "run/configs/run.yaml")
    runtime = OmegaConf.load(tmp_path / "run/configs/runtime.yaml")
    assert OmegaConf.to_yaml(authored, resolve=False) == before
    assert scientific.train.warmup_steps == 2
    assert "mdlm_identity" not in scientific.train
    assert "effective_precision" not in scientific.train
    assert runtime.effective_precision == "no"
    assert runtime.mdlm_identity.training_signature


def test_runtime_manifest_shares_exact_resume_contract(tmp_path):
    import copy
    from stok.utils.checkpoint import package_source_sha256, validate_resume_signature
    from tests.integration.test_mdlm_training import checkpoint

    source, codebook = training_fixture(tmp_path)
    payload = checkpoint(mdlm_config(tmp_path / "run", source, codebook))
    runtime, signature = payload["runtime"], payload["signature"]
    assert (
        runtime["source"] == signature["source"] == {"sha256": package_source_sha256()}
    )
    assert runtime["software"] == signature["software"]
    assert {
        "tokenizers",
        "transformers",
        "x-transformers",
        "hydra-core",
        "omegaconf",
    } <= runtime["software"].keys()
    assert runtime["execution"] == signature["execution"]
    assert runtime["components"] == {
        "objective": "mdlm",
        "model": "stok_mdlm",
        "sequence_tokenizer": "native",
        "structure_representation": "frozen_vq",
        "optimizer": "adamw",
        "scheduler": "warmup_linear",
    }
    for key in ("source", "software"):
        changed = copy.deepcopy(signature)
        changed[key][next(iter(changed[key]))] = "changed"
        with pytest.raises(ValueError, match="signature mismatch"):
            validate_resume_signature(payload, changed)
