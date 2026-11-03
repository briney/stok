"""Programmatic execution keeps the caller's config and records the resolved run."""

import pytest
from omegaconf import OmegaConf

from stok.training.engine import run_training
from tests.integration.test_mdlm_training import mdlm_config, training_fixture


def test_readonly_training_preserves_caller_config(tmp_path):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook)
    cfg.train.warmup_steps = "${train.max_steps}"
    before = OmegaConf.to_yaml(cfg, resolve=False)
    OmegaConf.set_readonly(cfg, True)
    run_training(cfg)
    assert OmegaConf.is_readonly(cfg)
    assert OmegaConf.to_yaml(cfg, resolve=False) == before
    scientific = OmegaConf.load(tmp_path / "run/configs/run.yaml")
    assert scientific.train.warmup_steps == 2
    assert "mdlm_identity" not in scientific.train


def test_runtime_manifest_shares_exact_resume_contract(tmp_path):
    import copy
    from stok.utils.checkpoint import validate_resume_signature
    from tests.integration.test_mdlm_training import checkpoint

    source, codebook = training_fixture(tmp_path)
    payload = checkpoint(mdlm_config(tmp_path / "run", source, codebook))
    runtime, signature = payload["runtime"], payload["signature"]
    assert runtime["execution"] == signature["execution"]
    for key in ("world_size", "precision", "device"):
        changed = copy.deepcopy(signature)
        changed["execution"][key] = "changed"
        with pytest.raises(ValueError, match="signature mismatch"):
            validate_resume_signature(payload, changed)
