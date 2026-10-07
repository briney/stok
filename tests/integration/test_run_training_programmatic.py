"""Programmatic execution freezes choices and records authored/runtime artifacts."""

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
