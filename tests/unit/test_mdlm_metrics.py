"""Tests for MDLM-specific evaluation metrics."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from stok.eval.metrics.mdlm import (
    MDLMSeqAccuracy,
    MDLMSeqPerplexity,
    MDLMStructAccuracy,
    MDLMStructPerplexity,
)


@pytest.fixture
def cfg():
    """Minimal config for metric tests."""
    return OmegaConf.create({
        "model": {"classifier": {"ignore_index": -100}},
    })


def _make_outputs(
    B=2, L=8, V_seq=32, V_struct=16, ignore_index=-100
):
    """Create synthetic model outputs for metric testing."""
    # Create seq logits where argmax matches targets at masked positions
    seq_targets = torch.full((B, L), ignore_index, dtype=torch.long)
    seq_targets[:, 2:5] = torch.randint(0, V_seq, (B, 3))

    seq_logits = torch.randn(B, L, V_seq)
    # Make predictions correct at masked positions
    for b in range(B):
        for l in range(L):
            if seq_targets[b, l] != ignore_index:
                seq_logits[b, l, seq_targets[b, l]] = 10.0

    # Create struct logits
    struct_targets = torch.full((B, L), ignore_index, dtype=torch.long)
    struct_targets[:, 1:4] = torch.randint(0, V_struct, (B, 3))

    struct_logits = torch.randn(B, L, V_struct)
    # Make predictions correct at masked positions
    for b in range(B):
        for l in range(L):
            if struct_targets[b, l] != ignore_index:
                struct_logits[b, l, struct_targets[b, l]] = 10.0

    return {
        "seq_logits": seq_logits,
        "struct_logits": struct_logits,
        "seq_targets": seq_targets,
        "struct_targets": struct_targets,
    }


class TestMDLMSeqAccuracy:
    def test_perfect_accuracy(self, cfg):
        metric = MDLMSeqAccuracy()
        outputs = _make_outputs()
        tokens = torch.randint(0, 32, (2, 8))
        labels = outputs["seq_targets"]

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert result["mdlm_seq_acc"] == pytest.approx(1.0)

    def test_wrong_predictions(self, cfg):
        metric = MDLMSeqAccuracy()
        outputs = _make_outputs()
        # Scramble logits so predictions are wrong
        outputs["seq_logits"] = torch.randn_like(outputs["seq_logits"])
        tokens = torch.randint(0, 32, (2, 8))
        labels = outputs["seq_targets"]

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        # With random logits, accuracy should be low (but not necessarily 0)
        assert 0.0 <= result["mdlm_seq_acc"] <= 1.0

    def test_empty_mask_returns_nan(self, cfg):
        metric = MDLMSeqAccuracy()
        # All targets are ignore_index
        outputs = {
            "seq_logits": torch.randn(2, 8, 32),
            "seq_targets": torch.full((2, 8), -100, dtype=torch.long),
        }
        tokens = torch.randint(0, 32, (2, 8))
        labels = outputs["seq_targets"]

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert math.isnan(result["mdlm_seq_acc"])

    def test_reset(self, cfg):
        metric = MDLMSeqAccuracy()
        outputs = _make_outputs()
        tokens = torch.randint(0, 32, (2, 8))
        labels = outputs["seq_targets"]

        metric.update(outputs, tokens, labels, None, cfg)
        metric.reset()
        result = metric.compute()
        assert math.isnan(result["mdlm_seq_acc"])

    def test_state_tensors_round_trip(self, cfg):
        metric = MDLMSeqAccuracy()
        outputs = _make_outputs()
        tokens = torch.randint(0, 32, (2, 8))
        labels = outputs["seq_targets"]

        metric.update(outputs, tokens, labels, None, cfg)
        tensors = metric.state_tensors()
        assert len(tensors) == 1

        metric2 = MDLMSeqAccuracy()
        metric2.load_state_tensors(tensors)
        assert metric.compute() == metric2.compute()


class TestMDLMStructAccuracy:
    def test_perfect_accuracy(self, cfg):
        metric = MDLMStructAccuracy()
        outputs = _make_outputs()
        tokens = torch.randint(0, 32, (2, 8))
        labels = outputs["seq_targets"]

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert result["mdlm_struct_acc"] == pytest.approx(1.0)

    def test_empty_mask_returns_nan(self, cfg):
        metric = MDLMStructAccuracy()
        outputs = {
            "struct_logits": torch.randn(2, 8, 16),
            "struct_targets": torch.full((2, 8), -100, dtype=torch.long),
        }
        tokens = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert math.isnan(result["mdlm_struct_acc"])

    def test_no_struct_logits_is_noop(self, cfg):
        metric = MDLMStructAccuracy()
        outputs = {"seq_logits": torch.randn(2, 8, 32)}
        tokens = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert math.isnan(result["mdlm_struct_acc"])


class TestMDLMSeqPerplexity:
    def test_finite_perplexity(self, cfg):
        metric = MDLMSeqPerplexity()
        outputs = _make_outputs()
        tokens = torch.randint(0, 32, (2, 8))
        labels = outputs["seq_targets"]

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert result["mdlm_seq_ppl"] > 0
        assert math.isfinite(result["mdlm_seq_ppl"])

    def test_empty_mask_returns_nan(self, cfg):
        metric = MDLMSeqPerplexity()
        outputs = {
            "seq_logits": torch.randn(2, 8, 32),
            "seq_targets": torch.full((2, 8), -100, dtype=torch.long),
        }
        tokens = torch.randint(0, 32, (2, 8))
        labels = outputs["seq_targets"]

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert math.isnan(result["mdlm_seq_ppl"])


class TestMDLMStructPerplexity:
    def test_finite_perplexity(self, cfg):
        metric = MDLMStructPerplexity()
        outputs = _make_outputs()
        tokens = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert result["mdlm_struct_ppl"] > 0
        assert math.isfinite(result["mdlm_struct_ppl"])

    def test_empty_mask_returns_nan(self, cfg):
        metric = MDLMStructPerplexity()
        outputs = {
            "struct_logits": torch.randn(2, 8, 16),
            "struct_targets": torch.full((2, 8), -100, dtype=torch.long),
        }
        tokens = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert math.isnan(result["mdlm_struct_ppl"])

    def test_no_struct_logits_is_noop(self, cfg):
        metric = MDLMStructPerplexity()
        outputs = {"seq_logits": torch.randn(2, 8, 32)}
        tokens = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()
        assert math.isnan(result["mdlm_struct_ppl"])
