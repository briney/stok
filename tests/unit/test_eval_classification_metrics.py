"""Unit tests for classification metrics."""

import math

import torch
from omegaconf import OmegaConf

from stok.eval.metrics.classification import (
    AccuracyMetric,
    MaskedAccuracyMetric,
    PerplexityMetric,
)


def _make_cfg(ignore_index: int = -100):
    """Create a minimal config for testing."""
    return OmegaConf.create(
        {"model": {"classifier": {"ignore_index": ignore_index}, "encoder": {"pad_id": 1}}}
    )


class TestAccuracyMetric:
    """Tests for AccuracyMetric."""

    def test_accuracy_metric_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = AccuracyMetric()
        assert metric.name == "acc"
        assert metric.objectives == {"codebook"}
        assert metric.requires_decoder is False
        assert metric.requires_coords is False

    def test_accuracy_metric_perfect_predictions(self):
        """Test accuracy with perfect predictions."""
        metric = AccuracyMetric()
        cfg = _make_cfg()

        # Perfect predictions: logits point to correct labels
        logits = torch.tensor(
            [
                [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],  # preds: 1, 0, 1
            ]
        )
        labels = torch.tensor([[1, 0, 1]])
        tokens = torch.zeros_like(labels)

        metric.update({"logits": logits}, tokens, labels, None, cfg)
        result = metric.compute()

        assert result["acc"] == 1.0

    def test_accuracy_metric_half_correct(self):
        """Test accuracy with half correct predictions."""
        metric = AccuracyMetric()
        cfg = _make_cfg()

        logits = torch.tensor(
            [
                [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]],  # preds: 0, 0, 1, 1
            ]
        )
        labels = torch.tensor([[0, 1, 0, 1]])  # correct: 0, wrong: 1, wrong: 0, correct: 1
        tokens = torch.zeros_like(labels)

        metric.update({"logits": logits}, tokens, labels, None, cfg)
        result = metric.compute()

        assert result["acc"] == 0.5

    def test_accuracy_metric_ignores_index(self):
        """Test that ignore_index positions are excluded."""
        metric = AccuracyMetric()
        cfg = _make_cfg(ignore_index=-100)

        logits = torch.tensor(
            [
                [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]],  # preds: 0, 1, 0
            ]
        )
        labels = torch.tensor([[0, 1, -100]])  # last is ignored
        tokens = torch.zeros_like(labels)

        metric.update({"logits": logits}, tokens, labels, None, cfg)
        result = metric.compute()

        # Only 2 valid positions, both correct
        assert result["acc"] == 1.0

    def test_accuracy_metric_accumulates_batches(self):
        """Test that accuracy accumulates across multiple batches."""
        metric = AccuracyMetric()
        cfg = _make_cfg()

        # Batch 1: 2/2 correct
        logits1 = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
        labels1 = torch.tensor([[0, 1]])
        tokens1 = torch.zeros_like(labels1)

        # Batch 2: 0/2 correct
        logits2 = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
        labels2 = torch.tensor([[1, 0]])
        tokens2 = torch.zeros_like(labels2)

        metric.update({"logits": logits1}, tokens1, labels1, None, cfg)
        metric.update({"logits": logits2}, tokens2, labels2, None, cfg)
        result = metric.compute()

        assert result["acc"] == 0.5  # 2/4 total correct

    def test_accuracy_metric_reset(self):
        """Test that reset clears state."""
        metric = AccuracyMetric()
        cfg = _make_cfg()

        logits = torch.tensor([[[0.9, 0.1]]])
        labels = torch.tensor([[0]])
        tokens = torch.zeros_like(labels)

        metric.update({"logits": logits}, tokens, labels, None, cfg)
        metric.reset()

        result = metric.compute()
        assert result["acc"] == 0.0  # No data after reset


class TestMaskedAccuracyMetric:
    """Tests for MaskedAccuracyMetric."""

    def test_masked_accuracy_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = MaskedAccuracyMetric()
        assert metric.name == "mask_acc"
        assert metric.objectives == {"mlm"}

    def test_masked_accuracy_computation(self):
        """Test masked accuracy computation is identical to regular accuracy."""
        acc_metric = AccuracyMetric()
        mask_metric = MaskedAccuracyMetric()
        cfg = _make_cfg()

        logits = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
        labels = torch.tensor([[0, 1]])
        tokens = torch.zeros_like(labels)

        acc_metric.update({"logits": logits}, tokens, labels, None, cfg)
        mask_metric.update({"logits": logits}, tokens, labels, None, cfg)

        # Values should be identical, just different names
        assert acc_metric.compute()["acc"] == mask_metric.compute()["mask_acc"]


class TestPerplexityMetric:
    """Tests for PerplexityMetric."""

    def test_perplexity_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = PerplexityMetric()
        assert metric.name == "ppl"
        assert metric.objectives is None  # Works for all objectives

    def test_perplexity_from_loss(self):
        """Test perplexity computation from loss."""
        metric = PerplexityMetric()
        cfg = _make_cfg()

        # Loss of 1.0 should give perplexity of e^1 ≈ 2.718
        outputs = {"loss": torch.tensor(1.0), "classification_loss": torch.tensor(1.0)}
        tokens = torch.zeros(1, 1)
        labels = torch.zeros(1, 1)

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()

        assert abs(result["ppl"] - math.e) < 0.01

    def test_perplexity_accumulates_batches(self):
        """Test perplexity accumulates across batches."""
        metric = PerplexityMetric()
        cfg = _make_cfg()

        tokens = torch.zeros(1, 1)
        labels = torch.zeros(1, 1)

        # Two batches with loss 1.0 and 2.0
        metric.update(
            {"loss": torch.tensor(1.0), "classification_loss": torch.tensor(1.0)},
            tokens,
            labels,
            None,
            cfg,
        )
        metric.update(
            {"loss": torch.tensor(2.0), "classification_loss": torch.tensor(2.0)},
            tokens,
            labels,
            None,
            cfg,
        )
        result = metric.compute()

        # Average loss is 1.5, perplexity is e^1.5
        expected_ppl = math.exp(1.5)
        assert abs(result["ppl"] - expected_ppl) < 0.01

    def test_perplexity_includes_cls_loss(self):
        """Test that cls_loss is also returned."""
        metric = PerplexityMetric()
        cfg = _make_cfg()

        outputs = {"loss": torch.tensor(2.0), "classification_loss": torch.tensor(2.0)}
        tokens = torch.zeros(1, 1)
        labels = torch.zeros(1, 1)

        metric.update(outputs, tokens, labels, None, cfg)
        result = metric.compute()

        assert "cls_loss" in result
        assert result["cls_loss"] == 2.0

    def test_perplexity_handles_no_batches(self):
        """Test perplexity with no data returns inf."""
        metric = PerplexityMetric()
        result = metric.compute()
        assert result["ppl"] == float("inf")

