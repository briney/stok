"""Unit tests for the Evaluator class."""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from stok.eval.evaluator import Evaluator


class MockModel(nn.Module):
    """A simple mock model for testing."""

    def __init__(self, vocab_size: int = 32, num_classes: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        # Simple linear layer
        self.linear = nn.Linear(vocab_size, num_classes)
        # Mock classifier with codebook
        self.classifier = nn.Module()
        self.classifier.E = nn.Parameter(torch.randn(num_classes, 32))

    def forward(self, tokens, labels=None, ignore_index=-100, **kwargs):
        B, L = tokens.shape
        # Return mock logits and loss
        logits = torch.randn(B, L, self.num_classes)
        loss = torch.tensor(1.0)
        return {
            "logits": logits,
            "loss": loss,
            "classification_loss": loss,
        }


def _make_cfg(objective: str = "codebook"):
    """Create a test configuration."""
    return OmegaConf.create(
        {
            "train": {
                "objective": objective,
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": True},
                        "masked_accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                },
                "decoding": {
                    "eval_enabled": False,
                    "eval_method": "argmax",
                    "temperature": 1.0,
                    "top_p": 0.9,
                },
            },
            "data": {"load_coords": False},
            "model": {
                "classifier": {"ignore_index": -100},
                "encoder": {"pad_id": 1, "vocab_size": 32},
            },
        }
    )


def _make_eval_loader(batch_size: int = 4, seq_len: int = 16, num_batches: int = 2):
    """Create a simple eval dataloader."""
    tokens = torch.randint(2, 30, (batch_size * num_batches, seq_len))
    labels = torch.randint(0, 64, (batch_size * num_batches, seq_len))
    dataset = TensorDataset(tokens, labels)
    return DataLoader(dataset, batch_size=batch_size)


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_evaluator_initialization(self):
        """Test Evaluator initializes correctly."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)

        assert evaluator.objective == "codebook"
        assert evaluator.decoder is None
        assert evaluator.has_coords is False

    def test_evaluator_builds_metrics_for_codebook(self):
        """Test Evaluator builds appropriate metrics for codebook objective."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        metrics = evaluator._get_metrics()

        metric_names = {type(m).__name__ for m in metrics}
        assert "AccuracyMetric" in metric_names
        assert "PerplexityMetric" in metric_names
        assert "MaskedAccuracyMetric" not in metric_names  # MLM only

    def test_evaluator_builds_metrics_for_mlm(self):
        """Test Evaluator builds appropriate metrics for MLM objective."""
        cfg = _make_cfg("mlm")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        metrics = evaluator._get_metrics()

        metric_names = {type(m).__name__ for m in metrics}
        assert "MaskedAccuracyMetric" in metric_names
        assert "PerplexityMetric" in metric_names
        assert "AccuracyMetric" not in metric_names  # Codebook only

    def test_evaluator_evaluate_returns_metrics(self):
        """Test that evaluate() returns metric values."""
        cfg = _make_cfg("codebook")
        model = MockModel()
        eval_loader = _make_eval_loader()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        results = evaluator.evaluate(eval_loader, "test")

        assert isinstance(results, dict)
        assert "acc" in results
        assert "ppl" in results
        assert isinstance(results["acc"], float)
        assert isinstance(results["ppl"], float)

    def test_evaluator_evaluate_all_multiple_datasets(self):
        """Test evaluate_all with multiple datasets."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        eval_loaders = {
            "val": _make_eval_loader(),
            "test": _make_eval_loader(),
        }

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        all_results = evaluator.evaluate_all(eval_loaders)

        assert "val" in all_results
        assert "test" in all_results
        assert "acc" in all_results["val"]
        assert "acc" in all_results["test"]

    def test_evaluator_caches_metrics(self):
        """Test that metrics are cached per eval dataset."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)

        metrics1 = evaluator._get_metrics("dataset1")
        metrics2 = evaluator._get_metrics("dataset1")
        metrics3 = evaluator._get_metrics("dataset2")

        # Same dataset should return cached metrics
        assert metrics1 is metrics2
        # Different dataset should have different metrics
        assert metrics1 is not metrics3

    def test_evaluator_clear_cache(self):
        """Test that clear_cache clears the metrics cache."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)

        metrics1 = evaluator._get_metrics()
        evaluator.clear_cache()
        metrics2 = evaluator._get_metrics()

        # After clear, should get new metrics
        assert metrics1 is not metrics2

    def test_evaluator_sets_model_to_eval_mode(self):
        """Test that evaluate sets model to eval mode."""
        cfg = _make_cfg("codebook")
        model = MockModel()
        model.train()
        eval_loader = _make_eval_loader()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)

        # Model starts in train mode
        assert model.training

        evaluator.evaluate(eval_loader, "test")

        # Model should be back in train mode after evaluate
        assert model.training

    def test_evaluator_handles_coords_in_batch(self):
        """Test that evaluate handles batches with coordinates."""
        cfg = _make_cfg("codebook")
        cfg.data.load_coords = True
        model = MockModel()

        # Create loader with coords
        B, L = 4, 16
        tokens = torch.randint(2, 30, (B, L))
        labels = torch.randint(0, 64, (B, L))
        coords = torch.randn(B, L, 3, 3)
        dataset = TensorDataset(tokens, labels, coords)
        eval_loader = DataLoader(dataset, batch_size=2)

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        results = evaluator.evaluate(eval_loader, "test")

        # Should complete without error
        assert "acc" in results


class TestEvaluatorDistributed:
    """Tests for distributed functionality (mocked)."""

    def test_evaluator_state_tensor_aggregation(self):
        """Test that metric state tensors can be aggregated."""
        from stok.eval.metrics.classification import AccuracyMetric

        # Simulate two "processes" with different states
        metric1 = AccuracyMetric()
        metric1._correct = 10.0
        metric1._total = 20.0

        metric2 = AccuracyMetric()
        metric2._correct = 15.0
        metric2._total = 30.0

        # Get state tensors
        state1 = metric1.state_tensors()
        state2 = metric2.state_tensors()

        # Simulate gather (sum)
        gathered = [state1[0] + state2[0]]

        # Load into new metric
        metric_combined = AccuracyMetric()
        metric_combined.load_state_tensors(gathered)

        # Check combined state
        assert metric_combined._correct == 25.0
        assert metric_combined._total == 50.0
        assert metric_combined.compute()["acc"] == 0.5

