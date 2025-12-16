"""Unit tests for the evaluation system base classes."""

import torch
from omegaconf import OmegaConf

from stok.eval.base import Metric, MetricBase


class DummyMetric(MetricBase):
    """A simple test metric for testing the base class."""

    name = "dummy"
    objectives = {"test"}
    requires_decoder = False
    requires_coords = False

    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self._sum = 0.0
        self._count = 0

    def update(self, outputs, tokens, labels, coords, cfg):
        self._sum += float(outputs.get("value", 0)) * self.scale
        self._count += 1

    def compute(self):
        return {self.name: self._sum / max(1, self._count)}

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def state_tensors(self):
        return [torch.tensor([self._sum, float(self._count)])]

    def load_state_tensors(self, tensors):
        if tensors:
            self._sum = float(tensors[0][0].item())
            self._count = int(tensors[0][1].item())


def test_metric_protocol_compliance():
    """Test that MetricBase subclasses satisfy the Metric protocol."""
    metric = DummyMetric()
    assert isinstance(metric, Metric)


def test_metric_base_initialization():
    """Test MetricBase initialization with kwargs."""
    metric = DummyMetric(scale=2.0)
    assert metric.scale == 2.0


def test_metric_update_and_compute():
    """Test the update and compute cycle."""
    metric = DummyMetric(scale=2.0)
    cfg = OmegaConf.create({})

    # Simulate multiple batch updates
    metric.update({"value": 1.0}, torch.zeros(1), torch.zeros(1), None, cfg)
    metric.update({"value": 2.0}, torch.zeros(1), torch.zeros(1), None, cfg)
    metric.update({"value": 3.0}, torch.zeros(1), torch.zeros(1), None, cfg)

    result = metric.compute()
    # (1*2 + 2*2 + 3*2) / 3 = 12/3 = 4.0
    assert result["dummy"] == 4.0


def test_metric_reset():
    """Test that reset clears accumulated state."""
    metric = DummyMetric()
    cfg = OmegaConf.create({})

    metric.update({"value": 10.0}, torch.zeros(1), torch.zeros(1), None, cfg)
    assert metric._sum == 10.0
    assert metric._count == 1

    metric.reset()
    assert metric._sum == 0.0
    assert metric._count == 0


def test_metric_state_tensors_roundtrip():
    """Test state serialization and deserialization."""
    metric1 = DummyMetric()
    cfg = OmegaConf.create({})

    metric1.update({"value": 5.0}, torch.zeros(1), torch.zeros(1), None, cfg)
    metric1.update({"value": 10.0}, torch.zeros(1), torch.zeros(1), None, cfg)

    # Get state and load into new metric
    state = metric1.state_tensors()
    metric2 = DummyMetric()
    metric2.load_state_tensors(state)

    # Both should compute the same result
    assert metric1.compute() == metric2.compute()


def test_metric_base_default_state_methods():
    """Test that default state methods work (no-op for simple metrics)."""

    class SimpleMetric(MetricBase):
        name = "simple"

        def __init__(self):
            super().__init__()
            self.value = 0.0

        def update(self, outputs, tokens, labels, coords, cfg):
            self.value += 1.0

        def compute(self):
            return {"simple": self.value}

        def reset(self):
            self.value = 0.0

    metric = SimpleMetric()
    # Default implementations should not raise
    assert metric.state_tensors() == []
    metric.load_state_tensors([])  # Should be no-op


def test_metric_class_attributes():
    """Test that class attributes are properly set."""
    assert DummyMetric.name == "dummy"
    assert DummyMetric.objectives == {"test"}
    assert DummyMetric.requires_decoder is False
    assert DummyMetric.requires_coords is False

