"""Modular evaluation metrics system for STok training."""

from stok.eval.base import Metric, MetricBase
from stok.eval.evaluator import Evaluator
from stok.eval.logger import MetricLogger
from stok.eval.registry import METRIC_REGISTRY, build_metrics, register_metric

# Import metrics to ensure they're registered
import stok.eval.metrics  # noqa: F401

__all__ = [
    "Metric",
    "MetricBase",
    "Evaluator",
    "MetricLogger",
    "METRIC_REGISTRY",
    "build_metrics",
    "register_metric",
]

