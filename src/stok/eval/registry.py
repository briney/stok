"""Metric registry and factory for the evaluation system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from stok.eval.base import Metric

# Global registry mapping metric names to their classes
METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(name: str):
    """Decorator to register a metric class in the global registry.

    Args:
        name: Unique identifier for the metric. This is used in config files
            to enable/configure the metric.

    Returns:
        Decorator function that registers the class.

    Example:
        @register_metric("accuracy")
        class AccuracyMetric(MetricBase):
            ...
    """

    def decorator(cls: type[Metric]) -> type[Metric]:
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered")
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator


def _get_metric_config(
    cfg: DictConfig,
    metric_name: str,
    eval_name: str | None = None,
) -> dict[str, Any]:
    """Get merged configuration for a specific metric.

    Configuration is merged in order of precedence (later overrides earlier):
    1. Default metric config from train.eval.metrics.{metric_name}
    2. Per-dataset overrides from data.eval.{eval_name}.metrics.{metric_name}

    Args:
        cfg: Full configuration object.
        metric_name: Name of the metric to get config for.
        eval_name: Optional name of the eval dataset for per-dataset overrides.

    Returns:
        Merged configuration dictionary for the metric.
    """
    result: dict[str, Any] = {"enabled": True}

    # Get default config from train.eval.metrics
    train_eval = cfg.get("train", {}).get("eval", {})
    if train_eval:
        metrics_cfg = train_eval.get("metrics", {})
        if metrics_cfg and metric_name in metrics_cfg:
            metric_cfg = metrics_cfg[metric_name]
            if isinstance(metric_cfg, (dict, DictConfig)):
                result.update(OmegaConf.to_container(metric_cfg, resolve=True))  # type: ignore
            elif metric_cfg is False:
                result["enabled"] = False

    # Get per-dataset overrides from data.eval.{eval_name}.metrics
    if eval_name:
        data_eval = cfg.get("data", {}).get("eval", {})
        if data_eval and eval_name in data_eval:
            eval_cfg = data_eval[eval_name]
            if isinstance(eval_cfg, (dict, DictConfig)):
                eval_metrics = eval_cfg.get("metrics", {})
                if eval_metrics and metric_name in eval_metrics:
                    override = eval_metrics[metric_name]
                    if isinstance(override, (dict, DictConfig)):
                        result.update(OmegaConf.to_container(override, resolve=True))  # type: ignore
                    elif override is False:
                        result["enabled"] = False

    return result


def build_metrics(
    cfg: DictConfig,
    objective: str,
    decoder: nn.Module | None = None,
    has_coords: bool = False,
    eval_name: str | None = None,
) -> list[Metric]:
    """Build a list of metric instances based on configuration.

    Args:
        cfg: Full configuration object.
        objective: Current training objective ("codebook" or "mlm").
        decoder: Optional decoder model (required for structure metrics).
        has_coords: Whether coordinate data is available.
        eval_name: Optional eval dataset name for per-dataset metric overrides.

    Returns:
        List of instantiated Metric objects that are enabled and compatible
        with the current objective and available resources.
    """
    # Import metrics to ensure they're registered
    # This import is deferred to avoid circular imports
    import stok.eval.metrics  # noqa: F401

    metrics: list[Metric] = []

    for name, cls in METRIC_REGISTRY.items():
        # Get merged config for this metric
        metric_cfg = _get_metric_config(cfg, name, eval_name)

        # Skip if explicitly disabled
        if not metric_cfg.get("enabled", True):
            continue

        # Check objective compatibility
        cls_objectives = getattr(cls, "objectives", None)
        if cls_objectives is not None and objective not in cls_objectives:
            continue

        # Check resource requirements
        if getattr(cls, "requires_decoder", False) and decoder is None:
            continue
        if getattr(cls, "requires_coords", False) and not has_coords:
            continue

        # Filter out meta-config keys before passing to constructor
        init_kwargs = {
            k: v
            for k, v in metric_cfg.items()
            if k not in ("enabled", "objectives", "requires_decoder", "requires_coords")
        }

        # Instantiate the metric
        try:
            metric = cls(**init_kwargs)
            metrics.append(metric)
        except Exception as e:
            # Log warning but don't fail - allows graceful degradation
            import warnings

            warnings.warn(f"Failed to instantiate metric '{name}': {e}")

    return metrics


def get_registered_metrics() -> dict[str, type[Metric]]:
    """Get a copy of the metric registry.

    Returns:
        Dictionary mapping metric names to their classes.
    """
    return dict(METRIC_REGISTRY)

