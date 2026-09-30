"""Metric registry and factory for the evaluation system."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch.nn as nn
from omegaconf import DictConfig

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


def resolve_eval_metrics(
    cfg: DictConfig, eval_name: str | None, *, objective: str
) -> dict[str, dict]:
    """Resolve intent before resource loading; dataset overrides win last."""
    import stok.eval.metrics  # noqa: F401

    global_cfg = cfg.get("train", {}).get("eval", {}).get("metrics", {})
    data_eval = cfg.get("data", {}).get("eval", {})
    dataset_cfg = (
        data_eval.get(eval_name, {})
        if isinstance(data_eval, (dict, DictConfig))
        else {}
    )
    overrides = (
        dataset_cfg.get("metrics", {})
        if isinstance(dataset_cfg, (dict, DictConfig))
        else {}
    )
    only = overrides.get("only")
    if isinstance(only, str):
        only = [only]
    unknown = (set(global_cfg) | (set(overrides) - {"only"}) | set(only or [])) - set(
        METRIC_REGISTRY
    )
    if unknown:
        raise ValueError(f"Unknown metric IDs: {sorted(unknown)}")
    auto = bool(cfg.get("train", {}).get("decoding", {}).get("eval_enabled", False))
    selected = {}
    for name, cls in METRIC_REGISTRY.items():
        settings = {
            "enabled": name in {"accuracy", "masked_accuracy", "perplexity", "p_at_l"}
        }
        raw = global_cfg.get(name, {})
        if isinstance(raw, (dict, DictConfig)):
            settings.update(dict(raw))
        elif isinstance(raw, bool):
            settings["enabled"] = raw
        explicit = bool(settings["enabled"] is True and cls.requires_decoder)
        if settings["enabled"] is None:
            settings["enabled"] = (
                auto if name in {"lddt", "tm_score", "rmsd"} else False
            )
        if only is not None:
            settings["enabled"] = name in only
            explicit = name in only
        override = overrides.get(name)
        if isinstance(override, (dict, DictConfig)):
            settings.update(dict(override))
            explicit |= override.get("enabled") is True
        elif isinstance(override, bool):
            settings["enabled"] = override
            explicit |= override
        if not settings["enabled"]:
            continue
        if cls.objectives is not None and objective not in cls.objectives:
            if explicit:
                raise ValueError(
                    f"Metric {name} does not support objective {objective}"
                )
            continue
        settings["explicit"] = explicit
        selected[name] = settings
    return selected


def build_metrics(
    cfg: DictConfig,
    objective: str,
    decoder: nn.Module | None = None,
    has_coords: bool = False,
    eval_name: str | None = None,
    *,
    has_labels: bool = True,
    resolved: dict[str, dict] | None = None,
) -> list[Metric]:
    """Construct resolved metrics against observed dataset capabilities.

    Resource flags describe actual loaded data, not configuration wishes.
    """
    if resolved is None:
        resolved = resolve_eval_metrics(cfg, eval_name, objective=objective)
    metrics = []
    for name, settings in resolved.items():
        cls = METRIC_REGISTRY[name]
        missing = []
        if cls.requires_decoder and decoder is None:
            missing.append("decoder")
        if cls.requires_coords and not has_coords:
            missing.append("coordinates")
        if name in {"accuracy", "masked_accuracy", "perplexity"} and not has_labels:
            missing.append("labels")
        if missing:
            if settings.get("explicit"):
                raise ValueError(
                    f"Dataset {eval_name}, metric {name}: missing {', '.join(missing)}"
                )
            continue
        kwargs = {
            k: v
            for k, v in settings.items()
            if k
            not in {
                "enabled",
                "explicit",
                "objectives",
                "requires_decoder",
                "requires_coords",
            }
        }
        if name == "p_at_l" and kwargs.get("num_layers") is None:
            kwargs["num_layers"] = math.ceil(
                cfg.get("model", {}).get("encoder", {}).get("n_layers", 12) * 0.1
            )
        metric = cls(**kwargs)
        metric.explicit = settings.get("explicit", False)
        metrics.append(metric)
    return metrics


def get_registered_metrics() -> dict[str, type[Metric]]:
    """Get a copy of the metric registry.

    Returns:
        Dictionary mapping metric names to their classes.
    """
    return dict(METRIC_REGISTRY)
