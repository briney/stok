"""Unit tests for the metric registry and factory."""

import pytest
from omegaconf import OmegaConf

from stok.eval.base import MetricBase
from stok.eval.registry import (
    METRIC_REGISTRY,
    build_metrics,
    get_registered_metrics,
    register_metric,
)


def test_metric_registry_not_empty():
    """Test that the registry contains metrics after import."""
    # Import to trigger registration
    import stok.eval.metrics  # noqa: F401

    assert len(METRIC_REGISTRY) > 0
    assert "accuracy" in METRIC_REGISTRY
    assert "perplexity" in METRIC_REGISTRY


def test_register_metric_decorator():
    """Test the register_metric decorator."""
    # Create a unique name to avoid conflicts
    test_name = "_test_metric_decorator"

    @register_metric(test_name)
    class TestMetric(MetricBase):
        name = test_name
        objectives = {"test"}

        def update(self, outputs, tokens, labels, coords, cfg):
            pass

        def compute(self):
            return {}

        def reset(self):
            pass

    assert test_name in METRIC_REGISTRY
    assert METRIC_REGISTRY[test_name] is TestMetric

    # Cleanup
    del METRIC_REGISTRY[test_name]


def test_register_metric_duplicate_raises():
    """Test that registering a duplicate name raises an error."""
    test_name = "_test_duplicate"

    @register_metric(test_name)
    class TestMetric1(MetricBase):
        name = test_name

        def update(self, outputs, tokens, labels, coords, cfg):
            pass

        def compute(self):
            return {}

        def reset(self):
            pass

    with pytest.raises(ValueError, match="already registered"):

        @register_metric(test_name)
        class TestMetric2(MetricBase):
            name = test_name

            def update(self, outputs, tokens, labels, coords, cfg):
                pass

            def compute(self):
                return {}

            def reset(self):
                pass

    # Cleanup
    del METRIC_REGISTRY[test_name]


def test_get_registered_metrics():
    """Test that get_registered_metrics returns a copy of the registry."""
    registry = get_registered_metrics()
    assert isinstance(registry, dict)
    assert len(registry) > 0

    # Should be a copy, not the original
    registry["_fake"] = None
    assert "_fake" not in METRIC_REGISTRY


def test_build_metrics_filters_by_objective():
    """Test that build_metrics filters metrics by objective."""
    cfg = OmegaConf.create(
        {
            "train": {"eval": {"metrics": {}}},
            "data": {},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Build for codebook objective
    codebook_metrics = build_metrics(cfg, objective="codebook")
    codebook_names = {type(m).__name__ for m in codebook_metrics}

    # Build for mlm objective
    mlm_metrics = build_metrics(cfg, objective="mlm")
    mlm_names = {type(m).__name__ for m in mlm_metrics}

    # AccuracyMetric should only be in codebook
    assert "AccuracyMetric" in codebook_names
    assert "AccuracyMetric" not in mlm_names

    # MaskedAccuracyMetric should only be in mlm
    assert "MaskedAccuracyMetric" in mlm_names
    assert "MaskedAccuracyMetric" not in codebook_names

    # Perplexity should be in both
    assert "PerplexityMetric" in codebook_names
    assert "PerplexityMetric" in mlm_names


def test_build_metrics_respects_enabled_flag():
    """Test that build_metrics respects the enabled flag."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": False},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(cfg, objective="codebook")
    metric_names = {type(m).__name__ for m in metrics}

    # Accuracy disabled, perplexity enabled
    assert "AccuracyMetric" not in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_filters_by_decoder_requirement():
    """Test that metrics requiring decoder are filtered when decoder is None."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "lddt": {"enabled": True},
                        "accuracy": {"enabled": True},
                    }
                }
            },
            "data": {},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Without decoder
    metrics_no_decoder = build_metrics(cfg, objective="codebook", decoder=None)
    names_no_decoder = {type(m).__name__ for m in metrics_no_decoder}

    # LDDTMetric requires decoder
    assert "LDDTMetric" not in names_no_decoder
    assert "AccuracyMetric" in names_no_decoder


def test_build_metrics_filters_by_coords_requirement():
    """Test that metrics requiring coords are filtered when coords unavailable."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "lddt": {"enabled": True},
                        "accuracy": {"enabled": True},
                    }
                }
            },
            "data": {},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Create a mock decoder
    class MockDecoder:
        pass

    decoder = MockDecoder()

    # Without coords
    metrics_no_coords = build_metrics(
        cfg, objective="codebook", decoder=decoder, has_coords=False
    )
    names_no_coords = {type(m).__name__ for m in metrics_no_coords}

    # LDDTMetric requires coords
    assert "LDDTMetric" not in names_no_coords


def test_build_metrics_passes_config_params():
    """Test that metric-specific config params are passed to constructor."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            "contact_threshold": 6.0,
                            "min_seq_sep": 8,
                        },
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(cfg, objective="mlm", has_coords=True)
    p_at_l = next((m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"), None)

    assert p_at_l is not None
    assert p_at_l.contact_threshold == 6.0
    assert p_at_l.min_seq_sep == 8

