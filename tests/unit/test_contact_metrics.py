"""Unit tests for contact prediction metrics, specifically num_layers functionality."""

import pytest
import torch
from omegaconf import OmegaConf

from stok.eval.metrics.contact import (
    PrecisionAtLMetric,
    _extract_attention_contacts,
)


class TestExtractAttentionContacts:
    """Tests for the _extract_attention_contacts function."""

    def test_single_layer_default(self):
        """Test default behavior with num_layers=1 (last layer only)."""
        B, H, L = 2, 4, 16
        n_layers = 6
        
        # Create mock attention weights for 6 layers
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}
        
        result = _extract_attention_contacts(
            outputs,
            layer="last",
            head_aggregation="mean",
            num_layers=1,
        )
        
        assert result is not None
        assert result.shape == (B, L, L)
        
        # Verify it's using only the last layer (manually compute expected)
        expected = attentions[-1].mean(dim=1)  # Average over heads
        expected = (expected + expected.transpose(-1, -2)) / 2  # Symmetrize
        torch.testing.assert_close(result, expected)

    def test_multi_layer_averaging(self):
        """Test averaging attention from final num_layers layers."""
        B, H, L = 2, 4, 16
        n_layers = 6
        num_layers_to_use = 3
        
        # Create mock attention weights
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}
        
        result = _extract_attention_contacts(
            outputs,
            layer="last",
            head_aggregation="mean",
            num_layers=num_layers_to_use,
        )
        
        assert result is not None
        assert result.shape == (B, L, L)
        
        # Manually compute expected result
        # Stack last 3 layers and average
        stacked = torch.stack(attentions[-num_layers_to_use:], dim=0)  # [3, B, H, L, L]
        layer_avg = stacked.mean(dim=0)  # [B, H, L, L]
        head_avg = layer_avg.mean(dim=1)  # [B, L, L]
        expected = (head_avg + head_avg.transpose(-1, -2)) / 2  # Symmetrize
        
        torch.testing.assert_close(result, expected)

    def test_num_layers_clamped_to_available(self):
        """Test that num_layers is clamped to available layers."""
        B, H, L = 2, 4, 16
        n_layers = 3
        
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}
        
        # Request more layers than available
        result = _extract_attention_contacts(
            outputs,
            layer="last",
            head_aggregation="mean",
            num_layers=10,  # More than available
        )
        
        assert result is not None
        assert result.shape == (B, L, L)
        
        # Should use all available layers (3)
        stacked = torch.stack(attentions, dim=0)  # All 3 layers
        layer_avg = stacked.mean(dim=0)
        head_avg = layer_avg.mean(dim=1)
        expected = (head_avg + head_avg.transpose(-1, -2)) / 2
        
        torch.testing.assert_close(result, expected)

    def test_num_layers_one_equals_last(self):
        """Test that num_layers=1 produces same result as layer='last' behavior."""
        B, H, L = 2, 4, 16
        n_layers = 6
        
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}
        
        result_num_layers_1 = _extract_attention_contacts(
            outputs,
            layer="last",
            head_aggregation="mean",
            num_layers=1,
        )
        
        # num_layers is ignored when layer is int, but for layer="last" with num_layers=1
        # it should be equivalent to using only the last layer
        expected = attentions[-1].mean(dim=1)
        expected = (expected + expected.transpose(-1, -2)) / 2
        
        torch.testing.assert_close(result_num_layers_1, expected)

    def test_layer_int_ignores_num_layers(self):
        """Test that specifying layer as int ignores num_layers parameter."""
        B, H, L = 2, 4, 16
        n_layers = 6
        
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}
        
        # Use layer=2 with num_layers=3 - should only use layer 2
        result = _extract_attention_contacts(
            outputs,
            layer=2,
            head_aggregation="mean",
            num_layers=3,
        )
        
        expected = attentions[2].mean(dim=1)
        expected = (expected + expected.transpose(-1, -2)) / 2
        
        torch.testing.assert_close(result, expected)

    def test_layer_mean_ignores_num_layers(self):
        """Test that layer='mean' ignores num_layers and uses all layers."""
        B, H, L = 2, 4, 16
        n_layers = 6
        
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}
        
        result = _extract_attention_contacts(
            outputs,
            layer="mean",
            head_aggregation="mean",
            num_layers=2,  # Should be ignored
        )
        
        # Should use ALL layers
        stacked = torch.stack(attentions, dim=0)
        layer_avg = stacked.mean(dim=0)
        head_avg = layer_avg.mean(dim=1)
        expected = (head_avg + head_avg.transpose(-1, -2)) / 2
        
        torch.testing.assert_close(result, expected)

    def test_returns_none_without_attentions(self):
        """Test that function returns None when attentions not in outputs."""
        outputs = {"logits": torch.randn(2, 16, 32)}
        
        result = _extract_attention_contacts(
            outputs,
            layer="last",
            head_aggregation="mean",
            num_layers=3,
        )
        
        assert result is None

    def test_head_aggregation_max(self):
        """Test max head aggregation with multi-layer averaging."""
        B, H, L = 2, 4, 16
        n_layers = 4
        num_layers_to_use = 2
        
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}
        
        result = _extract_attention_contacts(
            outputs,
            layer="last",
            head_aggregation="max",
            num_layers=num_layers_to_use,
        )
        
        assert result is not None
        assert result.shape == (B, L, L)
        
        # Manually compute with max aggregation
        stacked = torch.stack(attentions[-num_layers_to_use:], dim=0)
        layer_avg = stacked.mean(dim=0)
        head_max = layer_avg.max(dim=1).values
        expected = (head_max + head_max.transpose(-1, -2)) / 2
        
        torch.testing.assert_close(result, expected)


class TestPrecisionAtLMetricNumLayers:
    """Tests for PrecisionAtLMetric with num_layers parameter."""

    def test_init_accepts_num_layers(self):
        """Test that metric accepts num_layers parameter."""
        metric = PrecisionAtLMetric(
            contact_threshold=8.0,
            min_seq_sep=6,
            use_attention=True,
            num_layers=3,
        )
        
        assert metric.num_layers == 3

    def test_init_default_num_layers(self):
        """Test that default num_layers is 1."""
        metric = PrecisionAtLMetric()
        
        assert metric.num_layers == 1

    def test_update_uses_num_layers(self):
        """Test that update method passes num_layers to extraction function."""
        metric = PrecisionAtLMetric(
            contact_threshold=8.0,
            min_seq_sep=3,
            use_attention=True,
            num_layers=2,
        )
        
        B, H, L = 2, 4, 32
        n_layers = 4
        
        # Create inputs
        tokens = torch.randint(4, 24, (B, L))
        labels = torch.full((B, L), -100)
        coords = torch.randn(B, L, 3, 3) * 10.0
        
        attentions = tuple(
            torch.softmax(torch.randn(B, H, L, L), dim=-1)
            for _ in range(n_layers)
        )
        
        outputs = {
            "logits": torch.randn(B, L, 32),
            "loss": torch.tensor(2.5),
            "attentions": attentions,
        }
        
        cfg = OmegaConf.create({
            "model": {
                "encoder": {"pad_id": 1},
                "classifier": {"ignore_index": -100},
            }
        })
        
        # Should not raise
        metric.update(outputs, tokens, labels, coords, cfg)
        
        result = metric.compute()
        assert "p_at_l" in result
        assert 0.0 <= result["p_at_l"] <= 1.0


class TestPrecisionAtLMetricConfig:
    """Tests for config-based instantiation of P@L metric with num_layers."""

    def test_num_layers_from_config(self):
        """Test that num_layers is correctly passed from config."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            "contact_threshold": 8.0,
                            "min_seq_sep": 6,
                            "use_attention": True,
                            "num_layers": 4,
                        },
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {"classifier": {"ignore_index": -100}},
        })
        
        metrics = build_metrics(cfg, objective="mlm", has_coords=True)
        p_at_l = next(
            (m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"),
            None,
        )
        
        assert p_at_l is not None
        assert p_at_l.num_layers == 4

    def test_num_layers_default_from_config(self):
        """Test that num_layers defaults to 1 when not specified in config."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            # num_layers not specified
                        },
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {"classifier": {"ignore_index": -100}},
        })
        
        metrics = build_metrics(cfg, objective="mlm", has_coords=True)
        p_at_l = next(
            (m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"),
            None,
        )
        
        assert p_at_l is not None
        assert p_at_l.num_layers == 1

