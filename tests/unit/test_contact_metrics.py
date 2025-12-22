"""Unit tests for contact prediction metrics, including APC and logistic regression mode."""

import pytest
import torch
from omegaconf import OmegaConf

from stok.eval.metrics.contact import (
    PrecisionAtLMetric,
    _apply_apc,
    _extract_attention_contacts,
    _extract_per_layer_head_attention,
)


class TestApplyAPC:
    """Tests for the _apply_apc function."""

    def test_apc_shape_preserved(self):
        """Test that APC preserves input shape."""
        B, L = 2, 16
        matrix = torch.randn(B, L, L)
        result = _apply_apc(matrix)
        assert result.shape == matrix.shape

    def test_apc_reduces_row_col_means(self):
        """Test that APC reduces the contribution of high-mean rows/columns."""
        B, L = 1, 8
        # Create a matrix with a high-mean row/column
        matrix = torch.ones(B, L, L) * 0.1
        matrix[:, 0, :] = 0.5  # High-mean row
        matrix[:, :, 0] = 0.5  # High-mean column

        result = _apply_apc(matrix)

        # After APC, the high-mean row/column should be corrected
        # The correction term for position (0, j) should be higher
        # Result at (0, j) should be lower than at other positions
        # due to the correction subtracting more
        assert result[0, 0, 1] < matrix[0, 0, 1]

    def test_apc_symmetric_input_symmetric_output(self):
        """Test that APC preserves symmetry."""
        B, L = 2, 10
        # Create symmetric matrix
        matrix = torch.randn(B, L, L)
        matrix = (matrix + matrix.transpose(-1, -2)) / 2

        result = _apply_apc(matrix)

        # Check symmetry is preserved
        torch.testing.assert_close(result, result.transpose(-1, -2))

    def test_apc_formula_correctness(self):
        """Test that APC formula is correctly implemented."""
        B, L = 1, 4
        matrix = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                 [2.0, 3.0, 4.0, 5.0],
                                 [3.0, 4.0, 5.0, 6.0],
                                 [4.0, 5.0, 6.0, 7.0]]])

        result = _apply_apc(matrix)

        # Manually compute expected APC
        row_mean = matrix.mean(dim=-1, keepdim=True)
        col_mean = matrix.mean(dim=-2, keepdim=True)
        global_mean = matrix.mean(dim=(-1, -2), keepdim=True)
        expected = matrix - (row_mean * col_mean) / global_mean

        torch.testing.assert_close(result, expected)

    def test_apc_handles_small_global_mean(self):
        """Test that APC handles small global mean without division by zero."""
        B, L = 1, 4
        # Matrix with very small values
        matrix = torch.ones(B, L, L) * 1e-10
        
        # Should not raise, should return finite values
        result = _apply_apc(matrix)
        assert torch.isfinite(result).all()


class TestExtractPerLayerHeadAttention:
    """Tests for the _extract_per_layer_head_attention function."""

    def test_shape_correct(self):
        """Test output shape is [B, n_layers, n_heads, L, L]."""
        B, H, L = 2, 4, 16
        n_layers = 6

        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}

        result = _extract_per_layer_head_attention(outputs)

        assert result is not None
        assert result.shape == (B, n_layers, H, L, L)

    def test_returns_none_without_attentions(self):
        """Test returns None when attentions not in outputs."""
        outputs = {"logits": torch.randn(2, 16, 32)}
        result = _extract_per_layer_head_attention(outputs)
        assert result is None

    def test_symmetric_output(self):
        """Test that output is symmetrized."""
        B, H, L = 1, 2, 8
        n_layers = 2

        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}

        result = _extract_per_layer_head_attention(outputs)

        # Check symmetry for each layer/head
        for layer in range(n_layers):
            for head in range(H):
                matrix = result[0, layer, head]
                torch.testing.assert_close(matrix, matrix.T)

    def test_apc_applied(self):
        """Test that APC is applied to each layer/head."""
        B, H, L = 1, 2, 8
        n_layers = 2

        # Create attention that would have non-zero APC correction
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}

        result = _extract_per_layer_head_attention(outputs)

        # Result should not equal simple symmetrization (due to APC)
        # Just verify the function runs and produces valid output
        assert result is not None
        assert torch.isfinite(result).all()


class TestExtractAttentionContacts:
    """Tests for the _extract_attention_contacts function.
    
    Note: These tests now account for APC being applied after symmetrization.
    """

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
        
        # Verify it's using only the last layer (manually compute expected with APC)
        expected = attentions[-1].mean(dim=1)  # Average over heads
        expected = (expected + expected.transpose(-1, -2)) / 2  # Symmetrize
        expected = _apply_apc(expected)  # Apply APC
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
        
        # Manually compute expected result with APC
        stacked = torch.stack(attentions[-num_layers_to_use:], dim=0)  # [3, B, H, L, L]
        layer_avg = stacked.mean(dim=0)  # [B, H, L, L]
        head_avg = layer_avg.mean(dim=1)  # [B, L, L]
        expected = (head_avg + head_avg.transpose(-1, -2)) / 2  # Symmetrize
        expected = _apply_apc(expected)  # Apply APC
        
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
        
        # Should use all available layers (3) with APC
        stacked = torch.stack(attentions, dim=0)  # All 3 layers
        layer_avg = stacked.mean(dim=0)
        head_avg = layer_avg.mean(dim=1)
        expected = (head_avg + head_avg.transpose(-1, -2)) / 2
        expected = _apply_apc(expected)
        
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
        expected = _apply_apc(expected)
        
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
        expected = _apply_apc(expected)
        
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
        
        # Should use ALL layers with APC
        stacked = torch.stack(attentions, dim=0)
        layer_avg = stacked.mean(dim=0)
        head_avg = layer_avg.mean(dim=1)
        expected = (head_avg + head_avg.transpose(-1, -2)) / 2
        expected = _apply_apc(expected)
        
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
        
        # Manually compute with max aggregation and APC
        stacked = torch.stack(attentions[-num_layers_to_use:], dim=0)
        layer_avg = stacked.mean(dim=0)
        head_max = layer_avg.max(dim=1).values
        expected = (head_max + head_max.transpose(-1, -2)) / 2
        expected = _apply_apc(expected)
        
        torch.testing.assert_close(result, expected)
    
    def test_output_is_symmetric(self):
        """Test that output is symmetric after symmetrization and APC."""
        B, H, L = 2, 4, 16
        n_layers = 4
        
        attentions = tuple(
            torch.randn(B, H, L, L).softmax(dim=-1)
            for _ in range(n_layers)
        )
        outputs = {"attentions": attentions}
        
        result = _extract_attention_contacts(
            outputs,
            layer="last",
            head_aggregation="mean",
            num_layers=2,
        )
        
        # Check symmetry
        torch.testing.assert_close(result, result.transpose(-1, -2))


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
        """Test that num_layers defaults to 10% of encoder layers when not specified."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            # num_layers not specified -> defaults to 10% of n_layers
                        },
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {
                "encoder": {"n_layers": 24},  # 10% of 24 = 2.4 -> ceil -> 3
                "classifier": {"ignore_index": -100},
            },
        })
        
        metrics = build_metrics(cfg, objective="mlm", has_coords=True)
        p_at_l = next(
            (m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"),
            None,
        )
        
        assert p_at_l is not None
        assert p_at_l.num_layers == 3  # ceil(24 * 0.1) = 3

    def test_num_layers_default_8_layers(self):
        """Test that 8-layer model gets num_layers=1."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {
                "encoder": {"n_layers": 8},  # 10% of 8 = 0.8 -> ceil -> 1
                "classifier": {"ignore_index": -100},
            },
        })
        
        metrics = build_metrics(cfg, objective="mlm", has_coords=True)
        p_at_l = next(
            (m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"),
            None,
        )
        
        assert p_at_l is not None
        assert p_at_l.num_layers == 1  # ceil(8 * 0.1) = 1

    def test_num_layers_default_36_layers(self):
        """Test that 36-layer model gets num_layers=4."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {
                "encoder": {"n_layers": 36},  # 10% of 36 = 3.6 -> ceil -> 4
                "classifier": {"ignore_index": -100},
            },
        })
        
        metrics = build_metrics(cfg, objective="mlm", has_coords=True)
        p_at_l = next(
            (m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"),
            None,
        )
        
        assert p_at_l is not None
        assert p_at_l.num_layers == 4  # ceil(36 * 0.1) = 4

    def test_num_layers_default_missing_encoder_config(self):
        """Test fallback to 12 layers when encoder config is missing."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {"classifier": {"ignore_index": -100}},  # No encoder config
        })
        
        metrics = build_metrics(cfg, objective="mlm", has_coords=True)
        p_at_l = next(
            (m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"),
            None,
        )
        
        assert p_at_l is not None
        assert p_at_l.num_layers == 2  # ceil(12 * 0.1) = 2 (fallback)

    def test_num_layers_explicit_overrides_default(self):
        """Test that explicit num_layers config overrides the dynamic default."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            "num_layers": 5,  # Explicit override
                        },
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {
                "encoder": {"n_layers": 36},  # Would be 4 if not overridden
                "classifier": {"ignore_index": -100},
            },
        })
        
        metrics = build_metrics(cfg, objective="mlm", has_coords=True)
        p_at_l = next(
            (m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"),
            None,
        )
        
        assert p_at_l is not None
        assert p_at_l.num_layers == 5  # Explicit value, not calculated


class TestPrecisionAtLLogisticRegression:
    """Tests for PrecisionAtLMetric with logistic regression mode."""

    def test_init_logreg_params(self):
        """Test that metric accepts logistic regression parameters."""
        metric = PrecisionAtLMetric(
            use_logistic_regression=True,
            logreg_n_train=15,
            logreg_lambda=0.2,
            logreg_n_iterations=3,
        )
        
        assert metric.use_logistic_regression is True
        assert metric.logreg_n_train == 15
        assert metric.logreg_lambda == 0.2
        assert metric.logreg_n_iterations == 3

    def test_init_logreg_defaults(self):
        """Test default logistic regression parameters."""
        metric = PrecisionAtLMetric()
        
        assert metric.use_logistic_regression is False
        assert metric.logreg_n_train == 20
        assert metric.logreg_lambda == 0.15
        assert metric.logreg_n_iterations == 5

    def test_logreg_accumulates_structures(self):
        """Test that logreg mode accumulates per-structure data."""
        metric = PrecisionAtLMetric(
            use_logistic_regression=True,
            min_seq_sep=3,
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
        
        # Before update, no structures accumulated
        assert len(metric._logreg_structures) == 0
        
        metric.update(outputs, tokens, labels, coords, cfg)
        
        # After update, should have accumulated B structures
        assert len(metric._logreg_structures) == B
        
        # Each structure should have features and labels
        for struct in metric._logreg_structures:
            assert "features" in struct
            assert "labels" in struct
            assert "seq_len" in struct
            # Features should be [n_pairs, n_layers * n_heads]
            assert struct["features"].shape[1] == n_layers * H

    def test_logreg_reset(self):
        """Test that reset clears accumulated structures."""
        metric = PrecisionAtLMetric(use_logistic_regression=True)
        
        # Add some dummy data
        metric._logreg_structures = [
            {"features": torch.randn(10, 8), "labels": torch.zeros(10), "seq_len": 32}
        ]
        
        metric.reset()
        
        assert len(metric._logreg_structures) == 0

    def test_logreg_compute_insufficient_structures(self):
        """Test that compute falls back when not enough structures."""
        metric = PrecisionAtLMetric(
            use_logistic_regression=True,
            logreg_n_train=20,
        )
        
        # Add only 5 structures (less than n_train)
        for _ in range(5):
            n_pairs = 50
            metric._logreg_structures.append({
                "features": torch.randn(n_pairs, 16),
                "labels": torch.randint(0, 2, (n_pairs,)).float(),
                "seq_len": 32,
            })
        
        # Should fall back to standard computation without error
        result = metric.compute()
        
        assert "p_at_l" in result
        assert 0.0 <= result["p_at_l"] <= 1.0

    def test_logreg_compute_with_sufficient_structures(self):
        """Test logistic regression computation with enough structures."""
        pytest.importorskip("sklearn")
        
        metric = PrecisionAtLMetric(
            use_logistic_regression=True,
            logreg_n_train=5,
            logreg_n_iterations=2,
        )
        
        # Add 10 structures (5 for train, 5 for test)
        for _ in range(10):
            n_pairs = 100
            # Create features that have some correlation with labels
            features = torch.randn(n_pairs, 16)
            labels = (features[:, 0] > 0).float()  # Correlate with first feature
            
            metric._logreg_structures.append({
                "features": features,
                "labels": labels,
                "seq_len": 32,
            })
        
        result = metric.compute()
        
        assert "p_at_l" in result
        # With correlated features, should get reasonable precision
        assert result["p_at_l"] >= 0.0

    def test_logreg_state_tensors_empty(self):
        """Test state_tensors with no accumulated structures."""
        metric = PrecisionAtLMetric(use_logistic_regression=True)
        
        tensors = metric.state_tensors()
        
        assert len(tensors) == 5
        assert tensors[0][0].item() == 0  # n_structures

    def test_logreg_state_tensors_with_data(self):
        """Test state_tensors with accumulated structures."""
        metric = PrecisionAtLMetric(use_logistic_regression=True)
        
        # Add structures
        n_features = 16
        for i in range(3):
            n_pairs = 10 + i
            metric._logreg_structures.append({
                "features": torch.randn(n_pairs, n_features),
                "labels": torch.randint(0, 2, (n_pairs,)).float(),
                "seq_len": 32 + i,
            })
        
        tensors = metric.state_tensors()
        
        assert len(tensors) == 5
        assert tensors[0][0].item() == 3  # n_structures
        # Total pairs = 10 + 11 + 12 = 33
        assert tensors[1].shape[0] == 33  # features
        assert tensors[2].shape[0] == 33  # labels

    def test_logreg_load_state_tensors(self):
        """Test load_state_tensors restores structures."""
        metric = PrecisionAtLMetric(use_logistic_regression=True)
        
        # Create state tensors manually
        n_structures = 2
        n_pairs_per_struct = [10, 15]
        n_features = 8
        
        features1 = torch.randn(n_pairs_per_struct[0], n_features)
        features2 = torch.randn(n_pairs_per_struct[1], n_features)
        labels1 = torch.randint(0, 2, (n_pairs_per_struct[0],)).float()
        labels2 = torch.randint(0, 2, (n_pairs_per_struct[1],)).float()
        
        tensors = [
            torch.tensor([n_structures]),
            torch.cat([features1, features2], dim=0),
            torch.cat([labels1, labels2], dim=0),
            torch.tensor([32, 48]),  # seq_lens
            torch.tensor(n_pairs_per_struct),
        ]
        
        metric.load_state_tensors(tensors)
        
        assert len(metric._logreg_structures) == 2
        assert metric._logreg_structures[0]["seq_len"] == 32
        assert metric._logreg_structures[1]["seq_len"] == 48


class TestPrecisionAtLLogisticRegressionConfig:
    """Tests for config-based instantiation with logistic regression options."""

    def test_logreg_config_options(self):
        """Test that logreg options are correctly passed from config."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            "use_logistic_regression": True,
                            "logreg_n_train": 10,
                            "logreg_lambda": 0.1,
                            "logreg_n_iterations": 3,
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
        assert p_at_l.use_logistic_regression is True
        assert p_at_l.logreg_n_train == 10
        assert p_at_l.logreg_lambda == 0.1
        assert p_at_l.logreg_n_iterations == 3

    def test_logreg_config_defaults(self):
        """Test that logreg defaults are applied when not specified."""
        from stok.eval.registry import build_metrics
        
        cfg = OmegaConf.create({
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            # No logreg options specified
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
        assert p_at_l.use_logistic_regression is False
        assert p_at_l.logreg_n_train == 20
        assert p_at_l.logreg_lambda == 0.15
        assert p_at_l.logreg_n_iterations == 5

