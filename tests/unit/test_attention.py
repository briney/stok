"""Tests for MultiheadAttention with need_weights option.

Verifies that the optimized SDPA path and manual attention implementation
produce equivalent results, and that attention weights are correctly returned.
"""

import pytest
import torch

from stok.models.attention import MultiheadAttention
from stok.models.rope import RotaryEmbedding


@pytest.fixture
def attention_module():
    """Create a MultiheadAttention module for testing."""
    torch.manual_seed(42)
    d_model = 64
    n_heads = 4
    dropout = 0.0  # No dropout for deterministic comparison
    rope = RotaryEmbedding(base=10000.0)
    attn = MultiheadAttention(d_model, n_heads, dropout, rope)
    attn.eval()  # Ensure eval mode for deterministic behavior
    return attn


@pytest.fixture
def sample_input():
    """Create sample input tensors."""
    torch.manual_seed(42)
    B, L, d_model = 2, 16, 64
    x = torch.randn(B, L, d_model)
    return x


class TestNeedWeightsEquivalence:
    """Test that SDPA and manual attention produce equivalent outputs."""

    def test_outputs_equivalent_no_mask(self, attention_module, sample_input):
        """Both methods produce same output without any masks."""
        attn = attention_module
        x = sample_input

        # SDPA path (need_weights=False)
        out_sdpa = attn(x, need_weights=False)

        # Manual path (need_weights=True)
        out_manual, weights = attn(x, need_weights=True)

        assert out_sdpa.shape == out_manual.shape
        assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4), (
            f"Max diff: {(out_sdpa - out_manual).abs().max().item()}"
        )

    def test_outputs_equivalent_with_key_padding_mask(
        self, attention_module, sample_input
    ):
        """Both methods produce same output with key padding mask."""
        attn = attention_module
        x = sample_input
        B, L, _ = x.shape

        # Create a padding mask (last 4 positions are padding)
        key_padding_mask = torch.zeros(B, L, dtype=torch.bool)
        key_padding_mask[:, -4:] = True

        out_sdpa = attn(x, key_padding_mask=key_padding_mask, need_weights=False)
        out_manual, weights = attn(
            x, key_padding_mask=key_padding_mask, need_weights=True
        )

        assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4), (
            f"Max diff: {(out_sdpa - out_manual).abs().max().item()}"
        )

    def test_outputs_equivalent_with_additive_attn_mask(
        self, attention_module, sample_input
    ):
        """Both methods produce same output with additive attention mask."""
        attn = attention_module
        x = sample_input
        B, L, _ = x.shape
        n_heads = attn.n_heads

        # Create an additive mask (simulate some blocked attention patterns)
        attn_mask = torch.zeros(B, 1, L, L)
        # Block attention to some positions
        attn_mask[:, :, :, -2:] = float("-inf")

        out_sdpa = attn(x, attn_mask=attn_mask, need_weights=False)
        out_manual, weights = attn(x, attn_mask=attn_mask, need_weights=True)

        assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4), (
            f"Max diff: {(out_sdpa - out_manual).abs().max().item()}"
        )

    def test_outputs_equivalent_with_boolean_attn_mask(
        self, attention_module, sample_input
    ):
        """Both methods produce same output with boolean attention mask."""
        attn = attention_module
        x = sample_input
        B, L, _ = x.shape

        # Create a boolean mask (True = masked/blocked)
        attn_mask = torch.zeros(B, 1, L, L, dtype=torch.bool)
        attn_mask[:, :, :, -3:] = True

        out_sdpa = attn(x, attn_mask=attn_mask, need_weights=False)
        out_manual, weights = attn(x, attn_mask=attn_mask, need_weights=True)

        assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4), (
            f"Max diff: {(out_sdpa - out_manual).abs().max().item()}"
        )

    def test_outputs_equivalent_with_combined_masks(
        self, attention_module, sample_input
    ):
        """Both methods produce same output with both padding and attn masks."""
        attn = attention_module
        x = sample_input
        B, L, _ = x.shape

        key_padding_mask = torch.zeros(B, L, dtype=torch.bool)
        key_padding_mask[:, -3:] = True

        attn_mask = torch.zeros(B, 1, L, L, dtype=torch.bool)
        attn_mask[:, :, :, 0] = True  # Block first position

        out_sdpa = attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        out_manual, weights = attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
        )

        assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4), (
            f"Max diff: {(out_sdpa - out_manual).abs().max().item()}"
        )


class TestAttentionWeightsProperties:
    """Test properties of returned attention weights."""

    def test_weights_shape(self, attention_module, sample_input):
        """Attention weights have correct shape [B, H, L, S]."""
        attn = attention_module
        x = sample_input
        B, L, _ = x.shape

        _, weights = attn(x, need_weights=True)

        assert weights.shape == (B, attn.n_heads, L, L)

    def test_weights_sum_to_one(self, attention_module, sample_input):
        """Attention weights sum to 1 along the key dimension."""
        attn = attention_module
        x = sample_input

        _, weights = attn(x, need_weights=True)

        # Sum along the last dimension (keys)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_weights_nonnegative(self, attention_module, sample_input):
        """Attention weights are non-negative (post-softmax)."""
        attn = attention_module
        x = sample_input

        _, weights = attn(x, need_weights=True)

        assert (weights >= 0).all()

    def test_masked_positions_have_zero_weight(self, attention_module, sample_input):
        """Masked key positions receive zero attention weight."""
        attn = attention_module
        x = sample_input
        B, L, _ = x.shape

        # Mask the last 4 positions
        key_padding_mask = torch.zeros(B, L, dtype=torch.bool)
        key_padding_mask[:, -4:] = True

        _, weights = attn(x, key_padding_mask=key_padding_mask, need_weights=True)

        # Weights to masked positions should be ~0
        masked_weights = weights[:, :, :, -4:]
        assert torch.allclose(
            masked_weights, torch.zeros_like(masked_weights), atol=1e-6
        )

    def test_weights_dtype_matches_input(self, attention_module):
        """Attention weights dtype matches input dtype."""
        attn = attention_module
        x = torch.randn(2, 8, 64, dtype=torch.float32)

        _, weights = attn(x, need_weights=True)
        assert weights.dtype == x.dtype


class TestReturnTypes:
    """Test return type behavior based on need_weights parameter."""

    def test_need_weights_false_returns_tensor(self, attention_module, sample_input):
        """need_weights=False returns a single tensor."""
        attn = attention_module
        x = sample_input

        result = attn(x, need_weights=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_need_weights_true_returns_tuple(self, attention_module, sample_input):
        """need_weights=True returns a tuple of (output, weights)."""
        attn = attention_module
        x = sample_input

        result = attn(x, need_weights=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        output, weights = result
        assert isinstance(output, torch.Tensor)
        assert isinstance(weights, torch.Tensor)

    def test_default_need_weights_is_false(self, attention_module, sample_input):
        """Default behavior (no need_weights arg) returns tensor."""
        attn = attention_module
        x = sample_input

        result = attn(x)

        assert isinstance(result, torch.Tensor)


class TestGradientFlow:
    """Test that gradients flow correctly through both paths."""

    def test_gradients_flow_sdpa_path(self, attention_module, sample_input):
        """Gradients flow through SDPA path."""
        attn = attention_module
        attn.train()
        x = sample_input.clone().requires_grad_(True)

        out = attn(x, need_weights=False)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_gradients_flow_manual_path(self, attention_module, sample_input):
        """Gradients flow through manual attention path."""
        attn = attention_module
        attn.train()
        x = sample_input.clone().requires_grad_(True)

        out, weights = attn(x, need_weights=True)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_gradients_equivalent_between_paths(self, attention_module, sample_input):
        """Both paths produce equivalent gradients (with dropout=0)."""
        attn = attention_module
        attn.train()

        # SDPA path
        x1 = sample_input.clone().requires_grad_(True)
        out1 = attn(x1, need_weights=False)
        out1.sum().backward()

        # Manual path
        x2 = sample_input.clone().requires_grad_(True)
        out2, _ = attn(x2, need_weights=True)
        out2.sum().backward()

        # Gradients should be equivalent
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-4), (
            f"Max grad diff: {(x1.grad - x2.grad).abs().max().item()}"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_sequence(self):
        """Handle single-token sequences."""
        torch.manual_seed(42)
        d_model = 64
        n_heads = 4
        rope = RotaryEmbedding(base=10000.0)
        attn = MultiheadAttention(d_model, n_heads, dropout=0.0, rope=rope)
        attn.eval()

        x = torch.randn(2, 1, d_model)

        out_sdpa = attn(x, need_weights=False)
        out_manual, weights = attn(x, need_weights=True)

        assert out_sdpa.shape == (2, 1, d_model)
        assert weights.shape == (2, n_heads, 1, 1)
        assert torch.allclose(out_sdpa, out_manual, atol=1e-5)

    def test_batch_size_one(self):
        """Handle batch size of 1."""
        torch.manual_seed(42)
        d_model = 64
        n_heads = 4
        rope = RotaryEmbedding(base=10000.0)
        attn = MultiheadAttention(d_model, n_heads, dropout=0.0, rope=rope)
        attn.eval()

        x = torch.randn(1, 8, d_model)

        out_sdpa = attn(x, need_weights=False)
        out_manual, weights = attn(x, need_weights=True)

        assert out_sdpa.shape == (1, 8, d_model)
        assert torch.allclose(out_sdpa, out_manual, atol=1e-5)

    def test_different_dtypes(self):
        """Both paths work with different dtypes."""
        torch.manual_seed(42)
        d_model = 64
        n_heads = 4
        rope = RotaryEmbedding(base=10000.0)
        attn = MultiheadAttention(d_model, n_heads, dropout=0.0, rope=rope)
        attn.eval()

        for dtype in [torch.float32, torch.float64]:
            attn = attn.to(dtype)
            x = torch.randn(2, 8, d_model, dtype=dtype)

            out_sdpa = attn(x, need_weights=False)
            out_manual, weights = attn(x, need_weights=True)

            assert out_sdpa.dtype == dtype
            assert out_manual.dtype == dtype
            assert weights.dtype == dtype
            assert torch.allclose(out_sdpa, out_manual, atol=1e-4, rtol=1e-3)

    def test_all_positions_masked_except_one(self, attention_module):
        """Handle case where all but one position is masked."""
        attn = attention_module
        B, L, d_model = 2, 8, 64
        x = torch.randn(B, L, d_model)

        # Mask all positions except the first
        key_padding_mask = torch.ones(B, L, dtype=torch.bool)
        key_padding_mask[:, 0] = False

        out_sdpa = attn(x, key_padding_mask=key_padding_mask, need_weights=False)
        out_manual, weights = attn(
            x, key_padding_mask=key_padding_mask, need_weights=True
        )

        assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4)

        # All attention should be on the first position
        expected_weights = torch.zeros_like(weights)
        expected_weights[:, :, :, 0] = 1.0
        assert torch.allclose(weights, expected_weights, atol=1e-5)

