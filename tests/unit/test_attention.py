"""Tests for MultiheadAttention with need_weights option.

Verifies that the optimized SDPA path and manual attention implementation
produce equivalent results, and that attention weights are correctly returned.
Also tests propagation of output_attentions through EncoderBlock, Encoder,
and STokModel.
"""

import pytest
import torch

from stok.models.attention import MultiheadAttention
from stok.models.blocks import EncoderBlock
from stok.models.encoder import Encoder
from stok.models.rope import RotaryEmbedding
from stok.models.stok import STokModel


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


# ==============================================================================
# Tests for output_attentions propagation through model layers
# ==============================================================================


class TestEncoderBlockOutputAttentions:
    """Test output_attentions propagation through EncoderBlock."""

    @pytest.fixture
    def encoder_block(self):
        """Create an EncoderBlock for testing."""
        torch.manual_seed(42)
        d_model = 64
        n_heads = 4
        rope = RotaryEmbedding(base=10000.0)
        block = EncoderBlock(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=0.0,
            resid_dropout=0.0,
            rope=rope,
        )
        block.eval()
        return block

    def test_output_attentions_false_returns_tensor(self, encoder_block):
        """output_attentions=False returns only output tensor."""
        x = torch.randn(2, 16, 64)
        result = encoder_block(x, output_attentions=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_output_attentions_true_returns_tuple(self, encoder_block):
        """output_attentions=True returns tuple of (output, weights)."""
        x = torch.randn(2, 16, 64)
        result = encoder_block(x, output_attentions=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        output, weights = result
        assert output.shape == x.shape
        assert weights.shape == (2, 4, 16, 16)  # [B, H, L, L]

    def test_outputs_equivalent_with_and_without_attentions(self, encoder_block):
        """Output tensor is the same regardless of output_attentions."""
        x = torch.randn(2, 16, 64)

        out_no_attn = encoder_block(x, output_attentions=False)
        out_with_attn, _ = encoder_block(x, output_attentions=True)

        assert torch.allclose(out_no_attn, out_with_attn, atol=1e-5)

    def test_default_output_attentions_is_false(self, encoder_block):
        """Default behavior returns tensor (output_attentions=False)."""
        x = torch.randn(2, 16, 64)
        result = encoder_block(x)

        assert isinstance(result, torch.Tensor)


class TestEncoderOutputAttentions:
    """Test output_attentions propagation through Encoder stack."""

    @pytest.fixture
    def encoder(self):
        """Create an Encoder for testing."""
        torch.manual_seed(42)
        encoder = Encoder(
            d_model=64,
            n_heads=4,
            n_layers=3,
            dropout=0.0,
            attn_dropout=0.0,
        )
        encoder.eval()
        return encoder

    def test_output_attentions_false_returns_tensor(self, encoder):
        """output_attentions=False returns only output tensor."""
        x = torch.randn(2, 16, 64)
        result = encoder(x, output_attentions=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_output_attentions_true_returns_tuple(self, encoder):
        """output_attentions=True returns tuple of (output, all_attentions)."""
        x = torch.randn(2, 16, 64)
        result = encoder(x, output_attentions=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        output, all_attentions = result
        assert output.shape == x.shape
        assert isinstance(all_attentions, tuple)
        assert len(all_attentions) == 3  # n_layers

    def test_attention_shapes_per_layer(self, encoder):
        """Each layer's attention weights have correct shape."""
        B, L, d_model = 2, 16, 64
        n_heads = 4
        x = torch.randn(B, L, d_model)

        _, all_attentions = encoder(x, output_attentions=True)

        for layer_idx, attn_weights in enumerate(all_attentions):
            assert attn_weights.shape == (B, n_heads, L, L), (
                f"Layer {layer_idx} attention shape mismatch"
            )

    def test_outputs_equivalent_with_and_without_attentions(self, encoder):
        """Output tensor is the same regardless of output_attentions."""
        x = torch.randn(2, 16, 64)

        out_no_attn = encoder(x, output_attentions=False)
        out_with_attn, _ = encoder(x, output_attentions=True)

        assert torch.allclose(out_no_attn, out_with_attn, atol=1e-5)

    def test_default_output_attentions_is_false(self, encoder):
        """Default behavior returns tensor (output_attentions=False)."""
        x = torch.randn(2, 16, 64)
        result = encoder(x)

        assert isinstance(result, torch.Tensor)

    def test_attention_weights_properties(self, encoder):
        """Attention weights from encoder have valid properties."""
        x = torch.randn(2, 16, 64)
        _, all_attentions = encoder(x, output_attentions=True)

        for layer_idx, attn_weights in enumerate(all_attentions):
            # Non-negative
            assert (attn_weights >= 0).all(), f"Layer {layer_idx} has negative weights"
            # Sum to 1
            sums = attn_weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
                f"Layer {layer_idx} weights don't sum to 1"
            )


class TestSTokModelOutputAttentions:
    """Test output_attentions propagation through STokModel."""

    @pytest.fixture
    def stok_model(self):
        """Create a STokModel for testing."""
        torch.manual_seed(42)
        # Create a small codebook for testing
        codebook = torch.randn(32, 16)
        model = STokModel(
            vocab_size=30,
            pad_id=0,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.0,
            dropout=0.0,
            attn_dropout=0.0,
            codebook=codebook,
        )
        model.eval()
        return model

    def test_output_attentions_false_no_attentions_key(self, stok_model):
        """output_attentions=False does not include attentions in output."""
        tokens = torch.randint(1, 30, (2, 16))
        result = stok_model(tokens, output_attentions=False)

        assert "attentions" not in result
        assert "logits" in result

    def test_output_attentions_true_includes_attentions(self, stok_model):
        """output_attentions=True includes attentions in output dict."""
        tokens = torch.randint(1, 30, (2, 16))
        result = stok_model(tokens, output_attentions=True)

        assert "attentions" in result
        assert "logits" in result
        assert isinstance(result["attentions"], tuple)
        assert len(result["attentions"]) == 2  # n_layers

    def test_attention_shapes(self, stok_model):
        """Attention weights have correct shape [B, H, L, L]."""
        B, L = 2, 16
        n_heads = 4
        tokens = torch.randint(1, 30, (B, L))

        result = stok_model(tokens, output_attentions=True)

        for layer_idx, attn_weights in enumerate(result["attentions"]):
            assert attn_weights.shape == (B, n_heads, L, L), (
                f"Layer {layer_idx} attention shape mismatch"
            )

    def test_logits_equivalent_with_and_without_attentions(self, stok_model):
        """Logits are the same regardless of output_attentions."""
        tokens = torch.randint(1, 30, (2, 16))

        result_no_attn = stok_model(tokens, output_attentions=False)
        result_with_attn = stok_model(tokens, output_attentions=True)

        assert torch.allclose(
            result_no_attn["logits"], result_with_attn["logits"], atol=1e-5
        )

    def test_default_output_attentions_is_false(self, stok_model):
        """Default behavior does not include attentions."""
        tokens = torch.randint(1, 30, (2, 16))
        result = stok_model(tokens)

        assert "attentions" not in result

    def test_output_attentions_with_labels(self, stok_model):
        """output_attentions works correctly with loss computation."""
        tokens = torch.randint(1, 30, (2, 16))
        labels = torch.randint(0, 32, (2, 16))

        result = stok_model(tokens, labels=labels, output_attentions=True)

        assert "attentions" in result
        assert result["loss"] is not None
        assert len(result["attentions"]) == 2

    def test_output_attentions_with_padding_mask(self, stok_model):
        """Attention weights respect padding mask."""
        B, L = 2, 16
        tokens = torch.randint(1, 30, (B, L))

        # Create padding mask (last 4 positions are padding)
        key_padding_mask = torch.zeros(B, L, dtype=torch.bool)
        key_padding_mask[:, -4:] = True

        result = stok_model(
            tokens, key_padding_mask=key_padding_mask, output_attentions=True
        )

        # Check that attention to padded positions is ~0
        for attn_weights in result["attentions"]:
            padded_attention = attn_weights[:, :, :, -4:]
            assert torch.allclose(
                padded_attention, torch.zeros_like(padded_attention), atol=1e-6
            )


# ==============================================================================
# Tests for output_hidden_states functionality
# ==============================================================================


class TestEncoderOutputHiddenStates:
    """Test output_hidden_states in Encoder."""

    @pytest.fixture
    def encoder(self):
        """Create an Encoder for testing."""
        torch.manual_seed(42)
        encoder = Encoder(
            d_model=64,
            n_heads=4,
            n_layers=3,
            dropout=0.0,
            attn_dropout=0.0,
        )
        encoder.eval()
        return encoder

    def test_output_hidden_states_false_returns_tensor(self, encoder):
        """output_hidden_states=False returns only output tensor."""
        x = torch.randn(2, 16, 64)
        result = encoder(x, output_hidden_states=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_output_hidden_states_true_returns_tuple(self, encoder):
        """output_hidden_states=True returns tuple of (output, all_hidden_states)."""
        x = torch.randn(2, 16, 64)
        result = encoder(x, output_hidden_states=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        output, all_hidden_states = result
        assert output.shape == x.shape
        assert isinstance(all_hidden_states, tuple)

    def test_hidden_states_count(self, encoder):
        """Returns n_layers + 1 hidden states (including initial embeddings)."""
        x = torch.randn(2, 16, 64)
        _, all_hidden_states = encoder(x, output_hidden_states=True)

        # n_layers=3, so we expect 4 hidden states (initial + 3 layers)
        assert len(all_hidden_states) == 4

    def test_hidden_states_shapes(self, encoder):
        """Each hidden state has correct shape [B, L, d_model]."""
        B, L, d_model = 2, 16, 64
        x = torch.randn(B, L, d_model)
        _, all_hidden_states = encoder(x, output_hidden_states=True)

        for idx, hidden_state in enumerate(all_hidden_states):
            assert hidden_state.shape == (B, L, d_model), (
                f"Hidden state {idx} has wrong shape"
            )

    def test_first_hidden_state_equals_input(self, encoder):
        """First hidden state should equal the input embeddings."""
        x = torch.randn(2, 16, 64)
        _, all_hidden_states = encoder(x, output_hidden_states=True)

        assert torch.equal(all_hidden_states[0], x)

    def test_outputs_equivalent_with_and_without_hidden_states(self, encoder):
        """Output tensor is the same regardless of output_hidden_states."""
        x = torch.randn(2, 16, 64)

        out_without = encoder(x, output_hidden_states=False)
        out_with, _ = encoder(x, output_hidden_states=True)

        assert torch.allclose(out_without, out_with, atol=1e-5)

    def test_default_output_hidden_states_is_false(self, encoder):
        """Default behavior returns tensor only."""
        x = torch.randn(2, 16, 64)
        result = encoder(x)

        assert isinstance(result, torch.Tensor)

    def test_both_attentions_and_hidden_states(self, encoder):
        """Both output_attentions and output_hidden_states can be True."""
        x = torch.randn(2, 16, 64)
        result = encoder(x, output_attentions=True, output_hidden_states=True)

        assert isinstance(result, tuple)
        assert len(result) == 3

        output, all_attentions, all_hidden_states = result

        # Verify output
        assert output.shape == x.shape

        # Verify attentions (n_layers)
        assert isinstance(all_attentions, tuple)
        assert len(all_attentions) == 3

        # Verify hidden states (n_layers + 1)
        assert isinstance(all_hidden_states, tuple)
        assert len(all_hidden_states) == 4

    def test_outputs_equivalent_with_all_combinations(self, encoder):
        """Output is equivalent across all flag combinations."""
        x = torch.randn(2, 16, 64)

        out_none = encoder(x)
        out_attn, _ = encoder(x, output_attentions=True)
        out_hidden, _ = encoder(x, output_hidden_states=True)
        out_both, _, _ = encoder(x, output_attentions=True, output_hidden_states=True)

        assert torch.allclose(out_none, out_attn, atol=1e-5)
        assert torch.allclose(out_none, out_hidden, atol=1e-5)
        assert torch.allclose(out_none, out_both, atol=1e-5)


class TestSTokModelOutputHiddenStates:
    """Test output_hidden_states propagation through STokModel."""

    @pytest.fixture
    def stok_model(self):
        """Create a STokModel for testing."""
        torch.manual_seed(42)
        codebook = torch.randn(32, 16)
        model = STokModel(
            vocab_size=30,
            pad_id=0,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.0,
            dropout=0.0,
            attn_dropout=0.0,
            codebook=codebook,
        )
        model.eval()
        return model

    def test_output_hidden_states_false_no_key(self, stok_model):
        """output_hidden_states=False does not include hidden_states in output."""
        tokens = torch.randint(1, 30, (2, 16))
        result = stok_model(tokens, output_hidden_states=False)

        assert "hidden_states" not in result
        assert "logits" in result

    def test_output_hidden_states_true_includes_key(self, stok_model):
        """output_hidden_states=True includes hidden_states in output dict."""
        tokens = torch.randint(1, 30, (2, 16))
        result = stok_model(tokens, output_hidden_states=True)

        assert "hidden_states" in result
        assert "logits" in result
        assert isinstance(result["hidden_states"], tuple)

    def test_hidden_states_count(self, stok_model):
        """Returns n_layers + 1 hidden states."""
        tokens = torch.randint(1, 30, (2, 16))
        result = stok_model(tokens, output_hidden_states=True)

        # n_layers=2, so we expect 3 hidden states
        assert len(result["hidden_states"]) == 3

    def test_hidden_states_shapes(self, stok_model):
        """Each hidden state has correct shape [B, L, d_model]."""
        B, L = 2, 16
        d_model = 64
        tokens = torch.randint(1, 30, (B, L))

        result = stok_model(tokens, output_hidden_states=True)

        for idx, hidden_state in enumerate(result["hidden_states"]):
            assert hidden_state.shape == (B, L, d_model), (
                f"Hidden state {idx} has wrong shape"
            )

    def test_logits_equivalent_with_and_without_hidden_states(self, stok_model):
        """Logits are the same regardless of output_hidden_states."""
        tokens = torch.randint(1, 30, (2, 16))

        result_without = stok_model(tokens, output_hidden_states=False)
        result_with = stok_model(tokens, output_hidden_states=True)

        assert torch.allclose(
            result_without["logits"], result_with["logits"], atol=1e-5
        )

    def test_default_output_hidden_states_is_false(self, stok_model):
        """Default behavior does not include hidden_states."""
        tokens = torch.randint(1, 30, (2, 16))
        result = stok_model(tokens)

        assert "hidden_states" not in result

    def test_output_hidden_states_with_labels(self, stok_model):
        """output_hidden_states works correctly with loss computation."""
        tokens = torch.randint(1, 30, (2, 16))
        labels = torch.randint(0, 32, (2, 16))

        result = stok_model(tokens, labels=labels, output_hidden_states=True)

        assert "hidden_states" in result
        assert result["loss"] is not None
        assert len(result["hidden_states"]) == 3

    def test_both_attentions_and_hidden_states(self, stok_model):
        """Both output_attentions and output_hidden_states can be True."""
        tokens = torch.randint(1, 30, (2, 16))
        result = stok_model(
            tokens, output_attentions=True, output_hidden_states=True
        )

        assert "attentions" in result
        assert "hidden_states" in result
        assert "logits" in result

        # n_layers=2
        assert len(result["attentions"]) == 2
        # n_layers + 1
        assert len(result["hidden_states"]) == 3

    def test_logits_equivalent_all_combinations(self, stok_model):
        """Logits are equivalent across all flag combinations."""
        tokens = torch.randint(1, 30, (2, 16))

        result_none = stok_model(tokens)
        result_attn = stok_model(tokens, output_attentions=True)
        result_hidden = stok_model(tokens, output_hidden_states=True)
        result_both = stok_model(
            tokens, output_attentions=True, output_hidden_states=True
        )

        assert torch.allclose(result_none["logits"], result_attn["logits"], atol=1e-5)
        assert torch.allclose(result_none["logits"], result_hidden["logits"], atol=1e-5)
        assert torch.allclose(result_none["logits"], result_both["logits"], atol=1e-5)

