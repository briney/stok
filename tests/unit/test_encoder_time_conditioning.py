"""Tests for time conditioning in EncoderBlock and Encoder.

Verifies that adaLN time conditioning works correctly and that backward
compatibility is maintained when time_conditioning=None.
"""

import pytest
import torch

from stok.models.blocks import EncoderBlock, RMSNorm
from stok.models.encoder import Encoder
from stok.models.rope import RotaryEmbedding
from stok.models.time_embed import AdaptiveLayerNorm


D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
TIME_EMBED_DIM = 64


@pytest.fixture
def rope():
    return RotaryEmbedding()


@pytest.fixture
def sample_input():
    torch.manual_seed(42)
    B, L = 2, 16
    x = torch.randn(B, L, D_MODEL)
    t_embed = torch.randn(B, TIME_EMBED_DIM)
    return x, t_embed


class TestEncoderBlockAdaLN:
    """Tests for EncoderBlock with time_conditioning='adaln'."""

    @pytest.fixture
    def block(self, rope):
        torch.manual_seed(42)
        return EncoderBlock(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            attn_dropout=0.0,
            resid_dropout=0.0,
            rope=rope,
            time_conditioning="adaln",
            time_embed_dim=TIME_EMBED_DIM,
        )

    def test_norms_are_adaln(self, block):
        assert isinstance(block.norm1, AdaptiveLayerNorm)
        assert isinstance(block.norm2, AdaptiveLayerNorm)

    def test_output_shape(self, block, sample_input):
        x, t_embed = sample_input
        out = block(x, t_embed=t_embed)
        assert out.shape == x.shape

    def test_output_shape_with_attention(self, block, sample_input):
        x, t_embed = sample_input
        out, attn = block(x, t_embed=t_embed, output_attentions=True)
        assert out.shape == x.shape
        assert attn.shape == (2, N_HEADS, 16, 16)

    def test_gradients_flow_through_adaln(self, block, sample_input):
        x, t_embed = sample_input
        x = x.requires_grad_(True)
        t_embed = t_embed.requires_grad_(True)
        out = block(x, t_embed=t_embed)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert t_embed.grad is not None
        # adaLN linear layers should also have gradients
        assert block.norm1.linear.weight.grad is not None
        assert block.norm2.linear.weight.grad is not None


class TestEncoderBlockBackwardCompat:
    """Tests that time_conditioning=None preserves original behavior."""

    @pytest.fixture
    def block(self, rope):
        torch.manual_seed(42)
        return EncoderBlock(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            attn_dropout=0.0,
            resid_dropout=0.0,
            rope=rope,
        )

    def test_norms_are_standard(self, block):
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)

    def test_forward_without_t_embed(self, block, sample_input):
        x, _ = sample_input
        out = block(x)
        assert out.shape == x.shape

    def test_forward_with_none_t_embed(self, block, sample_input):
        x, _ = sample_input
        out = block(x, t_embed=None)
        assert out.shape == x.shape

    def test_rmsnorm_without_time_conditioning(self, rope):
        block = EncoderBlock(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            attn_dropout=0.0,
            resid_dropout=0.0,
            rope=rope,
            norm_type="rmsnorm",
        )
        assert isinstance(block.norm1, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)


class TestEncoderAdaLN:
    """Tests for Encoder with time_conditioning='adaln'."""

    @pytest.fixture
    def encoder(self):
        torch.manual_seed(42)
        return Encoder(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
            attn_dropout=0.0,
            time_conditioning="adaln",
            time_embed_dim=TIME_EMBED_DIM,
        )

    def test_output_shape(self, encoder, sample_input):
        x, t_embed = sample_input
        out = encoder(x, t_embed=t_embed)
        assert out.shape == x.shape

    def test_final_norm_is_standard(self, encoder):
        """final_norm should remain standard LayerNorm, not adaptive."""
        assert isinstance(encoder.final_norm, torch.nn.LayerNorm)
        assert not isinstance(encoder.final_norm, AdaptiveLayerNorm)

    def test_blocks_are_adaln(self, encoder):
        for layer in encoder.layers:
            assert isinstance(layer.norm1, AdaptiveLayerNorm)
            assert isinstance(layer.norm2, AdaptiveLayerNorm)

    def test_gradients_flow(self, encoder, sample_input):
        x, t_embed = sample_input
        x = x.requires_grad_(True)
        t_embed = t_embed.requires_grad_(True)
        out = encoder(x, t_embed=t_embed)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert t_embed.grad is not None

    def test_with_output_attentions(self, encoder, sample_input):
        x, t_embed = sample_input
        out, attns = encoder(x, t_embed=t_embed, output_attentions=True)
        assert out.shape == x.shape
        assert len(attns) == N_LAYERS

    def test_with_output_hidden_states(self, encoder, sample_input):
        x, t_embed = sample_input
        out, hidden = encoder(x, t_embed=t_embed, output_hidden_states=True)
        assert out.shape == x.shape
        assert len(hidden) == N_LAYERS + 1

    def test_with_padding_mask(self, encoder, sample_input):
        x, t_embed = sample_input
        pad_mask = torch.zeros(2, 16, dtype=torch.bool)
        pad_mask[:, -4:] = True
        out = encoder(x, key_padding_mask=pad_mask, t_embed=t_embed)
        assert out.shape == x.shape


class TestEncoderBackwardCompat:
    """Tests that Encoder without time_conditioning works as before."""

    @pytest.fixture
    def encoder(self):
        torch.manual_seed(42)
        return Encoder(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
            attn_dropout=0.0,
        )

    def test_forward_without_t_embed(self, encoder, sample_input):
        x, _ = sample_input
        out = encoder(x)
        assert out.shape == x.shape

    def test_forward_with_none_t_embed(self, encoder, sample_input):
        x, _ = sample_input
        out = encoder(x, t_embed=None)
        assert out.shape == x.shape

    def test_blocks_are_standard(self, encoder):
        for layer in encoder.layers:
            assert isinstance(layer.norm1, torch.nn.LayerNorm)

    def test_different_time_embed_dim(self):
        """time_embed_dim can differ from d_model."""
        encoder = Encoder(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
            attn_dropout=0.0,
            time_conditioning="adaln",
            time_embed_dim=128,
        )
        x = torch.randn(2, 16, D_MODEL)
        t_embed = torch.randn(2, 128)
        out = encoder(x, t_embed=t_embed)
        assert out.shape == x.shape
