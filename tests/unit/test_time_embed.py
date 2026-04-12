"""Tests for SinusoidalTimeEmbedding and AdaptiveLayerNorm."""

import pytest
import torch

from stok.models.time_embed import AdaptiveLayerNorm, SinusoidalTimeEmbedding


class TestSinusoidalTimeEmbedding:
    """Tests for the sinusoidal time embedding module."""

    @pytest.fixture
    def embed(self):
        torch.manual_seed(42)
        return SinusoidalTimeEmbedding(d_model=128)

    def test_output_shape(self, embed):
        t = torch.rand(4)
        out = embed(t)
        assert out.shape == (4, 128)

    def test_output_shape_single(self, embed):
        t = torch.tensor([0.5])
        out = embed(t)
        assert out.shape == (1, 128)

    def test_different_times_different_embeddings(self, embed):
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embed(t)
        # Embeddings for different times should differ
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[1], out[2])

    def test_gradients_flow(self, embed):
        t = torch.rand(4, requires_grad=False)
        out = embed(t)
        loss = out.sum()
        loss.backward()
        # Check MLP parameters have gradients
        for p in embed.mlp.parameters():
            assert p.grad is not None

    def test_odd_d_model(self):
        torch.manual_seed(42)
        embed = SinusoidalTimeEmbedding(d_model=65)
        t = torch.rand(2)
        out = embed(t)
        assert out.shape == (2, 65)

    def test_deterministic(self, embed):
        t = torch.tensor([0.3, 0.7])
        out1 = embed(t)
        out2 = embed(t)
        torch.testing.assert_close(out1, out2)


class TestAdaptiveLayerNorm:
    """Tests for AdaptiveLayerNorm."""

    @pytest.fixture
    def adaln(self):
        torch.manual_seed(42)
        return AdaptiveLayerNorm(d_model=64, time_embed_dim=64)

    def test_output_shape(self, adaln):
        x = torch.randn(2, 10, 64)
        t_embed = torch.randn(2, 64)
        out = adaln(x, t_embed)
        assert out.shape == (2, 10, 64)

    def test_reduces_to_layernorm_with_zero_params(self):
        """When scale=0 and shift=0, adaLN should equal standard LayerNorm."""
        d_model = 32
        adaln = AdaptiveLayerNorm(d_model=d_model, time_embed_dim=d_model)
        # Zero the linear layer so scale=0, shift=0
        # Then output = norm(x) * (1 + 0) + 0 = norm(x)
        torch.nn.init.zeros_(adaln.linear.weight)
        torch.nn.init.zeros_(adaln.linear.bias)

        x = torch.randn(2, 5, d_model)
        t_embed = torch.randn(2, d_model)
        out = adaln(x, t_embed)

        ln = torch.nn.LayerNorm(d_model, elementwise_affine=False)
        expected = ln(x)
        torch.testing.assert_close(out, expected)

    def test_gradients_flow(self, adaln):
        x = torch.randn(2, 10, 64, requires_grad=True)
        t_embed = torch.randn(2, 64, requires_grad=True)
        out = adaln(x, t_embed)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert t_embed.grad is not None
        assert adaln.linear.weight.grad is not None

    def test_different_t_embed_different_output(self, adaln):
        x = torch.randn(2, 10, 64)
        t1 = torch.randn(2, 64)
        t2 = torch.randn(2, 64)
        out1 = adaln(x, t1)
        out2 = adaln(x, t2)
        assert not torch.allclose(out1, out2)

    def test_time_embed_dim_differs_from_d_model(self):
        """time_embed_dim can differ from d_model."""
        adaln = AdaptiveLayerNorm(d_model=64, time_embed_dim=128)
        x = torch.randn(2, 10, 64)
        t_embed = torch.randn(2, 128)
        out = adaln(x, t_embed)
        assert out.shape == (2, 10, 64)
