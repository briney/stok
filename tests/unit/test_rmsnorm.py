import torch

from stok.models.blocks import RMSNorm


def _reference_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * inv_rms.to(dtype=x.dtype)
    return y * weight.to(dtype=x.dtype)


def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    d_model = 16
    x = torch.randn(2, 3, d_model, dtype=torch.float32)

    norm = RMSNorm(d_model=d_model, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, d_model))

    y = norm(x)
    y_ref = _reference_rmsnorm(x, norm.weight, eps=norm.eps)

    assert y.shape == x.shape
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)


def test_rmsnorm_scale_invariance_for_positive_scale():
    torch.manual_seed(0)
    d_model = 32
    x = torch.randn(4, 5, d_model, dtype=torch.float32)

    norm = RMSNorm(d_model=d_model, eps=1e-6)
    with torch.no_grad():
        norm.weight.fill_(1.0)

    y1 = norm(x)
    y2 = norm(2.0 * x)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)


def test_rmsnorm_preserves_input_dtype():
    torch.manual_seed(0)
    d_model = 16
    x = torch.randn(2, 3, d_model, dtype=torch.float16)

    norm = RMSNorm(d_model=d_model, eps=1e-6)
    y = norm(x)
    assert y.dtype == x.dtype
    assert y.shape == x.shape
