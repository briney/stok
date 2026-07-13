from __future__ import annotations

import torch

from .types import TensorComparison, classify_run

__all__ = ["classify_run", "compare_floats", "compare_indices"]


def _require_same_shape(name: str, ref: torch.Tensor, stok: torch.Tensor) -> None:
    if ref.shape != stok.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(ref.shape)} != {tuple(stok.shape)}")


def compare_indices(
    name: str,
    ref: torch.Tensor,
    stok: torch.Tensor,
    ref_valid: torch.Tensor,
    stok_valid: torch.Tensor,
) -> TensorComparison:
    _require_same_shape(name, ref, stok)
    _require_same_shape(f"{name}.valid", ref_valid, stok_valid)
    if ref.shape != ref_valid.shape:
        raise ValueError(f"{name} values and validity mask have different shapes")
    ref_valid = ref_valid.to(torch.bool).cpu()
    stok_valid = stok_valid.to(torch.bool).cpu()
    mask_equal = torch.equal(ref_valid, stok_valid)
    shared = ref_valid & stok_valid
    lhs = ref.detach().cpu()[shared]
    rhs = stok.detach().cpu()[shared]
    mismatched = int(torch.count_nonzero(lhs != rhs).item())
    exact = mask_equal and mismatched == 0
    return TensorComparison(
        name=name,
        shape=tuple(ref.shape),
        compared=int(shared.sum().item()),
        mismatched=mismatched,
        exact=exact,
        passed=exact,
        mask_equal=mask_equal,
        finite_pattern_equal=True,
        within_tolerance=exact,
    )


def compare_floats(
    name: str,
    ref: torch.Tensor,
    stok: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> TensorComparison:
    _require_same_shape(name, ref, stok)
    lhs = ref.detach().to(torch.float64).cpu()
    rhs = stok.detach().to(torch.float64).cpu()
    finite_lhs = torch.isfinite(lhs)
    finite_rhs = torch.isfinite(rhs)
    finite_pattern_equal = torch.equal(finite_lhs, finite_rhs)
    shared = finite_lhs & finite_rhs
    diffs = (lhs[shared] - rhs[shared]).abs()
    if diffs.numel() == 0:
        max_abs = max_rel = p50 = p95 = p99 = 0.0
    else:
        denom = lhs[shared].abs().clamp_min(torch.finfo(torch.float32).tiny)
        rel = diffs / denom
        quantiles = torch.quantile(diffs, torch.tensor([0.50, 0.95, 0.99], dtype=diffs.dtype))
        max_abs = float(diffs.max().item())
        max_rel = float(rel.max().item())
        p50, p95, p99 = (float(value.item()) for value in quantiles)
    within = finite_pattern_equal and torch.allclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=True)
    exact = torch.equal(lhs, rhs)
    return TensorComparison(
        name=name,
        shape=tuple(ref.shape),
        compared=int(shared.sum().item()),
        mismatched=int(torch.count_nonzero(diffs != 0).item()),
        exact=exact,
        passed=within,
        mask_equal=True,
        finite_pattern_equal=finite_pattern_equal,
        within_tolerance=within,
        max_abs=max_abs,
        max_rel=max_rel,
        p50_abs=p50,
        p95_abs=p95,
        p99_abs=p99,
    )
