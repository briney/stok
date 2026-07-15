"""Token-space metrics and paired protein-level bootstrap for Stage 1."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-residue negative log-likelihood, shape ``(N,)``."""
    return F.cross_entropy(logits, targets, reduction="none")


def topk_hits(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Whether the target is within the top-``k`` logits, per residue, shape ``(N,)``."""
    topk = logits.topk(k, dim=-1).indices  # (N, k)
    return (topk == targets.unsqueeze(-1)).any(dim=-1)


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired protein-level bootstrap CI for the mean of ``a - b``.

    Args:
        a, b: Per-protein metric arrays of equal length (paired by protein).

    Returns:
        ``(mean_diff, lo, hi)`` where the interval is the central ``1 - alpha`` band.
    """
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    diff = a - b
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = diff[rng.integers(0, n, size=n)].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(diff.mean()), lo, hi
