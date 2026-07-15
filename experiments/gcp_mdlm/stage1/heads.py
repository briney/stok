"""Stage 1 prediction heads: frequency floor, independent classifier, prototype factory.

The independent classifier is a throwaway DPLM-2-style validation baseline; the
prototype head reuses STōk's production ``CodebookClassifier``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from stok.models.head import CodebookClassifier


class FrequencyBaseline:
    """Predicts the (smoothed) marginal training-token distribution for every residue."""

    def __init__(self, log_probs: torch.Tensor) -> None:
        self.log_probs = log_probs  # (C,)

    @classmethod
    def fit(
        cls, token_ids: np.ndarray, num_classes: int, *, smoothing: float = 1.0
    ) -> FrequencyBaseline:
        counts = np.bincount(token_ids, minlength=num_classes).astype(np.float64) + smoothing
        probs = counts / counts.sum()
        return cls(torch.log(torch.from_numpy(probs).float()))

    def logits(self, n: int) -> torch.Tensor:
        """Return ``(n, C)`` constant log-probability rows (usable as logits)."""
        return self.log_probs.unsqueeze(0).expand(n, -1).contiguous()


class IndependentClassifier(nn.Module):
    """Free linear map from features to codebook classes (no codebook grounding)."""

    def __init__(self, d_in: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def build_prototype_head(d_in: int, codebook: torch.Tensor, **kwargs) -> CodebookClassifier:
    """Construct the production prototype-tied head."""
    return CodebookClassifier(d_in=d_in, codebook=codebook, **kwargs)


def head_predict(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Apply a ``(*, d_in)`` head to flattened ``(N, d_in)`` features -> ``(N, C)``."""
    return head(features.unsqueeze(0)).squeeze(0)
