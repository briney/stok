"""Deterministic per-protein evaluation of a Stage 1 arm over a feature cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import CachedFeatures
from .heads import FrequencyBaseline, head_predict
from .metrics import token_nll, topk_hits


def _logits_for(arm, feats: torch.Tensor) -> torch.Tensor:
    """Compute ``(n, C)`` logits for an arm on ``(n, d_in)`` features."""
    if isinstance(arm, FrequencyBaseline):
        return arm.logits(feats.shape[0])
    return head_predict(arm, feats)


def evaluate_arm(arm, cache: CachedFeatures, *, device: str | torch.device = "cpu") -> pd.DataFrame:
    """Return a per-protein metrics table (sequence_id, n_res, mean_nll, top1, top5)."""
    if isinstance(arm, nn.Module):
        arm = arm.to(device).eval()
    features = np.asarray(cache.features)
    rows: list[dict] = []
    with torch.no_grad():
        for sequence_id, start, length in cache.protein_ranges:
            feats = torch.from_numpy(features[start : start + length].astype(np.float32)).to(device)
            targets = torch.from_numpy(cache.token_ids[start : start + length]).to(device)
            logits = _logits_for(arm, feats)
            num_classes = logits.shape[-1]
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "n_res": int(length),
                    "mean_nll": float(token_nll(logits, targets).mean().cpu()),
                    "top1": float(topk_hits(logits, targets, 1).float().mean().cpu()),
                    "top5": float(
                        topk_hits(logits, targets, min(5, num_classes)).float().mean().cpu()
                    ),
                }
            )
    return pd.DataFrame(rows)
