"""Train a Stage 1 head on cached frozen features (per-residue token classification)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .features import CachedFeatures
from .heads import head_predict


def train_head(
    head: nn.Module,
    cache: CachedFeatures,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[float]:
    """Train ``head`` in place on cached features; return per-step loss history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = head.to(device).train()
    optimizer = torch.optim.Adam((p for p in head.parameters() if p.requires_grad), lr=lr)
    n = cache.token_ids.shape[0]
    features = np.asarray(cache.features)
    history: list[float] = []
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        feats = torch.from_numpy(features[idx].astype(np.float32)).to(device)
        targets = torch.from_numpy(cache.token_ids[idx]).to(device)
        logits = head_predict(head, feats)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history
