"""Batching helpers for graph-based structural encoders."""

from __future__ import annotations

import torch
from torch_geometric.utils import unbatch


def unbatch_and_pad(
    node_emb: torch.Tensor,
    batch_index: torch.Tensor,
    max_length: int,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Split a flat per-node tensor into per-graph rows padded to ``max_length``.

    Mirrors the ``separate_features`` + ``merge_features`` pair used by GCP-VQVAE's
    ``SuperModel.forward`` to bridge the GCPNet node output into a fixed-length
    transformer input.

    Args:
        node_emb: Flat node tensor of shape ``(sum(L_i), D)``.
        batch_index: Integer graph-assignment tensor of shape ``(sum(L_i),)``.
        max_length: Target padded length per graph.
        pad_value: Fill value used for padded positions.

    Returns:
        Tensor of shape ``(B, max_length, D)``.

    Raises:
        ValueError: If any graph is longer than ``max_length``.
    """
    per_graph = unbatch(node_emb, batch_index)
    batch_size = len(per_graph)
    dim = node_emb.size(-1)
    out = node_emb.new_full((batch_size, max_length, dim), pad_value)
    for i, x in enumerate(per_graph):
        n = x.size(0)
        if n > max_length:
            raise ValueError(
                f"Graph {i} has {n} nodes, exceeding max_length={max_length}"
            )
        out[i, :n] = x
    return out
