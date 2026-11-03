# this was drawn heavily from GCP-VQVAE (https://github.com/mahdip72/vq_encoder_decoder)
# which itself was heavily inspired by ProteinWorkshop (https://github.com/a-r-j/ProteinWorkshop)
# which is a re-implementation of the original GCPNet (https://github.com/BioinfoMachineLearning/GCPNet)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass(frozen=True, slots=True)
class ScalarVector:
    """Paired scalar features (..., S) and 3D vector features (..., V, 3)."""

    scalar: torch.Tensor
    vector: torch.Tensor

    def __iter__(self):
        yield self.scalar
        yield self.vector

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self.scalar
        if index == 1:
            return self.vector
        raise IndexError("ScalarVector only contains two elements")

    def __add__(self, other: "ScalarVector") -> "ScalarVector":
        return ScalarVector(self.scalar + other.scalar, self.vector + other.vector)


class CachedGaussianRBF(torch.nn.Module):
    """Gaussian radial basis expansion of edge distances."""

    _centers_cpu: torch.Tensor
    _sigma_cpu: torch.Tensor

    def __init__(
        self,
        min_distance: float = 0.0,
        max_distance: float = 10.0,
        num_rbf: int = 8,
    ):
        super().__init__()
        base = torch.linspace(
            min_distance,
            max_distance,
            num_rbf,
            dtype=torch.get_default_dtype(),
        ).view(1, -1)
        sigma = torch.tensor(
            (max_distance - min_distance) / num_rbf,
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("_centers_cpu", base, persistent=False)
        self.register_buffer("_sigma_cpu", sigma, persistent=False)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        centers = self._centers_cpu.to(device=distances.device, dtype=distances.dtype)
        sigma = self._sigma_cpu.to(device=distances.device, dtype=distances.dtype)
        expanded = distances.unsqueeze(-1)
        return torch.exp(-(((expanded - centers) / sigma) ** 2))


def centralize(batch: Data, key: str, batch_index: torch.Tensor) -> torch.Tensor:
    """Subtract each entity's mean from ``batch[key]`` (``batch_index`` maps rows to entities)."""
    lengths = torch.bincount(batch_index)
    centroid = scatter_reduce(
        batch[key], batch_index, dim=0, reduce="mean", dim_size=lengths.size(0)
    )
    return batch[key] - centroid[batch_index]


def localize(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    norm_pos_diff: bool = True,
) -> torch.Tensor:
    """Construct local 3D edge frames from node positions.

    For each edge ``(i, j)``, computes:

    - ``pos_diff = pos[i] - pos[j]``
    - ``pos_cross = cross(pos[i], pos[j])``
    - ``pos_vertical = cross(pos_diff, pos_cross)``

    If ``norm_pos_diff`` is ``True``, ``pos_diff`` and ``pos_cross`` are each
    normalized by their L2 norm plus ``1`` to avoid division by zero.

    Returns:
        torch.Tensor: Local edge frames of shape ``(E, 3, 3)`` ordered as
        ``(pos_diff, pos_cross, pos_vertical)``.
    """
    row, col = edge_index
    pos_diff = pos[row] - pos[col]
    pos_cross = torch.cross(pos[row], pos[col])

    if norm_pos_diff:
        pos_diff = pos_diff / (pos_diff.norm(dim=1, keepdim=True) + 1)
        pos_cross = pos_cross / (pos_cross.norm(dim=1, keepdim=True) + 1)

    pos_vertical = torch.cross(pos_diff, pos_cross)
    return torch.cat(
        (
            pos_diff.unsqueeze(1),
            pos_cross.unsqueeze(1),
            pos_vertical.unsqueeze(1),
        ),
        dim=1,
    )


@overload
def get_activations(
    act_name: str, return_functional: Literal[False] = False
) -> nn.Module: ...


@overload
def get_activations(act_name: str, return_functional: Literal[True]) -> Callable: ...


def get_activations(
    act_name: str, return_functional: bool = False
) -> nn.Module | Callable:
    if act_name == "none":
        return nn.Identity()
    if act_name in {"silu", "swish"}:
        return F.silu if return_functional else nn.SiLU()
    raise ValueError(f"Unknown activation function: {act_name}")


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True,
) -> torch.Tensor:
    norm = torch.sum(x**2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm.clamp_min(eps))
    return norm


def scatter_reduce(
    x: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int, reduce: str
) -> torch.Tensor:
    """Sum or mean rows of ``x`` into ``dim_size`` groups; empty groups are zero."""
    if reduce not in {"sum", "mean"}:
        raise ValueError(f"Unknown reduction function: {reduce}")
    out = x.new_zeros((dim_size, x.size(1)))
    idx = index.unsqueeze(-1).expand_as(x)
    out.scatter_reduce_(dim, idx, x, reduce=reduce, include_self=False)
    return out
