# this was drawn heavily from GCP-VQVAE (https://github.com/mahdip72/vq_encoder_decoder)
# which itself was heavily inspired by ProteinWorkshop (https://github.com/a-r-j/ProteinWorkshop)
# which is a re-implementation of the original GCPNet (https://github.com/BioinfoMachineLearning/GCPNet)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, NewType, Tuple, overload, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch_scatter
from torch_geometric.data import Batch, Data

EncoderOutput = NewType("EncoderOutput", dict[str, torch.Tensor])


@dataclass(frozen=True, slots=True)
class ScalarVector:
    """Container for paired scalar and 3D vector features.

    This structure bundles a scalar feature tensor and a matching vector
    feature tensor whose last dimension has size 3 to represent 3D components.

    Attributes:
        scalar (torch.Tensor): Tensor of scalar features with shape (..., S).
        vector (torch.Tensor): Tensor of vector features with shape (..., V, 3).
    """

    scalar: torch.Tensor
    vector: torch.Tensor

    def __iter__(self):
        """Iterate over the pair as ``(scalar, vector)``.

        Yields:
            torch.Tensor: The scalar tensor followed by the vector tensor.
        """
        yield self.scalar
        yield self.vector

    def __len__(self) -> int:
        """Return the number of contained elements.

        Returns:
            int: Always ``2`` for ``(scalar, vector)``.
        """
        return 2

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the scalar (index 0) or vector (index 1).

        Args:
            index (int): ``0`` for ``scalar``, ``1`` for ``vector``.

        Returns:
            torch.Tensor: The selected tensor.

        Raises:
            IndexError: If ``index`` is not 0 or 1.
        """
        if index == 0:
            return self.scalar
        if index == 1:
            return self.vector
        raise IndexError("ScalarVector only contains two elements")

    def __add__(self, other: "ScalarVector") -> "ScalarVector":
        """Element-wise addition with another ``ScalarVector``.

        Args:
            other (ScalarVector): The right-hand operand.

        Returns:
            ScalarVector: Sum of corresponding ``scalar`` and ``vector`` tensors.
        """
        if not isinstance(other, ScalarVector):  # pragma: no cover - defensive
            return NotImplemented
        return ScalarVector(self.scalar + other.scalar, self.vector + other.vector)

    def __mul__(self, other) -> "ScalarVector":
        """Element-wise multiplication.

        If ``other`` is a ``ScalarVector``, multiplies element-wise both
        components. Otherwise, treats ``other`` as a scalar/broadcastable
        tensor and multiplies both ``scalar`` and ``vector`` by it.

        Args:
            other (ScalarVector | Any): Right-hand operand.

        Returns:
            ScalarVector: Product of operands.
        """
        if isinstance(other, ScalarVector):
            return ScalarVector(self.scalar * other.scalar, self.vector * other.vector)
        return ScalarVector(self.scalar * other, self.vector * other)

    def concat(
        self, others: Iterable["ScalarVector"], dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate this pair with others along a tensor dimension.

        Concatenates the ``scalar`` tensors together and the ``vector`` tensors
        together along the same dimension.

        Args:
            others (Iterable[ScalarVector]): Iterable of additional pairs to
                concatenate.
            dim (int): Dimension along which to concatenate. Negative indices
                are resolved modulo ``scalar.dim()``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The concatenated scalar and
            vector tensors.

        Raises:
            TypeError: If any element in ``others`` is not a ``ScalarVector``.
        """
        dim %= self.scalar.dim()
        scalars = [self.scalar]
        vectors = [self.vector]
        for other in others:
            if not isinstance(other, ScalarVector):  # pragma: no cover - defensive
                raise TypeError("Expected ScalarVector instances in concat")
            scalars.append(other.scalar)
            vectors.append(other.vector)
        return torch.cat(scalars, dim=dim), torch.cat(vectors, dim=dim)

    def flatten(self) -> torch.Tensor:
        """Flatten the vector channel and concatenate with the scalars.

        Transforms ``vector`` from shape ``(..., V, 3)`` to ``(..., 3*V)`` and
        concatenates it to ``scalar`` along the last dimension.

        Returns:
            torch.Tensor: Tensor of shape ``(..., S + 3*V)``.
        """
        flat_vector = self.vector.reshape(*self.vector.shape[:-2], -1)
        return torch.cat((self.scalar, flat_vector), dim=-1)

    @staticmethod
    def recover(x: torch.Tensor, vector_dim: int) -> "ScalarVector":
        """Reconstruct a ``ScalarVector`` from a flattened feature tensor.

        This is the inverse of :meth:`flatten`. It splits the last dimension
        into scalar and vector parts, with the vector part reshaped to
        ``(..., V, 3)``.

        Args:
            x (torch.Tensor): Flattened features of shape ``(..., S + 3*V)``.
            vector_dim (int): Number of vector channels ``V``. If ``0``, an
                empty vector dimension of shape ``(..., 0, 3)`` is produced.

        Returns:
            ScalarVector: The reconstructed pair.
        """
        if vector_dim == 0:
            zero_vector = x.new_zeros(*x.shape[:-1], 0, 3)
            return ScalarVector(x, zero_vector)
        v = x[..., -3 * vector_dim :].reshape(*x.shape[:-1], vector_dim, 3)
        s = x[..., : -3 * vector_dim]
        return ScalarVector(s, v)

    def idx(self, index) -> "ScalarVector":
        """Index both components consistently.

        Args:
            index: Index or slice applied to both ``scalar`` and ``vector``.

        Returns:
            ScalarVector: Indexed view of the pair.
        """
        return ScalarVector(self.scalar[index], self.vector[index])

    def clone(self) -> "ScalarVector":
        """Deep-copy both components.

        Returns:
            ScalarVector: A cloned pair with detached storage.
        """
        return ScalarVector(self.scalar.clone(), self.vector.clone())

    def mask(self, node_mask: torch.Tensor) -> "ScalarVector":
        """Apply a node mask to both components.

        Multiplies ``scalar`` by ``node_mask[..., None]`` and ``vector`` by
        ``node_mask[..., None, None]`` to zero out entries where the mask is
        ``False``/``0``.

        Args:
            node_mask (torch.Tensor): Boolean or float mask broadcastable to
                ``scalar`` as ``(..., 1)`` and ``vector`` as ``(..., 1, 1)``.

        Returns:
            ScalarVector: Masked pair.
        """
        return ScalarVector(
            self.scalar * node_mask[..., None],
            self.vector * node_mask[..., None, None],
        )


class CachedGaussianRBF(torch.nn.Module):
    """Gaussian RBF sampler that caches basis centers on each device."""

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

        self._center_cache = {}
        self._sigma_cache = {}

    def _cached(
        self,
        cache: dict,
        tensor: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (device.type, device.index if device.type != "cpu" else -1, dtype)
        value = cache.get(key)
        if value is None:
            value = tensor.to(device=device, dtype=dtype)
            cache[key] = value
        return value

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        centers = self._cached(
            self._center_cache, self._centers_cpu, distances.device, distances.dtype
        )
        sigma = self._cached(
            self._sigma_cache, self._sigma_cpu, distances.device, distances.dtype
        )
        expanded = distances.unsqueeze(-1)
        return torch.exp(-(((expanded - centers) / sigma) ** 2))


# @dataclass
# class _CuGraphCSCCache:
#     perm: torch.Tensor
#     graph: Any
#     num_edges: int
#     num_nodes: int
#     offsets_cp: Any
#     indices_cp: Any


def centralize(
    batch: Data,
    key: str,
    batch_index: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-entity centroids and center features.

    Computes the mean of ``batch[key]`` for each entity indicated by
    ``batch_index`` and subtracts it to produce centered values.

    If ``node_mask`` is provided, only positions where ``node_mask`` is
    ``True`` contribute to the mean, and only those positions are centered;
    positions where ``node_mask`` is ``False`` in the returned centered tensor
    are filled with ``+inf``.

    Args:
        batch (Batch | ProteinBatch): Graph batch containing the tensor at
            ``key`` of shape ``(N, D)``.
        key (str): Key into ``batch`` that points to the tensor to center.
        batch_index (torch.Tensor): Integer tensor of shape ``(N,)`` mapping
            each row/node to an entity index in ``[0, num_entities)``.
        node_mask (torch.Tensor | None): Optional boolean or float mask of
            shape ``(N,)`` indicating valid nodes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - ``centroid``: Tensor of shape ``(num_entities, D)`` with per-entity means.
            - ``centered``: Tensor of shape ``(N, D)`` with means subtracted
              (or ``+inf`` where ``node_mask`` is ``False`` when provided).
    """
    lengths = torch.bincount(batch_index)
    dim_size = lengths.size(0)
    if node_mask is not None:
        # centroid = torch_scatter.scatter(
        #     batch[key][node_mask],
        #     batch_index[node_mask],
        #     dim=0,
        #     reduce="mean",
        #     dim_size=dim_size,
        # )
        centroid = scatter_reduce(
            batch[key][node_mask],
            batch_index[node_mask],
            dim=0,
            reduce="mean",
            dim_size=dim_size,
        )
        centered = torch.full_like(batch[key], torch.inf)
        centered[node_mask] = batch[key][node_mask] - centroid[batch_index][node_mask]
        return centroid, centered

    # centroid = torch_scatter.scatter(
    #     batch[key], batch_index, dim=0, reduce="mean", dim_size=dim_size
    # )
    centroid = scatter_reduce(
        batch[key], batch_index, dim=0, reduce="mean", dim_size=dim_size
    )
    centered = batch[key] - centroid[batch_index]
    return centroid, centered


def decentralize(
    batch: Data,
    key: str,
    batch_index: torch.Tensor,
    entities_centroid: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restore global features by adding per-entity centroids.

    This is the inverse of :func:`centralize` for the centered tensor. If
    ``node_mask`` is provided, only entries where ``node_mask`` is ``True`` are
    restored; other positions are filled with ``+inf``.

    Args:
        batch (Batch | ProteinBatch): Graph batch with tensor at ``key`` of
            shape ``(N, D)``.
        key (str): Key into ``batch`` for the centered tensor to restore.
        batch_index (torch.Tensor): Integer tensor of shape ``(N,)`` mapping
            rows to entities.
        entities_centroid (torch.Tensor): Tensor of shape ``(num_entities, D)``
            containing per-entity centroids.
        node_mask (torch.Tensor | None): Optional boolean mask of shape ``(N,)``.

    Returns:
        torch.Tensor: Restored tensor of shape ``(N, D)``.
    """
    if node_mask is not None:
        restored = torch.full_like(batch[key], torch.inf)
        restored[node_mask] = (
            batch[key][node_mask] + entities_centroid[batch_index][node_mask]
        )
        return restored
    return batch[key] + entities_centroid[batch_index]


def localize(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    norm_pos_diff: bool = True,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct local 3D edge frames from node positions.

    For each edge ``(i, j)``, computes:

    - ``pos_diff = pos[i] - pos[j]``
    - ``pos_cross = cross(pos[i], pos[j])``
    - ``pos_vertical = cross(pos_diff, pos_cross)``

    If ``norm_pos_diff`` is ``True``, ``pos_diff`` and ``pos_cross`` are each
    normalized by their L2 norm plus ``1`` to avoid division by zero. The
    returned tensor stacks these three vectors per edge.

    When ``node_mask`` is provided, only edges where both incident nodes have
    ``True`` in the mask are computed; other edges are filled with ``+inf``.

    Args:
        pos (torch.Tensor): Node positions of shape ``(N, 3)``.
        edge_index (torch.Tensor): Edge indices of shape ``(2, E)``.
        norm_pos_diff (bool): Whether to normalize the difference and cross
            vectors by their norms plus ``1``.
        node_mask (torch.Tensor | None): Optional boolean mask over nodes of
            shape ``(N,)``.

    Returns:
        torch.Tensor: Local edge frames of shape ``(E, 3, 3)`` ordered as
        ``(pos_diff, pos_cross, pos_vertical)``.
    """
    row, col = edge_index

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]
        pos_diff = torch.full((edge_index.size(1), 3), torch.inf, device=pos.device)
        pos_cross = torch.full_like(pos_diff, torch.inf)
        pos_diff[edge_mask] = pos[row][edge_mask] - pos[col][edge_mask]
        pos_cross[edge_mask] = torch.cross(pos[row][edge_mask], pos[col][edge_mask])
    else:
        pos_diff = pos[row] - pos[col]
        pos_cross = torch.cross(pos[row], pos[col])

    if norm_pos_diff:
        if node_mask is not None:
            norm = torch.ones((edge_index.size(1), 1), device=pos.device)
            norm[edge_mask] = pos_diff[edge_mask].norm(dim=1, keepdim=True) + 1
        else:
            norm = pos_diff.norm(dim=1, keepdim=True) + 1
        pos_diff = pos_diff / norm

        if node_mask is not None:
            cross_norm = torch.ones((edge_index.size(1), 1), device=pos.device)
            cross_norm[edge_mask] = pos_cross[edge_mask].norm(dim=1, keepdim=True) + 1
        else:
            cross_norm = pos_cross.norm(dim=1, keepdim=True) + 1
        pos_cross = pos_cross / cross_norm

    if node_mask is not None:
        pos_vertical = torch.full_like(pos_diff, torch.inf)
        pos_vertical[edge_mask] = torch.cross(pos_diff[edge_mask], pos_cross[edge_mask])
    else:
        pos_vertical = torch.cross(pos_diff, pos_cross)

    return torch.cat(
        (
            pos_diff.unsqueeze(1),
            pos_cross.unsqueeze(1),
            pos_vertical.unsqueeze(1),
        ),
        dim=1,
    )


def _extract_batch_info(
    batch_like: Batch | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if isinstance(batch_like, Batch):
        index = cast(Data, batch_like).batch
        assert index is not None
        num_graphs = batch_like.num_graphs
        lengths = torch.bincount(index, minlength=num_graphs)
    else:
        index = batch_like
        lengths = (
            torch.bincount(index)
            if index.numel()
            else index.new_zeros(0, dtype=torch.long)
        )
        num_graphs = lengths.size(0)
    if lengths.device != index.device:
        lengths = lengths.to(index.device)
    return index, lengths, num_graphs


def get_aggregation(aggregation: str) -> Callable:
    def pool_sum(x: torch.Tensor, batch_like: Batch | torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, x.size(-1)))
        index, _, num_graphs = _extract_batch_info(batch_like)
        if num_graphs == 0:
            return x.new_zeros((0, x.size(-1)))
        # return torch_scatter.scatter(x, index, dim=0, dim_size=num_graphs, reduce="sum")
        return scatter_reduce(x, index, dim=0, dim_size=num_graphs, reduce="sum")

    def pool_mean(x: torch.Tensor, batch_like: Batch | torch.Tensor) -> torch.Tensor:
        sums = pool_sum(x, batch_like)
        if sums.size(0) == 0:
            return sums
        _, lengths, _ = _extract_batch_info(batch_like)
        counts = lengths.to(sums.dtype).clamp_min(1).unsqueeze(-1)
        return sums / counts

    def pool_max(x: torch.Tensor, batch_like: Batch | torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, x.size(-1)))
        index, _, num_graphs = _extract_batch_info(batch_like)
        if num_graphs == 0:
            return x.new_zeros((0, x.size(-1)))
        # return torch_scatter.scatter(x, index, dim=0, dim_size=num_graphs, reduce="max")
        return scatter_reduce(x, index, dim=0, dim_size=num_graphs, reduce="max")

    if aggregation == "max":
        return pool_max
    if aggregation == "mean":
        return pool_mean
    if aggregation in {"sum", "add"}:
        return pool_sum
    raise ValueError(f"Unknown aggregation function: {aggregation}")


@overload
def get_activations(
    act_name: str, return_functional: Literal[False] = False
) -> nn.Module: ...


@overload
def get_activations(act_name: str, return_functional: Literal[True]) -> Callable: ...


def get_activations(
    act_name: str, return_functional: bool = False
) -> nn.Module | Callable:
    if act_name == "relu":
        return F.relu if return_functional else nn.ReLU()
    if act_name == "elu":
        return F.elu if return_functional else nn.ELU()
    if act_name == "leaky_relu":
        return F.leaky_relu if return_functional else nn.LeakyReLU()
    if act_name == "tanh":
        return F.tanh if return_functional else nn.Tanh()
    if act_name == "sigmoid":
        return F.sigmoid if return_functional else nn.Sigmoid()
    if act_name == "none":
        return nn.Identity()
    if act_name in {"silu", "swish"}:
        return F.silu if return_functional else nn.SiLU()
    raise ValueError(f"Unknown activation function: {act_name}")


def is_identity(obj: nn.Module | Callable) -> bool:
    return isinstance(obj, nn.Identity) or getattr(obj, "__name__", None) == "identity"


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
    """ """
    # torch.scatter_reduce does not support "max" and "min" reduction, so we use "amax" and "amin" instead
    if reduce in ["max", "amax"]:
        reduce = "amax"
        init_val = float("-inf")
    elif reduce in ["min", "amin"]:
        reduce = "amin"
        init_val = float("inf")
    elif reduce in ["sum", "mean"]:
        init_val = float(0)
    elif reduce == "prod":
        init_val = float(1)
    else:
        raise ValueError(f"Unknown reduction function: {reduce}")

    # build the output tensor
    out = x.new_full((dim_size, x.size(1)), init_val)

    # match the source shape along dim=0 so that broadcasting works
    idx = index.unsqueeze(-1).expand_as(x)

    out.scatter_reduce_(dim, idx, x, reduce=reduce, include_self=False)
    if reduce in ("amax", "amin"):
        occupied = torch.bincount(index, minlength=dim_size) > 0
        out = out.masked_fill(~occupied[:, None], 0)
    return out
    # return torch.scatter_reduce(out, dim, idx, x, reduce=reduce, include_self=False)
