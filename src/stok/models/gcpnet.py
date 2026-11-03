# this was drawn heavily from GCP-VQVAE (https://github.com/mahdip72/vq_encoder_decoder)
# which itself was heavily inspired by ProteinWorkshop (https://github.com/a-r-j/ProteinWorkshop)
# which is a re-implementation of the original GCPNet (https://github.com/BioinfoMachineLearning/GCPNet)
#
# Only the paths reachable from the pinned lite/large presets remain: SE(3) frames,
# vector gating, post-norm interactions, no position updates and no dropout.

import contextlib

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Data

from ..utils.gcp import (
    CachedGaussianRBF,
    ScalarVector,
    centralize,
    get_activations,
    localize,
    safe_norm,
    scatter_reduce,
)


class GCPNetModel(torch.nn.Module):
    def __init__(
        self,
        *,
        module_cfg: DictConfig,
        model_cfg: DictConfig,
        layer_cfg: DictConfig,
        # Remaining pinned top-level keys duplicate the nested configs above.
        **_duplicated,
    ):
        super().__init__()
        edge_input_dims = (model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = (model_cfg.h_input_dim, model_cfg.chi_input_dim)
        self.edge_dims = (model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = (model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)
        self.norm_pos_diff = module_cfg.norm_pos_diff

        self.gcp_embedding = GCPEmbedding(
            edge_input_dims,
            node_input_dims,
            self.edge_dims,
            self.node_dims,
            cfg=module_cfg,
        )
        self.interaction_layers = nn.ModuleList(
            GCPInteractions(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                num_message_layers=layer_cfg.mp_cfg.num_message_layers,
            )
            for _ in range(model_cfg.num_layers)
        )
        self.invariant_node_projection = nn.ModuleList(
            [
                GCPLayerNorm(self.node_dims),
                GCP(
                    self.node_dims,
                    (self.node_dims[0], 0),
                    nonlinearities=tuple(module_cfg.nonlinearities),
                ),
            ]
        )

    def forward(self, batch: Data) -> dict[str, torch.Tensor]:
        """Return ``{"node_embedding": [|V|, d]}`` for a featurised graph batch."""
        assert batch.batch is not None and batch.edge_index is not None
        # Centralize node positions to make them translation-invariant
        batch.pos = pos = centralize(batch, "pos", batch_index=batch.batch)
        frames = localize(pos, batch.edge_index, norm_pos_diff=self.norm_pos_diff)
        batch.f_ij = frames

        (h, chi), (e, xi) = self.gcp_embedding(batch)
        for layer in self.interaction_layers:
            h, chi = layer((h, chi), (e, xi), batch.edge_index, frames)

        out = self.invariant_node_projection[0](ScalarVector(h, chi))
        out = self.invariant_node_projection[1](
            out, batch.edge_index, frames, node_inputs=True
        )
        return {"node_embedding": out}


# ------------
#   Layers
# ------------


class GCP(nn.Module):
    def __init__(
        self,
        input_dims: tuple[int, int],
        output_dims: tuple[int, int],
        nonlinearities: tuple[str, str] = ("silu", "silu"),
        scalar_out_nonlinearity: str = "silu",
        feedforward_out: bool = False,
        bottleneck: int = 1,
        scalarization_vectorization_output_dim: int = 3,
    ):
        super().__init__()
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_activations(nonlinearities[0], return_functional=True),
            get_activations(nonlinearities[1], return_functional=True),
        )

        assert self.vector_input_dim % bottleneck == 0, (
            f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"
        )
        self.hidden_dim = (
            self.vector_input_dim // bottleneck
            if bottleneck > 1
            else max(self.vector_input_dim, self.vector_output_dim)
        )

        scalar_vector_frame_dim = scalarization_vectorization_output_dim * 3
        self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
        scalar_in = self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim
        self.scalar_out = (
            nn.Sequential(
                nn.Linear(scalar_in, self.scalar_output_dim),
                get_activations(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
            )
            if feedforward_out
            else nn.Linear(scalar_in, self.scalar_output_dim)
        )
        self.vector_down_frames = nn.Linear(
            self.vector_input_dim,
            scalarization_vectorization_output_dim,
            bias=False,
        )
        if self.vector_output_dim:
            self.vector_up = nn.Linear(
                self.hidden_dim, self.vector_output_dim, bias=False
            )
            self.vector_out_scale = nn.Linear(
                self.scalar_output_dim, self.vector_output_dim
            )

    @staticmethod
    def scalarize(
        vector_rep: torch.Tensor,
        edge_index: torch.Tensor,
        frames: torch.Tensor,
        node_inputs: bool,
        dim_size: int,
    ) -> torch.Tensor:
        row = edge_index[0]

        # gather source node features for each `entity` (i.e., node or edge)
        # note: edge inputs are already ordered according to source nodes
        vector_rep_i = vector_rep[row] if node_inputs else vector_rep

        # project equivariant values onto corresponding local frames
        vector_rep_i = vector_rep_i.transpose(-1, -2)
        local_scalar_rep_i = torch.matmul(frames, vector_rep_i).transpose(-1, -2)
        local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

        if node_inputs:
            # summarize according to source node indices due to the directional nature of GCP's equivariant frames
            return scatter_reduce(
                local_scalar_rep_i, row, dim=0, dim_size=dim_size, reduce="mean"
            )
        return local_scalar_rep_i

    def vectorize(
        self,
        scalar_rep: torch.Tensor,
        vector_hidden_rep: torch.Tensor,
    ) -> torch.Tensor:
        vector_rep = self.vector_up(vector_hidden_rep)
        vector_rep = vector_rep.transpose(-1, -2)
        gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
        return vector_rep * torch.sigmoid(gate).unsqueeze(-1)

    def forward(
        self,
        s_maybe_v: ScalarVector | tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        frames: torch.Tensor,
        node_inputs: bool = False,
    ) -> torch.Tensor | ScalarVector:
        scalar_rep, vector_rep = s_maybe_v
        v_pre = vector_rep.transpose(-1, -2)

        vector_hidden_rep = self.vector_down(v_pre)
        vector_norm = safe_norm(vector_hidden_rep, dim=-2)
        merged = torch.cat((scalar_rep, vector_norm), dim=-1)

        # curate direction-robust and chirality-aware scalar geometric features
        vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
        scalar_hidden_rep = self.scalarize(
            vector_down_frames_hidden_rep.transpose(-1, -2),
            edge_index,
            frames,
            node_inputs=node_inputs,
            dim_size=vector_down_frames_hidden_rep.shape[0],
        )
        merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            return self.scalar_nonlinearity(scalar_rep)

        # update vector features using row-wise scalar gating with complete local frames
        vector_rep = self.vectorize(scalar_rep, vector_hidden_rep)
        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        return ScalarVector(scalar_rep, vector_rep)


class GCPLayerNorm(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, dims: tuple[int, int], eps: float = 1e-8):
        super().__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = nn.LayerNorm(self.scalar_dims)
        self.eps = eps

    def norm_vector(self, v: torch.Tensor) -> torch.Tensor:
        vector_norm = torch.clamp(
            torch.sum(torch.square(v), dim=-1, keepdim=True), min=self.eps
        )
        vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
        return v / vector_norm

    def forward(self, x: ScalarVector) -> ScalarVector:
        if x.scalar.shape[0] == 0 or x.vector.shape[0] == 0:
            return x
        s, v = x
        return ScalarVector(self.scalar_norm(s), self.norm_vector(v))


class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: tuple[int, int],
        node_input_dims: tuple[int, int],
        edge_hidden_dims: tuple[int, int],
        node_hidden_dims: tuple[int, int],
        cfg: DictConfig,
        nonlinearities: tuple[str, str] = ("silu", "silu"),
    ):
        super().__init__()
        self.radial_embedding = CachedGaussianRBF(
            max_distance=cfg.r_max,
            num_rbf=cfg.num_rbf,
        )
        self.edge_normalization = GCPLayerNorm(edge_input_dims)
        self.node_normalization = GCPLayerNorm(node_input_dims)
        self.edge_embedding = GCP(
            edge_input_dims, edge_hidden_dims, nonlinearities=nonlinearities
        )
        self.node_embedding = GCP(
            node_input_dims, node_hidden_dims, nonlinearities=("none", "none")
        )

    def forward(self, batch: Data) -> tuple[ScalarVector, ScalarVector]:
        assert batch.pos is not None and batch.edge_index is not None
        assert batch.x is not None and batch.edge_attr is not None
        edge_vectors = (
            batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)  # [n_edges, 1]
        edge_rep = ScalarVector(
            torch.cat((batch.edge_attr, self.radial_embedding(edge_lengths)), dim=-1),
            batch.edge_vector_attr,
        )
        node_rep = ScalarVector(batch.x, batch.x_vector_attr)

        edge_rep = self.edge_normalization(edge_rep)
        node_rep = self.node_normalization(node_rep)
        edge_rep = self.edge_embedding(
            edge_rep, batch.edge_index, batch.f_ij, node_inputs=False
        )
        node_rep = self.node_embedding(
            node_rep, batch.edge_index, batch.f_ij, node_inputs=True
        )
        return node_rep, edge_rep


class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: tuple[int, int],
        output_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        cfg: DictConfig,
        num_message_layers: int,
    ):
        super().__init__()
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim
        nonlinearities = tuple(cfg.nonlinearities)

        # Outer layers use the default bottleneck, inner layers the configured one.
        module_list = [
            GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims,
                nonlinearities=nonlinearities,
                bottleneck=cfg.default_bottleneck,
            )
        ]
        for _ in range(num_message_layers - 2):
            module_list.append(
                GCP(
                    output_dims,
                    output_dims,
                    nonlinearities=nonlinearities,
                    bottleneck=cfg.bottleneck,
                )
            )
        module_list.append(
            GCP(
                output_dims,
                output_dims,
                nonlinearities=nonlinearities,
                bottleneck=cfg.default_bottleneck,
            )
        )
        self.message_fusion = nn.ModuleList(module_list)

        # learnable scalar message gating
        self.scalar_message_attention = nn.Sequential(
            nn.Linear(output_dims[0], 1), nn.Sigmoid()
        )

    def forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: torch.Tensor,
        frames: torch.Tensor,
    ) -> ScalarVector:
        row, col = edge_index
        scalar_inputs = torch.cat(
            (node_rep.scalar[row], edge_rep.scalar, node_rep.scalar[col]), dim=-1
        )
        vector_inputs = torch.cat(
            (node_rep.vector[row], edge_rep.vector, node_rep.vector[col]), dim=1
        )
        message = ScalarVector(scalar_inputs, vector_inputs)

        message_residual = self.message_fusion[0](
            message, edge_index, frames, node_inputs=False
        )
        for module in self.message_fusion[1:]:
            new_message = module(
                message_residual, edge_index, frames, node_inputs=False
            )
            message_residual = ScalarVector(
                message_residual.scalar + new_message.scalar,
                message_residual.vector + new_message.vector,
            )

        attention = self.scalar_message_attention(message_residual.scalar)
        message_residual = ScalarVector(
            message_residual.scalar * attention,
            message_residual.vector,
        )

        agg_scalar, agg_vector = self._sparse_reduce(
            row,
            message_residual.scalar,
            message_residual.vector,
            dim_size=node_rep.scalar.shape[0],
        )
        return ScalarVector(agg_scalar, agg_vector)

    def _sparse_reduce(
        self,
        row: torch.Tensor,
        scalar_message: torch.Tensor,
        vector_message: torch.Tensor,
        *,
        dim_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_edges = row.numel()
        device = row.device

        if num_edges == 0:
            return (
                scalar_message.new_zeros(dim_size, *scalar_message.shape[1:]),
                vector_message.new_zeros(
                    dim_size, vector_message.size(1), vector_message.size(2)
                ),
            )

        scalar_dtype = scalar_message.dtype
        autocast_context = (
            (lambda: torch.amp.autocast(device_type="cuda", enabled=False))
            if scalar_message.is_cuda
            else contextlib.nullcontext
        )

        counts = torch.bincount(row, minlength=dim_size)
        indptr = torch.zeros(dim_size + 1, device=device, dtype=torch.long)
        indptr[1:] = counts.cumsum(0)

        perm = torch.argsort(row)
        scalar_message = scalar_message.index_select(0, perm)
        vector_message = vector_message.index_select(0, perm)

        with autocast_context():
            aggregated_scalar_fp32 = torch.segment_reduce(
                scalar_message.float(),
                reduce="sum",
                offsets=indptr,
                axis=0,
            )
        aggregated_scalar = aggregated_scalar_fp32.to(scalar_dtype)

        vector_flat = vector_message.reshape(num_edges, -1)
        with autocast_context():
            aggregated_vector_flat_fp32 = torch.segment_reduce(
                vector_flat.float(),
                reduce="sum",
                offsets=indptr,
                axis=0,
            )
        aggregated_vector = aggregated_vector_flat_fp32.to(vector_message.dtype).view(
            dim_size, vector_message.size(1), vector_message.size(2)
        )
        return aggregated_scalar, aggregated_vector


class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        cfg: DictConfig,
        num_message_layers: int,
    ):
        super().__init__()
        self.interaction = GCPMessagePassing(
            node_dims,
            node_dims,
            edge_dims,
            cfg=cfg,
            num_message_layers=num_message_layers,
        )
        self.gcp_norm = nn.ModuleList([GCPLayerNorm(node_dims)])

        # Two-layer feedforward network, as in both pinned presets.
        hidden_dims = (4 * node_dims[0], 2 * node_dims[1])
        self.feedforward_network = nn.ModuleList(
            [
                GCP(
                    (node_dims[0] * 2, node_dims[1] * 2),
                    hidden_dims,
                    nonlinearities=tuple(cfg.nonlinearities),
                    bottleneck=cfg.bottleneck,
                ),
                GCP(
                    hidden_dims,
                    node_dims,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    bottleneck=cfg.bottleneck,
                ),
            ]
        )

    def forward(
        self,
        node_rep: ScalarVector | tuple[torch.Tensor, torch.Tensor],
        edge_rep: ScalarVector | tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        frames: torch.Tensor,
    ) -> ScalarVector:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        hidden_residual = self.interaction(node_rep, edge_rep, edge_index, frames)

        # aggregate input and hidden features
        hidden_residual = ScalarVector(
            torch.cat((hidden_residual.scalar, node_rep.scalar), dim=-1),
            torch.cat((hidden_residual.vector, node_rep.vector), dim=1),
        )
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual, edge_index, frames, node_inputs=True
            )

        # post-norm residual update
        return self.gcp_norm[0](node_rep + hidden_residual)
