# this was drawn heavily from GCP-VQVAE (https://github.com/mahdip72/vq_encoder_decoder)
# which itself was heavily inspired by ProteinWorkshop (https://github.com/a-r-j/ProteinWorkshop)
# which is a re-implementation of the original GCPNet (https://github.com/BioinfoMachineLearning/GCPNet)

import contextlib
from copy import copy
from functools import partial
from typing import Any

import torch
import torch.nn as nn

# import torch_scatter
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

from ..utils.gcp import (
    CachedGaussianRBF,
    EncoderOutput,
    ScalarVector,
    centralize,
    decentralize,
    get_activations,
    get_aggregation,
    is_identity,
    localize,
    safe_norm,
    scatter_reduce,
)


class GCPNetModel(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 5,
        node_s_emb_dim: int = 128,
        node_v_emb_dim: int = 16,
        edge_s_emb_dim: int = 32,
        edge_v_emb_dim: int = 4,
        r_max: float = 10.0,
        num_rbf: int = 8,
        activation: str = "silu",
        pool: str = "sum",
        # Note: Each of the arguments above are stored in the corresponding `kwargs` configs below
        # They are simply listed here to highlight key available arguments
        **kwargs,
    ):
        """
        Initializes an instance of the GCPNetModel class with the provided
        parameters.
        Note: Each of the model's keyword arguments listed here
        are also referenced in the corresponding `DictConfigs` within `kwargs`.
        They are simply listed here to highlight some of the key arguments available.
        See `models/gcpnet/config/encoder/gcpnet.yaml` for a full list of all available arguments.

        :param num_layers: Number of layers in the model (default: ``5``)
        :type num_layers: int
        :param node_s_emb_dim: Dimension of the node state embeddings (default: ``128``)
        :type node_s_emb_dim: int
        :param node_v_emb_dim: Dimension of the node vector embeddings (default: ``16``)
        :type node_v_emb_dim: int
        :param edge_s_emb_dim: Dimension of the edge state embeddings
            (default: ``32``)
        :type edge_s_emb_dim: int
        :param edge_v_emb_dim: Dimension of the edge vector embeddings
            (default: ``4``)
        :type edge_v_emb_dim: int
        :param r_max: Maximum distance for radial basis functions
            (default: ``10.0``)
        :type r_max: float
        :param num_rbf: Number of radial basis functions (default: ``8``)
        :type num_rbf: int
        :param activation: Activation function to use in each GCP layer (default: ``silu``)
        :type activation: str
        :param pool: Global pooling method to be used
            (default: ``"sum"``)
        :type pool: str
        :param kwargs: Primary model arguments in the form of the
            `DictConfig`s `module_cfg`, `model_cfg`, and `layer_cfg`, respectively
        :type kwargs: dict
        """
        super().__init__()

        assert all(
            [cfg in kwargs for cfg in ["module_cfg", "model_cfg", "layer_cfg"]]
        ), "All required GCPNet `DictConfig`s must be provided."
        module_cfg = kwargs["module_cfg"]
        model_cfg = kwargs["model_cfg"]
        layer_cfg = kwargs["layer_cfg"]

        self.predict_node_pos = module_cfg.predict_node_positions
        self.predict_node_rep = module_cfg.predict_node_rep

        # Feature dimensionalities
        edge_input_dims = (model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = (model_cfg.h_input_dim, model_cfg.chi_input_dim)
        self.edge_dims = (model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = (model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)

        # Position-wise operations
        self.centralize = partial(centralize, key="pos")
        self.localize = partial(localize, norm_pos_diff=module_cfg.norm_pos_diff)
        self.decentralize = partial(decentralize, key="pos")
        self._frame_update_eps = 1e-6

        # Input embeddings
        self.gcp_embedding = GCPEmbedding(
            edge_input_dims,
            node_input_dims,
            self.edge_dims,
            self.node_dims,
            cfg=module_cfg,
        )

        # Message-passing layers
        self.interaction_layers = nn.ModuleList(
            GCPInteractions(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.num_layers)
        )

        if self.predict_node_rep:
            # predictions
            self.invariant_node_projection = nn.ModuleList(
                [
                    GCPLayerNorm(self.node_dims),
                    GCP(
                        # Note: `GCPNet` defaults to providing SE(3) equivariance
                        # It is possible to provide E(3) equivariance by instead setting `module_cfg.enable_e3_equivariance=true`
                        self.node_dims,
                        (self.node_dims[0], 0),
                        nonlinearities=tuple(module_cfg.nonlinearities),
                        scalar_gate=module_cfg.scalar_gate,
                        vector_gate=module_cfg.vector_gate,
                        enable_e3_equivariance=module_cfg.enable_e3_equivariance,
                        node_inputs=True,
                    ),
                ]
            )

        # Global pooling/readout function
        self.readout = get_aggregation(
            module_cfg.pool
        )  # {"mean": global_mean_pool, "sum": global_add_pool}[pool]

    @property
    def required_batch_attributes(self) -> list[str]:
        return ["edge_index", "pos", "x", "batch"]

    def _ensure_edge_frames(
        self,
        batch: Data,
        *,
        force: bool = False,
        pos_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reuse cached edge-local frames when node positions are unchanged."""

        assert batch.edge_index is not None
        if pos_override is not None:
            frames = self.localize(pos_override, batch.edge_index)
            batch.f_ij = frames
            return frames

        pos = batch.pos
        assert pos is not None
        edge_index = batch.edge_index
        num_edges = edge_index.size(1)

        cached_pos = getattr(batch, "_f_ij_cache_pos", None)
        cached_frames = getattr(batch, "f_ij", None)

        # Always recompute if cache is missing, stale, or forced.
        need_full_recompute = (
            force
            or cached_pos is None
            or cached_frames is None
            or cached_frames.size(0) != num_edges
            or cached_pos.shape != pos.shape
            or cached_pos.device != pos.device
        )

        detached_pos = pos.detach()

        if need_full_recompute:
            frames = self.localize(pos, edge_index)
            batch.f_ij = frames
            batch._f_ij_cache_pos = detached_pos.clone()
            return frames

        # Identify nodes whose positions have changed beyond tolerance.
        pos_delta = torch.max(torch.abs(detached_pos - cached_pos), dim=1).values
        changed_nodes = pos_delta > self._frame_update_eps

        if changed_nodes.any():
            row, col = edge_index
            edge_mask = changed_nodes[row] | changed_nodes[col]

            if edge_mask.any():
                updated_edges = edge_index[:, edge_mask]
                updated_frames = self.localize(pos, updated_edges)

                # Clone only when an in-place update is required.
                frames = cached_frames.clone()
                frames[edge_mask] = updated_frames
                batch.f_ij = frames
            else:
                frames = cached_frames
        else:
            frames = cached_frames

        # Refresh cached positions in-place to avoid reallocations.
        cached_pos.copy_(detached_pos)
        return frames

    def forward(self, batch: Data) -> EncoderOutput:
        """Implements the forward pass of the GCPNet encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        assert batch.batch is not None
        # Centralize node positions to make them translation-invariant
        pos_centroid, batch.pos = self.centralize(batch, batch_index=batch.batch)

        # Install `h`, `chi`, `e`, and `xi` using corresponding features built by the `FeatureFactory`
        batch.h, batch.chi, batch.e, batch.xi = (
            batch.x,
            batch.x_vector_attr,
            batch.edge_attr,
            batch.edge_vector_attr,
        )

        # Craft complete local frames corresponding to each edge, reusing cached values when possible
        batch.f_ij = self._ensure_edge_frames(batch)

        # Embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # Update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi), batch.pos = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_pos=batch.pos,
            )

        # Record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # initialize encoder outputs
        encoder_outputs = {}

        # when updating node positions, decentralize updated positions to make their updates translation-equivariant
        if self.predict_node_pos:
            batch.pos = self.decentralize(
                batch, batch_index=batch.batch, entities_centroid=pos_centroid
            )
            if self.predict_node_rep:
                # prior to scalar node predictions, re-derive local frames after performing all node position updates
                _, centralized_node_pos = self.centralize(
                    batch, batch_index=batch.batch
                )
                batch.f_ij = self._ensure_edge_frames(
                    batch, force=True, pos_override=centralized_node_pos
                )
            assert batch.pos is not None
            encoder_outputs["pos"] = batch.pos  # (n, 3) -> (batch_size, 3)

        # Summarize intermediate node representations as final predictions
        out = h
        if self.predict_node_rep:
            out = self.invariant_node_projection[0](
                ScalarVector(h, chi)
            )  # e.g., GCPLayerNorm()
            out = self.invariant_node_projection[1](
                out, batch.edge_index, batch.f_ij, node_inputs=True
            )  # e.g., GCP((h, chi)) -> h'

        encoder_outputs["node_embedding"] = out
        encoder_outputs["graph_embedding"] = self.readout(
            out, batch
        )  # (n, d) -> (batch_size, d)
        return EncoderOutput(encoder_outputs)


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
        scalar_gate: int = 0,
        vector_gate: bool = True,
        feedforward_out: bool = False,
        bottleneck: int = 1,
        scalarization_vectorization_output_dim: int = 3,
        enable_e3_equivariance: bool = False,
        **kwargs,
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = ("none", "none")

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_activations(nonlinearities[0], return_functional=True),
            get_activations(nonlinearities[1], return_functional=True),
        )
        self.scalar_gate, self.vector_gate = scalar_gate, vector_gate
        self.enable_e3_equivariance = enable_e3_equivariance

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert self.vector_input_dim % bottleneck == 0, (
                f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"
            )

            self.hidden_dim = (
                self.vector_input_dim // bottleneck
                if bottleneck > 1
                else max(self.vector_input_dim, self.vector_output_dim)
            )

            scalar_vector_frame_dim = scalarization_vectorization_output_dim * 3
            self.vector_down = nn.Linear(
                self.vector_input_dim, self.hidden_dim, bias=False
            )
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(
                        self.hidden_dim
                        + self.scalar_input_dim
                        + scalar_vector_frame_dim,
                        self.scalar_output_dim,
                    ),
                    get_activations(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(
                    self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim,
                    self.scalar_output_dim,
                )
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
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(
                        self.scalar_output_dim, self.vector_output_dim
                    )
        else:
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                    get_activations(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)
            )

    def create_zero_vector(
        self,
        scalar_rep: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(
            scalar_rep.shape[0],
            self.vector_output_dim,
            3,
            device=scalar_rep.device,
        )

    @staticmethod
    def scalarize(
        vector_rep: torch.Tensor,
        edge_index: torch.Tensor,
        frames: torch.Tensor,
        node_inputs: bool,
        enable_e3_equivariance: bool,
        dim_size: int,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]

        # gather source node features for each `entity` (i.e., node or edge)
        # note: edge inputs are already ordered according to source nodes
        vector_rep_i = vector_rep[row] if node_inputs else vector_rep

        # project equivariant values onto corresponding local frames
        if vector_rep_i.ndim == 2:
            vector_rep_i = vector_rep_i.unsqueeze(-1)
        elif vector_rep_i.ndim == 3:
            vector_rep_i = vector_rep_i.transpose(-1, -2)

        if node_mask is not None:
            edge_mask = node_mask[row] & node_mask[col]
            mask = edge_mask.to(frames.dtype).view(-1, 1, 1)
            matmul_result = torch.matmul(frames, vector_rep_i)
            matmul_result = matmul_result * mask
        else:
            matmul_result = torch.matmul(frames, vector_rep_i)

        local_scalar_rep_i = matmul_result.transpose(-1, -2)

        # potentially enable E(3)-equivariance and, thereby, chirality-invariance
        if enable_e3_equivariance:
            # avoid corrupting gradients with an in-place operation
            local_scalar_rep_i_copy = local_scalar_rep_i.clone()
            local_scalar_rep_i_copy[:, :, 1] = torch.abs(local_scalar_rep_i[:, :, 1])
            local_scalar_rep_i = local_scalar_rep_i_copy

        # reshape frame-derived geometric scalars
        local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

        if node_inputs:
            # for node inputs, summarize all edge-wise geometric scalars using an average
            # return torch_scatter.scatter(
            #     local_scalar_rep_i,
            #     # summarize according to source node indices due to the directional nature of GCP's equivariant frames
            #     row,
            #     dim=0,
            #     dim_size=dim_size,
            #     reduce="mean",
            # )
            return scatter_reduce(
                local_scalar_rep_i,
                # summarize according to source node indices due to the directional nature of GCP's equivariant frames
                row,
                dim=0,
                dim_size=dim_size,
                reduce="mean",
            )

        return local_scalar_rep_i

    def vectorize(
        self,
        scalar_rep: torch.Tensor,
        vector_hidden_rep: torch.Tensor,
    ) -> torch.Tensor:
        vector_rep = self.vector_up(vector_hidden_rep)
        vector_rep = vector_rep.transpose(-1, -2)

        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(vector_rep, dim=-1, keepdim=True)
            )

        return vector_rep

    def forward(
        self,
        s_maybe_v: ScalarVector | tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        edge_index: torch.Tensor,
        frames: torch.Tensor,
        node_inputs: bool = False,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | ScalarVector:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            # curate direction-robust and (by default) chirality-aware scalar geometric features
            vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
            scalar_hidden_rep = self.scalarize(
                vector_down_frames_hidden_rep.transpose(-1, -2),
                edge_index,
                frames,
                node_inputs=node_inputs,
                enable_e3_equivariance=self.enable_e3_equivariance,
                dim_size=vector_down_frames_hidden_rep.shape[0],
                node_mask=node_mask,
            )
            merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        else:
            # update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.vectorize(scalar_rep, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        return ScalarVector(scalar_rep, vector_rep)


class VectorDropout(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class GCPDropout(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float, use_gcp_dropout: bool = True):
        super().__init__()
        self.scalar_dropout = (
            nn.Dropout(drop_rate) if use_gcp_dropout else nn.Identity()
        )
        self.vector_dropout = (
            VectorDropout(drop_rate) if use_gcp_dropout else nn.Identity()
        )

    def forward(self, x: torch.Tensor | ScalarVector) -> torch.Tensor | ScalarVector:
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (
            x.scalar.shape[0] == 0 or x.vector.shape[0] == 0
        ):
            return x
        elif isinstance(x, torch.Tensor):
            return self.scalar_dropout(x)
        return ScalarVector(self.scalar_dropout(x[0]), self.vector_dropout(x[1]))


class GCPLayerNorm(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(
        self, dims: tuple[int, int], eps: float = 1e-8, use_gcp_norm: bool = True
    ):
        super().__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = (
            nn.LayerNorm(self.scalar_dims) if use_gcp_norm else nn.Identity()
        )
        self.use_gcp_norm = use_gcp_norm
        self.eps = eps

    @staticmethod
    def norm_vector(
        v: torch.Tensor, use_gcp_norm: bool = True, eps: float = 1e-8
    ) -> torch.Tensor:
        v_norm = v
        if use_gcp_norm:
            vector_norm = torch.clamp(
                torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps
            )
            vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
            v_norm = v / vector_norm
        return v_norm

    def forward(self, x: torch.Tensor | ScalarVector) -> torch.Tensor | ScalarVector:
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (
            x.scalar.shape[0] == 0 or x.vector.shape[0] == 0
        ):
            return x
        elif not self.vector_dims:
            return self.scalar_norm(x)
        s, v = x
        return ScalarVector(
            self.scalar_norm(s),
            self.norm_vector(v, use_gcp_norm=self.use_gcp_norm, eps=self.eps),
        )


class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: tuple[int, int],
        node_input_dims: tuple[int, int],
        edge_hidden_dims: tuple[int, int],
        node_hidden_dims: tuple[int, int],
        num_atom_types: int = 0,
        nonlinearities: tuple[str, str] = ("silu", "silu"),
        cfg: DictConfig | None = None,
        pre_norm: bool = True,
        use_gcp_norm: bool = True,
    ):
        super().__init__()

        if num_atom_types > 0:
            self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        else:
            self.atom_embedding = None

        assert cfg is not None
        self.radial_embedding = CachedGaussianRBF(
            max_distance=cfg.r_max,
            num_rbf=cfg.num_rbf,
        )

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_normalization = GCPLayerNorm(
                edge_input_dims, use_gcp_norm=use_gcp_norm
            )
            self.node_normalization = GCPLayerNorm(
                node_input_dims, use_gcp_norm=use_gcp_norm
            )
        else:
            self.edge_normalization = GCPLayerNorm(
                edge_hidden_dims, use_gcp_norm=use_gcp_norm
            )
            self.node_normalization = GCPLayerNorm(
                node_hidden_dims, use_gcp_norm=use_gcp_norm
            )

        self.edge_embedding = GCP(
            edge_input_dims,
            edge_hidden_dims,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

        self.node_embedding = GCP(
            node_input_dims,
            node_hidden_dims,
            nonlinearities=("none", "none"),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

    def forward(
        self, batch: Data
    ) -> tuple[
        ScalarVector | torch.Tensor,
        ScalarVector | torch.Tensor,
    ]:
        if self.atom_embedding is not None:
            node_rep = ScalarVector(self.atom_embedding(batch.h), batch.chi)
        else:
            node_rep = ScalarVector(batch.h, batch.chi)

        edge_rep = ScalarVector(batch.e, batch.xi)

        assert batch.pos is not None and batch.edge_index is not None
        edge_vectors = (
            batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)  # [n_edges, 1]
        edge_rep = ScalarVector(
            torch.cat((edge_rep.scalar, self.radial_embedding(edge_lengths)), dim=-1),
            edge_rep.vector,
        )

        edge_rep = (
            edge_rep.scalar if not self.edge_embedding.vector_input_dim else edge_rep
        )
        node_rep = (
            node_rep.scalar if not self.node_embedding.vector_input_dim else node_rep
        )

        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None),
        )
        node_rep = self.node_embedding(
            node_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=getattr(batch, "mask", None),
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        return node_rep, edge_rep


class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: tuple[int, int],
        output_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        cfg: DictConfig,
        mp_cfg: DictConfig,
        reduce_function: str = "sum",
        use_scalar_message_attention: bool = True,
    ):
        super().__init__()

        # hyperparameters
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.conv_cfg = mp_cfg
        self.self_message = self.conv_cfg.self_message
        self.reduce_function = reduce_function
        self.use_scalar_message_attention = use_scalar_message_attention

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        # config instantiations
        soft_cfg = copy(cfg)
        soft_cfg.bottleneck = cfg.default_bottleneck

        primary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=soft_cfg)
        secondary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=cfg)

        # PyTorch modules #
        module_list = [
            primary_cfg_GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims,
                nonlinearities=cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
        ]

        for _ in range(self.conv_cfg.num_message_layers - 2):
            module_list.append(
                secondary_cfg_GCP(
                    output_dims,
                    output_dims,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        if self.conv_cfg.num_message_layers > 1:
            module_list.append(
                primary_cfg_GCP(
                    output_dims,
                    output_dims,
                    nonlinearities=cfg.nonlinearities,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.message_fusion = nn.ModuleList(module_list)

        # learnable scalar message gating
        if use_scalar_message_attention:
            self.scalar_message_attention = nn.Sequential(
                nn.Linear(output_dims[0], 1), nn.Sigmoid()
            )

    def forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: torch.Tensor,
        frames: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> ScalarVector:
        return self._fused_message_pass(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )

    def _fused_message_pass(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: torch.Tensor,
        frames: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> ScalarVector:
        row, col = edge_index

        scalar_src = node_rep.scalar[row]
        scalar_dst = node_rep.scalar[col]

        vector_src = node_rep.vector[row]
        vector_dst = node_rep.vector[col]

        scalar_inputs = torch.cat((scalar_src, edge_rep.scalar, scalar_dst), dim=-1)

        if node_rep.vector.shape[1] == 0 and edge_rep.vector.shape[1] == 0:
            vector_inputs = edge_rep.vector
        else:
            vector_inputs = torch.cat((vector_src, edge_rep.vector, vector_dst), dim=1)

        message = ScalarVector(scalar_inputs, vector_inputs)

        message_residual = self.message_fusion[0](
            message, edge_index, frames, node_inputs=False, node_mask=node_mask
        )
        for module in self.message_fusion[1:]:
            new_message = module(
                message_residual,
                edge_index,
                frames,
                node_inputs=False,
                node_mask=node_mask,
            )
            message_residual = ScalarVector(
                message_residual.scalar + new_message.scalar,
                message_residual.vector + new_message.vector,
            )

        if self.use_scalar_message_attention:
            attention = self.scalar_message_attention(message_residual.scalar)
            message_residual = ScalarVector(
                message_residual.scalar * attention,
                message_residual.vector,
            )

        agg_scalar, agg_vector = self._sparse_reduce(
            row,
            col,
            message_residual.scalar,
            message_residual.vector,
            dim_size=node_rep.scalar.shape[0],
            node_mask=node_mask,
        )

        return ScalarVector(agg_scalar, agg_vector)

    def _sparse_reduce(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        scalar_message: torch.Tensor,
        vector_message: torch.Tensor,
        *,
        dim_size: int,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_edges = row.numel()
        device = row.device

        if num_edges == 0:
            aggregated_scalar = scalar_message.new_zeros(
                dim_size, *scalar_message.shape[1:]
            )
            if vector_message.numel():
                aggregated_vector = vector_message.new_zeros(
                    dim_size, vector_message.size(1), vector_message.size(2)
                )
            else:
                aggregated_vector = vector_message.new_zeros(dim_size, 0, 3)
            if node_mask is not None:
                mask = node_mask.to(aggregated_scalar.dtype).unsqueeze(-1)
                aggregated_scalar = aggregated_scalar * mask
                if aggregated_vector.numel():
                    aggregated_vector = aggregated_vector * mask.unsqueeze(-1)
            return aggregated_scalar, aggregated_vector

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
        if vector_message.numel():
            vector_message = vector_message.index_select(0, perm)

        with autocast_context():
            scalar_fp32 = scalar_message.float()
            aggregated_scalar_fp32 = torch.segment_reduce(
                scalar_fp32,
                reduce="sum",
                offsets=indptr,
                axis=0,
            )

        aggregated_scalar = aggregated_scalar_fp32.to(scalar_dtype)

        if vector_message.numel():
            vector_flat = vector_message.reshape(num_edges, -1)
            with autocast_context():
                vector_fp32 = vector_flat.float()
                aggregated_vector_flat_fp32 = torch.segment_reduce(
                    vector_fp32,
                    reduce="sum",
                    offsets=indptr,
                    axis=0,
                )
            aggregated_vector_flat = aggregated_vector_flat_fp32.to(
                vector_message.dtype
            )
            aggregated_vector = aggregated_vector_flat.view(
                dim_size, vector_message.size(1), vector_message.size(2)
            )
        else:
            aggregated_vector = vector_message.new_zeros(dim_size, 0, 3)

        if node_mask is not None:
            mask = node_mask.to(aggregated_scalar.dtype).unsqueeze(-1)
            aggregated_scalar = aggregated_scalar * mask
            if aggregated_vector.numel():
                aggregated_vector = aggregated_vector * mask.unsqueeze(-1)

        return aggregated_scalar, aggregated_vector


class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        cfg: DictConfig,
        layer_cfg: DictConfig,
        dropout: float = 0.0,
        nonlinearities: tuple[str, str] | None = None,
    ):
        super().__init__()

        # hyperparameters
        if nonlinearities is None:
            nonlinearities = cfg.nonlinearities
        self.pre_norm = layer_cfg.pre_norm
        self.predict_node_positions = getattr(cfg, "predict_node_positions", False)
        self.node_positions_weight = getattr(cfg, "node_positions_weight", 1.0)
        self.update_positions_with_vector_sum = getattr(
            cfg, "update_positions_with_vector_sum", False
        )
        reduce_function = "sum"

        # geometry-complete message-passing neural network
        message_function = GCPMessagePassing

        self.interaction = message_function(
            node_dims,
            node_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg,
            reduce_function=reduce_function,
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention,
        )

        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_cfg)

        self.gcp_norm = nn.ModuleList(
            [GCPLayerNorm(node_dims, use_gcp_norm=layer_cfg.use_gcp_norm)]
        )
        self.gcp_dropout = nn.ModuleList(
            [GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout)]
        )

        # build out feedforward network modules
        hidden_dims = (
            (node_dims[0], node_dims[1])
            if layer_cfg.num_feedforward_layers == 1
            else (4 * node_dims[0], 2 * node_dims[1])
        )
        ff_interaction_layers = [
            ff_GCP(
                (node_dims[0] * 2, node_dims[1] * 2),
                hidden_dims,
                nonlinearities=("none", "none")
                if layer_cfg.num_feedforward_layers == 1
                else cfg.nonlinearities,
                feedforward_out=layer_cfg.num_feedforward_layers == 1,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
        ]

        interaction_layers = [
            ff_GCP(
                hidden_dims,
                hidden_dims,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_GCP(
                    hidden_dims,
                    node_dims,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # potentially build out node position update modules
        if self.predict_node_positions:
            # node position update GCP(s)
            position_output_dims = (
                node_dims
                if getattr(cfg, "update_positions_with_vector_sum", False)
                else (node_dims[0], 1)
            )
            self.node_position_update_gcp = ff_GCP(
                node_dims,
                position_output_dims,
                nonlinearities=cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )

    def derive_x_update(
        self,
        node_rep: ScalarVector,
        edge_index: torch.Tensor,
        f_ij: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # use vector-valued features to derive node position updates
        node_rep_update = self.node_position_update_gcp(
            node_rep, edge_index, f_ij, node_inputs=True, node_mask=node_mask
        )
        if self.update_positions_with_vector_sum:
            x_vector_update = node_rep_update.vector.sum(1)
        else:
            x_vector_update = node_rep_update.vector.squeeze(1)

        # (up/down) weight position updates
        x_update = x_vector_update * self.node_positions_weight

        return x_update

    def forward(
        self,
        node_rep: ScalarVector | tuple[torch.Tensor, torch.Tensor],
        edge_rep: ScalarVector | tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        frames: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        node_pos: torch.Tensor | None = None,
    ) -> tuple[ScalarVector, torch.Tensor | None]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # forward propagate with interaction module
        hidden_residual = self.interaction(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )

        # aggregate input and hidden features
        hidden_residual = ScalarVector(
            torch.cat((hidden_residual.scalar, node_rep.scalar), dim=-1),
            torch.cat((hidden_residual.vector, node_rep.vector), dim=1),
        )

        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                edge_index,
                frames,
                node_inputs=True,
                node_mask=node_mask,
            )

        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout[0](hidden_residual)

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask.float())

        # bypass updating node positions
        if not self.predict_node_positions:
            return node_rep, node_pos

        # update node positions
        assert node_pos is not None
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames, node_mask=node_mask
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        return node_rep, node_pos


# ------------
#   helpers
# ------------


def _cfg_to_dict(cfg: Any) -> Any:
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        return copy(OmegaConf.to_container(cfg, throw_on_missing=True))
    if isinstance(cfg, dict):
        return {key: _cfg_to_dict(value) for key, value in cfg.items()}
    if hasattr(cfg, "__dict__"):
        return {key: _cfg_to_dict(value) for key, value in vars(cfg).items()}
    if isinstance(cfg, list):
        return [_cfg_to_dict(item) for item in cfg]
    return cfg


def get_GCP_with_custom_cfg(input_dims: Any, output_dims: Any, cfg: Any, **kwargs):
    cfg_dict = _cfg_to_dict(cfg)
    if not isinstance(cfg_dict, dict):  # pragma: no cover - defensive guard
        raise TypeError("GCP configuration must be a mapping.")

    cfg_dict = copy(cfg_dict)
    cfg_dict.pop("scalar_nonlinearity", None)
    cfg_dict.pop("vector_nonlinearity", None)

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return GCP(input_dims, output_dims, **cfg_dict)
