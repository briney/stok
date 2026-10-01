# this was drawn heavily from GCP-VQVAE (https://github.com/mahdip72/vq_encoder_decoder)
# which itself was heavily inspired by ProteinWorkshop (https://github.com/a-r-j/ProteinWorkshop)
# which is a re-implementation of the original GCPNet (https://github.com/BioinfoMachineLearning/GCPNet)

import torch
import torch.nn as nn
import torch.nn.functional as F
from graphein.protein.tensor.angles import alpha, dihedrals, kappa
from graphein.protein.tensor.data import get_random_batch
from torch_geometric.data import Data
from torch_geometric.nn.encoding import PositionalEncoding


class ProteinFeaturiser(nn.Module):
    """Protein featuriser for graph-based structural representations.

    Builds scalar and vector node features, ensures edges and their metadata
    are present, and computes scalar and vector edge features. By default,
    node positions are set to C-alpha coordinates extracted from
    ``batch.coords`` and residue identities are read from ``batch.residue_type``.

    Args:
        representation (str): Target structural representation. Currently, this
            module operates on C-alpha backbone coordinates ("CA" / "ca_bb").
        scalar_node_features (list[str]): Per-node scalar features to compute.
            Supported: "amino_acid_one_hot", "alpha", "kappa",
            "dihedrals", and the required "sequence_positional_encoding".
        vector_node_features (list[str]): Per-node vector features to compute.
            Supported: "orientation".
        edge_types (list[str]): Edge construction strategy identifiers. This
            helper assumes ``edge_index`` is provided upstream; values are kept
            as metadata only.
        scalar_edge_features (list[str]): Per-edge scalar features to compute.
            Supported: "edge_distance".
        vector_edge_features (list[str]): Per-edge vector features to compute.
            Supported: "edge_vectors".

    Note:
        If "sequence_positional_encoding" is included, a
        :class:`torch_geometric.nn.encoding.PositionalEncoding` module is
        created as ``self.positional_encoding`` and its 16 features are prepended
        to the scalar features using the explicit dense residue positions.
    """

    def __init__(
        self,
        representation: str = "ca_bb",
        scalar_node_features: list[str] = [
            "amino_acid_one_hot",
            "sequence_positional_encoding",
            "alpha",
            "kappa",
            "dihedrals",
        ],
        vector_node_features: list[str] = ["orientation"],
        edge_types: list[str] = ["knn_16"],
        scalar_edge_features: list[str] = ["edge_distance"],
        vector_edge_features: list[str] = ["edge_vectors"],
    ):
        super(ProteinFeaturiser, self).__init__()
        if "sequence_positional_encoding" not in scalar_node_features:
            raise ValueError(
                "The supported featurizer requires sequence positional encoding"
            )
        self.representation = representation
        self.scalar_node_features = scalar_node_features
        self.vector_node_features = vector_node_features
        self.edge_types = edge_types
        self.scalar_edge_features = scalar_edge_features
        self.vector_edge_features = vector_edge_features

        if "sequence_positional_encoding" in self.scalar_node_features:
            self.positional_encoding = PositionalEncoding(16)

    def forward(self, batch: Data) -> Data:
        """Compute features and edges for a protein batch.

        Given a batch with fields like ``coords``, ``residue_type`` and
        optionally ``edge_index``, this method:

        - computes scalar node features and writes them to ``x``;
        - sets ``pos`` to C-alpha coordinates from ``coords``;
        - optionally adds vector node features to ``x_vector_attr``;
        - ensures ``edge_index``/``edge_type`` exist (no new edges are built);
        - optionally computes ``edge_attr`` and ``edge_vector_attr``.

        Args:
            batch (Batch | ProteinBatch): Input batch. Must define ``coords``
                with shape (N, A, 3) and ``residue_type`` with shape (N,). If
                ``edge_index`` is present, it is used; otherwise it must be
                provided upstream.

        Returns:
            Batch | ProteinBatch: The same object with features populated.
        """
        # scalar node features
        positional_features = self.positional_encoding(batch.seq_pos)
        batch.x = positional_features
        scalar_features = compute_scalar_node_features(batch, self.scalar_node_features)
        batch.x = torch.nan_to_num(
            torch.cat([positional_features, scalar_features], dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # transform representation
        batch.pos = batch.coords[:, 1, :]

        # vector node features
        if self.vector_node_features:
            vector_node_features = [
                orientations(batch.coords, batch._slice_dict["coords"])
            ]
            batch.x_vector_attr = torch.cat(vector_node_features, dim=0)
            if getattr(batch, "orientation_mask", None) is not None:
                batch.x_vector_attr = (
                    batch.x_vector_attr * batch.orientation_mask[..., None]
                )

        # edges
        if self.edge_types:
            batch.edge_index, batch.edge_type = compute_edges(batch)
            batch.num_relation = len(self.edge_types)

        # scalar edge features
        if self.scalar_edge_features:
            assert batch.pos is not None and batch.edge_index is not None
            scalars = [_edge_distance(batch.pos, batch.edge_index)]
            batch.edge_attr = torch.cat(scalars, dim=1)

        # vector edge features
        if self.vector_edge_features:
            assert batch.pos is not None and batch.edge_index is not None
            diff = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
            vectors = [_normalize(diff).unsqueeze(-2)]
            batch.edge_vector_attr = torch.cat(vectors, dim=0)

        return batch

    def _example(self, batch_size: int = 2):
        """Featurise a randomly generated batch for quick inspection.

        Args:
            batch_size (int): Number of protein graphs to generate.

        Returns:
            ProteinBatch: Featurised synthetic batch.
        """
        batch = get_random_batch(batch_size)
        return self(batch)

    def __repr__(self) -> str:
        """Return a readable string summary of the configuration.

        Returns:
            str: Human-readable representation.
        """
        repr = f"ProteinFeaturiser(\n\trepresentation={self.representation},"
        repr += f"\n\tscalar_node_features={self.scalar_node_features},"
        repr += f"\n\tvector_node_features={self.vector_node_features}, "
        repr += f"\n\tedge_types={self.edge_types},"
        repr += f"\n\tscalar_edge_features={self.scalar_edge_features},"
        repr += f"\n\tvector_edge_features={self.vector_edge_features}\n)"
        return repr


def compute_edges(batch: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure edge metadata exists for a batch.

    This helper assumes ``edge_index`` has been computed upstream (e.g., k-NN).
    If ``edge_type`` is missing or empty, it is created as a zeros tensor that
    aligns with ``edge_index``.

    Args:
        batch (Batch): Graph batch containing ``edge_index`` and optionally
            ``edge_type``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of (``edge_index``,
        ``edge_type``).
    """
    edge_index = batch.edge_index
    assert edge_index is not None
    edge_type = getattr(batch, "edge_type", None)
    if edge_type is None or edge_type.numel() == 0:
        edge_type = torch.zeros(
            edge_index.size(1), dtype=torch.long, device=edge_index.device
        )
    return edge_index, edge_type


def compute_scalar_node_features(
    x: Data,
    node_features: list[str],
) -> torch.Tensor:
    """Factory for scalar node features.

    Builds and concatenates the requested per-node scalar features:

    - "amino_acid_one_hot": 23-D one-hot vectors from ``x.residue_type``
    - "alpha": alpha angle embeddings derived from ``x.coords``
    - "kappa": kappa angle embeddings derived from ``x.coords``
    - "dihedrals": phi/psi/omega dihedral embeddings derived from ``x.coords``

    The "sequence_positional_encoding" keyword is recognised but ignored in
    this function (it may be applied elsewhere in the model).

    Args:
        x (Data | Batch | Protein | ProteinBatch): Protein graph object with at
            least ``coords`` (N, A, 3), ``batch`` (N,) and ``residue_type``
            (N,) fields.
        node_features (list[str]): Names of features to compute.

    Returns:
        torch.Tensor: Concatenated feature tensor of shape (N, F). If no
        features are requested, returns ``x.x`` unchanged.

    Raises:
        ValueError: If an unsupported feature name is provided.
    """
    feats = []
    for feature in node_features:
        if feature == "amino_acid_one_hot":
            feats.append(F.one_hot(x.residue_type, num_classes=23).float())
        elif feature == "alpha":
            feats.append(alpha(x.coords, x.batch, rad=True, embed=True))
        elif feature == "kappa":
            feats.append(kappa(x.coords, x.batch, rad=True, embed=True))
        elif feature == "dihedrals":
            feats.append(dihedrals(x.coords, x.batch, rad=True, embed=True))
        elif feature == "sequence_positional_encoding":
            continue
        else:
            raise ValueError(f"Node feature {feature} not recognised.")
        stencil_mask = getattr(x, "sequence_stencil_mask", None)
        if stencil_mask is not None and feature in {"alpha", "kappa", "dihedrals"}:
            columns = {
                "alpha": slice(0, 1),
                "kappa": slice(1, 2),
                "dihedrals": slice(2, 5),
            }
            valid = stencil_mask[:, columns[feature]].repeat_interleave(2, dim=-1)
            feats[-1] = feats[-1] * valid
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]

    if feats:
        return torch.cat(feats, dim=1)
    assert x.x is not None
    return x.x


def orientations(
    X: torch.Tensor, coords_slice_index: torch.Tensor, ca_idx: int = 1
) -> torch.Tensor:
    """Compute forward and backward orientation unit vectors per residue.

    For each node, returns the normalized forward difference vector to the next
    residue and the backward difference vector to the previous residue. For the
    last (first) node of each sequence in the batch, the forward (backward)
    vector is zero.

    Args:
        X (CoordTensor | AtomTensor): Coordinates of shape (N, 3) or (N, A, 3).
            If 3D, the C-alpha slice is selected via ``ca_idx``.
        coords_slice_index (torch.Tensor): ``Batch._slice_dict['coords']``
            slice index to identify per-graph boundaries in the batch.
        ca_idx (int, optional): C-alpha index along the atom axis when ``X`` is
            3D. Defaults to 1.

    Returns:
        torch.Tensor: Tensor of shape (N, 2, 3) with forward and backward unit
        vectors per node.
    """
    if X.ndim == 3:
        X = X[:, ca_idx, :]

    # the first item in the coords_slice_index is always 0,
    # and the last item is always the node count of the batch
    # batch_num_nodes = X.shape[0]
    slice_index = coords_slice_index[1:] - 1
    last_node_index = slice_index[:-1]
    first_node_index = slice_index[:-1] + 1

    # all the last (first) nodes in a subgraph have their
    # forward (backward) vectors set to a padding value (i.e., 0.0)
    # to mimic feature construction behavior with single input graphs
    forward_slice = X[1:] - X[:-1]
    backward_slice = X[:-1] - X[1:]

    if forward_slice.numel() > 0 and last_node_index.numel() > 0:
        max_forward_idx = forward_slice.size(0) - 1
        # zero the forward vectors for last nodes in each subgraph without boolean masks (torch.compile friendly)
        valid_forward_idx = (
            last_node_index.clamp_min(0).clamp_max(max_forward_idx).to(X.device)
        )
        forward_slice.index_fill_(0, valid_forward_idx, 0.0)

    if backward_slice.numel() > 0 and first_node_index.numel() > 0:
        max_backward_idx = backward_slice.size(0) - 1
        # zero the backward vectors for first nodes in each subgraph
        valid_backward_idx = (
            (first_node_index - 1).clamp_min(0).clamp_max(max_backward_idx).to(X.device)
        )
        backward_slice.index_fill_(0, valid_backward_idx, 0.0)

    # padding first and last nodes with zero vectors does not impact feature normalization
    forward = _normalize(forward_slice)
    backward = _normalize(backward_slice)
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])

    return torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)


def _edge_distance(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute Euclidean distances for edges.

    Args:
        pos (CoordTensor): Node positions of shape (N, 3).
        edge_index (EdgeTensor): Edge indices of shape (2, E).

    Returns:
        torch.Tensor: Edge distances of shape (E, 1).
    """
    return torch.pairwise_distance(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(-1)


def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Safely normalize a tensor along a dimension.

    Computes ``tensor / ||tensor||`` along ``dim`` with ``keepdim=True`` and
    replaces NaNs/Infs with zeros.

    Args:
        tensor (torch.Tensor): Input tensor of any shape.
        dim (int, optional): Dimension along which to compute the norm.
            Defaults to -1.

    Returns:
        torch.Tensor: Normalized tensor with the same shape as ``tensor``.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )
