"""Load PDB/mmCIF files into a :class:`torch_geometric.data.Batch` that the
:class:`~stok.models.structure_encoder.StructureEncoder` can consume.

Mirrors the batch construction in upstream GCP-VQVAE's
``custom_collate_pretrained_gcp`` so that the encoder sees the same layout:

- ``graph.coords``    : ``(N, num_atoms, 3)`` with N/CA/C in slots [0:3] and
                         the remaining atoms filled with a ``fill_value``
                         sentinel (the encoder only reads slots [0:3]).
- ``graph.x_bb``      : the N/CA/C subset used by kNN edge construction and
                         downstream Cα extraction.
- ``graph.residue_type``: flat ``(N,)`` integer amino-acid indices.
- ``graph.seq_pos``   : flat ``(N, 1)`` position indices.
- ``graph.edge_index``: kNN_16 edges built from Cα positions.
- ``graph._slice_dict['coords']`` mirrors ``_slice_dict['x_bb']`` so the
  featurizer's per-sample slicing works.

Also produces the ``(B, max_length)`` ``mask`` and ``nan_mask`` tensors the
encoder's forward pass takes as key-padding inputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from graphein.protein.resi_atoms import (
    PROTEIN_ATOMS,
    STANDARD_AMINO_ACID_MAPPING_1_TO_3,
    STANDARD_AMINO_ACIDS,
)
from torch_geometric.data import Batch, Data

from stok.utils.featurizer import ProteinFeaturiser
from stok.utils.gcp_vqvae_preprocessing import (
    GCPVQVAEStructureSample,
    parse_gcp_vqvae_samples,
    prepare_gcp_vqvae_sample,
)
from stok.utils.structure_parser import StructureData

__all__ = [
    "LoadedStructures",
    "NoAcceptedStructuresError",
    "load_structures",
    "structures_to_batch",
]


class NoAcceptedStructuresError(ValueError):
    """Raised when upstream-compatible filtering accepts no input samples."""


class LoadedStructures(NamedTuple):
    """Payload returned by :func:`load_structures`.

    Attributes:
        graph: PyG ``Batch`` whose attributes match what
            :class:`stok.utils.featurizer.ProteinFeaturiser` and
            :class:`stok.models.gcpnet.GCPNetModel` expect.
        mask: ``(B, max_length)`` bool key-padding mask (True = real residue).
        nan_mask: ``(B, max_length)`` bool mask (True = Cα coords finite).
        pids: Per-sample identifiers (filename stem by default).
        sequences: Per-sample one-letter amino-acid sequences (trimmed to
            ``max_length`` if longer).
    """

    graph: Batch
    mask: torch.Tensor
    nan_mask: torch.Tensor
    pids: list[str]
    sequences: list[str]


def load_structures(
    paths: list[Path | str] | Path | str,
    *,
    max_length: int = 1280,
    k: int = 16,
    fill_value: float = 1e-5,
    device: torch.device | str | None = None,
) -> LoadedStructures:
    """Parse and collate one or more PDB/mmCIF files for encoder input.

    Args:
        paths: File path(s) or a single directory (recursively scanned for
            ``.pdb`` / ``.ent`` / ``.cif`` / ``.mmcif``).
        max_length: Fixed padded length per sample (trims longer inputs,
            pads shorter). Must match the encoder's ``max_length``.
        k: kNN neighbor count for graph edges (upstream uses 16).
        fill_value: Sentinel for non-N/CA/C atom slots in ``graph.coords``.
            Must match upstream's ``custom_collate_pretrained_gcp`` default.
        device: Optional target device. If provided, all tensors are moved
            there before returning.

    Returns:
        :class:`LoadedStructures` ready to feed to
        :meth:`stok.models.structure_encoder.StructureEncoder.forward`.
    """
    resolved = _resolve_input_paths(paths)
    if not resolved:
        raise ValueError(f"No structure files found at: {paths}")

    samples = [
        sample
        for file_index, path in enumerate(resolved)
        for sample in parse_gcp_vqvae_samples(
            path,
            file_index=file_index,
            max_length=max_length,
        )
    ]
    if not samples:
        raise NoAcceptedStructuresError(f"No accepted structures found at: {paths}")

    return _gcp_vqvae_samples_to_batch(
        samples,
        max_length=max_length,
        k=k,
        fill_value=fill_value,
        device=device,
    )


def _gcp_vqvae_samples_to_batch(
    samples: list[GCPVQVAEStructureSample],
    *,
    max_length: int,
    k: int,
    fill_value: float,
    device: torch.device | str | None,
) -> LoadedStructures:
    """Prepare and featurize accepted GCP-VQVAE samples on CPU."""
    prepared_samples = [
        prepare_gcp_vqvae_sample(sample, max_length=max_length) for sample in samples
    ]

    data_list: list[Data] = []
    masks: list[torch.Tensor] = []
    nan_masks: list[torch.Tensor] = []
    for sample in prepared_samples:
        length = len(sample.sequence)
        x_bb = sample.coords[:, :3]
        node_scalar_features = _upstream_dihedrals(sample.coords)
        node_vector_features = torch.cat(
            [
                _upstream_orientations(sample.coords[:, 1]),
                _upstream_sidechains(sample.coords).unsqueeze(-2),
            ],
            dim=-2,
        )
        node_scalar_features, node_vector_features = map(
            torch.nan_to_num,
            (node_scalar_features, node_vector_features),
        )
        data_list.append(
            Data(
                x=x_bb[:, 1],
                x_bb=x_bb,
                seq=torch.tensor(
                    [_upstream_sequence_index(residue) for residue in sample.sequence],
                    dtype=torch.long,
                ),
                name=sample.pid,
                h=node_scalar_features,
                chi=node_vector_features,
                mask=torch.isfinite(sample.coords.sum(dim=(1, 2))),
            )
        )

        mask = torch.zeros(max_length, dtype=torch.bool)
        mask[:length] = True
        masks.append(mask)

        nan_mask = torch.zeros(max_length, dtype=torch.bool)
        nan_mask[:length] = sample.nan_mask
        nan_masks.append(nan_mask)

    batch: Batch = Batch.from_data_list(data_list)
    batch.edge_index = _knn_graph(
        batch.x_bb[:, 1].contiguous(),
        k=k,
        batch_index=batch.batch,
    )
    batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long)
    batch.edge_index_precomputed = True

    sequences = [sample.sequence for sample in prepared_samples]
    pids = [sample.pid for sample in prepared_samples]
    batch.fill_value = torch.full((batch.num_graphs,), fill_value)
    batch.atom_list = [PROTEIN_ATOMS for _ in range(batch.num_graphs)]
    batch.id = pids
    batch.residue_id = [
        [
            f"A:{STANDARD_AMINO_ACID_MAPPING_1_TO_3[residue]}:{index}"
            for index, residue in enumerate(sequence, start=1)
        ]
        for sequence in sequences
    ]
    batch.residue_type = torch.cat(
        [
            torch.tensor([STANDARD_AMINO_ACIDS.index(residue) for residue in sequence])
            for sequence in sequences
        ]
    )
    batch.residues = [
        [STANDARD_AMINO_ACID_MAPPING_1_TO_3[residue] for residue in sequence]
        for sequence in sequences
    ]
    batch.chains = torch.zeros_like(batch.batch)

    batch.coords = torch.full(
        (batch.num_nodes, len(PROTEIN_ATOMS), 3),
        fill_value,
        dtype=torch.float32,
    )
    batch.coords[:, :3] = batch.x_bb.float()
    batch._slice_dict["coords"] = batch._slice_dict["x_bb"]
    batch.seq_pos = torch.cat([torch.arange(len(sequence)).unsqueeze(1) for sequence in sequences])

    with torch.inference_mode():
        batch = ProteinFeaturiser().eval()(batch)
    batch.features_precomputed = True

    dev = torch.device(device) if device is not None else torch.device("cpu")
    batch = batch.to(dev)
    mask = torch.stack(masks).to(dev)
    nan_mask = torch.stack(nan_masks).to(dev)
    return LoadedStructures(
        graph=batch,
        mask=mask,
        nan_mask=nan_mask,
        pids=pids,
        sequences=sequences,
    )


def structures_to_batch(
    structures: list[StructureData],
    *,
    max_length: int = 1280,
    k: int = 16,
    fill_value: float = 1e-5,
    device: torch.device | str | None = None,
) -> LoadedStructures:
    """Turn pre-parsed :class:`StructureData` items into a batched graph.

    Public for callers that already have parsed structures in hand (e.g.,
    when streaming from a dataset rather than individual files).
    """
    if not structures:
        raise ValueError("structures list cannot be empty")

    dev = torch.device(device) if device is not None else torch.device("cpu")

    data_list: list[Data] = []
    pids: list[str] = []
    sequences: list[str] = []
    masks: list[torch.Tensor] = []
    nan_masks: list[torch.Tensor] = []

    num_protein_atoms = len(PROTEIN_ATOMS)

    for s in structures:
        seq = s.protein_sequence
        coords_np = s.coords  # (L_raw, 3, 3) float32, NaN for missing atoms

        # Trim to max_length (same as upstream).
        if len(seq) > max_length:
            seq = seq[:max_length]
            coords_np = coords_np[:max_length]
        length = len(seq)

        coords_bb = torch.as_tensor(coords_np, dtype=torch.float32)  # (L, 3, 3)

        # Per-residue validity: True when all N/CA/C atoms are finite.
        finite_per_residue = torch.isfinite(coords_bb).all(dim=(1, 2))  # (L,)

        # Recenter around per-sample finite centroid (mirrors upstream
        # ``recenter_coordinates`` but only over finite atoms to avoid
        # NaN poisoning).
        if finite_per_residue.any():
            centroid = coords_bb[finite_per_residue].reshape(-1, 3).mean(dim=0)
            coords_bb = coords_bb - centroid

        # Replace any remaining NaNs in the raw coords with ``fill_value`` so
        # ``torch_geometric.Batch`` doesn't propagate them into graph ops.
        # The nan_mask retains the original finiteness for downstream use.
        coords_bb_clean = torch.nan_to_num(
            coords_bb, nan=fill_value, posinf=fill_value, neginf=fill_value
        )

        # Pad non-backbone atom slots with fill_value (matches upstream
        # collate which reserves len(PROTEIN_ATOMS) channels but only uses
        # the first three for N/CA/C).
        coords_full = torch.full(
            (length, num_protein_atoms, 3),
            fill_value,
            dtype=torch.float32,
        )
        coords_full[:, :3, :] = coords_bb_clean

        residue_type = torch.tensor([_aa_to_index(ch) for ch in seq], dtype=torch.long)
        seq_pos = torch.arange(length, dtype=torch.long).unsqueeze(1)

        # Per-sample Data carrying the raw backbone and the padded 14-atom
        # coords; both will survive through ``Batch.from_data_list``.
        d = Data(
            x_bb=coords_bb_clean,
            coords=coords_full,
            residue_type=residue_type,
            seq_pos=seq_pos,
            num_nodes=length,
        )
        data_list.append(d)
        pids.append(s.pid)
        sequences.append(seq)

        m = torch.zeros(max_length, dtype=torch.bool)
        m[:length] = True
        masks.append(m)

        nm = torch.zeros(max_length, dtype=torch.bool)
        nm[:length] = finite_per_residue
        nan_masks.append(nm)

    batch: Batch = Batch.from_data_list(data_list)

    # ``Batch.from_data_list`` will have built a ``_slice_dict`` for every
    # attribute we passed (``x_bb``, ``coords``, ``residue_type``, ``seq_pos``).
    # The featurizer slices ``coords`` per graph via
    # ``batch._slice_dict['coords']`` — so mirror it from x_bb for safety in
    # case graphein's ``ProteinBatch`` semantics differ. PyG already fills
    # both, but being explicit keeps the contract visible.
    if "coords" not in batch._slice_dict:
        batch._slice_dict["coords"] = batch._slice_dict["x_bb"]

    # Build kNN_16 edges from Cα positions.
    ca = batch.x_bb[:, 1].contiguous()
    batch.edge_index = _knn_graph(ca, k=k, batch_index=batch.batch)
    batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long)
    batch.num_relation = 1

    mask = torch.stack(masks, dim=0)
    nan_mask = torch.stack(nan_masks, dim=0)

    batch = batch.to(dev)
    mask = mask.to(dev)
    nan_mask = nan_mask.to(dev)

    return LoadedStructures(
        graph=batch,
        mask=mask,
        nan_mask=nan_mask,
        pids=pids,
        sequences=sequences,
    )


def _resolve_input_paths(paths: list[Path | str] | Path | str) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    resolved: list[Path] = []
    suffixes = {".pdb", ".ent", ".cif", ".mmcif"}
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in suffixes:
                    resolved.append(f)
        elif p.is_file():
            resolved.append(p)
        else:
            raise FileNotFoundError(f"Structure input not found: {p}")
    return resolved


def _aa_to_index(ch: str) -> int:
    """Map a one-letter amino-acid code to its ``STANDARD_AMINO_ACIDS`` index.

    Matches upstream's ``STANDARD_AMINO_ACIDS.index(res)`` call. Unknown
    residues fall back to ``"X"`` (graphein reserves that slot for unknowns),
    and if ``"X"`` is absent for some reason, to ``"A"``.
    """
    try:
        return STANDARD_AMINO_ACIDS.index(ch)
    except ValueError:
        for fallback in ("X", "A"):
            try:
                return STANDARD_AMINO_ACIDS.index(fallback)
            except ValueError:
                continue
        return 0


def _upstream_sequence_index(residue: str) -> int:
    """Return the categorical index used by upstream graph samples."""
    return "ARNDCQEGHILKMFPSTWYVX".index(residue)


def _normalize_graph_vector(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    return torch.nan_to_num(tensor / torch.norm(tensor, dim=dim, keepdim=True).clamp_min(eps))


def _upstream_dihedrals(coords: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    flattened = torch.reshape(coords[:, :3], [3 * coords.shape[0], 3])
    unit = _normalize_graph_vector(flattened[1:] - flattened[:-1], dim=-1)
    unit_2 = unit[:-2]
    unit_1 = unit[1:-1]
    unit_0 = unit[2:]

    normal_2 = _normalize_graph_vector(torch.linalg.cross(unit_2, unit_1), dim=-1)
    normal_1 = _normalize_graph_vector(torch.linalg.cross(unit_1, unit_0), dim=-1)
    cosine = torch.sum(normal_2 * normal_1, -1).clamp(-1 + eps, 1 - eps)
    angles = torch.sign(torch.sum(unit_2 * normal_1, -1)) * torch.acos(cosine)
    angles = torch.reshape(F.pad(angles, [1, 2]), [-1, 3])
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)


def _upstream_orientations(ca_coords: torch.Tensor) -> torch.Tensor:
    forward = _normalize_graph_vector(ca_coords[1:] - ca_coords[:-1])
    backward = _normalize_graph_vector(ca_coords[:-1] - ca_coords[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], dim=-2)


def _upstream_sidechains(coords: torch.Tensor) -> torch.Tensor:
    n_coords, origin, c_coords = coords[:, 0], coords[:, 1], coords[:, 2]
    c_vector = _normalize_graph_vector(c_coords - origin)
    n_vector = _normalize_graph_vector(n_coords - origin)
    bisector = _normalize_graph_vector(c_vector + n_vector)
    perpendicular = _normalize_graph_vector(torch.linalg.cross(c_vector, n_vector))
    return -bisector * math.sqrt(1 / 3) - perpendicular * math.sqrt(2 / 3)


def _knn_graph(
    x: torch.Tensor,
    k: int,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch equivalent of ``torch_cluster.knn_graph(loop=False)``.

    Builds directed edges ``(source, target)`` where each residue points to
    its ``k`` nearest neighbors (by Euclidean distance) within the same
    sub-batch. Matches upstream kNN_16 semantics; tie-breaking on exactly
    equal distances follows ``torch.topk`` order, which is effectively
    never observed for real Cα coordinates.

    Args:
        x: ``(N, 3)`` Cα positions concatenated across the batch.
        k: Neighbor count (upstream: 16).
        batch_index: ``(N,)`` graph-assignment tensor.

    Returns:
        Edge index of shape ``(2, E)`` where ``E = sum_i L_i * min(k, L_i - 1)``.
        Row 0 is source (the neighbor), row 1 is target (the query residue),
        matching upstream ``flow='source_to_target'``.
    """
    device = x.device
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for b in batch_index.unique():
        idx = torch.where(batch_index == b)[0]
        n = idx.numel()
        if n < 2:
            continue
        xs = x[idx]
        dists = torch.cdist(xs, xs)
        dists.fill_diagonal_(float("inf"))
        eff_k = min(k, n - 1)
        _, nn_local = dists.topk(eff_k, largest=False, dim=1)  # (n, eff_k)
        nn_global = idx[nn_local]  # (n, eff_k)
        tgt = idx.unsqueeze(1).expand(-1, eff_k)
        sources.append(nn_global.reshape(-1))
        targets.append(tgt.reshape(-1))
    if not sources:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.stack([torch.cat(sources), torch.cat(targets)], dim=0).to(
        device=device, dtype=torch.long
    )
