"""Shared structure preparation for the released GCP-VQVAE components.

Reference filling/correction adapted from Mahdi Pourmirzaei's MIT-licensed
vq_encoder_decoder at 68c4c284 (demo/dataset.py); see LICENSE.
The reference file route intentionally retains upstream numbering heuristics.
"""

import math
from pathlib import Path
from typing import cast

from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
from Bio.PDB.Polypeptide import protein_letters_3to1
from graphein.protein.resi_atoms import STANDARD_AMINO_ACIDS
import torch
from torch_geometric.data import Batch, Data

BOND_LENGTHS = {"N-CA": 1.458, "CA-C": 1.525, "C-O": 1.231, "C-N": 1.329}


class StructureExclusion(ValueError):
    """Expected source/policy rejection, with a stable reason for reports."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def enforce_backbone_bonds(
    coords: torch.Tensor, changed: torch.Tensor | None = None
) -> torch.Tensor:
    n_res = coords.size(0)
    for i in range(n_res):
        for a, b, key in ((0, 1, "N-CA"), (1, 2, "CA-C"), (2, 3, "C-O")):
            if changed is None or changed[i, a] or changed[i, b]:
                if torch.isnan(coords[i, b]).any():
                    if a > 0:
                        v = coords[i, a] - coords[i, a - 1]
                    else:
                        v = coords[i, a + 1] - coords[i, a]
                else:
                    v = coords[i, b] - coords[i, a]
                norm = v.norm(dim=-1, keepdim=True)
                if norm.item() > 1e-6:
                    coords[i, b] = coords[i, a] + v / norm * BOND_LENGTHS[key]
        if i < n_res - 1:
            if changed is None or changed[i, 2] or changed[i + 1, 0]:
                v = coords[i + 1, 0] - coords[i, 2]
                norm = v.norm(dim=-1, keepdim=True)
                if norm > 1e-6:
                    coords[i + 1, 0] = coords[i, 2] + v / norm * BOND_LENGTHS["C-N"]
    return coords


def enforce_ca_spacing(
    coords: torch.Tensor, changed: torch.Tensor | None = None, ideal: float = 3.8
) -> torch.Tensor:
    n_res = coords.size(0)
    if changed is not None:
        res_changed = changed.any(dim=1)
    else:
        res_changed = torch.ones(n_res, dtype=torch.bool, device=coords.device)

    for i in range(n_res - 1):
        if changed is not None and not res_changed[i] and not res_changed[i + 1]:
            continue
        ca_i = coords[i, 1]
        ca_j = coords[i + 1, 1]
        v = ca_j - ca_i
        d = v.norm()
        if d > 1e-6:
            delta = v * ((d - ideal) / d)
            if changed is not None and not res_changed[i]:
                coords[i + 1] = coords[i + 1] - delta
            elif changed is not None and not res_changed[i + 1]:
                coords[i] = coords[i] + delta
            else:
                coords[i] = coords[i] + delta / 2
                coords[i + 1] = coords[i + 1] - delta / 2
    return coords


def impute_reference_coordinates(coords: torch.Tensor):
    nan_mask = torch.isnan(coords).any(dim=-1)
    if not nan_mask.any():
        return coords, coords.new_ones(coords.size(0), dtype=torch.bool)

    coords = coords.clone()
    n_res, n_atoms, _ = coords.shape

    for atom in range(n_atoms):
        flat = coords[:, atom].clone()
        mask = nan_mask[:, atom]
        valid = torch.where(~mask)[0]
        if valid.numel() == 0:
            flat[:] = 0.0
        else:
            first, last = int(valid[0].item()), int(valid[-1].item())
            flat[:first] = flat[first]
            flat[last + 1 :] = flat[last]
            for a, b in zip(valid[:-1], valid[1:]):
                gap = b - a - 1
                if gap > 0:
                    v = flat[b] - flat[a]
                    dist = v.norm().item()
                    l_target = gap * 3.8
                    if l_target > dist:
                        u = v / dist
                        temp = torch.tensor([1, 0, 0], device=v.device, dtype=v.dtype)
                        if abs((u * temp).sum()) > 0.9:
                            temp = torch.tensor(
                                [0, 1, 0], device=v.device, dtype=v.dtype
                            )
                        n = torch.linalg.cross(u, temp)
                        n = n / n.norm()

                        h_squared = (l_target / math.pi) ** 2 - (dist / 2) ** 2
                        if h_squared < 0:
                            w = torch.linspace(
                                0, 1, gap + 2, device=flat.device, dtype=flat.dtype
                            )[1:-1].unsqueeze(1)
                            flat[a + 1 : b] = flat[a] * (1 - w) + flat[b] * w
                            continue

                        h = h_squared**0.5
                        for j in range(1, gap + 1):
                            theta = j * math.pi / (gap + 1)
                            d_j = dist * j / (gap + 1)
                            h_j = h * math.sin(theta)
                            flat[a + j] = flat[a] + d_j * u + h_j * n
                    else:
                        w = torch.linspace(
                            0, 1, gap + 2, device=flat.device, dtype=flat.dtype
                        )[1:-1].unsqueeze(1)
                        flat[a + 1 : b] = flat[a] * (1 - w) + flat[b] * w
        coords[:, atom] = flat

    coords = enforce_ca_spacing(coords, changed=nan_mask)
    coords = enforce_backbone_bonds(coords, changed=nan_mask)
    return coords, ~nan_mask.any(dim=1)


def _check_reference_coverage(coords: torch.Tensor, *, detail: str) -> None:
    length = coords.size(0)
    if length < 25 or length > 1280:
        raise StructureExclusion(
            "chains_too_short" if length < 25 else "chains_too_long", detail
        )
    missing = ~torch.isfinite(coords[:, 1]).all(dim=-1)
    if missing.float().mean() > 0.2:
        raise StructureExclusion("missing_ratio_exceeded", detail)
    longest = current = 0
    for flag in missing.tolist():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    if longest > 15:
        raise StructureExclusion("missing_block_exceeded", detail)


def _reference_rows(path: Path, chain_id: str) -> tuple[str, torch.Tensor]:
    parser = (
        MMCIFParser(QUIET=True, auth_chains=False)
        if path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure(path.stem, str(path))
    if chain_id not in structure[0]:
        raise StructureExclusion("chain_not_found", f"{path}, chain {chain_id}")
    chain = structure[0][chain_id]
    if sum(len(peptide) for peptide in PPBuilder().build_peptides(chain)) < 25:
        raise StructureExclusion("chains_too_short", f"{path}, chain {chain_id}")
    residues = [residue for residue in chain if residue.id[0] == " "]
    # Upstream replaces ASX/GLX/SEC/PYL with X before featurization.
    sequence = [protein_letters_3to1.get(residue.resname, "X") for residue in residues]
    coords = [
        [
            residue[atom].coord.tolist() if atom in residue else [float("nan")] * 3
            for atom in ("N", "CA", "C", "O")
        ]
        for residue in residues
    ]
    # Upstream demo heuristic only: this does not establish polymer correspondence.
    for i in range(len(residues) - 1, 0, -1):
        gap = residues[i].id[1] - residues[i - 1].id[1] - 1
        if gap > 5:
            before, after = coords[i - 1][1], coords[i][1]
            if all(math.isfinite(value) for value in (*before, *after)):
                distance = math.sqrt(
                    sum((float(b) - float(a)) ** 2 for a, b in zip(before, after))
                )
                gap = min(gap, max(0, math.floor((distance / 3.8) * 1.2) - 1))
            else:
                gap = 5
        if gap > 0:
            sequence[i:i] = ["X"] * gap
            coords[i:i] = [[[float("nan")] * 3 for _ in range(4)] for _ in range(gap)]
    tensor = torch.tensor(coords, dtype=torch.float32)
    tensor[~torch.isfinite(tensor).all(dim=(1, 2))] = float("nan")
    _check_reference_coverage(tensor, detail=f"{path}, chain {chain_id}")
    return "".join(sequence), tensor


def build_structure_graph(
    coordinates: torch.Tensor, sequence: str, *, positions: torch.Tensor | None = None
) -> Data:
    """Build native directed kNN edges: source neighbor → target, no self edges."""
    if (
        coordinates.shape != (len(sequence), 4, 3)
        or not torch.isfinite(coordinates).all()
    ):
        raise ValueError("Graph coordinates must be finite [N,4,3] matching sequence")
    length = len(sequence)
    if not length:
        raise StructureExclusion("no_usable_structure", "Graph has no nodes")
    if positions is None:
        positions = torch.arange(length, device=coordinates.device)
    if positions.shape != (length,) or positions.dtype != torch.int64:
        raise ValueError("Node positions must be long [N]")
    ca = coordinates[:, 1]
    # ponytail: quadratic distances, bounded by the released 1280-residue policy.
    distances = (ca[:, None] - ca[None]).square().sum(-1)
    distances.fill_diagonal_(float("inf"))
    k = min(16, length - 1)
    neighbors = distances.argsort(dim=-1, stable=True)[:, :k]
    targets = torch.arange(length, device=ca.device)[:, None].expand(-1, k)
    edges = torch.stack([neighbors.flatten(), targets.flatten()])
    atoms = coordinates.new_full((length, 37, 3), 1e-5)
    atoms[:, :3] = coordinates[:, :3]
    identities = torch.tensor(
        [STANDARD_AMINO_ACIDS.index(aa) for aa in sequence], device=ca.device
    )
    return Data(
        coords=atoms,
        residue_type=identities,
        seq_pos=positions[:, None],
        edge_index=edges,
        edge_type=torch.zeros(edges.size(1), dtype=torch.long, device=ca.device),
        num_nodes=length,
        prepared_coordinates=coordinates,
    )


def batch_structure_graphs(graphs: list[Data]) -> Batch:
    batch = Batch.from_data_list([graph for graph in graphs])
    data = cast(Data, batch)
    data.residue_index = data.seq_pos[:, 0].long()
    return batch


def prepare_reference_structure(
    path: str | Path, *, chain_id: str
) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    """Prepare one explicitly selected chain under the pinned observed-file policy."""
    sequence, parsed = _reference_rows(Path(path), chain_id)
    prepared, available = impute_reference_coordinates(parsed)
    prepared = prepared - prepared.reshape(-1, 3).mean(dim=0)
    graph = build_structure_graph(prepared, sequence)
    graph.parsed_coordinates = parsed
    batch = batch_structure_graphs([graph])
    residues = torch.arange(1280)[None] < len(sequence)
    tokens = torch.zeros_like(residues)
    tokens[0, : len(sequence)] = available
    return batch, residues, tokens
