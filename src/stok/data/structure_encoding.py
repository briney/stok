"""Shared structure preparation for the released GCP-VQVAE components.

Reference filling/correction adapted from Mahdi Pourmirzaei's MIT-licensed
vq_encoder_decoder at 68c4c284 (demo/dataset.py); see LICENSE.
The reference file route intentionally retains upstream numbering heuristics.
"""

import math
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, cast

from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
from Bio.PDB.Polypeptide import protein_letters_3to1
from graphein.protein.resi_atoms import STANDARD_AMINO_ACIDS
import torch
from torch_geometric.data import Batch, Data

from ..utils.structure_parser import PolymerStructure

if TYPE_CHECKING:
    from ..models.gcp_vqvae import GCPVQTokenizer

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


def impute_linear_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """Fill each missing atom along residue positions without moving observations."""
    if coordinates.ndim != 3 or coordinates.shape[1:] != (4, 3):
        raise ValueError("Coordinates must have shape [L,4,3]")
    filled = coordinates.clone()
    for atom in range(4):
        values = filled[:, atom]
        anchors = torch.where(torch.isfinite(values).all(-1))[0]
        if not anchors.numel():
            raise StructureExclusion("no_usable_anchors", f"Backbone atom {atom}")
        first, last = int(anchors[0]), int(anchors[-1])
        values[:first] = values[first]
        values[last + 1 :] = values[last]
        for left, right in zip(anchors[:-1].tolist(), anchors[1:].tolist()):
            weights = torch.arange(
                1, right - left, dtype=values.dtype, device=values.device
            )[:, None] / (right - left)
            values[left + 1 : right] = torch.lerp(values[left], values[right], weights)
    return filled


def _sequence_stencil_masks(
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require contiguous polymer positions throughout each geometric stencil."""
    rows = torch.arange(positions.numel(), device=positions.device)

    def contiguous(offsets):
        valid = torch.ones_like(positions, dtype=torch.bool)
        for offset in offsets:
            neighbors = rows + offset
            inside = (neighbors >= 0) & (neighbors < positions.numel())
            valid &= inside & (
                positions[neighbors.clamp(0, positions.numel() - 1)]
                == positions + offset
            )
        return valid

    before, after = contiguous([-1, 0]), contiguous([0, 1])
    scalars = torch.stack(
        [
            contiguous([-1, 0, 1, 2]),
            contiguous([-2, -1, 0, 1, 2]),
            before,
            after,
            after,
        ],
        dim=-1,
    )
    return scalars, torch.stack([after, before], dim=-1)


def validate_structure_limits(
    *,
    min_length: int = 25,
    max_length: int = 1280,
    max_missing_ratio: float | None = 0.2,
    max_missing_block: int | None = 15,
) -> None:
    """Validate admission settings within the supported 1280-position tensors."""
    if (
        type(min_length) is not int
        or type(max_length) is not int
        or not 4 <= min_length <= max_length <= 1280
    ):
        raise ValueError(
            "Structure policy lengths must satisfy 4 <= min_length <= max_length <= 1280"
        )
    if max_missing_ratio is not None and (
        type(max_missing_ratio) not in {int, float} or not 0 <= max_missing_ratio <= 1
    ):
        raise ValueError(
            "Structure policy max_missing_ratio must be null or a finite number in [0, 1]"
        )
    if max_missing_block is not None and (
        type(max_missing_block) is not int or max_missing_block < 0
    ):
        raise ValueError(
            "Structure policy max_missing_block must be null or a nonnegative integer"
        )


def _check_reference_coverage(
    coords: torch.Tensor,
    *,
    detail: str,
    min_length: int = 25,
    max_length: int = 1280,
    max_missing_ratio: float | None = 0.2,
    max_missing_block: int | None = 15,
) -> None:
    validate_structure_limits(
        min_length=min_length,
        max_length=max_length,
        max_missing_ratio=max_missing_ratio,
        max_missing_block=max_missing_block,
    )
    length = coords.size(0)
    if length < min_length or length > max_length:
        raise StructureExclusion(
            "chains_too_short" if length < min_length else "chains_too_long", detail
        )
    if max_missing_ratio is None and max_missing_block is None:
        return
    missing = ~torch.isfinite(coords[:, 1]).all(dim=-1)
    if max_missing_ratio is not None and missing.float().mean() > max_missing_ratio:
        raise StructureExclusion("missing_ratio_exceeded", detail)
    if max_missing_block is None:
        return
    longest = current = 0
    for flag in missing.tolist():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    if longest > max_missing_block:
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


def prepare_structure(
    structure: PolymerStructure,
    *,
    sequence_mode: Literal["native", "unknown", "polymer"],
    imputation: Literal["reference", "linear", "observed_only"],
    min_length: int = 25,
    max_length: int = 1280,
    max_missing_ratio: float | None = 0.2,
    max_missing_block: int | None = 15,
) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    """Prepare aligned observations; null limits disable optional coverage filters.

    Defaults preserve the reference experiments; dataset writers supply their
    explicit policy limits. Tensor padding remains fixed at 1280 positions.
    """
    if sequence_mode not in {"native", "unknown", "polymer"}:
        raise ValueError("sequence_mode must be native, unknown, or polymer")
    if imputation not in {"reference", "linear", "observed_only"}:
        raise ValueError("imputation must be reference, linear, or observed_only")
    required_metadata = {"path", "sha256", "sequence_source", "model_index"}
    metadata = structure.source
    if (
        not required_metadata <= metadata.keys()
        or not isinstance(metadata.get("path"), str)
        or not metadata["path"]
        or not isinstance(metadata.get("sha256"), str)
        or len(metadata["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in metadata["sha256"])
        or type(metadata.get("model_index")) is not int
        or metadata["model_index"] < 0
        or not isinstance(metadata.get("sequence_source"), str)
        or metadata["sequence_source"]
        not in {"entity_poly_seq", "poly_seq_scheme", "seqres", "supplied", "observed"}
    ):
        raise ValueError("Structure source metadata is incomplete")
    coordinates = torch.tensor(structure.coordinates)
    original_atoms = torch.tensor(structure.atom_mask)
    available = original_atoms.all(dim=-1)
    if not available.any():
        raise StructureExclusion("no_usable_structure", structure.sequence_id)
    propagated = coordinates.clone()
    propagated[~available] = float("nan")
    _check_reference_coverage(
        propagated,
        detail=structure.sequence_id,
        min_length=min_length,
        max_length=max_length,
        max_missing_ratio=max_missing_ratio,
        max_missing_block=max_missing_block,
    )
    positions = torch.arange(len(structure.sequence))
    if imputation == "reference":
        coordinates, _ = impute_reference_coordinates(propagated)
    elif imputation == "linear":
        coordinates = impute_linear_coordinates(coordinates)
    else:
        positions = torch.where(available)[0]
        coordinates = coordinates[positions]
    # Graphein's kappa feature requires at least four graph nodes.
    if coordinates.size(0) < 4:
        raise StructureExclusion("too_few_graph_nodes", structure.sequence_id)
    if not torch.isfinite(coordinates).all():
        raise StructureExclusion("imputation_nonfinite", structure.sequence_id)
    original = torch.tensor(structure.coordinates)[positions]
    displacement = (
        (coordinates - original).norm(dim=-1)[original_atoms[positions]].max()
    )
    coordinates = coordinates - coordinates.reshape(-1, 3).mean(dim=0)
    if sequence_mode == "unknown":
        identities = "X" * len(structure.sequence)
    elif sequence_mode == "polymer":
        identities = structure.sequence
    else:
        identities = "".join(
            row.get("observed_one_letter") or "X" for row in structure.residue_map
        )
    identities = "".join(identities[position] for position in positions.tolist())
    graph = build_structure_graph(coordinates, identities, positions=positions)
    graph.max_observed_atom_displacement = displacement[None]
    if imputation == "observed_only":
        graph.sequence_stencil_mask, graph.orientation_mask = _sequence_stencil_masks(
            positions
        )
    batch = batch_structure_graphs([graph])
    residues = torch.arange(1280)[None] < len(structure.sequence)
    tokens = torch.zeros_like(residues)
    tokens[0, : len(structure.sequence)] = available
    data = cast(Data, batch)
    data.atom_mask = torch.zeros((1, 1280, 4), dtype=torch.bool)
    data.atom_mask[0, : len(structure.sequence)] = original_atoms
    data.geometry_mask = data.atom_mask[:, :, :3].all(-1)
    data.graph_node_mask = (
        tokens.clone() if imputation == "observed_only" else residues.clone()
    )
    return batch, residues, tokens


@torch.inference_mode()
def tokenize_structures(
    tokenizer: "GCPVQTokenizer",
    structures: Sequence[PolymerStructure],
    *,
    sequence_mode: str,
    imputation: str,
    min_length: int = 25,
    max_length: int = 1280,
    max_missing_ratio: float | None = 0.2,
    max_missing_block: int | None = 15,
) -> list[torch.Tensor]:
    """Return aligned CPU IDs in order, using independent singleton chain context."""
    device = next(tokenizer.parameters()).device
    results = []
    # Mixed-length upstream batches alter terminal angles. Keep dataset identity
    # independent of its batching/sharding until true batched context is validated.
    for structure in structures:
        graph, residues, tokens = prepare_structure(
            structure,
            sequence_mode=cast(Any, sequence_mode),
            imputation=cast(Any, imputation),
            min_length=min_length,
            max_length=max_length,
            max_missing_ratio=max_missing_ratio,
            max_missing_block=max_missing_block,
        )
        cast(Data, graph).to(device)
        with torch.autocast(device.type, enabled=False):
            indices = tokenizer.encode(
                graph,
                residue_mask=residues.to(device),
                token_mask=tokens.to(device),
            )
        results.append(indices[0, : len(structure.sequence)].cpu())
    return results


def iter_structure_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    """Validate JSONL rows; resolve paths relative to the manifest, never the cwd."""
    path = Path(path).resolve()
    seen = set()
    fields = {
        "sequence_id",
        "path",
        "chain_id",
        "chain_namespace",
        "model_index",
        "sequence",
    }
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            context = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{context}: malformed JSON") from error
            if not isinstance(row, dict) or row.keys() - fields:
                raise ValueError(
                    f"{context}: expected an object with known manifest fields"
                )
            for field in ("sequence_id", "path"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{context}: {field} must be a nonempty string")
            if row["sequence_id"] in seen:
                raise ValueError(
                    f"{context}: duplicate sequence_id {row['sequence_id']}"
                )
            seen.add(row["sequence_id"])
            row.setdefault("chain_namespace", "author")
            row.setdefault("model_index", 0)
            if not isinstance(row["chain_namespace"], str) or row[
                "chain_namespace"
            ] not in {"author", "label"}:
                raise ValueError(f"{context}: invalid chain_namespace")
            if type(row["model_index"]) is not int or row["model_index"] < 0:
                raise ValueError(
                    f"{context}: model_index must be a nonnegative integer"
                )
            if row.get("chain_id") is not None and (
                not isinstance(row["chain_id"], str) or not row["chain_id"]
            ):
                raise ValueError(
                    f"{context}: chain_id must be a nonempty string or null"
                )
            if row.get("sequence") is not None and (
                not isinstance(row["sequence"], str)
                or not row["sequence"]
                or any(
                    letter not in "ACDEFGHIKLMNPQRSTVWYX" for letter in row["sequence"]
                )
            ):
                raise ValueError(
                    f"{context}: sequence must contain uppercase amino acids or X"
                )
            resolved = (path.parent / row["path"]).resolve()
            if not resolved.is_file():
                raise ValueError(f"{context}: structure file not found: {resolved}")
            row["path"] = str(resolved)
            yield row
