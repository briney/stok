"""Phase-1 featurization: per-structure (B=1) parse+featurize, then collate for batched forward.

Each dataset item featurizes exactly ONE structure via
``stok.utils.structure_loader.load_structures([one_path])`` (never batches multiple
structures through the featurizer -- mixed-length batches make graphein produce different
per-node features for the shorter protein than it would alone). ``collate_featurized`` then
stacks the independently-featurized single-structure graphs into one PyG ``Batch`` for a
later batched GPU forward.

PyG round-trip note: ``load_structures`` builds its returned ``graph`` by assigning most
attributes (``pos``, ``x_vector_attr``, ``edge_index``, ``edge_attr``, ``edge_vector_attr``,
``coords``, ``residue_type``, ``seq_pos``, ``chains``, ``fill_value``, ``atom_list``, ``id``,
``residue_id``, ``residues``, ``features_precomputed``, ``edge_index_precomputed``) directly
onto the already-constructed ``Batch`` object rather than passing them through
``Batch.from_data_list``. Those attributes are therefore absent from the batch's
``_slice_dict``/``_inc_dict``, so ``Batch.to_data_list()`` either drops them silently or
raises (``coords`` is registered in ``_slice_dict`` but not ``_inc_dict``, so
``to_data_list()`` raises ``KeyError`` outright). Feeding the resulting single-graph
``Batch`` objects straight into ``Batch.from_data_list`` for re-collation is equally broken:
PyG's default ``__inc__`` increments any attribute whose name contains ``"index"``, so the
boolean flag ``edge_index_precomputed`` gets treated as an edge-index-like attribute and
``value.add_(incs)`` fails with a bool/long dtype mismatch.

``_extract_single_data`` below works around both problems by manually copying every
attribute off the (always num_graphs == 1) loaded batch into a plain ``Data`` -- skipping the
batch-bookkeeping keys and the two "*_precomputed" flags/`num_relation` (reset once on the
merged batch in ``collate_featurized`` instead of being carried, and re-collated, per item)
and unwrapping the length-1 list attributes (`id`, `atom_list`, `residue_id`, `residues`).
The resulting ``Data`` objects round-trip cleanly through the standard
``Batch.from_data_list`` path used by ``collate_featurized``. Verified against the real
fixtures: every derived feature (``pos``, ``x_vector_attr``, ``edge_attr``,
``edge_vector_attr``, ``coords``, ``residue_type``, ``seq_pos``, ``chains``) survives with
values identical to the un-merged single-structure graph, and ``edge_index`` is correctly
offset per subgraph after merging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from stok.utils.structure_loader import NoAcceptedStructuresError, load_structures

from .cif_source import accession_from_path, decompressed_cif
from .filters import CorpusFilters, classify, mean_plddt_from_cif

# Attributes that only make sense on a multi-graph Batch (bookkeeping) or that are reset
# once on the merged batch in `collate_featurized` rather than carried per item.
_BATCH_ONLY_KEYS = frozenset(
    {"batch", "ptr", "edge_index_precomputed", "features_precomputed", "num_relation"}
)
# List-valued attributes that `load_structures` stores as a length-1 list (one entry per
# graph) when num_graphs == 1; unwrap to the bare value so re-collation across many items
# reconstructs the length-N list the same way `load_structures` would for N paths at once.
_UNWRAP_SINGLETON_LIST_KEYS = frozenset({"id", "atom_list", "residue_id", "residues"})


@dataclass
class StructureItem:
    sequence_id: str
    status: str
    mean_plddt: float
    sequence: str | None = None
    data: object | None = None
    mask: torch.Tensor | None = None
    nan_mask: torch.Tensor | None = None


@dataclass
class StructureOutcome:
    sequence_id: str
    status: str


@dataclass
class CollatedBatch:
    graph: object | None
    mask: torch.Tensor | None
    nan_mask: torch.Tensor | None
    metas: list[tuple[str, str, float]] = field(default_factory=list)
    outcomes: list[StructureOutcome] = field(default_factory=list)


def _extract_single_data(graph: Batch) -> Data:
    """Flatten a ``load_structures`` num_graphs==1 batch into a plain, re-collatable ``Data``.

    See module docstring for why this can't simply be ``graph.to_data_list()[0]``.
    """
    kwargs = {}
    for key in graph.keys():
        if key in _BATCH_ONLY_KEYS:
            continue
        value = getattr(graph, key)
        if key in _UNWRAP_SINGLETON_LIST_KEYS:
            value = value[0]
        kwargs[key] = value
    kwargs["num_nodes"] = graph.num_nodes
    return Data(**kwargs)


class StructureFeatureDataset(Dataset):
    """Featurizes one structure per item (B=1) so batching never contaminates features."""

    def __init__(self, paths: list[str], filters: CorpusFilters, *, max_length: int = 1280) -> None:
        self.paths = list(paths)
        self.filters = filters
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> StructureItem:
        path = self.paths[index]
        sid = accession_from_path(path)
        try:
            with decompressed_cif(path) as cif:
                plddt = mean_plddt_from_cif(cif)
                if classify(mean_plddt=plddt, filters=self.filters) == "rejected_plddt":
                    return StructureItem(sid, "rejected_plddt", plddt)
                loaded = load_structures([str(cif)], max_length=self.max_length, device="cpu")
        except NoAcceptedStructuresError:
            return StructureItem(sid, "rejected_parser", float("nan"))
        except Exception:  # noqa: BLE001 - any parse failure is a recorded per-file outcome
            return StructureItem(sid, "parse_error", float("nan"))

        data = _extract_single_data(loaded.graph)
        return StructureItem(
            sequence_id=sid,
            status="accepted",
            mean_plddt=plddt,
            sequence=loaded.sequences[0],
            data=data,
            mask=loaded.mask[0],
            nan_mask=loaded.nan_mask[0],
        )


def collate_featurized(items: list[StructureItem]) -> CollatedBatch:
    """Stack accepted pre-featurized graphs into a batch; collect rejected outcomes."""
    accepted = [it for it in items if it.status == "accepted"]
    outcomes = [
        StructureOutcome(it.sequence_id, it.status) for it in items if it.status != "accepted"
    ]
    if not accepted:
        return CollatedBatch(None, None, None, [], outcomes)
    graph = Batch.from_data_list([it.data for it in accepted])
    graph.features_precomputed = True  # encoder must skip re-featurization
    graph.edge_index_precomputed = True
    graph.num_relation = 1
    mask = torch.stack([it.mask for it in accepted])
    nan_mask = torch.stack([it.nan_mask for it in accepted])
    metas = [(it.sequence_id, it.sequence, it.mean_plddt) for it in accepted]
    return CollatedBatch(graph, mask, nan_mask, metas, outcomes)
