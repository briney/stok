from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_cluster")

from experiments.gcp_mdlm.corpus.cif_source import accession_from_path  # noqa: E402
from experiments.gcp_mdlm.corpus.filters import CorpusFilters  # noqa: E402
from experiments.gcp_mdlm.corpus.tokenize import (  # noqa: E402
    StructureFeatureDataset,
    collate_featurized,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _paths():
    return sorted(str(p) for p in FIXTURES.glob("*.cif.gz"))


def test_dataset_item_accepted_shapes():
    ds = StructureFeatureDataset(_paths(), CorpusFilters(min_mean_plddt=0.0), max_length=1280)
    item = ds[0]
    assert item.sequence_id == accession_from_path(_paths()[0])
    if item.status == "accepted":
        assert item.sequence is not None and len(item.sequence) > 0
        assert item.mask.shape == (1280,) and item.nan_mask.shape == (1280,)


def test_collate_stacks_accepted():
    ds = StructureFeatureDataset(_paths(), CorpusFilters(min_mean_plddt=0.0), max_length=1280)
    items = [ds[i] for i in range(len(_paths()))]
    batch = collate_featurized(items)
    n_ok = sum(1 for it in items if it.status == "accepted")
    assert len(batch.metas) == n_ok
    if n_ok:
        assert batch.mask.shape == (n_ok, 1280)
        assert getattr(batch.graph, "features_precomputed", False) is True


def test_plddt_rejection_recorded():
    # impossible threshold -> every structure rejected_plddt, no featurization
    ds = StructureFeatureDataset(_paths(), CorpusFilters(min_mean_plddt=101.0), max_length=1280)
    items = [ds[i] for i in range(len(_paths()))]
    assert all(it.status == "rejected_plddt" for it in items)
    batch = collate_featurized(items)
    assert batch.graph is None and len(batch.outcomes) == len(items)
