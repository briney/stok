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

    # Accept-all fixtures must actually produce accepted structures, or the assertions
    # below (which need a real merged batch) would vacuously pass.
    assert n_ok > 0
    graph = batch.graph
    assert graph.num_graphs == n_ok

    # These four fields are what stok.models.structure_encoder's features_precomputed
    # path requires -- it raises ValueError if any is missing.
    for feature_name in ("x", "x_vector_attr", "edge_attr", "edge_vector_attr"):
        assert hasattr(graph, feature_name), f"merged batch missing {feature_name!r}"

    # Per-subgraph edge_index confinement: every edge sourced from subgraph i must have
    # both endpoints inside [ptr[i], ptr[i + 1]) -- proves edge_index was re-offset with
    # no cross-graph bleed during collation.
    ptr = graph.ptr
    edge_index = graph.edge_index
    for i in range(graph.num_graphs):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        src_in_graph = (edge_index[0] >= lo) & (edge_index[0] < hi)
        assert src_in_graph.any(), f"graph {i} has no edges"
        endpoints = edge_index[:, src_in_graph]
        assert endpoints.min() >= lo
        assert endpoints.max() < hi

    # Per-node value equality for graph 0: a single-item batch built from just the first
    # accepted path must match, bit-for-bit, the corresponding node-range slice of the
    # merged batch -- this is the "bit-for-bit" claim that was previously only checked
    # manually.
    first_accepted_index = next(i for i, it in enumerate(items) if it.status == "accepted")
    single_batch = collate_featurized([ds[first_accepted_index]])
    lo, hi = int(ptr[0]), int(ptr[1])
    assert torch.equal(single_batch.graph.x, graph.x[lo:hi])
    assert torch.equal(single_batch.graph.pos, graph.pos[lo:hi])


def test_plddt_rejection_recorded():
    # impossible threshold -> every structure rejected_plddt, no featurization
    ds = StructureFeatureDataset(_paths(), CorpusFilters(min_mean_plddt=101.0), max_length=1280)
    items = [ds[i] for i in range(len(_paths()))]
    assert all(it.status == "rejected_plddt" for it in items)
    batch = collate_featurized(items)
    assert batch.graph is None and len(batch.outcomes) == len(items)


def _load_encoder():
    import os

    from stok.models.structure_encoder import load_pretrained_encoder

    try:
        return load_pretrained_encoder(
            "base", path=os.environ.get("STOK_ENCODER_CHECKPOINT"), device="cpu", freeze=True
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"encoder weights unavailable: {exc}")


@pytest.mark.slow
def test_tokenize_rows_aligned_and_in_range():
    from experiments.gcp_mdlm.corpus.tokenize import tokenize_paths

    encoder = _load_encoder()
    rows, outcomes = tokenize_paths(
        _paths(),
        encoder,
        CorpusFilters(min_mean_plddt=0.0),
        batch_size=4,
        num_workers=0,
        device="cpu",
    )
    assert rows, "no structures tokenized"
    for r in rows:
        assert r.length == len(r.sequence) == len(r.structure_tokens)
        assert all(0 <= t < 4096 for t in r.structure_tokens)


@pytest.mark.slow
def test_batched_matches_b1_parity():
    from experiments.gcp_mdlm.corpus.tokenize import tokenize_paths

    encoder = _load_encoder()
    f = CorpusFilters(min_mean_plddt=0.0)
    batched, _ = tokenize_paths(
        _paths(), encoder, f, batch_size=4, num_workers=0, device="cpu", batch_forward=True
    )
    one_at_a_time, _ = tokenize_paths(
        _paths(), encoder, f, batch_size=1, num_workers=0, device="cpu", batch_forward=False
    )
    bx = {r.sequence_id: r.structure_tokens for r in batched}
    for r in one_at_a_time:
        assert bx[r.sequence_id] == r.structure_tokens, f"batched tokens differ for {r.sequence_id}"
