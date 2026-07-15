"""End-to-end corpus build on the bundled fixtures (slow; encoder-gated)."""

import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch_cluster")

from experiments.gcp_mdlm.corpus.cluster_split import assign_splits
from experiments.gcp_mdlm.corpus.filters import CorpusFilters
from experiments.gcp_mdlm.corpus.partition import partition
from experiments.gcp_mdlm.corpus.run_tokenize import write_shard
from experiments.gcp_mdlm.corpus.tokenize import tokenize_paths

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.slow
def test_end_to_end_corpus_build(tmp_path):
    from stok.models.structure_encoder import load_pretrained_encoder

    try:
        encoder = load_pretrained_encoder(
            "base", path=os.environ.get("STOK_ENCODER_CHECKPOINT"), device="cpu", freeze=True
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"encoder weights unavailable: {exc}")

    paths = sorted(str(p) for p in FIXTURES.glob("*.cif.gz"))
    # Phase 1
    rows, outcomes = tokenize_paths(
        paths, encoder, CorpusFilters(min_mean_plddt=0.0), batch_size=2, num_workers=0, device="cpu"
    )
    assert rows
    for r in rows:
        assert r.length == len(r.sequence) == len(r.structure_tokens)
    staging = tmp_path / "_staging"
    write_shard(rows, outcomes, staging, 0)
    # Phase 2 (fake single-member clusters so the split logic runs without mmseqs)
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("".join(f"{r.sequence_id}\t{r.sequence_id}\n" for r in rows))
    split = assign_splits(tsv, val_size=1, test_size=1, seed=0)
    # Phase 3
    counts = partition(staging, split, tmp_path / "out", rows_per_shard=5000)
    assert counts["dropped"] == 0
    assert sum(counts[s] for s in ("train", "val", "test")) == len(rows)
    for s in ("train", "val", "test"):
        if counts[s]:
            df = pd.read_parquet(tmp_path / "out" / s)
            assert list(df.columns) == [
                "sequence_id",
                "sequence",
                "structure_tokens",
                "length",
                "mean_plddt",
            ]
