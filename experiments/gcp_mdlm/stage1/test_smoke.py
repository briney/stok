"""End-to-end Stage 1 smoke test on a tiny real-schema sample (marked slow)."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("tokenizers")

from experiments.gcp_mdlm.stage1.evaluate import evaluate_arm
from experiments.gcp_mdlm.stage1.features import CachedFeatures, write_feature_cache
from experiments.gcp_mdlm.stage1.heads import (
    FrequencyBaseline,
    IndependentClassifier,
    build_prototype_head,
)
from experiments.gcp_mdlm.stage1.promote import assert_promotion, build_report
from experiments.gcp_mdlm.stage1.train_head import train_head
from stok.data.paired_records import load_paired_records
from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.utils.tokenizer import Tokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "sample_corpus.parquet"
C = 64  # tiny codebook classes


@pytest.mark.slow
def test_stage1_end_to_end(tmp_path):
    torch.manual_seed(0)
    tokenizer = Tokenizer()
    ns = NoiseSchedule(schedule_type="cosine")
    backbone = MDLMModel(
        tracks="seq_only",
        seq_vocab_size=tokenizer.vocab_size,
        seq_pad_id=tokenizer.pad_token_id,
        seq_mask_id=tokenizer.mask_token_id,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns,
        time_conditioning="adaln",
    ).eval()
    codebook = torch.randn(C, 8)

    records = load_paired_records(FIXTURE)
    assert all(len(r.sequence) == len(r.structure_tokens) for r in records)

    def encode_fn(seq_tokens, kpm):
        return backbone.encode_features(seq_tokens, key_padding_mask=kpm)

    write_feature_cache(tmp_path, records, encode_fn, tokenizer, d_model=32, batch_size=4)
    cache = CachedFeatures.load(tmp_path)
    assert cache.features.shape[1] == 32
    assert cache.token_ids.shape[0] == cache.features.shape[0]

    # three arms
    freq = FrequencyBaseline.fit(cache.token_ids, num_classes=C)
    indep = IndependentClassifier(d_in=32, num_classes=C)
    proto = build_prototype_head(d_in=32, codebook=codebook)
    train_head(indep, cache, steps=2000, batch_size=16, lr=1e-2, seed=0)
    train_head(proto, cache, steps=2000, batch_size=16, lr=1e-2, seed=0)

    tables = {
        "frequency": evaluate_arm(freq, cache),
        "independent": evaluate_arm(indep, cache),
        "prototype": evaluate_arm(proto, cache),
    }
    for df in tables.values():
        assert len(df) == len(records)
        assert np.all(np.isfinite(df["mean_nll"]))

    report = build_report(tables, n_boot=200, seed=0)
    # On tiny random data the learned heads should still fit the (memorized) tokens
    # well enough to beat the marginal floor; this exercises the full assertion path.
    assert_promotion(report)
    assert report["verdict"] in {"grounding_wins", "grounding_ties", "grounding_loses"}
