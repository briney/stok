import numpy as np

from experiments.gcp_mdlm.stage1.evaluate import evaluate_arm
from experiments.gcp_mdlm.stage1.features import CachedFeatures
from experiments.gcp_mdlm.stage1.heads import FrequencyBaseline


def _cache():
    feats = np.zeros((5, 3), dtype=np.float16)
    tokens = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    ranges = [("p1", 0, 2), ("p2", 2, 3)]
    return CachedFeatures(feats, tokens, ranges, {})


def test_frequency_arm_per_protein_rows():
    cache = _cache()
    fb = FrequencyBaseline.fit(cache.token_ids, num_classes=2, smoothing=0.0)
    df = evaluate_arm(fb, cache)
    assert df["sequence_id"].tolist() == ["p1", "p2"]
    assert df["n_res"].tolist() == [2, 3]
    # top1 accuracy: class 1 is most frequent (3/5), so p1 (all class 0) scores 0, p2 scores 1
    assert df.loc[df.sequence_id == "p1", "top1"].iloc[0] == 0.0
    assert df.loc[df.sequence_id == "p2", "top1"].iloc[0] == 1.0
    assert np.all(np.isfinite(df["mean_nll"]))
