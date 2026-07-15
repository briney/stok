import numpy as np

from experiments.gcp_mdlm.stage1.features import CachedFeatures
from experiments.gcp_mdlm.stage1.heads import IndependentClassifier
from experiments.gcp_mdlm.stage1.train_head import train_head


def _linearly_separable_cache(n=200, d=6, c=4, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, c, size=n)
    centers = rng.normal(size=(c, d)) * 3.0
    feats = centers[tokens] + rng.normal(size=(n, d)) * 0.1
    ranges = [("p0", 0, n)]
    return CachedFeatures(feats.astype(np.float16), tokens.astype(np.int64), ranges, {})


def test_training_reduces_loss():
    cache = _linearly_separable_cache()
    head = IndependentClassifier(d_in=6, num_classes=4)
    history = train_head(head, cache, steps=100, batch_size=32, lr=1e-2, seed=0)
    assert len(history) == 100
    assert all(np.isfinite(history))
    assert history[-1] < history[0]  # learned something on separable data
