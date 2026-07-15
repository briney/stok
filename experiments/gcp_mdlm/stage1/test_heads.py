import numpy as np
import torch

from experiments.gcp_mdlm.stage1.heads import (
    FrequencyBaseline,
    IndependentClassifier,
    build_prototype_head,
    head_predict,
)


def test_frequency_baseline_matches_empirical_logprob():
    token_ids = np.array([0, 0, 0, 1])  # class 0 thrice, class 1 once
    fb = FrequencyBaseline.fit(token_ids, num_classes=2, smoothing=0.0)
    logits = fb.logits(1)  # (1, 2) log-probs
    probs = logits.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(probs[0], torch.tensor([0.75, 0.25]), atol=1e-5)


def test_independent_classifier_shape():
    head = IndependentClassifier(d_in=8, num_classes=16)
    out = head_predict(head, torch.randn(5, 8))
    assert out.shape == (5, 16)


def test_prototype_head_shape():
    codebook = torch.randn(16, 8)
    head = build_prototype_head(d_in=12, codebook=codebook)
    out = head_predict(head, torch.randn(5, 12))
    assert out.shape == (5, 16)
