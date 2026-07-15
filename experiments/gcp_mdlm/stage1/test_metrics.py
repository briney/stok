import numpy as np
import torch

from experiments.gcp_mdlm.stage1.metrics import paired_bootstrap_ci, token_nll, topk_hits


def test_token_nll_matches_manual():
    logits = torch.tensor([[0.0, 0.0]])  # uniform over 2 classes -> nll = ln 2
    nll = token_nll(logits, torch.tensor([0]))
    assert torch.allclose(nll, torch.tensor([np.log(2.0)], dtype=nll.dtype), atol=1e-5)


def test_topk_hits():
    logits = torch.tensor([[3.0, 1.0, 2.0]])  # ranking: 0, 2, 1
    assert topk_hits(logits, torch.tensor([2]), k=1).tolist() == [False]
    assert topk_hits(logits, torch.tensor([2]), k=2).tolist() == [True]


def test_paired_bootstrap_ci_is_deterministic_and_signed():
    a = np.array([0.2, 0.3, 0.25, 0.28])  # e.g. prototype NLL (lower is better)
    b = np.array([0.4, 0.5, 0.45, 0.48])  # independent NLL
    mean_diff, lo, hi = paired_bootstrap_ci(a, b, n_boot=1000, seed=0)
    assert mean_diff < 0  # a is better (lower NLL)
    assert lo <= mean_diff <= hi
    # determinism
    assert paired_bootstrap_ci(a, b, n_boot=1000, seed=0) == (mean_diff, lo, hi)
