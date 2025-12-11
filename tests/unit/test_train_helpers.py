import math

import pytest
import torch
from omegaconf import OmegaConf

from stok.cli.train import _build_scheduler, _compute_accuracy, _parse_eval_configs


def test_build_scheduler_warmup_then_cosine_decay():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    warmup_steps, total_steps = 3, 10
    sched = _build_scheduler(
        opt,
        decay="cosine",
        warmup_steps=warmup_steps,
        stable_steps=0,
        decay_steps=None,
        total_steps=total_steps,
    )

    lrs: list[float] = []
    for _ in range(total_steps):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # warmup monotonic increasing
    assert lrs[0] < lrs[1] <= lrs[2] <= 1.0 + 1e-6
    # post-warmup non-increasing
    for i in range(warmup_steps + 1, total_steps):
        assert lrs[i] <= lrs[i - 1] + 1e-6
    # bounds
    assert all(0.0 - 1e-6 <= lr <= 1.0 + 1e-6 for lr in lrs)


def test_build_scheduler_linear_with_stable_and_auto_decay_steps():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    warmup_steps, stable_steps, total_steps = 2, 3, 12
    # decay_steps auto: 12 - 2 - 3 = 7
    sched = _build_scheduler(
        opt,
        decay="linear",
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=None,
        total_steps=total_steps,
    )

    lrs: list[float] = []
    for _ in range(total_steps):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # warmup monotonic increasing into 1.0 plateau
    assert lrs[0] < lrs[1] <= 1.0 + 1e-6
    # stable plateau
    assert all(abs(lrs[i] - 1.0) <= 1e-6 for i in range(warmup_steps, warmup_steps + stable_steps))
    # decay non-increasing thereafter
    for i in range(warmup_steps + stable_steps + 1, total_steps):
        assert lrs[i] <= lrs[i - 1] + 1e-6
    # bounds
    assert all(0.0 - 1e-6 <= lr <= 1.0 + 1e-6 for lr in lrs)


def test_build_scheduler_warmup_then_stable_only_when_zero_decay_steps():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    warmup_steps, total_steps = 2, 10
    sched = _build_scheduler(
        opt,
        decay="cosine",
        warmup_steps=warmup_steps,
        stable_steps=0,
        decay_steps=0,
        total_steps=total_steps,
    )

    lrs: list[float] = []
    for _ in range(total_steps):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # after warmup, stay at 1.0
    assert all(abs(lr - 1.0) <= 1e-6 for lr in lrs[warmup_steps:])


def test_compute_accuracy_with_ignore_index():
    ignore_index = -100
    logits = torch.tensor(
        [
            [2.0, 1.0],  # pred 0, ignored
            [0.1, 0.9],  # pred 1, correct
            [0.9, 0.1],  # pred 0, incorrect
        ]
    )
    labels = torch.tensor([ignore_index, 1, 1])
    acc = _compute_accuracy(logits, labels, ignore_index)
    assert math.isclose(acc, 0.5, rel_tol=1e-6, abs_tol=1e-6)


def test_parse_eval_configs_supports_legacy_string():
    cfg = OmegaConf.create({"data": {"eval": "/path/to/eval"}})
    parsed = _parse_eval_configs(cfg)
    assert parsed == {"default": {"path": "/path/to/eval"}}


def test_parse_eval_configs_handles_dict_of_paths():
    cfg = OmegaConf.create({"data": {"eval": {"val": "/p1", "test": "/p2"}}})
    parsed = _parse_eval_configs(cfg)
    assert parsed == {"val": {"path": "/p1"}, "test": {"path": "/p2"}}


def test_parse_eval_configs_handles_nested_configs():
    cfg = OmegaConf.create(
        {"data": {"eval": {"val": {"path": "/p1", "batch_size": 8}}}}
    )
    parsed = _parse_eval_configs(cfg)
    assert parsed == {"val": {"path": "/p1", "batch_size": 8}}


def test_parse_eval_configs_rejects_invalid_type():
    cfg = OmegaConf.create({"data": {"eval": 123}})
    with pytest.raises(ValueError):
        _parse_eval_configs(cfg)


