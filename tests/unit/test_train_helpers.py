import pytest
import torch
from omegaconf import OmegaConf

from stok.training.engine import _build_scheduler
from stok.data.loaders import _parse_eval_configs


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
    assert all(
        abs(lrs[i] - 1.0) <= 1e-6
        for i in range(warmup_steps, warmup_steps + stable_steps)
    )
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


def test_parse_eval_configs_supports_single_path():
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


def test_accelerator_initialization_failure_is_not_hidden(monkeypatch):
    import accelerate
    from stok.training.engine import _maybe_get_accelerator

    def fail(**kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(accelerate, "Accelerator", fail)
    with pytest.raises(RuntimeError, match="initialization failed"):
        _maybe_get_accelerator()


def test_accumulation_windows_keep_partial_tail():
    from stok.training.engine import iter_windows

    assert list(iter_windows(range(5), 4)) == [[0, 1, 2, 3], [4]]
    assert list(iter_windows([], 4)) == []
    with pytest.raises(ValueError):
        list(iter_windows([], 0))


def test_manual_accumulation_is_not_divided_by_accelerate_environment(monkeypatch):
    from stok.training.engine import _maybe_get_accelerator

    monkeypatch.setenv("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "4")
    accelerator = _maybe_get_accelerator()
    value = torch.tensor(1.0, requires_grad=True)
    accelerator.backward(value * 16)
    assert value.grad.item() == 16.0
