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




def test_accelerator_initialization_failure_is_not_hidden(monkeypatch):
    import accelerate
    from stok.cli.train import _maybe_get_accelerator
    def fail():
        raise RuntimeError('initialization failed')
    monkeypatch.setattr(accelerate, 'Accelerator', fail)
    with pytest.raises(RuntimeError, match='initialization failed'):
        _maybe_get_accelerator()


def test_accumulation_windows_keep_partial_tail():
    from stok.cli.train import iter_windows
    assert list(iter_windows(range(5), 4)) == [[0, 1, 2, 3], [4]]
    assert list(iter_windows([], 4)) == []
    with pytest.raises(ValueError):
        list(iter_windows([], 0))


@pytest.mark.parametrize('load_coords', [None, False, True])
def test_required_coordinate_loading_policy(tmp_path, load_coords):
    from pathlib import Path
    import pandas as pd
    from hydra import compose, initialize_config_dir
    from stok.cli.train import _build_dataloaders
    source = tmp_path/'train.parquet'
    pd.DataFrame([{'pid': 'p', 'protein_sequence': 'LAG', 'indices': [0, 1, 2],
        'coordinates': [[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]]*3}]*2).to_parquet(source)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'src/stok/configs'), version_base=None):
        cfg = compose(config_name='config', overrides=['data.batch_size=2', 'data.max_len=6',
            'data.num_workers=0', f'data.train={source}', 'train.fape.enabled=true'])
    cfg.data.load_coords = load_coords
    if load_coords is False:
        with pytest.raises(ValueError, match='load_coords'):
            _build_dataloaders(cfg, codebook_size=128, pad_id=1)
    else:
        loader, _ = _build_dataloaders(cfg, codebook_size=128, pad_id=1)
        batch = next(iter(loader))
        assert len(batch) == 3 and torch.isfinite(batch[2][:, 1:4]).all()


def test_coordinate_alias_conflict_and_missing_source(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from stok.cli.train import _build_dataloaders
    source = tmp_path/'seq.csv'
    source.write_text('pid,protein_sequence,indices\np,LAG,0 1 2\np,LAG,0 1 2\n')
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'src/stok/configs'), version_base=None):
        cfg = compose(config_name='config', overrides=[f'data.train={source}', 'data.num_workers=0'])
    cfg.data.eval = {'val': {'path': str(source), 'has_coords': True, 'load_coords': False}}
    with pytest.raises(ValueError, match='Conflicting'):
        _build_dataloaders(cfg, codebook_size=128, pad_id=1)
    cfg.data.eval = {'val': {'path': str(source), 'has_coords': True}}
    with pytest.raises(ValueError, match='coordinate-capable'):
        _build_dataloaders(cfg, codebook_size=128, pad_id=1)


@pytest.mark.parametrize('override,match', [
    ('model.classifier.tie_to_codebook=false', 'tie_to_codebook'),
    ('model.codebook.trainable=true', 'codebook.trainable'),
    ('model.decoder.freeze=false', 'decoder.freeze'),
    ('train.optimizer.name=sgd', 'optimizer.name'),
])
def test_unsupported_options_fail_before_accelerator(monkeypatch, override, match):
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import stok.cli.train as train
    with initialize_config_dir(config_dir=str(Path(train.__file__).parents[1]/'configs'), version_base=None):
        cfg = compose(config_name='config', overrides=[override])
    def should_not_initialize():
        raise AssertionError('Validation must precede accelerator/downloads')
    monkeypatch.setattr(train, '_maybe_get_accelerator', should_not_initialize)
    with pytest.raises(ValueError, match=match):
        train.run_training(cfg)


@pytest.mark.parametrize('option', ['train.fape.enabled', 'model.decoder.enabled', 'train.decoding.eval_enabled'])
def test_mlm_rejects_explicit_geometry_before_initialization(monkeypatch, option):
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import stok.cli.train as train
    with initialize_config_dir(config_dir=str(Path(train.__file__).parents[1]/'configs'), version_base=None):
        cfg = compose(config_name='config', overrides=['train.objective=mlm', f'{option}=true'])
    def should_not_initialize():
        raise AssertionError('MLM geometry must be rejected before initialization')
    monkeypatch.setattr(train, '_maybe_get_accelerator', should_not_initialize)
    with pytest.raises(ValueError, match='MLM'):
        train.run_training(cfg)
