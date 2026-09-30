import os
from pathlib import Path
import subprocess
import sys

import pytest


def training_command(project, *overrides):
    return [sys.executable, '-m', 'stok.train',
            'model.encoder.d_model=16', 'model.encoder.n_heads=2',
            'model.encoder.n_layers=1', 'model.encoder.ffn_mult=1.0',
            'model.encoder.dropout=0.0', 'model.codebook.preset=lite',
            'data.num_workers=0', 'data.batch_size=2', 'data.max_len=8',
            'train.num_steps=3', 'train.checkpoint_steps=1',
            'train.wandb.enabled=false', 'train.console.enabled=false',
            f'train.project_path={project}', *overrides]


def training_env():
    return {**os.environ, 'ACCELERATE_USE_CPU': 'true', 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '',
            'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': '',
            'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')}


def test_undersized_loader_fails_promptly(tmp_path):
    csv = tmp_path / 'small.csv'
    csv.write_text('pid,protein_sequence,indices\np,LAG,0 1 2\n')
    result = subprocess.run(training_command(tmp_path / 'run', f'data.train={csv}'),
                            env=training_env(), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'complete batches' in result.stderr


@pytest.mark.parametrize('override', ['train.grad_accum_steps=0', 'train.log_steps=0',
                                       'train.eval.steps=0', 'train.num_steps=-1'])
def test_invalid_progress_configuration(tmp_path, override):
    result = subprocess.run(training_command(tmp_path, override), env=training_env(),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'ValueError' in result.stderr


def run_checkpoint(project, *overrides):
    import torch
    result = subprocess.run(training_command(project, *overrides), env=training_env(),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return torch.load(project/'model/final.pt', map_location='cpu', weights_only=False)


def test_num_steps_counts_optimizer_updates(tmp_path):
    checkpoint = run_checkpoint(tmp_path, 'train.grad_accum_steps=4')
    states = checkpoint['optimizer']['state'].values()
    assert states and all(state['step'].item() == 3 for state in states)
    assert checkpoint['scheduler']['last_epoch'] == 3
    assert checkpoint['global_step'] == 3
    assert checkpoint['micro_step'] == 12
    assert checkpoint['step_unit'] == 'optimizer_update'


def write_labeled_csv(path, count=16, empty=False):
    rows = ['pid,protein_sequence,indices']
    for i in range(count):
        labels = '-1 -1 -1' if empty else ('0 -1 -1' if i % 3 else '0 1 2')
        rows.append(f'{i},LAG,{labels}')
    path.write_text('\n'.join(rows)+'\n')


def test_epoch_flushes_final_partial_window(tmp_path):
    source = tmp_path/'train.csv'
    write_labeled_csv(source, count=10)
    checkpoint = run_checkpoint(tmp_path/'run', f'data.train={source}',
                                'train.epochs=1', 'train.grad_accum_steps=4')
    assert checkpoint['global_step'] == 2
    assert checkpoint['micro_step'] == 5
    assert checkpoint['scheduler']['last_epoch'] == 2
    assert all(s['step'].item() == 2 for s in checkpoint['optimizer']['state'].values())


def test_accumulation_matches_large_batch_with_unequal_supervision(tmp_path):
    import torch
    source = tmp_path/'train.csv'
    write_labeled_csv(source)
    common = [f'data.train={source}', 'train.num_steps=2', 'train.scheduler.warmup_steps=0',
              'train.optimizer.lr=0.001', 'train.grad_clip_norm=0']
    small = run_checkpoint(tmp_path/'small', *common, 'data.batch_size=2', 'train.grad_accum_steps=4')
    large = run_checkpoint(tmp_path/'large', *common, 'data.batch_size=8', 'train.grad_accum_steps=1')
    for name in small['model']:
        torch.testing.assert_close(small['model'][name], large['model'][name], atol=2e-6, rtol=2e-5)


def test_fully_unsupervised_pass_fails_without_checkpoint(tmp_path):
    source = tmp_path/'empty.csv'
    write_labeled_csv(source, count=4, empty=True)
    project = tmp_path/'run'
    result = subprocess.run(training_command(project, f'data.train={source}'),
                            env=training_env(), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'no successful optimizer update' in result.stderr
    assert not list((project/'checkpoints').glob('step_*.pt'))


def test_skipped_optimizer_step_does_not_advance_schedule(tmp_path, monkeypatch):
    import torch
    from accelerate.optimizer import AcceleratedOptimizer
    from hydra import compose, initialize_config_dir
    from stok.cli.train import run_training
    original = AcceleratedOptimizer.step
    attempts = []
    def skip_once(self, *args, **kwargs):
        attempts.append(1)
        self._is_overflow = len(attempts) == 1
        if not self._is_overflow:
            return original(self, *args, **kwargs)
    monkeypatch.setattr(AcceleratedOptimizer, 'step', skip_once)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'src/stok/configs'), version_base=None):
        cfg = compose(config_name='config', overrides=training_command(tmp_path, 'train.num_steps=2')[3:])
    run_training(cfg)
    checkpoint = torch.load(tmp_path/'model/final.pt', weights_only=False, map_location='cpu')
    assert len(attempts) == 3
    assert checkpoint['global_step'] == checkpoint['scheduler']['last_epoch'] == 2
    assert checkpoint['micro_step'] == 3
    assert all(s['step'].item() == 2 for s in checkpoint['optimizer']['state'].values())
