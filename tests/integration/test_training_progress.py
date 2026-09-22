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
