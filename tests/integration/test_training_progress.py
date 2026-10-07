"""Process launch helpers and MDLM progress preflight."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def training_command(project, *overrides):
    from tests.integration.test_mdlm_training import training_fixture

    fixture = project.parent / "launch-fixture"
    if not fixture.exists():
        fixture.mkdir()
        training_fixture(fixture)
    return [
        sys.executable,
        "-m",
        "stok.train",
        "model.encoder.d_model=16",
        "model.encoder.n_heads=2",
        "model.encoder.n_layers=1",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        f"model.codebook.path={fixture / 'codebook.pt'}",
        f"data.train={fixture / 'data'}",
        "data.num_workers=0",
        "train.batch_size=2",
        "data.max_len=8",
        "train.max_steps=3",
        "train.save_every=1",
        "train.wandb.enabled=false",
        "train.console.enabled=false",
        f"train.output_dir={project}",
        *overrides,
    ]


def training_env():
    return {
        **os.environ,
        "ACCELERATE_USE_CPU": "true",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }


def test_undersized_loader_fails_promptly(tmp_path):
    from tests.integration.test_mdlm_training import training_fixture

    source, codebook = training_fixture(tmp_path, n=1)
    result = subprocess.run(
        training_command(
            tmp_path / "run", f"data.train={source}", f"model.codebook.path={codebook}"
        ),
        env=training_env(),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "complete batches" in result.stderr


@pytest.mark.parametrize(
    "override",
    [
        "train.gradient_accumulation_steps=0",
        "train.log_every=0",
        "train.eval.steps=0",
        "train.max_steps=-1",
    ],
)
def test_invalid_progress_configuration(tmp_path, override):
    result = subprocess.run(
        training_command(tmp_path / "run", override),
        env=training_env(),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr
