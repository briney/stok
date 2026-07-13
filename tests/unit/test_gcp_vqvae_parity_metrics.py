from pathlib import Path

import torch

import scripts.gcp_vqvae_parity as parity
from scripts.gcp_vqvae_parity.metrics import (
    classify_run,
    compare_floats,
    compare_indices,
)
from scripts.gcp_vqvae_parity.types import RunStatus


def test_gcp_vqvae_parity_resolves_from_repository_scripts_package():
    project_root = Path(__file__).resolve().parents[2]

    assert Path(parity.__file__).resolve().is_relative_to(project_root / "scripts")


def test_compare_indices_requires_exact_valid_tokens_and_exact_mask():
    ref = torch.tensor([[5, 6, -1, 9]])
    stok = torch.tensor([[5, 7, 42, 9]])
    valid = torch.tensor([[True, True, False, True]])

    result = compare_indices("indices", ref, stok, valid, valid.clone())

    assert result.compared == 3
    assert result.mismatched == 1
    assert result.mask_equal is True
    assert result.exact is False
    assert result.passed is False


def test_compare_indices_fails_when_masks_differ():
    values = torch.tensor([[5, 6]])
    result = compare_indices(
        "indices",
        values,
        values.clone(),
        torch.tensor([[True, False]]),
        torch.tensor([[True, True]]),
    )
    assert result.mismatched == 0
    assert result.mask_equal is False
    assert result.passed is False


def test_compare_floats_uses_float32_qualification_tolerance():
    ref = torch.tensor([1.0, 10.0], dtype=torch.float32)
    stok = ref + torch.tensor([5e-7, 5e-5], dtype=torch.float32)

    result = compare_floats("encoder_head", ref, stok, rtol=1e-5, atol=1e-6)

    assert result.exact is False
    assert result.within_tolerance is True
    assert result.passed is True
    assert result.max_abs > 0
    assert result.p99_abs > 0


def test_compare_floats_rejects_nonfinite_pattern_drift():
    result = compare_floats(
        "encoder_head",
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, 0.0]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.finite_pattern_equal is False
    assert result.passed is False


def test_classify_run_distinguishes_core_and_public_parity():
    assert classify_run(weights_pass=True, core_pass=True, public_pass=True) is RunStatus.FULLY_QUALIFIED
    assert classify_run(weights_pass=True, core_pass=True, public_pass=False) is RunStatus.CORE_QUALIFIED
    assert classify_run(weights_pass=False, core_pass=True, public_pass=True) is RunStatus.NOT_QUALIFIED
    assert classify_run(weights_pass=True, core_pass=False, public_pass=True) is RunStatus.NOT_QUALIFIED
