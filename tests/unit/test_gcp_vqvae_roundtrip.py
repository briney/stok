import numpy as np
import pytest


def test_rmsd_uses_original_mask_and_proper_rigid_alignment():
    from experiments.gcp_vqvae_roundtrip import backbone_rmsd

    original = np.random.default_rng(7).normal(size=(5, 3, 3))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    predicted = original @ rotation + [10, 20, 30]
    mask = np.array([True, False, True, True, True])
    original[1] = np.nan
    predicted[1] = 1000
    assert backbone_rmsd(predicted, original, mask) == pytest.approx(0, abs=1e-12)
    reflected = original.copy()
    reflected[..., 0] *= -1
    assert backbone_rmsd(reflected, original, mask) > 0.1
