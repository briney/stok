"""Unit tests for structure metrics."""

import torch

from stok.eval.metrics.structure import STRUCTURE_SCORES


def _stable_ncac_coords(batch: int, length: int) -> torch.Tensor:
    """Generate geometrically stable N-CA-C coordinates [B, L, 3, 3]."""
    g = torch.Generator().manual_seed(1234)

    ca = torch.randn((batch, length, 3), generator=g)
    x_dir = torch.randn((batch, length, 3), generator=g)
    y_dir = torch.randn((batch, length, 3), generator=g)
    x_dir = x_dir / (torch.linalg.norm(x_dir, dim=-1, keepdim=True).clamp_min(1e-3))
    y_dir = y_dir / (torch.linalg.norm(y_dir, dim=-1, keepdim=True).clamp_min(1e-3))
    n = ca - 1.45 * x_dir
    c = ca + 1.52 * y_dir
    return torch.stack([n, ca, c], dim=-2)


def test_identical_structures_score_perfectly():
    coords = _stable_ncac_coords(1, 10)
    mask = torch.ones(1, 10, dtype=torch.bool)
    expected = {"lddt": 1.0, "rmsd": 0.0, "tm": 1.0}
    for name, metric in STRUCTURE_SCORES.items():
        assert abs(float(metric(coords.clone(), coords, mask)) - expected[name]) < 0.01


def test_missing_targets_are_not_evaluable():
    coords = _stable_ncac_coords(1, 6)
    missing = torch.full_like(coords, float("nan"))
    mask = torch.ones(1, 6, dtype=torch.bool)
    for metric in STRUCTURE_SCORES.values():
        assert metric(coords, missing, mask) is None
