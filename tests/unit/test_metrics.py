import torch

from stok.utils.metrics import lddt_ca, tm_score, rmsd, true_aligned_error
from stok.utils.geometry import Affine3D, RotationMatrix


def _stable_ncac_coords(
    batch: int, length: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate geometrically stable random N–CA–C coordinates [B, L, 3, 3]."""
    g = torch.Generator(device=device).manual_seed(1234)

    ca = torch.randn((batch, length, 3), generator=g, device=device)
    x_dir = torch.randn((batch, length, 3), generator=g, device=device)
    y_dir = torch.randn((batch, length, 3), generator=g, device=device)
    x_dir = x_dir / (torch.linalg.norm(x_dir, dim=-1, keepdim=True).clamp_min(1e-3))
    y_dir = y_dir / (torch.linalg.norm(y_dir, dim=-1, keepdim=True).clamp_min(1e-3))
    n = ca - 1.45 * x_dir
    c = ca + 1.52 * y_dir
    coords = torch.stack([n, ca, c], dim=-2)
    return coords


def test_lddt_identity_one():
    B, L = 2, 8
    true = _stable_ncac_coords(B, L)
    pred = true.clone()
    mask = torch.ones((B, L), dtype=torch.bool)
    lddt_b, per_res = lddt_ca(pred, true, residue_mask=mask, return_per_residue=True)
    assert torch.allclose(lddt_b, torch.ones_like(lddt_b)), (
        "lDDT should be 1.0 for identical structures"
    )
    assert per_res is not None and torch.allclose(
        per_res[mask], torch.ones_like(per_res[mask])
    )


def test_tm_and_rmsd_identity_and_rigid_invariance():
    B, L = 2, 10
    true = _stable_ncac_coords(B, L)
    pred = true.clone()
    mask = torch.ones((B, L), dtype=torch.bool)

    tm_b, _ = tm_score(pred, true, residue_mask=mask)
    rmsd_b = rmsd(pred, true, residue_mask=mask, align=True, atom_set="CA")
    assert torch.allclose(tm_b, torch.ones_like(tm_b), atol=1e-5)
    assert torch.allclose(rmsd_b, torch.zeros_like(rmsd_b), atol=1e-6)

    # Apply same rigid transform to both
    rot = RotationMatrix.random((B, 1))
    trans = torch.randn((B, 1, 3))
    T = Affine3D(trans=trans, rot=rot)

    def _apply_affine_all(T, coords):
        p_in = coords.permute(1, 2, 0, 3).unsqueeze(-2)  # [L, 3a, B, 1, 3]
        applied = T.apply(p_in)  # [L, 3a, B, 1, 3]
        return applied.squeeze(-2).permute(2, 0, 1, 3)  # [B, L, 3a, 3]

    pred_t = _apply_affine_all(T, pred)
    true_t = _apply_affine_all(T, true)

    tm_b_t, _ = tm_score(pred_t, true_t, residue_mask=mask)
    rmsd_b_t = rmsd(pred_t, true_t, residue_mask=mask, align=True, atom_set="CA")
    assert torch.allclose(tm_b, tm_b_t, atol=1e-5)
    assert torch.allclose(rmsd_b, rmsd_b_t, atol=1e-6)


def test_true_aligned_error_identity_and_invariance():
    B, L = 1, 6
    true = _stable_ncac_coords(B, L)
    pred = true.clone()
    mask = torch.ones((B, L), dtype=torch.bool)
    tae, pair_mask = true_aligned_error(pred, true, residue_mask=mask)
    assert tae.shape == (B, L, L)
    assert torch.allclose(tae[pair_mask], torch.zeros_like(tae[pair_mask]), atol=1e-6)

    # Global rigid transform invariance
    rot = RotationMatrix.random((B, 1))
    trans = torch.randn((B, 1, 3))
    T = Affine3D(trans=trans, rot=rot)

    def _apply_affine_all(T, coords):
        p_in = coords.permute(1, 2, 0, 3).unsqueeze(-2)
        applied = T.apply(p_in)
        return applied.squeeze(-2).permute(2, 0, 1, 3)

    pred_t = _apply_affine_all(T, pred)
    true_t = _apply_affine_all(T, true)
    tae_t, pm_t = true_aligned_error(pred_t, true_t, residue_mask=mask)
    assert torch.allclose(tae[pm_t], tae_t[pm_t], atol=1e-5)


def test_noise_effects_on_metrics():
    B, L = 1, 12
    true = _stable_ncac_coords(B, L)
    mask = torch.ones((B, L), dtype=torch.bool)

    pred_low_noise = true + 0.1 * torch.randn_like(true)
    pred_high_noise = true + 1.0 * torch.randn_like(true)

    lddt_low, _ = lddt_ca(pred_low_noise, true, residue_mask=mask)
    lddt_high, _ = lddt_ca(pred_high_noise, true, residue_mask=mask)
    tm_low, _ = tm_score(pred_low_noise, true, residue_mask=mask)
    tm_high, _ = tm_score(pred_high_noise, true, residue_mask=mask)
    r_low = rmsd(pred_low_noise, true, residue_mask=mask, align=True, atom_set="CA")
    r_high = rmsd(pred_high_noise, true, residue_mask=mask, align=True, atom_set="CA")

    assert (lddt_low >= lddt_high).all()  # lDDT decreases with noise
    assert (tm_low >= tm_high).all()  # TM decreases with noise
    assert (r_low <= r_high).all()  # RMSD increases with noise


def test_alignment_nan_padding_and_reflection():
    true = _stable_ncac_coords(1, 6)
    pred = true + 0.2 * torch.randn_like(true)
    reference_r = rmsd(pred[:, :5], true[:, :5])
    reference_tm = tm_score(pred[:, :5], true[:, :5])[0]
    true[:, -1] = float("nan")
    pred[:, -1] = float("nan")
    mask = torch.ones(1, 6, dtype=torch.bool)
    torch.testing.assert_close(rmsd(pred, true, mask), reference_r)
    torch.testing.assert_close(tm_score(pred, true, mask)[0], reference_tm)
    reflected = true[:, :5].clone()
    reflected[..., 0] *= -1
    assert rmsd(reflected, true[:, :5]).item() > 0.01


def test_alignment_requires_adequate_targets_but_scores_collapsed_predictions():
    true = _stable_ncac_coords(1, 5)
    collapsed = torch.zeros_like(true)
    assert torch.isfinite(rmsd(collapsed, true)).all()
    assert torch.isfinite(tm_score(collapsed, true)[0]).all()
    assert torch.isnan(rmsd(true[:, :2], true[:, :2])).all()
    line = true.clone()
    line[:, :, 1] = torch.arange(5)[:, None]
    assert torch.isnan(tm_score(line, line)[0]).all()


def test_structure_metrics_reject_bad_predictions():
    import pytest

    true = _stable_ncac_coords(1, 5)
    pred = true.clone()
    pred[:, 2] = float("nan")
    for metric in (rmsd, tm_score, lddt_ca, true_aligned_error):
        with pytest.raises(ValueError, match="Nonfinite predictions"):
            metric(pred, true)


def test_fixed_cameo_subset_matches_independent_ca_references():
    """Biopython alignment plus NumPy distance definitions, with missing slots."""
    import numpy as np
    from pathlib import Path
    from Bio.SVDSuperimposer import SVDSuperimposer
    from stok.utils.structure_parser import parse_structure

    root = Path(__file__).parents[1] / "test_data/cameo"
    for filename in ("7YPD_B.pdb", "8JVC_A.pdb"):
        true = torch.as_tensor(parse_structure(root / filename).coords[:48]).float()
        true[7] = float("nan")
        ca = true[:, 1].double().numpy()
        valid = np.isfinite(ca).all(-1)
        predicted_ca = ca.copy()
        predicted_ca[:, 0] += np.sin(np.arange(len(ca))) * 0.7
        predicted_ca += np.array([2.0, -3.0, 1.0])
        pred = true.clone()
        pred[:, 1] = torch.tensor(predicted_ca, dtype=torch.float32)
        reference = SVDSuperimposer()
        reference.set(ca[valid], predicted_ca[valid])
        reference.run()
        distance = np.linalg.norm(reference.get_transformed() - ca[valid], axis=-1)
        assert np.isclose(rmsd(pred, true).item(), reference.get_rms(), atol=2e-5)
        d0 = max(0.5, 1.24 * max(int(valid.sum()) - 15, 1) ** (1 / 3) - 1.8)
        assert np.isclose(
            tm_score(pred, true)[0].item(),
            np.mean(1 / (1 + (distance / d0) ** 2)),
            atol=2e-5,
        )
        dt = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
        dp = np.linalg.norm(predicted_ca[:, None] - predicted_ca[None, :], axis=-1)
        local = []
        for i in np.flatnonzero(valid):
            neighbors = valid & (dt[i] <= 15) & (np.arange(len(ca)) != i)
            if neighbors.any():
                errors = abs(dp[i, neighbors] - dt[i, neighbors])
                local.append(np.mean(errors[:, None] < np.array([0.5, 1.0, 2.0, 4.0])))
        assert np.isclose(lddt_ca(pred, true)[0].item(), np.mean(local), atol=2e-5)
