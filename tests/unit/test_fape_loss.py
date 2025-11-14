import torch

from stok.utils.losses import fape_loss
from stok.utils.geometry import Affine3D, RotationMatrix


def _stable_ncac_coords(batch: int, length: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generate geometrically stable random N–CA–C coordinates [B, L, 3, 3]."""
    g = torch.Generator(device=device).manual_seed(42)

    ca = torch.randn((batch, length, 3), generator=g, device=device)

    # x-axis direction (CA − N) and xy-plane vector (C − CA)
    x_dir = torch.randn((batch, length, 3), generator=g, device=device)
    y_dir = torch.randn((batch, length, 3), generator=g, device=device)

    # normalize to avoid degenerate frames and set reasonable bond-length scales
    x_dir = x_dir / (torch.linalg.norm(x_dir, dim=-1, keepdim=True).clamp_min(1e-3))
    y_dir = y_dir / (torch.linalg.norm(y_dir, dim=-1, keepdim=True).clamp_min(1e-3))

    n = ca - 1.45 * x_dir  # approx CA−N bond length
    c = ca + 1.52 * y_dir  # approx CA−C bond length

    coords = torch.stack([n, ca, c], dim=-2)  # [B, L, 3_atoms, 3]
    return coords


def test_fape_identity_zero():
    B, L = 2, 5
    true_coords = _stable_ncac_coords(B, L)
    pred_coords = true_coords.clone()

    loss = fape_loss(pred_coords, true_coords)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(0.0, dtype=loss.dtype)), f"Expected 0, got {loss.item()}"


def test_fape_rigid_motion_invariance():
    B, L = 2, 6
    true_coords = _stable_ncac_coords(B, L)
    # introduce a small systematic error to predictions
    delta = torch.zeros_like(true_coords)
    delta[..., 0, :] += torch.tensor([0.2, -0.1, 0.05])  # shift N
    pred_coords = true_coords + delta

    base = fape_loss(pred_coords, true_coords)

    # apply the same random rigid transform to both pred and true
    rot = RotationMatrix.random((B, 1), device=true_coords.device)
    trans = torch.randn((B, 1, 3), device=true_coords.device)
    T = Affine3D(trans=trans, rot=rot)

    def _apply_affine_all(T, coords):
        p_in = coords.permute(1, 2, 0, 3).unsqueeze(-2)
        applied = T.apply(p_in)  # [L, 3a, B, 1, 3]
        return applied.squeeze(-2).permute(2, 0, 1, 3)  # [B, L, 3a, 3]

    pred_t = _apply_affine_all(T, pred_coords)
    true_t = _apply_affine_all(T, true_coords)

    moved = fape_loss(pred_t, true_t)
    assert torch.allclose(base, moved, atol=1e-5, rtol=1e-5)


def test_fape_nan_in_ground_truth_equals_explicit_mask():
    B, L = 1, 4
    true_coords = _stable_ncac_coords(B, L)
    pred_coords = true_coords + 0.1 * torch.randn_like(true_coords)

    # mask out last two residues by inserting NaNs into GT
    true_with_nans = true_coords.clone()
    true_with_nans[:, 2:, :, :] = float("nan")

    mask_valid = torch.zeros((B, L), dtype=torch.bool)
    mask_valid[:, :2] = True

    loss_inferred = fape_loss(pred_coords, true_with_nans)
    loss_explicit = fape_loss(pred_coords, true_coords, residue_mask=mask_valid)

    assert torch.isfinite(loss_inferred)
    assert torch.allclose(loss_inferred, loss_explicit, atol=1e-6, rtol=1e-6)


def test_fape_nan_in_predictions_all_nan_returns_zero_and_finite():
    B, L = 2, 6
    true_coords = _stable_ncac_coords(B, L)
    pred_coords = torch.full_like(true_coords, float("nan"))

    loss = fape_loss(pred_coords, true_coords)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(0.0, dtype=loss.dtype), atol=0.0, rtol=0.0)


def test_fape_nan_in_predictions_partial():
    B, L = 2, 8
    true_coords = _stable_ncac_coords(B, L)
    pred_coords = true_coords.clone()
    # Make first half finite but slightly off; second half NaN
    delta = torch.zeros_like(pred_coords)
    delta[:, : L // 2, :, :] = 0.2
    pred_coords[:, : L // 2, :, :] = true_coords[:, : L // 2, :, :] + delta[:, : L // 2, :, :]
    pred_coords[:, L // 2 :, :, :] = float("nan")

    loss = fape_loss(pred_coords, true_coords)
    assert torch.isfinite(loss)
    # With perturbations on valid half, expect positive loss
    assert loss.item() > 0.0

