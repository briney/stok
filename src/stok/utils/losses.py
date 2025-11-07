import torch
import torch.nn.functional as F

from .geometry import Affine3D


def token_ce_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Compute cross-entropy loss over structure tokens.

    Args:
        logits: Logits tensor of shape [B, L, C].
        labels: Target labels of shape [B, L].
        ignore_index: Index to ignore in loss computation. Defaults to -100.

    Returns:
        Scalar loss tensor.
    """
    C = int(logits.size(-1))
    logits_flat = logits.view(-1, C)
    labels_flat = labels.view(-1)
    # Treat any label outside [0, C) as ignore_index to avoid device asserts
    invalid = (labels_flat < 0) | (labels_flat >= C)
    if invalid.any():
        labels_flat = labels_flat.clone()
        labels_flat[invalid] = ignore_index
    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
    )


def _frames_from_ncac(coords: torch.Tensor) -> Affine3D:
    """Build per-residue backbone frames from N-CA-C coordinates.

    Args:
        coords: Tensor of shape [B, L, 3, 3] with atoms ordered [N(0), CA(1), C(2)].

    Returns:
        ``Affine3D`` transforms of shape [B, L] whose origin is at CA and x-axis
        is along CA−N with the xy-plane defined by C.
    """
    n = coords[..., 0, :]
    ca = coords[..., 1, :]
    c = coords[..., 2, :]
    return Affine3D.from_graham_schmidt(neg_x_axis=n, origin=ca, xy_plane=c)


def fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    clamp: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """AlphaFold-style global pairwise Frame Aligned Point Error (FAPE).

    Computes per-example mean FAPE by comparing all residues' atoms j in the
    local frames of all residues i, then averages per-example means across the
    batch to return a single scalar.

    Args:
        pred_coords: Predicted N-CA-C coordinates, shape [B, L, 3, 3].
        true_coords: Ground-truth N-CA-C coordinates, shape [B, L, 3, 3].
        residue_mask: Optional [B, L] mask (True/1 for valid residues).
        clamp: Max distance before normalization.
        length_scale: Normalization factor for distances.
        eps: Small value for numerical safety.

    Returns:
        Scalar tensor: batch-averaged FAPE.
    """
    if pred_coords.shape != true_coords.shape:
        raise ValueError("pred_coords and true_coords must have the same shape")

    if pred_coords.ndim != 4 or pred_coords.shape[-2:] != (3, 3):
        raise ValueError("coords must be shaped [B, L, 3, 3] with atoms (N, CA, C)")

    # 1) Per-residue frames for predicted and true
    T_pred = _frames_from_ncac(pred_coords)
    T_true = _frames_from_ncac(true_coords)

    # 2) Build masks (valid residues/atoms). Default: infer from GT NaNs
    if residue_mask is None:
        residue_valid = ~torch.isnan(true_coords).any(dim=(-2, -1))  # [B, L]
    else:
        residue_valid = residue_mask.to(torch.bool)

    # Set invalid frames to identity to avoid NaNs in transforms
    T_pred = T_pred.mask(~residue_valid)
    T_true = T_true.mask(~residue_valid)

    pair_mask = residue_valid[:, :, None] & residue_valid[:, None, :]  # [B, L, L]
    atom_valid = ~torch.isnan(true_coords).any(dim=-1)  # [B, L, 3]
    full_mask = pair_mask[:, :, :, None] & atom_valid[:, None, :, :]  # [B, Li, Lj, 3]

    # 3) Transform all atoms j into all frames i (global pairwise)
    Ti_pred_inv = T_pred.invert()
    Ti_true_inv = T_true.invert()

    # Broadcast transforms across all j, atoms by arranging dims so that
    # the transform's batch dims [B, Li] are the trailing ellipsis dims.
    # Shapes through the pipeline:
    #   pred_coords: [B, Lj, 3a, 3]
    #   p_in:        [Lj, 3a, B, 1, 3]  (the 1 will broadcast to Li)
    #   applied:     [Lj, 3a, B, Li, 3]
    #   permuted:    [B, Li, Lj, 3a, 3]
    p_pred_in = pred_coords.permute(1, 2, 0, 3).unsqueeze(-2)
    p_true_in = true_coords.permute(1, 2, 0, 3).unsqueeze(-2)

    P_pred_local = Ti_pred_inv.apply(p_pred_in).permute(2, 3, 0, 1, 4)
    P_true_local = Ti_true_inv.apply(p_true_in).permute(2, 3, 0, 1, 4)

    # 4) Per-pair atom errors with clamping and length scaling
    d = torch.linalg.norm(P_pred_local - P_true_local, dim=-1)  # [B, Li, Lj, 3]
    per = torch.clamp(d, max=clamp) / (length_scale + eps)

    # 5) Reductions: per-example mean over valid pairs/atoms, then batch mean
    per = torch.where(full_mask, per, torch.zeros_like(per))
    denom = full_mask.sum(dim=(1, 2, 3)).clamp_min(1).to(per.dtype)
    loss_b = per.sum(dim=(1, 2, 3)) / denom  # [B]
    return loss_b.mean()
