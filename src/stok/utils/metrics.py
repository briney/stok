from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch

from .geometry import Affine3D, frames_from_ncac

__all__ = ["lddt_ca", "tm_score", "rmsd", "true_aligned_error"]


def _ensure_batch(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim == 3 and coords.shape[-2:] == (3, 3):
        return coords.unsqueeze(0)
    return coords


def _infer_residue_mask(true_coords: torch.Tensor) -> torch.Tensor:
    # Valid if none of the backbone atoms are NaN
    return ~torch.isnan(true_coords).any(dim=(-2, -1))


def _expand_mask_for_atoms(
    residue_mask: torch.Tensor, num_atoms: int = 3
) -> torch.Tensor:
    # [B, L] -> [B, L, num_atoms]
    return residue_mask[..., None].expand(*residue_mask.shape, num_atoms)


def _get_atom(coords: torch.Tensor, name: str = "CA") -> torch.Tensor:
    name = name.upper()
    idx = 1 if name == "CA" else (0 if name == "N" else (2 if name == "C" else None))
    if idx is None:
        raise ValueError(f"Unsupported atom name: {name}. Use one of 'N', 'CA', 'C'.")
    return coords[..., idx, :]


def _masked_mean(
    x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim: bool = False
) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    if dim is None:
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / denom
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return (x * mask).sum(dim=dim, keepdim=keepdim) / denom


def _kabsch(
    P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute optimal rigid transform aligning P to Q under squared error.

    Args:
        P: Pred points [B, L, 3]
        Q: True points [B, L, 3]
        mask: Optional validity mask [B, L] (True = use)

    Returns:
        R: Rotation matrices [B, 3, 3]
        t: Translations [B, 3]
    """
    B, L, _ = P.shape
    device = P.device
    dtype = torch.float32

    if mask is None:
        mask = torch.ones((B, L), device=device, dtype=torch.bool)
    w = mask.to(P.dtype)[..., None]  # [B, L, 1]

    # Weighted centroids
    wsum = w.sum(dim=1).clamp_min(1.0)  # [B, 1]
    mu_P = (P * w).sum(dim=1) / wsum  # [B, 3]
    mu_Q = (Q * w).sum(dim=1) / wsum  # [B, 3]

    P_centered = (P - mu_P[:, None, :]) * w
    Q_centered = Q - mu_Q[:, None, :]

    # Cross-covariance H = P^T Q
    # Use fp32 for stability (works under AMP)
    H = torch.einsum(
        "bli,blj->bij", P_centered.to(dtype), Q_centered.to(dtype)
    )  # [B, 3, 3]

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.transpose(-1, -2)
    # Reflection correction
    det = torch.linalg.det(V @ U.transpose(-1, -2))
    D = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    D[:, 2, 2] = torch.where(
        det < 0,
        torch.tensor(-1.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )
    R = (V @ D @ U.transpose(-1, -2)).to(P.dtype)  # [B, 3, 3]
    t = (mu_Q - torch.einsum("bij,bj->bi", R, mu_P)).to(P.dtype)  # [B, 3]

    # If fewer than 3 valid points, fall back to identity
    enough = mask.sum(dim=1) >= 3
    if not torch.all(enough):
        R_id = torch.eye(3, device=device, dtype=P.dtype).unsqueeze(0).repeat(B, 1, 1)
        t_zeros = torch.zeros((B, 3), device=device, dtype=P.dtype)
        R = torch.where(enough[:, None, None], R, R_id)
        t = torch.where(enough[:, None], t, t_zeros)

    return R, t


def lddt_ca(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    thresholds: Iterable[float] = (0.5, 1.0, 2.0, 4.0),
    cutoff: float = 15.0,
    min_seq_sep: int = 1,
    return_per_residue: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute lDDT (Cα-only), superposition-free.

    Returns:
        (lddt_per_example [B], lddt_per_residue [B, L] | None)
    """
    pred = _ensure_batch(pred_coords)
    true = _ensure_batch(true_coords)
    if pred.shape != true.shape or pred.ndim != 4 or pred.shape[-2:] != (3, 3):
        raise ValueError("coords must be shaped [B, L, 3, 3] with atoms (N, CA, C)")
    B, L = pred.shape[:2]
    device = pred.device

    if residue_mask is None:
        res_mask = _infer_residue_mask(true)  # [B, L]
    else:
        res_mask = (
            _ensure_batch(residue_mask) if residue_mask.ndim == 1 else residue_mask
        )
        res_mask = res_mask.to(torch.bool)

    ca_pred = _get_atom(pred, "CA")  # [B, L, 3]
    ca_true = _get_atom(true, "CA")

    # Pairwise distances
    def pairwise_dists(x: torch.Tensor) -> torch.Tensor:
        # [B, L, L]
        x2 = (x**2).sum(dim=-1)  # [B, L]
        # dist^2 = ||x_i - x_j||^2
        d2 = x2[:, :, None] + x2[:, None, :] - 2.0 * torch.einsum("bid,bjd->bij", x, x)
        return d2.clamp_min(0.0).sqrt()

    Dp = pairwise_dists(ca_pred)
    Dt = pairwise_dists(ca_true)

    # Pair mask: valid residues, exclude i==j and short seq separation
    pair_mask = res_mask[:, :, None] & res_mask[:, None, :]  # [B, L, L]
    eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & ~eye
    if min_seq_sep > 0:
        idx = torch.arange(L, device=device)
        sep = (idx[None, :] - idx[:, None]).abs()  # [L, L]
        sep_mask = (sep >= min_seq_sep).unsqueeze(0)  # [1, L, L]
        pair_mask = pair_mask & sep_mask

    # Only consider neighbors within cutoff in true structure
    neighbor_mask = Dt <= cutoff
    nbr_mask = pair_mask & neighbor_mask  # [B, L, L]

    if not isinstance(thresholds, (list, tuple)):
        thresholds = tuple(thresholds)
    T = torch.tensor(list(thresholds), device=device, dtype=Dp.dtype)  # [T]

    # Broadcast thresholds over pairs
    delta = (Dp - Dt).abs().unsqueeze(-1)  # [B, L, L, 1]
    passed = (delta < T.view(1, 1, 1, -1)) & nbr_mask.unsqueeze(-1)  # [B, L, L, T]

    # Per-residue fraction over neighbors and thresholds
    num_pass = passed.sum(dim=(2, 3))  # [B, L]
    denom = (nbr_mask.sum(dim=2) * T.numel()).clamp_min(1)  # [B, L]
    lddt_per_res = num_pass.to(Dp.dtype) / denom.to(Dp.dtype)  # [B, L]

    # Per-example mean over valid residues
    lddt_b = _masked_mean(lddt_per_res, res_mask, dim=1)  # [B]
    return (lddt_b, lddt_per_res if return_per_residue else None)


def tm_score(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    d0: float | None = None,
    length_norm: str = "valid",
    return_aligned: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute TM-score on Cα after Kabsch alignment of pred to true.

    Returns:
        (tm_per_example [B], pred_ca_aligned [B, L, 3] | None)
    """
    pred = _ensure_batch(pred_coords)
    true = _ensure_batch(true_coords)
    if pred.shape != true.shape or pred.ndim != 4 or pred.shape[-2:] != (3, 3):
        raise ValueError("coords must be shaped [B, L, 3, 3] with atoms (N, CA, C)")
    B, L = pred.shape[:2]

    if residue_mask is None:
        res_mask = _infer_residue_mask(true)  # [B, L]
    else:
        res_mask = (
            _ensure_batch(residue_mask) if residue_mask.ndim == 1 else residue_mask
        )
        res_mask = res_mask.to(torch.bool)

    ca_pred = _get_atom(pred, "CA")  # [B, L, 3]
    ca_true = _get_atom(true, "CA")  # [B, L, 3]

    R, t = _kabsch(ca_pred, ca_true, mask=res_mask)  # [B,3,3], [B,3]
    ca_pred_aligned = (
        torch.einsum("bij,blj->bli", R, ca_pred) + t[:, None, :]
    )  # [B,L,3]

    d = torch.linalg.norm(ca_pred_aligned - ca_true, dim=-1)  # [B, L]
    n_valid = res_mask.sum(dim=1).clamp_min(1)  # [B]

    if d0 is None:
        # Zhang–Skolnick length-dependent normalization (per-example)
        L0 = n_valid.to(d.dtype)
        # Protect small L0 to keep real root; d0 >= 0.5Å
        safe = torch.clamp(L0 - 15.0, min=1.0)
        d0_b = torch.clamp(1.24 * torch.pow(safe, 1.0 / 3.0) - 1.8, min=0.5)
    else:
        d0_b = torch.full_like(n_valid.to(d.dtype), float(d0))

    contrib = 1.0 / (1.0 + (d / d0_b[:, None]) ** 2)  # [B, L]
    tm_b = _masked_mean(contrib, res_mask, dim=1)  # [B]
    return tm_b, (ca_pred_aligned if return_aligned else None)


def rmsd(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    align: bool = True,
    atom_set: str = "CA",
) -> torch.Tensor:
    """Compute per-example RMSD in Å. Align by Kabsch if requested.

    atom_set: 'CA' or 'backbone'. Alignment is performed using Cα for stability.
    """
    pred = _ensure_batch(pred_coords)
    true = _ensure_batch(true_coords)
    if pred.shape != true.shape or pred.ndim != 4 or pred.shape[-2:] != (3, 3):
        raise ValueError("coords must be shaped [B, L, 3, 3] with atoms (N, CA, C)")

    if residue_mask is None:
        res_mask = _infer_residue_mask(true)  # [B, L]
    else:
        res_mask = (
            _ensure_batch(residue_mask) if residue_mask.ndim == 1 else residue_mask
        )
        res_mask = res_mask.to(torch.bool)

    ca_pred = _get_atom(pred, "CA")
    ca_true = _get_atom(true, "CA")

    if align:
        R, t = _kabsch(ca_pred, ca_true, mask=res_mask)
        pred_aligned = torch.einsum("bij,blaj->blai", R, pred) + t[:, None, None, :]
    else:
        pred_aligned = pred

    if atom_set.lower() == "ca":
        P = _get_atom(pred_aligned, "CA")
        Q = ca_true
        mask = res_mask
    elif atom_set.lower() == "backbone":
        # Flatten N/CA/C, but mask invalid residues and any NaN atoms
        atom_valid = ~torch.isnan(true).any(dim=-1)  # [B, L, 3]
        mask = (atom_valid & _expand_mask_for_atoms(res_mask)).reshape(
            res_mask.shape[0], -1
        )  # [B, L*3]
        P = pred_aligned.reshape(pred.shape[0], -1, 3)  # [B, L*3, 3]
        Q = true.reshape(true.shape[0], -1, 3)
    else:
        raise ValueError("atom_set must be 'CA' or 'backbone'")

    diffs = (P - Q).norm(dim=-1)  # [B, N]
    rmsd_b = torch.sqrt(_masked_mean(diffs**2, mask, dim=1))
    return rmsd_b


def true_aligned_error(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    atom: str = "CA",
    clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute True Aligned Error (per-pair PAE target) in Å.

    For each residue j, align the predicted structure to the true structure using
    the local frames at residue j, then measure the distance between all residues i.

    Returns:
        (tae: [B, L, L], pair_mask: [B, L, L])
    """
    pred = _ensure_batch(pred_coords)
    true = _ensure_batch(true_coords)
    if pred.shape != true.shape or pred.ndim != 4 or pred.shape[-2:] != (3, 3):
        raise ValueError("coords must be shaped [B, L, 3, 3] with atoms (N, CA, C)")
    B, L = pred.shape[:2]
    device = pred.device

    if residue_mask is None:
        res_mask = _infer_residue_mask(true)  # [B, L]
    else:
        res_mask = (
            _ensure_batch(residue_mask) if residue_mask.ndim == 1 else residue_mask
        )
        res_mask = res_mask.to(torch.bool)

    atom_pred = _get_atom(pred, atom)  # [B, L, 3]
    atom_true = _get_atom(true, atom)  # [B, L, 3]

    # Per-residue frames
    T_pred = frames_from_ncac(pred)  # [B, L]
    T_true = frames_from_ncac(true)  # [B, L]
    T_align = T_true.compose(T_pred.invert())  # align pred->true by residue j

    # Transform all predicted atoms by frames of each residue j
    p_in = atom_pred.permute(1, 0, 2).unsqueeze(-2)  # [Lj, B, 1, 3]
    applied = T_align.apply(p_in)  # [Li, B, Lj, 3] after broadcasting
    pred_aligned = applied.permute(1, 0, 2, 3)  # [B, Li, Lj, 3]

    # True atoms expanded across j
    true_exp = atom_true[:, :, None, :]  # [B, Li, 1, 3]
    d = torch.linalg.norm(pred_aligned - true_exp, dim=-1)  # [B, L, L]
    if clamp is not None:
        d = d.clamp(max=float(clamp))

    pair_mask = res_mask[:, :, None] & res_mask[:, None, :]  # [B, L, L]
    return d, pair_mask
