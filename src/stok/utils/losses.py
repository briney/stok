from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import frames_from_ncac


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

    # When no valid labels exist, return zero to avoid NaN loss
    valid_count = (labels_flat != ignore_index).sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
    )


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
    T_pred = frames_from_ncac(pred_coords)
    T_true = frames_from_ncac(true_coords)

    # 2) Build masks (valid residues/atoms). Always intersect with finiteness.
    residue_valid_true = torch.isfinite(true_coords).all(dim=(-2, -1))  # [B, L]
    if residue_mask is not None:
        residue_valid = residue_mask.to(torch.bool) & residue_valid_true
    else:
        residue_valid = residue_valid_true

    # Also require predicted residues to be finite
    residue_valid_pred = torch.isfinite(pred_coords).all(dim=(-2, -1))  # [B, L]
    residue_valid = residue_valid & residue_valid_pred

    # Set invalid frames to identity to avoid NaNs in transforms
    T_pred = T_pred.mask(~residue_valid)
    T_true = T_true.mask(~residue_valid)

    pair_mask = residue_valid[:, :, None] & residue_valid[:, None, :]  # [B, L, L]
    # Atom validity requires both true and predicted atoms to be finite
    atom_valid_true = torch.isfinite(true_coords).all(dim=-1)  # [B, L, 3]
    atom_valid_pred = torch.isfinite(pred_coords).all(dim=-1)  # [B, L, 3]
    atom_valid = atom_valid_true & atom_valid_pred  # [B, L, 3]
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
    denom = full_mask.sum(dim=(1, 2, 3)).to(per.dtype)

    # If entire batch is invalid, return 0.0
    if (denom == 0).all():
        return torch.tensor(0.0, device=per.device, dtype=per.dtype, requires_grad=True)

    denom = denom.clamp_min(1)
    loss_b = per.sum(dim=(1, 2, 3)) / denom  # [B]
    return loss_b.mean()


class MDLMLoss(nn.Module):
    """Rao-Blackwellized MDLM loss for a single track.

    Computes weighted cross-entropy at masked positions, where the weight
    depends on the noise schedule derivative at the sampled diffusion time.

    Args:
        noise_schedule: Noise schedule providing loss_weight(t).
        ignore_index: Label index to ignore (padding / unmasked positions).
    """

    def __init__(self, noise_schedule, ignore_index: int = -100):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        padding_mask: torch.Tensor,
        position_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MDLM loss.

        Args:
            logits: Model predictions, shape [B, L, V].
            targets: Ground-truth token IDs, shape [B, L].
            mask: Boolean mask, True at masked (noised) positions, shape [B, L].
            t: Diffusion times per sample, shape [B].
            padding_mask: Boolean mask, True at padding positions, shape [B, L].
            position_weights: Per-position loss weights (unused, future extension).

        Returns:
            Scalar loss tensor (mean over valid masked non-padding positions).
        """
        # If no positions are masked, return zero loss
        valid = mask & ~padding_mask  # [B, L]
        num_valid = valid.sum()
        if num_valid == 0:
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        # Loss weight from noise schedule: w(t) = |alpha'(t)| / (1 - alpha(t))
        # Shape: [B] -> [B, 1] for broadcasting
        w = self.noise_schedule.loss_weight(t).unsqueeze(-1)  # [B, 1]

        # Per-position CE loss (no reduction)
        V = logits.size(-1)
        per_pos_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
            reduction="none",
        ).reshape_as(targets)  # [B, L]

        # Zero out loss at padding and unmasked positions
        per_pos_loss = per_pos_loss * valid.float()

        # Apply time-dependent weight (broadcast [B, 1] over [B, L])
        weighted = per_pos_loss * w

        # Mean over valid positions
        return weighted.sum() / num_valid.float().clamp(min=1.0)
