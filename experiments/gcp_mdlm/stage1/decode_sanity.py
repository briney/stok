"""Decode predicted vs. ground-truth tokens and compare backbones (native-token ceiling)."""

from __future__ import annotations

import torch

from stok.utils.decoding import decode_coords, indices_to_codes
from stok.utils.metrics import lddt_ca, rmsd


def decode_sanity_row(
    pred_tokens: torch.Tensor,
    gt_tokens: torch.Tensor,
    codebook: torch.Tensor,
    decoder,
    *,
    device: str | torch.device = "cpu",
) -> dict:
    """Compare decoded predicted vs. decoded ground-truth backbone for one protein.

    Args:
        pred_tokens, gt_tokens: 1-D valid-residue token-id tensors of equal length.
        codebook: ``(C, d_code)`` prototypes.
        decoder: frozen ``GeometricDecoder`` (or compatible callable).

    Returns:
        ``{"lddt", "rmsd", "identical_tokens"}`` comparing predicted to ground-truth coords.
    """
    codebook = codebook.to(device)
    length = int(gt_tokens.shape[0])
    mask = torch.ones(1, length, dtype=torch.bool, device=device)
    pred_codes = indices_to_codes(codebook, pred_tokens.view(1, -1).to(device))
    gt_codes = indices_to_codes(codebook, gt_tokens.view(1, -1).to(device))
    with torch.no_grad():
        pred_coords = decode_coords(decoder, pred_codes, mask)  # (1, L, 3, 3)
        gt_coords = decode_coords(decoder, gt_codes, mask)
    residue_mask = mask
    lddt_val, _ = lddt_ca(pred_coords, gt_coords, residue_mask)
    rmsd_val = rmsd(pred_coords, gt_coords, residue_mask)
    return {
        "lddt": float(lddt_val.mean().cpu()),
        "rmsd": float(rmsd_val.mean().cpu()),
        "identical_tokens": bool(torch.equal(pred_tokens.cpu(), gt_tokens.cpu())),
    }
