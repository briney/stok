import torch


def indices_to_codes(
    codebook: torch.Tensor, indices: torch.Tensor, *, allow_missing: bool = False
) -> torch.Tensor:
    """
    Gather code vectors by index.

    Args:
        codebook: [C, d_code] codebook matrix.
        indices: [B, L] integer indices in [0, C).
        allow_missing: Permit -1 as an unavailable slot and return a zero vector.

    Returns:
        [B, L, d_code] tensor of code vectors.
    """
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape [C, d_code]")
    if indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be integer [B, L]")
    C = codebook.size(0)
    if (indices < (-1 if allow_missing else 0)).any() or (indices >= C).any():
        raise ValueError("indices out of range for provided codebook")
    codes = codebook[indices.clamp_min(0).long()]
    return codes.masked_fill((indices == -1)[..., None], 0)


def decode_structure_tokens(
    decoder,
    codebook: torch.Tensor,
    indices: torch.Tensor,
    *,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    """Decode available labels in place; return NaN for holes and padding."""
    if residue_mask.dtype != torch.bool or residue_mask.shape != indices.shape:
        raise ValueError("residue_mask must be boolean [B,L] matching indices")
    codes = indices_to_codes(codebook, indices, allow_missing=True)
    if ((indices >= 0) & ~residue_mask).any():
        raise ValueError("Available tokens must belong to real residue positions")
    available = (indices >= 0) & residue_mask
    result = codes.new_full((*indices.shape, 3, 3), float("nan"))
    active = available.any(dim=1)
    if active.any():
        decoded = decode_coords(decoder, codes[active], available[active])
        result[active] = decoded.masked_fill(
            ~available[active, :, None, None], float("nan")
        )
    return result


def decode_coords(
    decoder,
    codes: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Run the geometric decoder and return [B, L, 3, 3] backbone coordinates.

    Args:
        decoder: GeometricDecoder instance.
        codes: [B, L, d_code] code vectors.
        mask: [B, L] boolean mask (True = valid).

    Returns:
        [B, L, 3, 3] coordinates for N, CA, C atoms.
    """
    if codes.ndim != 3:
        raise ValueError("codes must have shape [B, L, d_code]")
    if mask.ndim != 2 or mask.shape[:2] != codes.shape[:2]:
        raise ValueError("mask must be [B, L] matching codes")
    bb = decoder(codes, mask=mask)  # [B, L, 9]
    return bb.view(bb.size(0), bb.size(1), 3, 3)


def decode_token_aligned_coords(
    decoder, codes: torch.Tensor, residue_mask: torch.Tensor
) -> torch.Tensor:
    """Decode residue slots only, then restore the full token-aligned shape."""
    result = codes.new_full((*codes.shape[:2], 3, 3), float("nan"))
    active = residue_mask.any(dim=1)
    if active.any():
        decoded = decode_coords(
            decoder, codes[active, 1:-1], residue_mask[active, 1:-1]
        )
        result[active, 1:-1] = decoded.masked_fill(
            ~residue_mask[active, 1:-1, None, None], float("nan")
        )
    return result
