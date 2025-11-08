import torch
import torch.nn.functional as F


def logits_to_soft_codes_gumbel(
    logits: torch.Tensor,
    codebook: torch.Tensor,
    *,
    tau: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """
    Map per-position logits over codes to a (soft) mixture of code embeddings using Gumbel-Softmax.

    Args:
        logits: [B, L, C] unnormalized logits over code indices.
        codebook: [C, d_code] codebook matrix (classifier.E).
        tau: Gumbel-Softmax temperature.
        hard: If True, uses straight-through one-hot selections.

    Returns:
        Tensor of shape [B, L, d_code] representing code vectors per position.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, L, C]")
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape [C, d_code]")
    if logits.size(-1) != codebook.size(0):
        raise ValueError("logits last dim (C) must match codebook rows (C)")
    weights = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # [B, L, C]
    return weights @ codebook  # [B, L, d_code]


def sample_indices_top_p(
    probs: torch.Tensor,
    *,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Per-position nucleus (top-p) sampling over probabilities.

    Args:
        probs: [B, L, C] probabilities (must be non-negative and sum to 1 over C).
        top_p: cumulative probability mass to retain.
        temperature: softens/sharpens the distribution (acts on probs).

    Returns:
        Long tensor [B, L] of sampled indices.
    """
    if probs.ndim != 3:
        raise ValueError("probs must have shape [B, L, C]")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    # temperature scaling on probabilities (approximate; keep numerically stable)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature != 1.0:
        # avoid zero^a issues
        probs = (probs + 1e-8).pow(1.0 / float(temperature))
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # sort by descending prob
    sorted_p, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # [B,L,C]
    cum = sorted_p.cumsum(dim=-1)
    keep = cum <= top_p
    # ensure at least one kept
    keep[..., 0] = True
    masked = sorted_p * keep
    # renorm
    masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    # sample
    flat = masked.reshape(-1, masked.size(-1))  # [B*L, C]
    sampled = torch.distributions.Categorical(flat).sample()  # [B*L]
    sampled = sampled.view(masked.size(0), masked.size(1))  # [B, L]
    # map back to original indices
    idx = sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return idx.long()


def indices_to_codes(codebook: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather code vectors by index.

    Args:
        codebook: [C, d_code] codebook matrix.
        indices: [B, L] integer indices in [0, C).

    Returns:
        [B, L, d_code] tensor of code vectors.
    """
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape [C, d_code]")
    if indices.ndim != 2:
        raise ValueError("indices must have shape [B, L]")
    C = codebook.size(0)
    if (indices < 0).any() or (indices >= C).any():
        raise ValueError("indices out of range for provided codebook")
    return codebook[indices]  # type: ignore[index]


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


