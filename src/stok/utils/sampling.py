"""Sampling, noising, and time sampling utilities."""

from __future__ import annotations

import torch


def top_p_sample(
    logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0
) -> torch.Tensor:
    """Top-p (nucleus) sampling from logits.

    Args:
        logits: Logit tensor, shape [B, L, C].
        p: Cumulative probability threshold.
        temperature: Temperature scaling factor.

    Returns:
        Sampled token indices, shape [B, L].
    """
    if temperature != 1.0:
        logits = logits / temperature
    B, L, C = logits.shape
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    to_remove = cumprobs > p
    to_remove[..., 0] = False  # keep at least the top-1
    sorted_probs = torch.where(to_remove, torch.zeros_like(sorted_probs), sorted_probs)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    ranks = torch.multinomial(sorted_probs.reshape(-1, C), 1).view(B, L, 1)
    sampled = torch.gather(sorted_idx, -1, ranks).squeeze(-1)
    return sampled


# ---------------------------------------------------------------------------
# MDLM diffusion utilities
# ---------------------------------------------------------------------------


def sample_t_antithetic(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample diffusion times with antithetic pairs for variance reduction.

    For each pair of samples, draws t and 1 - t. If batch_size is odd,
    the last sample gets a plain uniform draw.

    Args:
        batch_size: Number of time samples to draw.
        device: Device for the output tensor.

    Returns:
        Tensor of diffusion times in (0, 1), shape [batch_size].
    """
    n_pairs = batch_size // 2
    u = torch.rand(n_pairs, device=device)
    # Clamp away from exact 0 and 1 for numerical safety
    u = u.clamp(min=1e-5, max=1.0 - 1e-5)
    pairs = torch.stack([u, 1.0 - u], dim=-1).reshape(-1)  # [2 * n_pairs]
    if batch_size % 2 == 1:
        extra = torch.rand(1, device=device).clamp(min=1e-5, max=1.0 - 1e-5)
        pairs = torch.cat([pairs, extra])
    return pairs


def guarantee_min_mask(
    mask: torch.Tensor,
    padding_mask: torch.Tensor,
    special_token_mask: torch.Tensor | None,
    min_masked: int = 1,
) -> torch.Tensor:
    """Ensure at least min_masked tokens are masked per sequence.

    For any sequence with fewer than min_masked masked positions, randomly
    selects eligible (non-padding, non-special) positions to force-mask.

    Args:
        mask: Current mask, True at masked positions, shape [B, L].
        padding_mask: True at padding positions, shape [B, L].
        special_token_mask: True at special token positions, shape [B, L].
            May be None (no special tokens excluded).
        min_masked: Minimum number of masked positions per sequence.

    Returns:
        Updated mask with at least min_masked True positions per sequence
        (where enough eligible positions exist).
    """
    mask = mask.clone()
    # Eligible = not padding, not special, not already masked
    eligible = ~padding_mask
    if special_token_mask is not None:
        eligible = eligible & ~special_token_mask

    per_seq_count = mask.sum(dim=-1)  # [B]
    needs_more = per_seq_count < min_masked  # [B]

    if not needs_more.any():
        return mask

    for i in range(mask.size(0)):
        if not needs_more[i]:
            continue
        current = int(per_seq_count[i].item())
        need = min_masked - current
        candidates = (eligible[i] & ~mask[i]).nonzero(as_tuple=True)[0]
        if len(candidates) == 0:
            continue
        n_pick = min(need, len(candidates))
        perm = torch.randperm(len(candidates), device=mask.device)[:n_pick]
        mask[i, candidates[perm]] = True

    return mask


def apply_noise(
    tokens: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    noise_schedule,
    padding_mask: torch.Tensor,
    special_token_mask: torch.Tensor | None = None,
    position_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply forward diffusion: independently mask each token with P(mask) = 1 - alpha(t).

    Args:
        tokens: Clean token IDs, shape [B, L].
        t: Diffusion time per sample, shape [B].
        mask_token_id: Token ID to use for masked positions.
        noise_schedule: NoiseSchedule instance providing alpha(t).
        padding_mask: True at padding positions, shape [B, L].
        special_token_mask: True at positions that should never be masked
            (e.g., CLS, EOS), shape [B, L]. May be None.
        position_weights: Per-position masking weights (future, unused).

    Returns:
        Tuple of (noised_tokens, mask) where:
        - noised_tokens: Tokens with some replaced by mask_token_id, shape [B, L].
        - mask: Boolean tensor, True at masked positions, shape [B, L].
    """
    B, L = tokens.shape
    # alpha(t): probability token is NOT masked. Shape [B] -> [B, 1]
    alpha = noise_schedule.alpha(t, position_weights=position_weights)
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)  # [B, 1]

    # Sample Bernoulli mask: True where token IS masked (with prob 1 - alpha)
    rand = torch.rand(B, L, device=tokens.device)
    mask = rand >= alpha  # [B, L]

    # Exclude padding and special tokens from masking
    mask = mask & ~padding_mask
    if special_token_mask is not None:
        mask = mask & ~special_token_mask

    # Guarantee at least one masked token per sequence
    mask = guarantee_min_mask(mask, padding_mask, special_token_mask, min_masked=1)

    # Apply mask token
    noised = tokens.clone()
    noised[mask] = mask_token_id

    return noised, mask
