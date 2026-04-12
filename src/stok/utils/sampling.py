"""Sampling, noising, and time sampling utilities."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# MDLM iterative unmasking sampler
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample(
    model,
    length: int,
    num_samples: int = 1,
    num_steps: int = 100,
    condition_seq: torch.Tensor | None = None,
    condition_struct: torch.Tensor | None = None,
    seq_mask_positions: torch.Tensor | None = None,
    struct_mask_positions: torch.Tensor | None = None,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Generate sequences (and structures) via iterative unmasking.

    Supports multiple generation modes based on conditioning:
    - **Codesign**: both tracks fully masked, generate both.
    - **Forward**: sequence conditioned, generate structure.
    - **Inverse**: structure conditioned, generate sequence.
    - **Scaffold**: partial conditioning on either/both tracks via
      ``seq_mask_positions`` / ``struct_mask_positions``.

    Args:
        model: An MDLMModel instance (seq_only or joint).
        length: Sequence length (excluding special tokens handled internally).
        num_samples: Number of samples to generate.
        num_steps: Number of discrete unmasking steps.
        condition_seq: Pre-filled sequence tokens [num_samples, length].
            Positions NOT in ``seq_mask_positions`` are held fixed.
        condition_struct: Pre-filled structure tokens [num_samples, length].
            Positions NOT in ``struct_mask_positions`` are held fixed.
        seq_mask_positions: Boolean mask [num_samples, length] indicating
            which sequence positions to generate (True = generate).
            If None and condition_seq is None, all positions are generated.
            If None and condition_seq is provided, no positions are generated.
        struct_mask_positions: Boolean mask [num_samples, length] for struct.
        temperature: Sampling temperature (1.0 = unmodified logits).
        device: Device to run generation on.

    Returns:
        Dict with keys ``"seq_tokens"`` (and ``"struct_tokens"`` for joint
        mode), containing generated token IDs of shape [num_samples, length].
    """
    model.eval()
    B, L = num_samples, length
    is_joint = getattr(model, "tracks", "seq_only") == "joint"

    seq_mask_id = model.seq_mask_id
    seq_pad_id = model.seq_pad_id

    # Initialize sequence tokens
    if condition_seq is not None:
        seq_tokens = condition_seq.clone().to(device)
        if seq_mask_positions is None:
            # Condition is fully provided — nothing to generate for seq
            seq_gen_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        else:
            seq_gen_mask = seq_mask_positions.to(device)
            seq_tokens[seq_gen_mask] = seq_mask_id
    else:
        # Fully masked — generate everything
        seq_tokens = torch.full((B, L), seq_mask_id, dtype=torch.long, device=device)
        seq_gen_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Initialize structure tokens (joint mode only)
    struct_tokens = None
    struct_gen_mask = None
    if is_joint:
        struct_mask_id = model.struct_mask_id
        if condition_struct is not None:
            struct_tokens = condition_struct.clone().to(device)
            if struct_mask_positions is None:
                struct_gen_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            else:
                struct_gen_mask = struct_mask_positions.to(device)
                struct_tokens[struct_gen_mask] = struct_mask_id
        else:
            struct_tokens = torch.full(
                (B, L), struct_mask_id, dtype=torch.long, device=device
            )
            struct_gen_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Padding mask: none during generation (no padding in generated sequences)
    key_padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

    # Time steps: iterate from t=1 (fully masked) to t=0 (fully unmasked)
    time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for step_idx in range(num_steps):
        t_now = time_steps[step_idx]
        t_next = time_steps[step_idx + 1]

        # Current mask: positions still masked
        seq_mask = seq_tokens == seq_mask_id
        struct_mask = None
        if is_joint and struct_tokens is not None:
            struct_mask = struct_tokens == struct_mask_id

        # Diffusion time for this step
        t_seq = torch.full((B,), t_now.item(), device=device)
        t_struct = t_seq.clone() if is_joint else None

        # Forward pass
        outputs = model(
            seq_tokens=seq_tokens,
            t_seq=t_seq,
            seq_mask=seq_mask,
            key_padding_mask=key_padding_mask,
            struct_tokens=struct_tokens,
            t_struct=t_struct,
            struct_mask=struct_mask,
        )

        # Compute unmasking probability for this step
        # At each step, unmask a fraction of remaining masked positions
        if t_next > 0:
            # Fraction of tokens that should be unmasked at t_next
            noise_schedule = model.loss_fn_seq.noise_schedule
            alpha_now = noise_schedule.alpha(t_now.unsqueeze(0)).item()
            alpha_next = noise_schedule.alpha(t_next.unsqueeze(0)).item()
            # P(unmask) = (alpha_next - alpha_now) / (1 - alpha_now)
            unmask_prob = (alpha_next - alpha_now) / max(1.0 - alpha_now, 1e-8)
            unmask_prob = min(max(unmask_prob, 0.0), 1.0)
        else:
            # Last step: unmask everything
            unmask_prob = 1.0

        # Sample and unmask sequence positions
        seq_logits = outputs["seq_logits"]
        _unmask_positions(
            seq_tokens, seq_logits, seq_mask, seq_gen_mask,
            unmask_prob, temperature, seq_mask_id,
        )

        # Sample and unmask structure positions (joint mode)
        if is_joint and struct_tokens is not None and struct_mask is not None:
            struct_logits = outputs.get("struct_logits")
            if struct_logits is not None and struct_gen_mask is not None:
                _unmask_positions(
                    struct_tokens, struct_logits, struct_mask, struct_gen_mask,
                    unmask_prob, temperature, struct_mask_id,
                )

    # Final force-unmask: ensure no mask tokens remain at generated positions
    remaining_seq = (seq_tokens == seq_mask_id) & seq_gen_mask
    if remaining_seq.any():
        # One more forward pass to get logits for any remaining positions
        t_zero = torch.zeros(B, device=device)
        seq_mask_final = seq_tokens == seq_mask_id
        struct_mask_final = None
        if is_joint and struct_tokens is not None:
            struct_mask_final = struct_tokens == struct_mask_id
        outputs = model(
            seq_tokens=seq_tokens,
            t_seq=t_zero,
            seq_mask=seq_mask_final,
            key_padding_mask=key_padding_mask,
            struct_tokens=struct_tokens,
            t_struct=t_zero if is_joint else None,
            struct_mask=struct_mask_final,
        )
        # Force unmask all remaining
        if temperature != 1.0:
            final_logits = outputs["seq_logits"] / temperature
        else:
            final_logits = outputs["seq_logits"]
        sampled = final_logits.argmax(dim=-1)
        seq_tokens[remaining_seq] = sampled[remaining_seq]

        if is_joint and struct_tokens is not None:
            remaining_struct = (
                struct_tokens == struct_mask_id
            ) & struct_gen_mask  # type: ignore[operator]
            if remaining_struct.any():
                struct_logits_final = outputs.get("struct_logits")
                if struct_logits_final is not None:
                    sampled_struct = struct_logits_final.argmax(dim=-1)
                    struct_tokens[remaining_struct] = sampled_struct[remaining_struct]

    result: dict[str, torch.Tensor] = {"seq_tokens": seq_tokens}
    if is_joint and struct_tokens is not None:
        result["struct_tokens"] = struct_tokens

    return result


def _unmask_positions(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    current_mask: torch.Tensor,
    gen_mask: torch.Tensor,
    unmask_prob: float,
    temperature: float,
    mask_token_id: int,
) -> None:
    """Probabilistically unmask positions in-place.

    Args:
        tokens: Current token tensor [B, L], modified in-place.
        logits: Model logits [B, L, V].
        current_mask: Boolean mask of currently masked positions [B, L].
        gen_mask: Boolean mask of positions eligible for generation [B, L].
        unmask_prob: Probability of unmasking each eligible position.
        temperature: Sampling temperature.
        mask_token_id: The mask token ID.
    """
    eligible = current_mask & gen_mask  # [B, L]
    if not eligible.any():
        return

    # Decide which positions to unmask this step
    unmask_draw = torch.rand_like(eligible, dtype=torch.float)
    to_unmask = eligible & (unmask_draw < unmask_prob)

    if not to_unmask.any():
        return

    # Sample tokens at positions to unmask
    if temperature != 1.0:
        scaled_logits = logits / temperature
    else:
        scaled_logits = logits

    # Clamp logits to prevent inf/nan in softmax (SUBS uses dtype min/max)
    scaled_logits = scaled_logits.clamp(min=-1e9, max=1e9)
    probs = torch.softmax(scaled_logits, dim=-1)  # [B, L, V]
    # Ensure numerical safety for multinomial
    probs = probs.clamp(min=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Sample from categorical distribution
    B, L, V = probs.shape
    flat_probs = probs.reshape(-1, V)
    sampled_flat = torch.multinomial(flat_probs, 1).squeeze(-1)  # [B*L]
    sampled = sampled_flat.reshape(B, L)

    tokens[to_unmask] = sampled[to_unmask]
