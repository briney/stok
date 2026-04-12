from __future__ import annotations

from typing import Any

import torch

from ..models.noise_schedule import NoiseSchedule
from ..utils.sampling import apply_noise, sample_t_antithetic


def mlm_collate(
    batch: list[dict[str, Any]],
    tokenizer,
    *,
    max_len: int,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    pad_id: int = 1,
    mask_id: int = 31,
    ignore_index: int = -100,
    special_token_ids: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate batch for masked language modeling.

    Applies BERT-style masking:
    - mask_prob fraction of tokens are selected for prediction
    - Of selected tokens:
      - mask_token_prob are replaced with <mask>
      - random_token_prob are replaced with random token
      - remaining are kept unchanged

    Args:
        batch: List of dicts with 'seq' key containing amino acid sequences.
            May also contain 'coords' key with coordinate tensors [L, 3, 3].
        tokenizer: Tokenizer instance for encoding sequences.
        max_len: Maximum sequence length.
        mask_prob: Probability of selecting a token for masking.
        mask_token_prob: Probability of replacing selected token with <mask>.
        random_token_prob: Probability of replacing selected token with random token.
        pad_id: Padding token ID.
        mask_id: Mask token ID.
        ignore_index: Index to use for non-masked positions in labels.
        special_token_ids: Set of token IDs to never mask (e.g., CLS, EOS, PAD).

    Returns:
        Tuple of (input_ids, labels) tensors with shape [B, L], or
        (input_ids, labels, coords) if coordinates are present in batch items.
    """
    if special_token_ids is None:
        special_token_ids = set(tokenizer.all_special_ids)

    # Derive amino acid token IDs from tokenizer for random replacement
    _standard_aa = "LAGVSERTIDPKQNFYMHWC"
    aa_token_ids = [
        tokenizer.convert_tokens_to_ids(aa)
        for aa in _standard_aa
        if tokenizer.convert_tokens_to_ids(aa) is not None
    ]

    input_ids_list = []
    labels_list = []
    coords_list: list[torch.Tensor] = []

    for item in batch:
        seq: str = item["seq"]

        enc = tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"][0].clone()  # [L]
        labels = torch.full_like(ids, ignore_index)

        # Create mask for positions that CAN be masked (not special tokens)
        maskable = torch.ones_like(ids, dtype=torch.bool)
        for special_id in special_token_ids:
            maskable &= ids != special_id

        # Randomly select positions to mask
        probs = torch.rand_like(ids, dtype=torch.float)
        mask_positions = (probs < mask_prob) & maskable

        # Store original tokens as labels for masked positions
        labels[mask_positions] = ids[mask_positions]

        # Guarantee at least one masked token per sequence
        mask_indices = mask_positions.nonzero(as_tuple=True)[0]
        num_masked = len(mask_indices)
        if num_masked == 0 and maskable.any():
            eligible = maskable.nonzero(as_tuple=True)[0]
            force_idx = eligible[torch.randint(len(eligible), (1,))]
            mask_positions[force_idx] = True
            labels[force_idx] = ids[force_idx]
            ids[force_idx] = mask_id
            mask_indices = mask_positions.nonzero(as_tuple=True)[0]
            num_masked = len(mask_indices)

        if num_masked > 0:
            rand = torch.rand(num_masked)

            # 80% -> <mask> token
            mask_token_mask = rand < mask_token_prob
            # 10% -> random token (amino acids only: indices 4-23 in DEFAULT_VOCAB)
            random_token_mask = (rand >= mask_token_prob) & (
                rand < mask_token_prob + random_token_prob
            )
            # 10% -> keep original (no change needed)

            # Apply <mask> token
            ids[mask_indices[mask_token_mask]] = mask_id

            # Apply random tokens (sample from amino acid token IDs)
            num_random = random_token_mask.sum().item()
            if num_random > 0:
                aa_ids_t = torch.tensor(aa_token_ids)
                random_tokens = aa_ids_t[torch.randint(len(aa_ids_t), (num_random,))]
                ids[mask_indices[random_token_mask]] = random_tokens

        input_ids_list.append(ids)
        labels_list.append(labels)

        # Extract optional coordinates tensor
        coords = item.get("coords")
        if coords is not None and isinstance(coords, torch.Tensor):
            coords_list.append(coords)

    tokens = torch.stack(input_ids_list)
    labels = torch.stack(labels_list)

    if len(coords_list) > 0:
        return tokens, labels, torch.stack(coords_list)
    return tokens, labels


def mdlm_collate(
    batch: list[dict[str, Any]],
    tokenizer,
    noise_schedule_seq: NoiseSchedule,
    noise_schedule_struct: NoiseSchedule | None = None,
    *,
    max_len: int,
    seq_mask_id: int,
    seq_pad_id: int,
    struct_mask_id: int | None = None,
    struct_pad_id: int | None = None,
    ignore_index: int = -100,
    antithetic_time_sampling: bool = True,
    independent_track_times: bool = True,
    tracks: str = "joint",
) -> dict[str, torch.Tensor | None]:
    """Collate batch for MDLM diffusion training.

    Tokenizes sequences, samples diffusion times, applies forward noising,
    and builds targets for the MDLM loss.

    Args:
        batch: List of dicts with ``"seq"`` key (and optionally ``"indices"``
            for joint mode).
        tokenizer: Tokenizer instance.
        noise_schedule_seq: Noise schedule for the sequence track.
        noise_schedule_struct: Noise schedule for structure track (joint only).
        max_len: Maximum sequence length (including special tokens).
        seq_mask_id: Mask token ID for sequences.
        seq_pad_id: Padding token ID for sequences.
        struct_mask_id: Mask token ID for structures (joint only).
        struct_pad_id: Padding token ID for structures (joint only).
        ignore_index: Label value for positions excluded from loss.
        antithetic_time_sampling: Use antithetic time pairs for variance reduction.
        independent_track_times: Sample independent times per track (joint only).
        tracks: Operating mode, ``"seq_only"`` or ``"joint"``.

    Returns:
        Dict with keys ``seq_tokens``, ``t_seq``, ``seq_targets``, ``seq_mask``,
        ``key_padding_mask``, and ``None`` placeholders for struct fields when
        in seq_only mode.
    """
    special_token_ids = set(tokenizer.all_special_ids)

    # --- Tokenize all sequences ---
    all_ids: list[torch.Tensor] = []
    for item in batch:
        enc = tokenizer(
            item["seq"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        all_ids.append(enc["input_ids"][0])  # [L]

    clean_tokens = torch.stack(all_ids)  # [B, L]
    B, L = clean_tokens.shape

    # Padding mask: True at pad positions
    key_padding_mask = clean_tokens == seq_pad_id  # [B, L]

    # Special token mask: True at positions that must not be masked
    special_mask = torch.zeros(B, L, dtype=torch.bool)
    for sid in special_token_ids:
        special_mask |= clean_tokens == sid

    # --- Sample diffusion times ---
    device = clean_tokens.device
    if antithetic_time_sampling:
        t_seq = sample_t_antithetic(B, device)
    else:
        t_seq = torch.rand(B, device=device).clamp(min=1e-5, max=1.0 - 1e-5)

    # --- Apply forward noising ---
    noised_tokens, seq_mask = apply_noise(
        tokens=clean_tokens,
        t=t_seq,
        mask_token_id=seq_mask_id,
        noise_schedule=noise_schedule_seq,
        padding_mask=key_padding_mask,
        special_token_mask=special_mask,
    )

    # --- Build targets: clean tokens at masked positions, ignore elsewhere ---
    seq_targets = torch.full_like(clean_tokens, ignore_index)
    seq_targets[seq_mask] = clean_tokens[seq_mask]

    result: dict[str, torch.Tensor | None] = {
        "seq_tokens": noised_tokens,
        "t_seq": t_seq,
        "seq_targets": seq_targets,
        "seq_mask": seq_mask,
        "key_padding_mask": key_padding_mask,
        # Struct fields (populated in Phase 3 for joint mode)
        "struct_tokens": None,
        "t_struct": None,
        "struct_targets": None,
        "struct_mask": None,
    }
    return result
