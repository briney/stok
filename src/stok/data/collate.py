import torch
from typing import Any


def simple_pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate batch of token/label pairs.

    Currently assumes all sequences are equal length and stacks them.
    For variable-length sequences, this function would need to pad to max length.

    Args:
        batch: List of (tokens, labels) tuples where each tensor has shape [L].
        pad_id: Padding token ID (currently unused).

    Returns:
        Tuple of (tokens, labels) with shapes [B, L] and [B, L].
    """
    tokens, labels = zip(*batch)
    return torch.stack(tokens, dim=0), torch.stack(labels, dim=0)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate batch for masked language modeling.

    Applies BERT-style masking:
    - mask_prob fraction of tokens are selected for prediction
    - Of selected tokens:
      - mask_token_prob are replaced with <mask>
      - random_token_prob are replaced with random token
      - remaining are kept unchanged

    Args:
        batch: List of dicts with 'seq' key containing amino acid sequences.
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
        Tuple of (input_ids, labels) tensors with shape [B, L].
    """
    if special_token_ids is None:
        # Default special tokens: <cls>=0, <pad>=1, <eos>=2, <unk>=3
        special_token_ids = {0, 1, 2, 3}

    input_ids_list = []
    labels_list = []

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

        # Apply masking strategy to selected positions
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

            # Apply random tokens (sample from amino acid range)
            num_random = random_token_mask.sum().item()
            if num_random > 0:
                random_tokens = torch.randint(4, 24, (num_random,))  # AA tokens
                ids[mask_indices[random_token_mask]] = random_tokens

        input_ids_list.append(ids)
        labels_list.append(labels)

    return torch.stack(input_ids_list), torch.stack(labels_list)
