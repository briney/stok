import hashlib
import json

import torch
from typing import Any


def tokenize_residues(seq: str, tokenizer, max_len: int) -> torch.Tensor:
    """Encode exactly one token per biological position, plus BOS and EOS."""
    if max_len < 3:
        raise ValueError("max_len must be >= 3 (BOS, residue, EOS)")
    if not isinstance(seq, str) or any(
        c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ.-" for c in seq
    ):
        raise ValueError(
            "Sequence must contain single-character residues, not control tokens"
        )
    raw = tokenizer(seq, add_special_tokens=False)["input_ids"]
    if len(raw) != len(seq):
        raise ValueError("Tokenizer must encode exactly one token per residue")
    return tokenizer(
        seq,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"][0]


def align_coords(
    coords: torch.Tensor | None, *, residue_count: int, token_length: int
) -> torch.Tensor:
    """Shift raw residue coordinates past BOS; preserve missing sample rows."""
    aligned = torch.full((token_length, 3, 3), float("nan"))
    if coords is not None:
        if coords.ndim != 3 or coords.shape[1:] != (3, 3):
            raise ValueError("Coordinates must have shape [residues,3,3]")
        n = min(residue_count, token_length - 2)
        if len(coords) < n:
            raise ValueError("Coordinate length is shorter than the residue sequence")
        aligned[1 : 1 + n] = coords[:n]
    return aligned


def mlm_collate(
    batch: list[dict[str, Any]],
    tokenizer,
    *,
    max_len: int,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    pad_id: int | None = None,
    mask_id: int | None = None,
    ignore_index: int = -100,
    special_token_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
    eval_seed: int | None = None,
    dataset_name: str = "",
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Collate batch for masked language modeling.

    Applies BERT-style masking:
    - mask_prob fraction of tokens are selected for prediction
    - Of selected tokens:
      - mask_token_prob are replaced with <mask>
      - random_token_prob are replaced with random token
      - remaining are kept unchanged

    Args:
        batch: List of dicts with 'sequence' key containing amino acid sequences.
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
    if any(not 0 <= p <= 1 for p in (mask_prob, mask_token_prob, random_token_prob)):
        raise ValueError("MLM probabilities must be in [0, 1]")
    if mask_token_prob + random_token_prob > 1:
        raise ValueError("mask_token_prob + random_token_prob must be <= 1")
    if pad_id is not None and pad_id != tokenizer.pad_token_id:
        raise ValueError("pad_id must match tokenizer")
    if mask_id is not None and mask_id != tokenizer.mask_token_id:
        raise ValueError("mask_id must match tokenizer")
    pad_id, mask_id = tokenizer.pad_token_id, tokenizer.mask_token_id
    if pad_id is None or mask_id is None:
        raise ValueError("Tokenizer requires pad and mask tokens")
    special_token_ids = set(tokenizer.all_special_ids) | (special_token_ids or set())
    aa_ids = []
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        encoded = tokenizer(aa, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1 or encoded[0] in special_token_ids:
            raise ValueError(
                f"Tokenizer must encode amino acid {aa} as one known token"
            )
        aa_ids.append(encoded[0])
    aa_ids = torch.tensor(aa_ids)

    input_ids_list = []
    labels_list = []
    coords_list: list[torch.Tensor] = []

    for item in batch:
        seq: str = item["sequence"]
        sample_generator = generator
        if eval_seed is not None:
            identity = json.dumps(
                [eval_seed, dataset_name, item.get("sequence_id"), seq],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            seed = int.from_bytes(
                hashlib.blake2b(identity, digest_size=8).digest(), "big"
            )
            sample_generator = torch.Generator().manual_seed(seed)

        ids = tokenize_residues(seq, tokenizer, max_len).clone()
        labels = torch.full_like(ids, ignore_index)

        # Create mask for positions that CAN be masked (not special tokens)
        maskable = torch.ones_like(ids, dtype=torch.bool)
        for special_id in special_token_ids:
            maskable &= ids != special_id

        # Randomly select positions to mask
        probs = torch.rand(ids.shape, generator=sample_generator)
        mask_positions = (probs < mask_prob) & maskable

        # Store original tokens as labels for masked positions
        labels[mask_positions] = ids[mask_positions]

        # Apply masking strategy to selected positions
        mask_indices = mask_positions.nonzero(as_tuple=True)[0]
        num_masked = len(mask_indices)

        if num_masked > 0:
            rand = torch.rand(num_masked, generator=sample_generator)

            # 80% -> <mask> token
            mask_token_mask = rand < mask_token_prob
            # 10% -> random canonical amino-acid token
            random_token_mask = (rand >= mask_token_prob) & (
                rand < mask_token_prob + random_token_prob
            )
            # 10% -> keep original (no change needed)

            # Apply <mask> token
            ids[mask_indices[mask_token_mask]] = mask_id

            # Apply random tokens (sample from amino acid range)
            num_random = int(random_token_mask.sum().item())
            if num_random > 0:
                random_tokens = aa_ids[
                    torch.randint(
                        len(aa_ids), (num_random,), generator=sample_generator
                    )
                ]
                ids[mask_indices[random_token_mask]] = random_tokens

        input_ids_list.append(ids)
        labels_list.append(labels)

        # Extract optional coordinates tensor
        coords = item.get("coords")
        coords_list.append(
            align_coords(coords, residue_count=len(seq), token_length=max_len)
        )

    tokens = torch.stack(input_ids_list)
    labels = torch.stack(labels_list)

    if any(item.get("coords") is not None for item in batch):
        return tokens, labels, torch.stack(coords_list)
    return tokens, labels
