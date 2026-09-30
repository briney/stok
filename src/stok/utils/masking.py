import torch


def key_padding_mask_from_tokens(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Build key padding mask from token IDs.

    Args:
        tokens: Token IDs tensor of shape [B, L].
        pad_id: Padding token ID.

    Returns:
        Boolean mask of shape [B, L] where True marks padding positions.
    """
    return tokens == pad_id


def residue_mask_from_tokens(
    tokens: torch.Tensor, *, pad_id: int, bos_id: int, eos_id: int
) -> torch.Tensor:
    """Keep biological positions, including unknown and MLM-masked residues."""
    return (tokens != pad_id) & (tokens != bos_id) & (tokens != eos_id)
