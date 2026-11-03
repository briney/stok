import torch


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
    ids = tokenizer.convert_tokens_to_ids(list(seq[: max_len - 2]))
    tokens = torch.full((max_len,), tokenizer.pad_token_id, dtype=torch.long)
    tokens[: len(ids) + 2] = torch.tensor(
        [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]
    )
    return tokens
