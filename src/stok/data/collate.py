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
