"""Random-init full-train cross-check arms built on the one-shot STokModel.

``head_type="codebook"`` is the prototype-tied arm; ``head_type="mlm"`` with
``vocab_size=num_classes`` is the independent (DPLM-2-style) arm. Both train the
backbone and head jointly from random init on structure-token targets.
"""

from __future__ import annotations

import torch

from stok.models.stok import STokModel


def build_crosscheck_model(
    head_type: str,
    *,
    vocab_size: int,
    pad_id: int,
    num_classes: int,
    codebook: torch.Tensor | None,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ffn_mult: float = 2.667,
) -> STokModel:
    """Build a random-init one-shot seq->structure-token model for the cross-check."""
    if head_type == "codebook":
        return STokModel(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_mult=ffn_mult,
            dropout=0.0,
            attn_dropout=0.0,
            codebook=codebook,
            head_type="codebook",
        )
    if head_type == "mlm":
        # Independent classifier: LMHead sized to the codebook classes.
        return STokModel(
            vocab_size=num_classes,
            pad_id=pad_id,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_mult=ffn_mult,
            dropout=0.0,
            attn_dropout=0.0,
            head_type="mlm",
            tie_word_embeddings=False,
        )
    raise ValueError(f"unknown head_type: {head_type!r}")
