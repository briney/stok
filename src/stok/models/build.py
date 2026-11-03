"""Construct the paired MDLM model from composed scientific settings."""

from omegaconf import DictConfig
from torch import Tensor

from ..utils.tokenizer import Tokenizer
from .mdlm import STokMDLM


def build_model(cfg: DictConfig, *, codebook: Tensor | None) -> STokMDLM:
    """Keep artifact loading, RNG seeding, weight loading and devices in callers."""
    if codebook is None:
        raise ValueError("MDLM requires a structure codebook")
    enc, tokenizer = cfg.model.encoder, Tokenizer()
    model = STokMDLM(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        codebook=codebook,
        d_model=enc.d_model,
        n_heads=enc.n_heads,
        n_layers=enc.n_layers,
        ffn_mult=enc.ffn_mult,
        dropout=enc.dropout,
        attn_dropout=enc.attn_dropout,
        norm_type=enc.norm,
    )
    model.mdlm_regime_weights = dict(cfg.train.mdlm.get("regime_weights") or {})
    return model
