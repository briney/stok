"""Construct the paired MDLM model from composed scientific settings."""

from omegaconf import DictConfig
from torch import Tensor

from .mdlm import STokMDLM


def build_model(cfg: DictConfig, *, codebook: Tensor | None) -> STokMDLM:
    """Keep artifact loading, RNG seeding, weight loading and devices in callers."""
    objective = str(cfg.train.objective).lower()
    if objective != "mdlm":
        raise ValueError(f"Unknown train.objective: {objective}")
    enc = cfg.model.encoder
    kwargs = dict(
        vocab_size=enc.vocab_size,
        pad_id=enc.pad_id,
        d_model=enc.d_model,
        n_heads=enc.n_heads,
        n_layers=enc.n_layers,
        ffn_mult=enc.ffn_mult,
        dropout=enc.dropout,
        attn_dropout=enc.attn_dropout,
        norm_type=enc.norm,
    )
    if codebook is None:
        raise ValueError("codebook is required when train.objective='mdlm'")
    model = STokMDLM(**kwargs, codebook=codebook)
    model.mdlm_regime_weights = dict(cfg.train.mdlm.get("regime_weights") or {})
    return model
