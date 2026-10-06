"""Construct the existing objective-specific models from composed settings."""

from omegaconf import DictConfig
from torch import Tensor

from .mdlm import STokMDLM
from .stok import STokModel


def build_model(cfg: DictConfig, *, codebook: Tensor | None) -> STokModel | STokMDLM:
    """Keep artifact loading, RNG seeding, weight loading and devices in callers."""
    objective = str(cfg.train.get("objective", "codebook")).lower()
    if objective not in {"mdlm", "mlm", "codebook"}:
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
    if objective == "mdlm":
        if codebook is None:
            raise ValueError("codebook is required when train.objective='mdlm'")
        model = STokMDLM(**kwargs, codebook=codebook)
        model.mdlm_regime_weights = dict(cfg.train.mdlm.get("regime_weights") or {})
        return model
    is_mlm = objective == "mlm"
    return STokModel(
        **kwargs,
        codebook=None if is_mlm else codebook,
        classifier_kwargs=(
            dict(
                use_cosine=cfg.model.classifier.use_cosine,
                learnable_temperature=cfg.model.classifier.learnable_temperature,
                bias_from_code_norm=cfg.model.classifier.bias_from_code_norm,
                projector_dim=cfg.model.classifier.projector_dim,
            )
            if not is_mlm
            else None
        ),
        head_type=objective,
        tie_word_embeddings=(
            cfg.train.mlm.get("tie_word_embeddings", True) if is_mlm else True
        ),
    )
