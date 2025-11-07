import sys

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from stok.models.stok import STokModel
from stok.utils.codebook import load_codebook


def run_smoke_test(cfg: DictConfig):
    """Build the model and run a tiny forward pass for smoke testing.

    Args:
        cfg: Hydra configuration dictionary.
    """
    # # Warn if deprecated/unused decoder config is present
    # try:
    #     if "decoder" in cfg.model:
    #         print(
    #             "Warning: cfg.model.decoder is ignored. Decoder presets are selected "
    #             "via model.codebook.preset and loaded with load_pretrained_decoder()."
    #         )
    # except Exception:
    #     pass

    print(OmegaConf.to_yaml(cfg))

    # Load codebook from config (preset, path, or fallback to random)
    codebook = load_codebook(
        preset=cfg.model.codebook.get("preset"),
        path=cfg.model.codebook.get("path"),
    )

    # Infer codebook size from the loaded tensor
    codebook_size = codebook.shape[0]

    model = STokModel(
        vocab_size=cfg.model.encoder.vocab_size,
        pad_id=cfg.model.encoder.pad_id,
        d_model=cfg.model.encoder.d_model,
        n_heads=cfg.model.encoder.n_heads,
        n_layers=cfg.model.encoder.n_layers,
        ffn_mult=cfg.model.encoder.ffn_mult,
        dropout=cfg.model.encoder.dropout,
        attn_dropout=cfg.model.encoder.attn_dropout,
        codebook=codebook,
        classifier_kwargs=dict(
            use_cosine=cfg.model.classifier.use_cosine,
            learnable_temperature=cfg.model.classifier.learnable_temperature,
            bias_from_code_norm=cfg.model.classifier.bias_from_code_norm,
            projector_dim=cfg.model.classifier.projector_dim,
        ),
        norm_type=cfg.model.encoder.norm,
    )

    if cfg.print_model_summary:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_params/1e6:.2f}M")

    # Tiny forward sanity check
    B, L = 2, 16
    tokens = torch.randint(low=1, high=cfg.model.encoder.vocab_size, size=(B, L))
    tokens[:, -2:] = cfg.model.encoder.pad_id
    labels = torch.randint(low=0, high=codebook_size, size=(B, L))
    labels[:, -2:] = (
        cfg.model.classifier.ignore_index
        if "ignore_index" in cfg.model.classifier
        else -100
    )

    out = model(
        tokens=tokens, labels=labels, ignore_index=cfg.model.classifier.ignore_index
    )
    print("logits:", out["logits"].shape, "loss:", float(out["loss"].item()))
    print("OK")


if __name__ == "__main__":
    overrides = sys.argv[1:]
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=overrides)
        run_smoke_test(cfg)
