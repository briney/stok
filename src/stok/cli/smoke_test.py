import sys

from omegaconf import DictConfig, OmegaConf

from stok.config import load_training_config
from stok.models.build import build_model
from stok.utils.codebook import load_codebook


def run_smoke_test(cfg: DictConfig):
    """Build the model and run a tiny forward pass for smoke testing.

    Args:
        cfg: Hydra configuration dictionary.
    """
    print(OmegaConf.to_yaml(cfg))

    # Load codebook from config (preset, path, or fallback to random)
    codebook = load_codebook(
        preset=cfg.model.codebook.get("preset"),
        path=cfg.model.codebook.get("path"),
    )

    # Infer codebook size from the loaded tensor
    codebook_size = codebook.shape[0]

    model = build_model(cfg, codebook=codebook)
    # Explicit synthetic forward fixture; training still requires completed real sources.
    from stok.data.mdlm import prepare_mdlm_batch
    from stok.utils.tokenizer import Tokenizer

    batch = prepare_mdlm_batch(
        [
            {
                "dataset": "synthetic-smoke",
                "sequence_id": "0",
                "sequence": "ACDE",
                "structure_tokens": [0, 1, None, 0],
            }
        ],
        Tokenizer(),
        max_len=6,
        codebook_size=codebook_size,
        crop="center",
        seeds=[0],
    )
    out = model(batch["sequence_tokens"], batch["structure_tokens"])
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(
        "sequence_logits:",
        out["sequence_logits"].shape,
        "structure_logits:",
        out["structure_logits"].shape,
    )
    print("OK")
    return


if __name__ == "__main__":
    overrides = sys.argv[1:]
    run_smoke_test(load_training_config(overrides))
