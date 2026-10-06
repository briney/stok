"""Compatibility imports for the shared training engine and loader helpers."""

import sys

from omegaconf import DictConfig

from stok.data.loaders import (
    MixtureSampler as MixtureSampler,
    _build_dataloaders as _build_dataloaders,
    _parse_eval_configs as _parse_eval_configs,
    _parse_train_configs as _parse_train_configs,
    _tokenize_and_align as _tokenize_and_align,
)
from stok.training import engine as _engine
from stok.training.engine import (
    _maybe_get_accelerator as _maybe_get_accelerator,
    _get_model_device as _get_model_device,
    _build_scheduler as _build_scheduler,
    _TeeIO as _TeeIO,
    _resolve_project_dirs as _resolve_project_dirs,
    _ensure_dirs as _ensure_dirs,
    _save_config_snapshot as _save_config_snapshot,
    _unwrap_model as _unwrap_model,
    _save_checkpoint as _save_checkpoint,
    _load_pretrained_encoder as _load_pretrained_encoder,
    _compute_accuracy as _compute_accuracy,
    _maybe_init_wandb as _maybe_init_wandb,
    iter_windows as iter_windows,
)


def run_training(cfg: DictConfig) -> None:
    _engine.run_training(cfg)


if __name__ == "__main__":
    print(
        "This module is intended to be invoked via the CLI: `stok train ...`",
        file=sys.stderr,
    )
