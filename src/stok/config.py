"""Shared training configuration: defaults, YAML overlays, then Hydra overrides."""

from copy import deepcopy
from importlib.resources import as_file, files
from pathlib import Path
import re
from typing import Any, Sequence, cast

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


_TRAIN_RENAMES = {
    "num_steps": "max_steps",
    "epochs": "max_epochs",
    "grad_accum_steps": "gradient_accumulation_steps",
    "grad_clip_norm": "max_grad_norm",
    "precision": "mixed_precision",
    "project_path": "output_dir",
    "log_steps": "log_every",
    "checkpoint_steps": "save_every",
}


def _rewrite_training_references(value, path=()):
    """Keep scalar interpolations live when their target or container moves."""
    renamed = {f"train.{old}": f"train.{new}" for old, new in _TRAIN_RENAMES.items()}
    renamed.update(
        {
            "data.batch_size": "train.batch_size",
            "train.optimizer.name": "train.optimizer",
            "train.optimizer.lr": "train.lr",
            "train.optimizer.weight_decay": "train.weight_decay",
            "train.optimizer.eps": "train.adam_eps",
            "train.optimizer.betas.0": "train.adam_beta1",
            "train.optimizer.betas.1": "train.adam_beta2",
            **{
                f"train.scheduler.{key}": f"train.{key}"
                for key in ("warmup_steps", "stable_steps", "decay_steps")
            },
        }
    )
    if isinstance(value, dict):
        return {
            key: _rewrite_training_references(item, (*path, str(key)))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_training_references(item, (*path, str(i)))
            for i, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    def replace(match):
        reference = match[3]
        dots = len(reference) - len(reference.lstrip("."))
        if dots:
            if dots > len(path):
                return match[0]
            reference = ".".join((*path[:-dots], reference[dots:]))
        replacements = {
            "train.optimizer.betas": "train.adam_beta1 and train.adam_beta2",
            "train.scheduler.decay": "the new train.scheduler names",
        }
        if reference in replacements:
            raise ValueError(
                f"Update interpolation for {reference}: use {replacements[reference]}"
            )
        # Absolute references also preserve relative targets when the source moves.
        return match[1] + match[2] + renamed.get(reference, reference) + match[2]

    return re.sub(
        r"(?<!\\)(\$\{(?:oc\.select:\s*)?)(['\"]?)([\w.]+)\2(?=\s*[,}])", replace, value
    )


def normalize_training_config(config: dict, *, checkpoint: bool = False) -> dict:
    """Upgrade legacy YAML/checkpoint fields without mutating the original."""
    config = _rewrite_training_references(deepcopy(config))
    train = config.get("train", {})
    data = config.get("data", {})
    if not isinstance(train, dict):
        raise ValueError("train must be a mapping")

    def put(key, value):
        if key in train and train[key] != value:
            raise ValueError(f"Conflicting legacy and current values for train.{key}")
        train[key] = value

    if isinstance(data, dict) and "batch_size" in data:
        put("batch_size", data.pop("batch_size"))
        config["train"] = train
    for old, new in _TRAIN_RENAMES.items():
        if old in train:
            put(new, train.pop(old))
    if isinstance(train.get("optimizer"), dict):
        optimizer = train.pop("optimizer")
        for old, new in (
            ("name", "optimizer"),
            ("lr", "lr"),
            ("weight_decay", "weight_decay"),
            ("eps", "adam_eps"),
        ):
            if old in optimizer:
                put(new, optimizer.pop(old))
        if "betas" in optimizer:
            beta1, beta2 = optimizer.pop("betas")
            put("adam_beta1", beta1)
            put("adam_beta2", beta2)
        if checkpoint and "optimizer" in train:
            train.setdefault("adam_eps", 1e-8)
        if optimizer:
            raise ValueError(
                f"Unsupported legacy optimizer fields: {sorted(optimizer)}"
            )
    if isinstance(train.get("scheduler"), dict):
        scheduler = train.pop("scheduler")
        for key in ("warmup_steps", "stable_steps", "decay_steps"):
            if key in scheduler:
                put(key, scheduler.pop(key))
        if "decay" in scheduler:
            prefix = "wsd" if train.get("stable_steps") else "warmup"
            put("scheduler", f"{prefix}_{scheduler.pop('decay')}")
        if scheduler:
            raise ValueError(
                f"Unsupported legacy scheduler fields: {sorted(scheduler)}"
            )
    return config


def load_training_config(
    overrides: Sequence[str] = (),
    *,
    base_config: str | Path | None = None,
    model_config: str | Path | None = None,
    train_config: str | Path | None = None,
    data_config: str | Path | None = None,
) -> DictConfig:
    """Compose groups, full YAML, section YAML, then CLI values (highest priority).

    Section files contain the section's contents, without a model/train/data wrapper.
    Files are value overlays; select Hydra groups with model=..., train=..., etc.
    """
    overlay = OmegaConf.create({})
    legacy_schedule = None
    for section, path in (
        (None, base_config),
        ("model", model_config),
        ("train", train_config),
        ("data", data_config),
    ):
        if path is None:
            continue
        loaded = OmegaConf.load(path)
        if not isinstance(loaded, DictConfig) or "defaults" in loaded:
            raise ValueError(
                f"{path}: expected a YAML mapping without a Hydra defaults list"
            )
        values = cast(dict[str, Any], OmegaConf.to_container(loaded, resolve=False))
        if section is not None:
            values = {section: values}
        training = values.get("train", {})
        if isinstance(training, dict) and "scheduler" in training:
            scheduler = training["scheduler"]
            if not isinstance(scheduler, dict):
                legacy_schedule = False
            elif "decay" in scheduler or legacy_schedule is not False:
                legacy_schedule = True
        overlay = OmegaConf.merge(overlay, normalize_training_config(values))

    with as_file(files("stok").joinpath("configs")) as config_dir:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            layers = []
            if overlay:
                # An appended defaults layer lets Hydra apply its native override
                # grammar AFTER the YAML, including additions and deletions.
                ConfigStore.instance().store(
                    group="_stok_overlay",
                    name="custom",
                    node=overlay,
                    package="_global_",
                )
                layers.append("+_stok_overlay=custom")
            cfg = compose(config_name="config", overrides=[*layers, *overrides])
    # Old scheduler mappings always supported a plateau, including when its
    # length and decay were inherited from different files. Infer WSD only when
    # the user has not explicitly selected a new scheduler on the command line.
    if (
        legacy_schedule
        and not any(
            item.split("=", 1)[0].lstrip("+~") == "train.scheduler"
            for item in overrides
        )
        and str(OmegaConf.select(cfg, "train.scheduler", default="")).startswith(
            ("warmup_", "wsd_")
        )
    ):
        prefix = "wsd" if int(cfg.train.get("stable_steps", 0)) else "warmup"
        cfg.train.scheduler = prefix + "_" + cfg.train.scheduler.rsplit("_", 1)[1]
    return cfg
