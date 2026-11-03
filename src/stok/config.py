"""Canonical training choices: groups, YAML overlays, then Hydra overrides."""

from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Sequence, cast

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


def reject_unknown_keys(values, schema, path=""):
    """Typos and obsolete keys fail loudly instead of silently doing nothing."""
    for key, value in values.items():
        name = f"{path}.{key}" if path else str(key)
        if key not in schema:
            while isinstance(value, DictConfig) and value:
                child = next(iter(value))
                name += f".{child}"
                value = value[child]
            raise ValueError(f"Unsupported configuration field: {name}")
        if isinstance(schema[key], DictConfig) and isinstance(value, DictConfig):
            reject_unknown_keys(value, schema[key], name)


def validate_training_config(cfg: DictConfig) -> None:
    """Reject unknown keys and choices the loop cannot honor, before any side effect."""
    from stok.eval.mdlm import resolve_mdlm_eval_config
    from stok.data.loaders import _parse_eval_configs
    from stok.utils.mdlm import validate_mdlm_config

    with as_file(files("stok").joinpath("configs")) as directory:
        schema = OmegaConf.create(
            {
                "model": OmegaConf.load(directory / "model/arch.yaml"),
                "train": OmegaConf.load(directory / "train/base.yaml"),
                "data": OmegaConf.load(directory / "data/base.yaml"),
            }
        )
    values = cast(
        DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    )
    # Named sources are authored populations; the evaluator checks its own section.
    for section, option in (("train", "fraction"), ("eval", "batch_size")):
        sources = values.data.pop(section, None)
        if not isinstance(sources, DictConfig):
            continue
        for name, source in sources.items():
            if not isinstance(source, DictConfig):
                continue
            path = f"data.{section}.{name}"
            reject_unknown_keys(
                source, {"path": None, "load_coords": None, option: None}, path
            )
            if "batch_size" in source and (
                type(source.batch_size) is not int or source.batch_size < 1
            ):
                raise ValueError(f"{path}.batch_size must be a positive integer")
        if (
            section == "train"
            and sources
            and all(
                isinstance(source, DictConfig) and source.get("fraction") == 0
                for source in sources.values()
            )
        ):
            raise ValueError("data.train requires a nonzero source fraction")
    resolved_eval = resolve_mdlm_eval_config(values)
    values.train.eval.pop("mdlm", None)
    reject_unknown_keys(values, schema)

    train, data, enc = values.train, values.data, values.model.encoder
    if (train.get("max_steps") is None) == (train.get("max_epochs") is None):
        raise ValueError(
            "Choose exactly one training budget: max_steps or max_epochs; epoch runs require max_steps=null"
        )
    # Zero cadences divide by zero mid-run; negative budgets silently train nothing.
    for name, value, minimum in (
        ("train.batch_size", train.batch_size, 1),
        ("train.gradient_accumulation_steps", train.gradient_accumulation_steps, 1),
        ("train.max_steps", train.get("max_steps"), 0),
        ("train.max_epochs", train.get("max_epochs"), 0),
        ("train.log_every", train.log_every, 1),
        ("train.save_every", train.save_every, 1),
        ("train.eval.steps", train.eval.steps, 1),
        ("data.max_len", data.max_len, 3),
    ):
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if train.scheduler not in {
        "warmup_linear",
        "warmup_cosine",
        "wsd_linear",
        "wsd_cosine",
    }:
        raise ValueError("Unsupported train.scheduler")
    if train.scheduler.startswith("warmup_") and train.stable_steps:
        raise ValueError("train.stable_steps requires a wsd scheduler")
    if enc.d_model % enc.n_heads:
        raise ValueError("model.encoder.d_model must be divisible by n_heads")
    if (enc.d_model // enc.n_heads) % 2:
        raise ValueError("model.encoder head width must be even for RoPE")
    if (
        resolved_eval.generation.enabled
        and resolved_eval.generation.decode
        and any(
            source.get("load_coords", data.load_coords) is False
            for source in _parse_eval_configs(cfg).values()
        )
    ):
        raise ValueError(
            "Effective evaluation source load_coords=false conflicts with decoded MDLM evaluation"
        )
    validate_mdlm_config(cfg.train.mdlm)


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
        overlay = OmegaConf.merge(overlay, values)

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
    validate_training_config(cfg)
    return cfg
