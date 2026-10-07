"""Canonical training choices: groups, YAML overlays, then Hydra overrides."""

from importlib.resources import as_file, files
import math
from pathlib import Path
from typing import Any, Sequence, cast

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf


def _check_fields(values, defaults, path="", *, required=False):
    """Use the shipped YAML tree as the field contract, with dynamic populations."""
    if not isinstance(values, DictConfig):
        raise ValueError(f"{path or 'configuration'} must be a mapping")
    if required:
        for key in defaults:
            if key not in values:
                name = f"{path}.{key}" if path else str(key)
                raise ValueError(f"Missing required configuration field: {name}")
    for key, value in values.items():
        name = f"{path}.{key}" if path else str(key)
        if key not in defaults:
            while isinstance(value, DictConfig) and value:
                child = next(iter(value))
                name += f".{child}"
                value = value[child]
            raise ValueError(f"Unsupported configuration field: {name}")
        expected = defaults[key]
        if isinstance(expected, DictConfig):
            _check_fields(value, expected, name, required=required)
        elif expected is not None:
            valid = (
                type(value) in {int, float} and math.isfinite(value)
                if type(expected) is float
                else isinstance(value, ListConfig)
                if isinstance(expected, ListConfig)
                else type(value) is type(expected)
            )
            if not valid:
                raise ValueError(f"{name} has an invalid type or nonfinite value")


def validate_training_config(cfg: DictConfig) -> None:
    """Reject unsupported choices before devices, artifacts, logging or outputs."""
    from stok.eval.mdlm import resolve_mdlm_eval_config
    from stok.utils.mdlm import validate_mdlm_config
    from stok.utils.tokenizer import Tokenizer

    if not isinstance(cfg, DictConfig):
        raise ValueError("Training configuration must be a DictConfig mapping")
    for section in ("train", "data", "model"):
        if not isinstance(cfg.get(section), DictConfig):
            raise ValueError(f"{section} must be a mapping")
    if OmegaConf.select(cfg, "train.pretrained_encoder") is not None:
        raise ValueError(
            "train.pretrained_encoder requires a supported adapter; none is selected"
        )
    with as_file(files("stok").joinpath("configs")) as directory:
        defaults = OmegaConf.create(
            {
                "print_model_summary": True,
                "model": OmegaConf.load(directory / "model/arch.yaml"),
                "train": OmegaConf.load(directory / "train/base.yaml"),
                "data": OmegaConf.load(directory / "data/base.yaml"),
            }
        )
    if OmegaConf.select(cfg, "train.mdlm.placement") == "span":
        defaults.train.mdlm.span_mean = 8.0
    if OmegaConf.select(cfg, "train.mdlm.noise.name") == "power":
        defaults.train.mdlm.noise.power = 2.0
    defaults.train.max_steps = None  # Either step or epoch governance is selected.
    # Source/case names are authored populations, not fixed schema fields.
    values = cast(
        DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    )
    for section in ("train", "eval"):
        if section not in values.data:
            raise ValueError(f"Missing required configuration field: data.{section}")
        sources = values.data.pop(section)
        defaults.data.pop(section)
        if isinstance(sources, str):
            if not sources.strip():
                raise ValueError(f"data.{section} must be a nonempty path")
            continue
        if not isinstance(sources, DictConfig):
            raise ValueError(f"data.{section} must be a path or named source mapping")
        for name, source in sources.items():
            path = f"data.{section}.{name}"
            if not isinstance(name, str) or not name or "/" in name:
                raise ValueError(f"Invalid source name: {path}")
            if isinstance(source, str):
                if not source.strip():
                    raise ValueError(f"{path} must be a nonempty path")
                continue
            allowed = {"path": None, "load_coords": None}
            allowed["fraction" if section == "train" else "batch_size"] = None
            _check_fields(source, allowed, path)
            if not isinstance(source.get("path"), str) or not source.path.strip():
                raise ValueError(f"{path}.path must be a nonempty path")
            coords = source.get("load_coords")
            if coords is not None and type(coords) is not bool:
                raise ValueError(f"{path}.load_coords must be a bool or null")
            fraction = source.get("fraction")
            if fraction is not None and (
                type(fraction) not in {int, float}
                or not math.isfinite(fraction)
                or fraction < 0
            ):
                raise ValueError(f"{path}.fraction must be finite and nonnegative")
            batch_size = source.get("batch_size")
            if batch_size is not None and (
                type(batch_size) is not int or batch_size < 1
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
    # The evaluator owns its optional benchmark matrix and validates nested cases.
    resolved_eval = resolve_mdlm_eval_config(values)
    values.train.eval.pop("mdlm")
    defaults.train.eval.pop("mdlm")
    _check_fields(values, defaults)
    # Hydra deletions may remove optional W&B labels, but never scientific choices.
    for name in ("entity", "group", "name", "mode"):
        if name not in values.train.wandb:
            defaults.train.wandb.pop(name)
    # Active span/power parameters inherit only their component's documented default.
    for name in defaults.train.mdlm.regime_weights:
        values.train.mdlm.regime_weights.setdefault(name, 0.0)
    if values.train.mdlm.get("placement") == "span":
        values.train.mdlm.setdefault("span_mean", 8.0)
    if values.train.mdlm.noise.get("name") == "power":
        values.train.mdlm.noise.setdefault("power", 2.0)
    _check_fields(values, defaults, required=True)
    train, data, enc = values.train, values.data, values.model.encoder
    if (train.max_steps is None) == (train.max_epochs is None):
        raise ValueError(
            "Choose exactly one training budget: max_steps or max_epochs; epoch runs require max_steps=null"
        )
    if train.objective != "mdlm":
        raise ValueError("train.objective only supports paired mdlm")
    if train.optimizer != "adamw":
        raise ValueError("train.optimizer only supports adamw")
    if train.scheduler not in {
        "warmup_linear",
        "warmup_cosine",
        "wsd_linear",
        "wsd_cosine",
    }:
        raise ValueError("Unsupported train.scheduler")
    for name, value, minimum in (
        ("train.batch_size", train.batch_size, 1),
        ("train.gradient_accumulation_steps", train.gradient_accumulation_steps, 1),
        ("train.max_steps", train.max_steps, 0),
        ("train.max_epochs", train.max_epochs, 0),
        ("train.log_every", train.log_every, 1),
        ("train.save_every", train.save_every, 1),
        ("train.warmup_steps", train.warmup_steps, 0),
        ("train.stable_steps", train.stable_steps, 0),
        ("train.decay_steps", train.decay_steps, 0),
        ("train.seed", train.seed, 0),
        ("train.eval.seed", train.eval.seed, 0),
        ("train.eval.steps", train.eval.steps, 1),
        ("data.num_workers", data.num_workers, 0),
        ("data.max_len", data.max_len, 3),
        ("data.prefetch_factor", data.prefetch_factor, 1),
        *(
            (f"model.encoder.{key}", enc[key], 1)
            for key in ("d_model", "n_heads", "n_layers")
        ),
    ):
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if train.scheduler.startswith("warmup_") and train.stable_steps:
        raise ValueError("train.stable_steps requires a wsd scheduler")
    for key in (
        "lr",
        "adam_eps",
        "weight_decay",
        "max_grad_norm",
        "adam_beta1",
        "adam_beta2",
    ):
        value = train[key]
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or value < 0
            or (key in {"lr", "adam_eps"} and value == 0)
            or (key.startswith("adam_beta") and value >= 1)
        ):
            raise ValueError(f"train.{key} is outside its supported range")
    if train.mixed_precision not in {None, "no", "fp16", "bf16"}:
        raise ValueError("train.mixed_precision must be no, fp16, bf16 or null")
    for name in ("output_dir", "resume_from"):
        if train[name] is not None and (
            not isinstance(train[name], str) or not train[name].strip()
        ):
            raise ValueError(f"train.{name} must be a nonempty path or null")
    if data.load_coords is not None and type(data.load_coords) is not bool:
        raise ValueError("data.load_coords must be a bool or null")
    if data.split_manifest is not None and not isinstance(data.split_manifest, str):
        raise ValueError("data.split_manifest must be a path or null")
    if enc.d_model % enc.n_heads or enc.norm not in {"layernorm", "rmsnorm"}:
        raise ValueError("model.encoder heads/d_model/norm are unsupported")
    for key in ("dropout", "attn_dropout"):
        if not 0 <= enc[key] < 1:
            raise ValueError(f"model.encoder.{key} must be in [0,1)")
    if not math.isfinite(enc.ffn_mult) or enc.ffn_mult <= 0:
        raise ValueError("model.encoder.ffn_mult must be finite and positive")
    tokenizer = Tokenizer()
    for key, expected in (
        ("vocab_size", len(tokenizer)),
        ("pad_id", tokenizer.pad_token_id),
        ("bos_id", tokenizer.bos_token_id),
        ("eos_id", tokenizer.eos_token_id),
    ):
        if enc[key] != expected:
            raise ValueError(
                f"model.encoder.{key} must match the native sequence tokenizer"
            )
    for component in ("codebook", "decoder"):
        settings = values.model[component]
        if settings.preset not in {None, "base", "lite"} or (
            settings.path is not None
            and (not isinstance(settings.path, str) or not settings.path.strip())
        ):
            raise ValueError(f"model.{component} requires a supported preset/path")
    for name in ("project", "entity", "group", "name", "mode"):
        value = train.wandb.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"train.wandb.{name} must be a string or null")
    if train.wandb.get("mode") not in {None, "online", "offline", "disabled"}:
        raise ValueError("train.wandb.mode is unsupported")
    if any(not isinstance(tag, str) for tag in train.wandb.tags):
        raise ValueError("train.wandb.tags must contain strings")
    if (
        resolved_eval.generation.enabled
        and resolved_eval.generation.decode
        and data.load_coords is False
    ):
        raise ValueError(
            "data.load_coords=false conflicts with decoded MDLM evaluation"
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
