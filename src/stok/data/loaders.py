"""Training and evaluation loader construction and residue alignment."""

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from stok.data.dataset import (
    InterleavedIterableDataset,
    IterableTokenizedDataset,
    distributed_rank,
    _usable_samples,
)


def _parse_eval_configs(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    """
    Normalize eval config into {name: {path, **options}}.

    Supports:
      - Single path: data.eval="/path" -> {"default": {"path": "/path"}}
      - Dict of paths: data.eval.val="/p" -> {"val": {"path": "/p"}}
      - Dict of configs: data.eval.val.path="/p" -> {"val": {"path": "/p", ...}}
    """
    raw_eval = cfg.data.get("eval")
    if raw_eval is None:
        return {}
    if isinstance(raw_eval, str):
        return {"default": {"path": raw_eval}}
    if isinstance(raw_eval, (dict, DictConfig)):
        result: dict[str, dict[str, Any]] = {}
        for name, value in raw_eval.items():
            if value is None:
                continue
            if isinstance(value, str):
                result[name] = {"path": value}
            elif isinstance(value, (dict, DictConfig)):
                if value.get("path") is None:
                    continue
                result[name] = dict(value)
            else:
                raise ValueError(f"Invalid eval config for '{name}': {type(value)}")
        return result
    raise ValueError(f"Invalid data.eval config type: {type(raw_eval)}")


def _parse_train_configs(cfg: DictConfig) -> list[dict[str, Any]]:
    """
    Normalize train config into a list of {name, path, fraction, **options}.

    Supports:
      - Single path: data.train="/path" -> [{"name": "default", "path": "/path", "fraction": 1.0}]
      - Dict of datasets:
          data.train.ds1="/p" -> [{"name": "ds1", "path": "/p", "fraction": ...}]
          data.train.ds1.path="/p" -> same, with optional data.train.ds1.fraction

    Fractions are normalized to sum to 1.0. If some fractions are omitted, they
    share the remaining mass equally (when positive).
    """
    raw_train = cfg.data.get("train")
    if raw_train is None:
        return []
    if isinstance(raw_train, str):
        return [{"name": "default", "path": raw_train, "fraction": 1.0}]
    if isinstance(raw_train, (dict, DictConfig)):
        if len(raw_train) == 0:
            return []
        out: list[dict[str, Any]] = []
        for name, value in raw_train.items():
            if value is None:
                continue
            if isinstance(value, str):
                out.append({"name": str(name), "path": value, "fraction": None})
            elif isinstance(value, (dict, DictConfig)):
                p = value.get("path")
                if p is None:
                    continue
                entry: dict[str, Any] = {"name": str(name), "path": str(p)}
                entry["fraction"] = value.get("fraction")
                # optional per-dataset toggles (currently only load_coords is supported)
                if "load_coords" in value:
                    entry["load_coords"] = value.get("load_coords")
                out.append(entry)
            else:
                raise ValueError(f"Invalid train config for '{name}': {type(value)}")

        if len(out) == 0:
            return []
        if len(out) == 1:
            out[0]["fraction"] = 1.0
            return out

        # validate and fill missing fractions
        specified = []
        for e in out:
            f = e.get("fraction")
            if f is None:
                continue
            ff = float(f)
            if ff < 0:
                raise ValueError(f"data.train.{e['name']}.fraction must be >= 0")
            e["fraction"] = ff
            specified.append(ff)

        total_specified = float(sum(specified))
        unspecified = [e for e in out if e.get("fraction") is None]
        if len(unspecified) > 0:
            remaining = 1.0 - total_specified
            # If the specified fractions already exceed 1.0, give unspecified 0.0
            default_frac = (
                (remaining / float(len(unspecified))) if remaining > 0 else 0.0
            )
            for e in unspecified:
                e["fraction"] = float(default_frac)

        total = float(sum(float(e["fraction"]) for e in out))
        if total <= 0:
            # fallback: equal mixing
            eq = 1.0 / float(len(out))
            for e in out:
                e["fraction"] = eq
            return out

        # normalize
        for e in out:
            e["fraction"] = float(e["fraction"]) / total
        return out

    raise ValueError(f"Invalid data.train config type: {type(raw_train)}")


def _build_dataloaders(
    cfg: DictConfig,
    *,
    identity: Mapping[str, Any],
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Build raw aligned MDLM loaders; all derived state stays in consumers."""
    rank, _ = distributed_rank()
    batch_size = cfg.train.batch_size
    num_workers = cfg.data.num_workers
    eval_configs = _parse_eval_configs(cfg)
    train_configs = _parse_train_configs(cfg)
    if not train_configs:
        raise ValueError("MDLM requires real paired Parquet training sources")

    def coordinate_setting(options, needed=False):
        value = options.get("load_coords", cfg.data.load_coords)
        if value is False and needed:
            raise ValueError("load_coords=false conflicts with decoded MDLM evaluation")
        return needed if value is None else value, value is True or needed

    def dataset(path, options, name, *, training):
        needed = not training and bool(
            OmegaConf.select(cfg, "train.eval.mdlm.generation.decode", default=False)
        )
        load_coords, require_coords = coordinate_setting(options, needed)
        kwargs: dict[str, Any] = dict(
            dataset_path=path,
            max_length=None,
            dataset_name=identity["sources"][name]["replay_sha256"]
            if training
            else name,
            load_coords=load_coords,
            require_structure_tokens=training,
        )
        ds = IterableTokenizedDataset(
            **kwargs,
            shuffle_shards=cfg.data.shuffle_shards if training else False,
            shuffle_rows=cfg.data.shuffle_rows if training else False,
        )
        if require_coords and not ds.has_coords:
            raise ValueError(
                f"{name}: load_coords=true requires a coordinate-capable source"
            )
        return ds

    pairs = [
        (
            dataset(str(source["path"]), source, source["name"], training=True),
            source["fraction"],
        )
        for source in train_configs
    ]
    if len(pairs) == 1:
        train_ds = pairs[0][0]
    else:
        train_ds = InterleavedIterableDataset(
            [ds for ds, _ in pairs],
            [fraction for _, fraction in pairs],
            seed=cfg.train.seed,
        )

    def loader_kwargs(size):
        kwargs: dict[str, Any] = dict(
            batch_size=size,
            num_workers=num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=list,
            persistent_workers=False,
        )
        if num_workers:
            kwargs["prefetch_factor"] = cfg.data.prefetch_factor
        return kwargs

    train_ds.training_batch_size = batch_size
    train_ds.num_workers = num_workers
    dropped = train_ds.num_samples - _usable_samples(train_ds)
    if rank == 0 and dropped:
        print(
            f"Training stream drops {dropped} samples per pass for complete rank/worker batches"
        )
    train_loader = DataLoader(
        train_ds,
        shuffle=False,
        drop_last=True,
        **loader_kwargs(batch_size),
    )
    eval_loaders = {}
    for name, options in eval_configs.items():
        eval_loaders[name] = DataLoader(
            dataset(options["path"], options, name, training=False),
            shuffle=False,
            drop_last=False,
            generator=torch.Generator().manual_seed(
                identity["shared_cases"]["request"]["seed"]
                if identity.get("shared_cases")
                else 0
            ),
            **loader_kwargs(options.get("batch_size", batch_size)),
        )
    return train_loader, eval_loaders
