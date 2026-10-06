"""Training and evaluation loader construction and residue alignment."""

from functools import partial
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    IterableDataset,
    Sampler,
    DistributedSampler,
)

from stok.data.collate import align_coords, mlm_collate, tokenize_residues
from stok.data.dataset import (
    PARQUET_EXTENSIONS,
    DummyMLMDataset,
    DummySequenceDataset,
    InterleavedIterableDataset,
    IterableTokenizedDataset,
    MapAsIterableDataset,
    TokenizedDataset,
    distributed_rank,
    _usable_samples,
)
from stok.eval.registry import METRIC_REGISTRY, resolve_eval_metrics
from stok.utils.tokenizer import Tokenizer


def _tokenize_and_align(
    batch: list[dict[str, Any]] | list[tuple[torch.Tensor, torch.Tensor]],
    tokenizer: Optional[Tokenizer],
    *,
    max_len: int,
    ignore_index: int,
    pad_id: int,
    num_classes: int | None = None,
):
    # if using DummySequenceDataset, batch is tuples(tokens, labels)
    if tokenizer is None:
        tokens, labels = zip(*cast(list[tuple[torch.Tensor, torch.Tensor]], batch))
        return torch.stack(tokens, dim=0), torch.stack(labels, dim=0)

    # else TokenizedDataset dicts with 'sequence' and optionally 'structure_tokens'
    input_ids = []
    label_ids = []
    coords_batch: list[torch.Tensor] = []
    batch = cast(list[dict[str, Any]], batch)
    for item in batch:
        seq: str = item["sequence"]

        if pad_id != tokenizer.pad_token_id:
            raise ValueError("Model pad_id must match tokenizer padding")
        ids = tokenize_residues(seq, tokenizer, max_len)

        # build labels aligned to tokens: CLS/EOS/PAD -> ignore_index
        L = ids.size(0)
        labels = torch.full((L,), ignore_index, dtype=torch.long)

        # Handle indices if present (may be absent for structure folder datasets)
        indices_raw = item.get("structure_tokens")
        if indices_raw is not None:
            indices: torch.Tensor = indices_raw.long()
            if indices.numel() != len(seq):
                raise ValueError(
                    f"Sample {item.get('sequence_id', '?')}: structure_tokens length must match sequence length"
                )
            if num_classes is not None and (indices >= num_classes).any():
                raise ValueError(
                    f"Sample {item.get('sequence_id', '?')}: class ID out of range"
                )
            copy_len = min(len(seq), L - 2)
            values = indices[:copy_len]
            labels[1 : 1 + copy_len] = values.masked_fill(values < 0, ignore_index)

        input_ids.append(ids)
        label_ids.append(labels)
        # optional coords tensor [max_len, 3, 3]
        c = item.get("coords")
        coords_batch.append(align_coords(c, residue_count=len(seq), token_length=L))

    tokens = torch.stack(input_ids, dim=0)
    labels = torch.stack(label_ids, dim=0)
    if any(item.get("coords") is not None for item in batch):
        return tokens, labels, torch.stack(coords_batch, dim=0)
    else:
        return tokens, labels


def _parse_eval_configs(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    """
    Normalize eval config into {name: {path, **options}}.

    Supports:
      - Legacy single path: data.eval="/path" -> {"default": {"path": "/path"}}
      - Dict of paths: data.eval.val="/p" -> {"val": {"path": "/p"}}
      - Dict of configs: data.eval.val.path="/p" -> {"val": {"path": "/p", ...}}
      - Structure folder format: data.eval.pdb.format="structure" for PDB/mmCIF folders
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
                entry = dict(value)
                # Preserve format-related keys for structure folder support
                for key in ("format", "chain_id", "recursive"):
                    if key in value:
                        entry[key] = value.get(key)
                result[name] = entry
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


class MixtureSampler(Sampler[int]):
    """
    Efficient sampler for a ConcatDataset that samples per-dataset according to
    dataset-level fractions, then uniformly within the chosen dataset.

    This avoids allocating per-sample weight vectors (which can be huge).
    Sampling is with replacement.
    """

    def __init__(
        self,
        *,
        lengths: list[int],
        fractions: list[float],
        seed: int = 0,
        num_samples: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        if len(lengths) == 0:
            raise ValueError("MixtureSampler requires at least one dataset length")
        if len(lengths) != len(fractions):
            raise ValueError("lengths and fractions must have the same length")
        if any(int(L) <= 0 for L in lengths):
            raise ValueError("All dataset lengths must be positive for MixtureSampler")
        fr = np.asarray([float(f) for f in fractions], dtype=np.float64)
        if np.any(fr < 0) or float(fr.sum()) <= 0:
            raise ValueError(
                "fractions must be non-negative and sum to a positive value"
            )
        fr = fr / float(fr.sum())

        self.rank, self.world_size = rank, world_size
        self.lengths = [int(L) for L in lengths]
        self.fractions = fr.tolist()
        self.seed = int(seed)
        self._epoch = 0
        self.offsets = np.cumsum([0] + self.lengths[:-1]).tolist()
        self.num_samples = (
            int(num_samples) if num_samples is not None else int(sum(self.lengths))
        )

    def __len__(self) -> int:
        return self.num_samples // self.world_size

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState((self.seed + (self._epoch * 1009)) & 0xFFFFFFFF)
        self._epoch += 1
        fr = np.asarray(self.fractions, dtype=np.float64)
        usable = len(self) * self.world_size
        for position in range(usable):
            ds_idx = int(rng.choice(len(self.lengths), p=fr))
            j = int(rng.randint(0, self.lengths[ds_idx]))
            if position % self.world_size == self.rank:
                yield int(self.offsets[ds_idx] + j)


def _build_dataloaders(
    cfg: DictConfig,
    *,
    codebook_size: int | None,
    pad_id: int,
    is_mlm: bool | None = None,
    objective: str | None = None,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    if objective is not None and (
        objective not in {"mlm", "codebook", "mdlm"}
        or (is_mlm is not None and is_mlm != (objective == "mlm"))
    ):
        raise ValueError("Conflicting objective/is_mlm flags or unknown objective")
    objective = objective or ("mlm" if is_mlm else "codebook")
    is_mlm, is_mdlm = objective == "mlm", objective == "mdlm"
    rank, world_size = distributed_rank()
    batch_size: int = cfg.train.batch_size
    max_len: int = cfg.data.max_len
    num_workers: int = cfg.data.num_workers
    pin_memory: bool = cfg.data.pin_memory
    ignore_index: int = -100 if is_mdlm else cfg.model.classifier.ignore_index

    # resolve dataloader buffering
    prefetch_factor: int = int(getattr(cfg.data, "prefetch_factor", 2))

    # resolve whether to load 3D coordinates from disk
    user_load_coords = getattr(cfg.data, "load_coords", None)

    eval_configs = _parse_eval_configs(cfg)
    train_configs = _parse_train_configs(cfg)
    fape_required = objective == "codebook" and bool(
        cfg.train.get("fape", {}).get("enabled", False)
    )

    def coordinate_setting(options, needed, required=False):
        value = options.get("load_coords", user_load_coords)
        alias = options.get("has_coords")
        if alias is not None:
            if (
                "load_coords" in options
                and value is not None
                and bool(value) != bool(alias)
            ):
                raise ValueError("Conflicting has_coords and load_coords settings")
            value = alias
        if value is False and required:
            raise ValueError(
                "load_coords=false conflicts with requested structure supervision/metrics"
            )
        return bool(needed) if value is None else bool(value), value is True

    tokenizer: Optional[Tokenizer] = None
    collate_fn = None
    train_sampler: Optional[Sampler[int]] = None

    # MLM-specific config
    if is_mlm:
        mlm_cfg = cfg.train.get("mlm", {})
        mask_prob = float(mlm_cfg.get("mask_prob", 0.15))
        mask_token_prob = float(mlm_cfg.get("mask_token_prob", 0.8))
        random_token_prob = float(mlm_cfg.get("random_token_prob", 0.1))

    # Supported structure file extensions for auto-detection
    structure_exts = {".pdb", ".ent", ".cif", ".mmcif"}

    # dataset picker usable for train/eval
    def _pick_dataset(
        path: str,
        load_coords: bool,
        require_structure_tokens: bool = True,
        *,
        dataset_format: str | None = None,
        chain_id: str | None = None,
        recursive: bool = False,
        allow_structure_folders: bool = False,
        dataset_name: str | None = None,
    ):
        p = Path(path)

        # Explicit structure folder format
        if allow_structure_folders and dataset_format == "structure":
            from stok.data.structure_dataset import StructureFolderDataset

            ds = StructureFolderDataset(
                folder_path=str(p),
                max_length=max_len,
                chain_id=chain_id,
                recursive=recursive,
                load_coords=load_coords,
            )
            return ds

        # heuristic: directory containing parquet shards -> Iterable; else map-style
        if p.is_dir():
            has_parquet = any(
                f.is_file() and f.suffix.lower() in PARQUET_EXTENSIONS
                for f in p.iterdir()
            )
            if has_parquet:
                shuffle_shards = bool(getattr(cfg.data, "shuffle_shards", True))
                shuffle_rows = bool(getattr(cfg.data, "shuffle_rows", True))
                return IterableTokenizedDataset(
                    dataset_path=str(p),
                    max_length=None if is_mdlm else max_len,
                    dataset_name=dataset_name,
                    shuffle_shards=shuffle_shards,
                    shuffle_rows=shuffle_rows,
                    load_coords=bool(load_coords),
                    require_structure_tokens=require_structure_tokens,
                )

            # Auto-detect structure folder (no parquet, has structure files)
            has_structures = any(
                f.suffix.lower() in structure_exts for f in p.iterdir() if f.is_file()
            )
            if allow_structure_folders and has_structures:
                from stok.data.structure_dataset import StructureFolderDataset

                return StructureFolderDataset(
                    folder_path=str(p),
                    max_length=max_len,
                    chain_id=chain_id,
                    recursive=recursive,
                    load_coords=load_coords,
                )

            raise ValueError(f"{p}: expected a directory containing Parquet shards")

        return TokenizedDataset(
            dataset_path=str(path),
            max_length=None if is_mdlm else max_len,
            dataset_name=dataset_name,
            load_coords=bool(load_coords),
            require_structure_tokens=require_structure_tokens,
        )

    if len(train_configs) > 0:
        # Real dataset(s); tokenize in collate
        tokenizer = Tokenizer()

        if is_mdlm:
            collate_fn = list
        elif is_mlm:

            def collate(batch):
                return mlm_collate(
                    batch,
                    tokenizer,
                    max_len=max_len,
                    mask_prob=mask_prob,
                    mask_token_prob=mask_token_prob,
                    random_token_prob=random_token_prob,
                    pad_id=pad_id,
                    ignore_index=ignore_index,
                )

            collate_fn = collate
        else:

            def collate(batch):
                return _tokenize_and_align(
                    batch,
                    tokenizer,
                    max_len=max_len,
                    ignore_index=ignore_index,
                    pad_id=pad_id,
                    num_classes=codebook_size,
                )

            collate_fn = collate

        if len(train_configs) == 1:
            # Single dataset (backwards compatible)
            train_load, force_coords = coordinate_setting(
                train_configs[0], fape_required, fape_required
            )
            train_ds = _pick_dataset(
                str(train_configs[0]["path"]),
                train_load,
                require_structure_tokens=not is_mlm,
                dataset_name=cfg.train.mdlm_identity.sample_key_namespaces[
                    train_configs[0]["name"]
                ]
                if is_mdlm
                else None,
            )
            if force_coords and not train_ds.has_coords:
                raise ValueError(
                    "load_coords=true requires a coordinate-capable training source"
                )
        else:
            # Multiple datasets with fractions
            ds_pairs = []
            for tcfg in train_configs:
                t_load_coords, force_coords = coordinate_setting(
                    tcfg, fape_required, fape_required
                )
                ds = _pick_dataset(
                    str(tcfg["path"]),
                    t_load_coords,
                    require_structure_tokens=not is_mlm,
                    dataset_name=cfg.train.mdlm_identity.sample_key_namespaces[
                        tcfg["name"]
                    ]
                    if is_mdlm
                    else None,
                )
                if force_coords and not ds.has_coords:
                    raise ValueError(
                        f"load_coords=true requires coordinates: {tcfg['path']}"
                    )
                ds_pairs.append((ds, float(tcfg["fraction"])))

            any_iterable = any(isinstance(ds, IterableDataset) for ds, _ in ds_pairs)
            if any_iterable:
                # Convert map-style datasets to iterable wrappers, then interleave
                iterables: list[IterableTokenizedDataset | MapAsIterableDataset] = []
                fracs: list[float] = []
                total_samples = 0
                for ds, frac in ds_pairs:
                    if isinstance(ds, IterableTokenizedDataset):
                        itds = ds
                    else:
                        itds = MapAsIterableDataset(
                            ds,
                            num_samples=len(ds),
                            seed=int(cfg.train.get("seed", 1337)),
                        )
                    iterables.append(itds)
                    fracs.append(float(frac))
                    try:
                        total_samples += itds.num_samples
                    except Exception:
                        total_samples = 0
                train_ds = InterleavedIterableDataset(
                    iterables,
                    fracs,
                    num_samples=total_samples if total_samples > 0 else None,
                    seed=int(cfg.train.get("seed", 1337)),
                )
            else:
                # Efficient mixture sampler over a ConcatDataset
                map_datasets = [ds for ds, _ in ds_pairs]
                lengths = [int(len(ds)) for ds in map_datasets]
                fracs = [float(fr) for _, fr in ds_pairs]
                concat = ConcatDataset(map_datasets)
                sampler = MixtureSampler(
                    lengths=lengths,
                    rank=rank,
                    world_size=world_size,
                    fractions=fracs,
                    seed=int(cfg.train.get("seed", 1337)),
                )
                setattr(
                    concat,
                    "has_coords",
                    any(getattr(ds, "has_coords", False) for ds in map_datasets),
                )
                setattr(
                    concat,
                    "has_labels",
                    any(getattr(ds, "has_labels", True) for ds in map_datasets),
                )
                train_ds = concat
                train_sampler = sampler
    else:
        # fallback dummy data for quick smoke test
        if is_mdlm:
            raise ValueError("MDLM requires real paired Parquet training sources")
        if is_mlm:
            train_ds = DummyMLMDataset(
                num_samples=512,
                seq_len=min(max_len, 256) - 2,  # Account for CLS/EOS tokens
            )
            tokenizer = Tokenizer()

            def collate(batch):
                return mlm_collate(
                    batch,
                    tokenizer,
                    max_len=max_len,
                    mask_prob=mask_prob,
                    mask_token_prob=mask_token_prob,
                    random_token_prob=random_token_prob,
                    pad_id=pad_id,
                    ignore_index=ignore_index,
                )

            collate_fn = collate
        else:
            assert codebook_size is not None
            train_ds = DummySequenceDataset(
                num_samples=512,
                seq_len=min(max_len, 256),
                vocab_size=cfg.model.encoder.vocab_size,
                num_classes=codebook_size,
                pad_id=pad_id,
            )

    if fape_required and not getattr(train_ds, "has_coords", False):
        raise ValueError("FAPE requires a training source with coordinates")
    cfg.data.load_coords = bool(getattr(train_ds, "has_coords", False))
    train_collate_fn = collate_fn

    # configure shuffle depending on dataset type / sampler usage
    is_iterable = isinstance(train_ds, IterableDataset)
    # only meaningful for multi-process loading
    if tokenizer is None and len(eval_configs) > 0:
        tokenizer = Tokenizer()

        if is_mlm:

            def collate(batch):
                return mlm_collate(
                    batch,
                    tokenizer,
                    max_len=max_len,
                    mask_prob=mask_prob,
                    mask_token_prob=mask_token_prob,
                    random_token_prob=random_token_prob,
                    pad_id=pad_id,
                    ignore_index=ignore_index,
                )

            collate_fn = collate
        else:

            def collate(batch):
                return _tokenize_and_align(
                    batch,
                    tokenizer,
                    max_len=max_len,
                    ignore_index=ignore_index,
                    pad_id=pad_id,
                    num_classes=codebook_size,
                )

            collate_fn = collate

    if max_len < 3:
        raise ValueError("data.max_len must be >= 3")
    if tokenizer is not None:
        if tokenizer.pad_token_id != pad_id or len(tokenizer) != int(
            cfg.model.encoder.vocab_size
        ):
            raise ValueError("Model vocabulary and pad_id must match tokenizer")
        for key in ("bos_id", "eos_id"):
            OmegaConf.update(
                cfg,
                f"model.encoder.{key}",
                getattr(tokenizer, key.replace("_id", "_token_id")),
                force_add=True,
            )

    def _make_dl_kwargs(batch_sz: int):
        kwargs = {
            "batch_size": batch_sz,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn,
            "persistent_workers": False,
        }
        if num_workers > 0 and prefetch_factor is not None and prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        return kwargs

    if isinstance(
        train_ds,
        (IterableTokenizedDataset, MapAsIterableDataset, InterleavedIterableDataset),
    ):
        train_ds.training_batch_size = batch_size
        train_ds.num_workers = num_workers
        dropped = train_ds.num_samples - _usable_samples(train_ds)
        if rank == 0 and dropped:
            print(
                f"Training stream drops {dropped} samples per pass for complete rank/worker batches"
            )
    elif train_sampler is None:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg.train.get("seed", 1337)),
            drop_last=True,
        )
    if train_sampler is not None:
        train_loader = DataLoader(
            train_ds,  # type: ignore[arg-type]
            sampler=train_sampler,
            drop_last=True,
            **_make_dl_kwargs(batch_size),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=(not is_iterable),
            drop_last=True,
            **_make_dl_kwargs(batch_size),
        )
    train_loader.collate_fn = train_collate_fn or train_loader.collate_fn
    if isinstance(train_ds, DummySequenceDataset):
        from torch.utils.data import default_collate

        train_loader.collate_fn = default_collate
    eval_loaders: dict[str, DataLoader] = {}
    for name, eval_cfg in eval_configs.items():
        eval_path = eval_cfg["path"]
        eval_batch_size = int(eval_cfg.get("batch_size", batch_size))
        resolved = (
            {} if is_mdlm else resolve_eval_metrics(cfg, name, objective=objective)
        )
        needs_coords = (
            bool(
                OmegaConf.select(
                    cfg, "train.eval.mdlm.generation.decode", default=False
                )
            )
            if is_mdlm
            else any(METRIC_REGISTRY[key].requires_coords for key in resolved)
        )
        requires_coords = any(
            settings["explicit"] and METRIC_REGISTRY[key].requires_coords
            for key, settings in resolved.items()
        )
        eval_load_coords, force_coords = coordinate_setting(
            eval_cfg, needs_coords, requires_coords
        )
        # Extract structure folder format options
        eval_format = eval_cfg.get("format")
        eval_chain_id = eval_cfg.get("chain_id")
        eval_recursive = bool(eval_cfg.get("recursive", False))

        # Structure folders always have coords, don't require indices
        ds = _pick_dataset(
            eval_path,
            eval_load_coords,
            require_structure_tokens=False,
            dataset_format=eval_format,
            chain_id=eval_chain_id,
            recursive=eval_recursive,
            allow_structure_folders=not is_mdlm,
            dataset_name=cfg.train.mdlm_identity.sample_key_namespaces[name]
            if is_mdlm
            else None,
        )
        if force_coords and not ds.has_coords:
            raise ValueError(
                f"Dataset {name}: load_coords=true requires a coordinate-capable source"
            )
        for metric_name, settings in resolved.items():
            if not settings["explicit"]:
                continue
            if METRIC_REGISTRY[metric_name].requires_coords and not ds.has_coords:
                raise ValueError(
                    f"Dataset {name}, metric {metric_name}: missing coordinates"
                )
            if (
                not is_mlm
                and metric_name in {"accuracy", "perplexity"}
                and not ds.has_labels
            ):
                raise ValueError(
                    f"Dataset {name}, metric {metric_name}: missing labels"
                )
        eval_cfg["load_coords"] = bool(ds.has_coords)
        if isinstance(ds, IterableDataset):
            ds.shuffle_shards = False
            ds.shuffle_rows = False
            eval_sampler = None
        else:
            eval_sampler = range(rank, len(ds), world_size)
        eval_kwargs = _make_dl_kwargs(eval_batch_size)
        eval_seed = int(
            cfg.train.get("eval", {}).get(
                "seed", 1729 if is_mdlm else cfg.train.get("seed", 1337)
            )
        )
        eval_kwargs["generator"] = torch.Generator().manual_seed(eval_seed)
        if is_mlm:
            eval_kwargs["collate_fn"] = partial(
                mlm_collate,
                tokenizer=tokenizer,
                max_len=max_len,
                mask_prob=mask_prob,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
                pad_id=pad_id,
                ignore_index=ignore_index,
                eval_seed=eval_seed,
                dataset_name=name,
            )
        eval_loaders[name] = DataLoader(
            ds,
            sampler=eval_sampler,
            shuffle=False,
            drop_last=False,
            **eval_kwargs,
        )
        eval_loaders[name].metric_configs = resolved
    cfg.data.eval = OmegaConf.create(eval_configs)
    return train_loader, eval_loaders
