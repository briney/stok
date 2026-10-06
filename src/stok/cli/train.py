from functools import partial
from contextlib import nullcontext
from itertools import islice
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import gather_object, set_seed
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
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
from stok.eval import Evaluator, MetricLogger
from stok.eval.registry import METRIC_REGISTRY, resolve_eval_metrics
from stok.models.decoder import load_pretrained_decoder
from stok.models.build import build_model
from stok.data.mdlm import validate_mdlm_sources
from stok.utils.mdlm import validate_mdlm_config
from stok.models.head import CodebookClassifier
from stok.utils.codebook import load_codebook
from stok.utils.console import ConsoleLogger
from stok.utils.flops import compute_flops_6n, count_parameters
from stok.utils.tokenizer import Tokenizer
from stok.training.tasks import ClassificationTask, MDLMTask
from stok.utils.checkpoint import (
    atomic_destination as _atomic_destination,
    collect_rng_state as _collect_rng_state,
    epoch_iterator,
    rank_errors as _raise_rank_errors,
    read_training_checkpoint,
    restore_rng_state,
    restore_training_state,
    resume_signature,
    validate_resume_signature,
    ResumeWandb,
)


def _maybe_get_accelerator(precision=None):
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision=precision)
    if accelerator.distributed_type.name not in {"NO", "MULTI_CPU", "MULTI_GPU"}:
        raise ValueError(
            f"Unsupported distributed backend: {accelerator.distributed_type}; use replicated DDP"
        )
    accelerator.gradient_accumulation_steps = 1
    return accelerator


def _get_model_device(model: nn.Module, accelerator) -> torch.device:
    """
    Resolve the device to place tensors on, compatible with both plain nn.Module
    and models wrapped by Accelerate/DDP.
    """
    if accelerator is not None:
        return accelerator.device
    # fall back to the device of the first parameter (or CPU if model is empty)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    decay: str,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: Optional[int],
    total_steps: int,
):
    # derive decay_steps if not provided
    if decay_steps is None:
        decay_steps = max(0, int(total_steps) - int(warmup_steps) - int(stable_steps))

    decay = str(decay).lower()
    if decay not in {"cosine", "linear"}:
        raise ValueError(f"Unknown scheduler.decay: {decay}")

    if warmup_steps < 0 or stable_steps < 0 or decay_steps < 0:
        raise ValueError("scheduler step counts must be non-negative")

    def lr_lambda(current_step: int):
        # warmup phase (0 -> 1)
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # stable hold at 1.0
        post_warmup = current_step - warmup_steps
        if stable_steps > 0 and post_warmup < stable_steps:
            return 1.0

        # decay phase (1 -> 0)
        t = post_warmup - stable_steps
        if decay_steps <= 0:
            return 1.0
        progress = min(max(float(t) / float(decay_steps), 0.0), 1.0)
        if decay == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # linear
            return 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class _TeeIO:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str):
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self._streams:
            st.flush()


def _resolve_project_dirs(cfg: DictConfig) -> dict[str, Path]:
    root = Path(str(cfg.train.get("output_dir") or Path.cwd())).resolve()
    model_dir = root / "model"
    ckpt_dir = root / "checkpoints"
    logs_dir = root / "logs"
    configs_dir = root / "configs"
    return {
        "root": root,
        "model": model_dir,
        "checkpoints": ckpt_dir,
        "logs": logs_dir,
        "configs": configs_dir,
    }


def _ensure_dirs(dirs: list[Path]):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _save_config_snapshot(cfg: DictConfig, dst_file: Path):
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with dst_file.open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))


def _unwrap_model(model: nn.Module, accelerator) -> nn.Module:
    return accelerator.unwrap_model(model) if accelerator is not None else model


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
    cfg: DictConfig,
    accelerator,
    micro_step: int = 0,
    residues_seen: int = 0,
    executed_positions: int = 0,
    training_state: dict | None = None,
):
    payload = None
    rank_state = None
    error = None
    try:
        payload = {
            "model": _unwrap_model(model, accelerator).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": int(global_step),
            "micro_step": int(micro_step),
            "step_unit": "optimizer_update",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "rng_state": _collect_rng_state(
                device=accelerator.device if accelerator else torch.device("cpu")
            ),
            "residues_seen": int(residues_seen),
            "executed_positions": int(executed_positions),
        }
        payload["optimizer_initialized"] = [
            key for key, state in payload["optimizer"]["state"].items() if state
        ]
        # Minimal helper callers may still write initialization artifacts; production
        # always supplies complete rank state and only v2 supports full resume.
        if training_state is not None:
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError(
                    "Checkpoint requires a completed accumulation boundary with no pending gradients"
                )
            scaler = getattr(accelerator, "scaler", None)
            rank_state = {
                **training_state["local"],
                "rank": accelerator.process_index if accelerator else 0,
                "rng": payload["rng_state"],
                "scaler": scaler.state_dict() if scaler is not None else None,
            }
    except Exception as exc:
        if accelerator is None:
            raise
        error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(error, accelerator, "Collecting checkpoint state failed")
    assert payload is not None
    if training_state is not None:
        payload.update(
            format_version=2,
            rank_states=gather_object([rank_state]) if accelerator else [rank_state],
            signature=training_state["signature"],
            wandb_run_id=training_state["wandb_run_id"],
        )
    error = None
    if accelerator is None or accelerator.is_main_process:
        try:
            with _atomic_destination(path) as temporary:
                torch.save(payload, temporary)
        except Exception as exc:
            if accelerator is None:
                raise
            error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(error, accelerator, "Checkpoint failed")


def _load_pretrained_encoder(
    model: nn.Module,
    checkpoint_path: str,
    *,
    accelerator,
    printer,
) -> None:
    """Load encoder weights from a pre-trained checkpoint (e.g., from MLM pre-training).

    Only loads embedding and encoder weights, leaving head weights randomly initialized.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)

    # Filter to only encoder and embedding weights
    encoder_keys = {
        k: v for k, v in state_dict.items() if k.startswith(("embed.", "encoder."))
    }

    missing, unexpected = _unwrap_model(model, accelerator).load_state_dict(
        encoder_keys, strict=False
    )

    # Log what was loaded
    printer(
        f"Loaded {len(encoder_keys)} encoder/embedding weights from {checkpoint_path}"
    )
    if missing:
        # Filter out expected missing keys (head-specific)
        missing_encoder = [k for k in missing if k.startswith(("embed.", "encoder."))]
        if missing_encoder:
            printer(f"  Warning: Missing encoder keys: {missing_encoder}")


def _compute_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = labels != ignore_index
        if mask.sum().item() == 0:
            return 0.0
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return float(correct) / float(total)


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


def _maybe_init_wandb(
    cfg: DictConfig,
    *,
    is_main_process: bool,
    logs_dir: Optional[Path] = None,
    resume_payload: dict | None = None,
):
    if (
        not is_main_process
        or not cfg.train.get("wandb")
        or not cfg.train.wandb.get("enabled", True)
    ):
        return None
    import wandb

    options = cfg.train.wandb
    serialized_config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(serialized_config, dict)
    run_id = resume_payload["wandb_run_id"] if resume_payload else None
    run = wandb.init(
        project=options.get("project", "stok"),
        entity=options.get("entity"),
        group=options.get("group"),
        name=options.get("name"),
        tags=list(options.get("tags", [])),
        config=cast(dict[str, Any], serialized_config),
        mode=options.get("mode") or None,
        dir=str(logs_dir) if logs_dir is not None else None,
        id=run_id,
        resume="must" if run_id else None,
    )
    if run is None or (run_id and run.id != run_id):
        raise RuntimeError("W&B did not resume the recorded run ID")
    watermark = -1
    if resume_payload:
        watermark = max(int(run.step) - 1, int(run.summary.get("_step", -1)))
        run.summary["resume/rollback"] = {
            "checkpoint_step": resume_payload["global_step"],
            "history_watermark": watermark,
        }
    return ResumeWandb(wandb, watermark)


def iter_windows(loader, size: int, accelerator=None):
    if size < 1:
        raise ValueError("Accumulation window size must be positive")
    iterator = None
    while True:
        error = None
        window = []
        try:
            if iterator is None:
                iterator = iter(loader)
            window = list(islice(iterator, size))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _raise_rank_errors(error, accelerator, "Loading training window failed")
        if accelerator:
            sizes = gather_object([len(window)])
            if len(set(sizes)) != 1:
                raise RuntimeError(
                    f"Unequal accumulation window lengths across ranks: {sizes}"
                )
        if not window:
            return
        yield window


def run_training(cfg: DictConfig):
    objective = str(cfg.train.get("objective", "codebook")).lower()
    if objective not in {"codebook", "mlm", "mdlm"}:
        raise ValueError(f"Unknown train.objective: {objective}")
    is_mdlm = objective == "mdlm"
    if is_mdlm:
        project = cfg.train.get("output_dir")
        if not isinstance(project, str) or not project.strip():
            raise ValueError("MDLM requires a nonempty unique train.output_dir")
        destination = Path(project).resolve()
        if (
            not cfg.train.get("resume_from")
            and destination.exists()
            and (not destination.is_dir() or any(destination.iterdir()))
        ):
            raise ValueError(
                "Fresh MDLM output_dir is populated; choose a unique directory or resume"
            )
        if cfg.train.get("mixed_precision") not in {None, "no", "fp16", "bf16"}:
            raise ValueError("Unsupported MDLM precision; use no, fp16, or bf16")
    if str(cfg.train.get("objective", "codebook")).lower() == "mlm":
        for key in (
            "train.fape.enabled",
            "model.decoder.enabled",
            "train.decoding.eval_enabled",
        ):
            if OmegaConf.select(cfg, key, default=False):
                raise ValueError(
                    f"{key}=true is unsupported for MLM; geometry requires the codebook objective"
                )
    for key, supported in (
        ()
        if is_mdlm
        else (
            ("model.classifier.tie_to_codebook", True),
            ("model.codebook.trainable", False),
            ("model.decoder.freeze", True),
        )
    ):
        if OmegaConf.select(cfg, key, default=supported) != supported:
            raise ValueError(f"{key} only supports {supported} in the training CLI")
    if str(cfg.train.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("train.optimizer only supports adamw")
    schedule = str(cfg.train.scheduler).lower()
    if schedule not in {"warmup_linear", "warmup_cosine", "wsd_linear", "wsd_cosine"}:
        raise ValueError(f"Unknown train.scheduler: {schedule}")
    decay = schedule.rsplit("_", 1)[1]
    warmup_steps = int(cfg.train.get("warmup_steps", 0))
    stable_steps = int(cfg.train.get("stable_steps", 0))
    decay_steps_raw = cfg.train.get("decay_steps")
    decay_steps = int(decay_steps_raw) if decay_steps_raw is not None else None
    if (
        warmup_steps < 0
        or stable_steps < 0
        or (decay_steps is not None and decay_steps < 0)
    ):
        raise ValueError("scheduler step counts must be non-negative")
    if schedule.startswith("warmup_") and stable_steps:
        raise ValueError(
            "train.stable_steps requires a wsd_linear or wsd_cosine scheduler"
        )
    if OmegaConf.select(cfg, "model.init.std") is not None:
        raise ValueError(
            "model.init.std is unsupported; initialization follows module defaults"
        )
    for name, value, minimum in (
        (
            "gradient_accumulation_steps",
            cfg.train.get("gradient_accumulation_steps", 1),
            1,
        ),
        ("log_every", cfg.train.get("log_every", 1), 1),
        ("eval.steps", cfg.train.eval.get("steps", 1), 1),
        ("max_steps", cfg.train.get("max_steps", 0), 0),
        ("max_epochs", cfg.train.get("max_epochs"), 0),
    ):
        if value is not None and int(value) < minimum:
            raise ValueError(f"train.{name} must be >= {minimum}")
    os.environ["DS_LOG_LEVEL"] = "warn"  # set DeepSpeed log level to warn

    # set global seed (BEFORE Accelerator init)
    seed = int(cfg.train.get("seed", cfg.get("seed", 1337)))
    set_seed(seed)

    precision = cfg.train.get("mixed_precision")
    accelerator = (
        _maybe_get_accelerator(precision)
        if precision is not None
        else _maybe_get_accelerator()
    )
    is_main = accelerator.is_main_process if accelerator else True
    printer = accelerator.print if accelerator else print

    # allow dynamic config additions (e.g., data.eval.<name> overrides)
    try:
        OmegaConf.set_struct(cfg, False)
        if "data" in cfg:
            OmegaConf.set_struct(cfg.data, False)
            if "eval" in cfg.data:
                OmegaConf.set_struct(cfg.data.eval, False)
            if "train" in cfg.data and isinstance(cfg.data.train, DictConfig):
                OmegaConf.set_struct(cfg.data.train, False)
    except Exception:
        pass

    is_mlm = objective == "mlm"

    if is_main:
        printer(f"Training objective: {objective}")

    cfg.train.effective_precision = accelerator.mixed_precision if accelerator else "no"
    codebook = None
    codebook_size = None
    if is_mdlm:
        from stok.eval.mdlm import (
            evaluate_mdlm,
            resolve_mdlm_eval_config,
            validate_mdlm_decoder,
        )

        decoder = None
        preflight_error = None
        try:
            device = accelerator.device if accelerator else torch.device("cpu")
            effective_precision = cfg.train.effective_precision
            requested_precision = cfg.train.get("mixed_precision")
            if effective_precision not in {"no", "fp16", "bf16"} or (
                requested_precision is not None
                and requested_precision != effective_precision
            ):
                raise ValueError("Unsupported requested/effective MDLM precision")
            if effective_precision == "fp16" and device.type != "cuda":
                raise ValueError("MDLM fp16 precision requires CUDA")
            if effective_precision == "bf16" and (
                device.type not in {"cpu", "cuda"}
                or (device.type == "cuda" and not torch.cuda.is_bf16_supported())
            ):
                raise ValueError("MDLM bf16 precision is unsupported on this device")
            if OmegaConf.select(cfg, "train.fape.enabled", default=False):
                raise ValueError("train.fape.enabled is unsupported for MDLM")
            mlm = cfg.train.get("mlm") or {}
            defaults = OmegaConf.load(
                Path(__file__).resolve().parents[1] / "configs/train/base.yaml"
            )
            if mlm.get("enabled", False) or any(
                key in mlm and mlm[key] != defaults.mlm[key]
                for key in ("mask_prob", "mask_token_prob", "random_token_prob")
            ):
                raise ValueError(
                    "MLM masking/replacement options are unsupported for MDLM"
                )
            gumbel = cfg.train.get("gumbel") or {}
            if gumbel.get("enabled", False) or any(
                key not in defaults.gumbel or value != defaults.gumbel[key]
                for key, value in gumbel.items()
            ):
                raise ValueError("Gumbel loss options are unsupported for MDLM")
            for key, expected in (
                ("model.codebook.trainable", False),
                ("model.decoder.freeze", True),
            ):
                if OmegaConf.select(cfg, key, default=expected) != expected:
                    raise ValueError(f"{key} only supports {expected} for MDLM")
            validate_mdlm_config(cfg.train.mdlm)
            codebook = load_codebook(
                preset=cfg.model.codebook.get("preset"),
                path=cfg.model.codebook.get("path"),
            )
            codebook_size = codebook.shape[0]
            train_sources = {item["name"]: item for item in _parse_train_configs(cfg)}
            mdlm_identity = validate_mdlm_sources(
                train_sources,
                _parse_eval_configs(cfg),
                codebook=codebook,
                split_manifest=cfg.data.get("split_manifest"),
                eval_cohort=OmegaConf.select(cfg, "train.eval.mdlm.cohort"),
                generation_cohort=OmegaConf.select(
                    cfg, "train.eval.mdlm.generation_cohort"
                ),
            )
            cfg.train.mdlm_identity = mdlm_identity
            mdlm_eval = resolve_mdlm_eval_config(cfg)
            train_loader, eval_loaders = _build_dataloaders(
                cfg,
                codebook_size=codebook_size,
                pad_id=cfg.model.encoder.pad_id,
                objective="mdlm",
            )
            model = build_model(cfg, codebook=codebook)
            if mdlm_eval.generation.enabled and mdlm_eval.generation.decode:
                from stok.utils.sampling import inference_context

                with inference_context(model):
                    decoder = load_pretrained_decoder(
                        preset=cfg.model.decoder.get("preset")
                        or cfg.model.codebook.get("preset")
                        or "base",
                        path=cfg.model.decoder.get("path"),
                        device=accelerator.device if accelerator else "cpu",
                        freeze=True,
                        progress=is_main,
                    )
                    validate_mdlm_decoder(
                        decoder, codebook, mdlm_identity["codebook_sha256"]
                    )
            optimizer = AdamW(
                model.parameters(),
                lr=cfg.train.lr,
                betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
                eps=cfg.train.adam_eps,
                weight_decay=cfg.train.weight_decay,
            )
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"
        _raise_rank_errors(preflight_error, accelerator, "MDLM preflight failed")

    if (
        accelerator
        and is_main
        and accelerator.num_processes == 1
        and torch.cuda.device_count() > 1
    ):
        printer(
            "Multiple CUDA devices detected but only one process is active. "
            "Launch multi-GPU with: accelerate launch -m stok.train <overrides>"
        )

    io_dirs = _resolve_project_dirs(cfg)

    # Load codebook only for codebook objective
    if objective == "codebook":
        codebook = load_codebook(
            preset=cfg.model.codebook.get("preset"),
            path=cfg.model.codebook.get("path"),
        )
        codebook_size = codebook.shape[0]

    # Shared encoder configuration; MDLM uses paired tied heads.
    if not is_mdlm:
        model = build_model(cfg, codebook=codebook)

    # Load pre-trained encoder if specified (typically for codebook training after MLM)
    pretrained_encoder_path = cfg.train.get("pretrained_encoder")
    if pretrained_encoder_path is not None and is_main and not is_mdlm:
        _load_pretrained_encoder(
            model,
            str(pretrained_encoder_path),
            accelerator=accelerator,
            printer=printer,
        )

    # Count trainable parameters for FLOPs tracking (6N approximation)
    num_params = count_parameters(model, trainable_only=True)
    if is_main:
        printer(f"Trainable parameters: {num_params:,}")
        if is_mdlm:
            device = accelerator.device if accelerator else torch.device("cpu")
            hardware = (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            )
            mdlm_startup = [
                f"Trainable parameters: {num_params:,}",
                f"MDLM hardware: device={device}, hardware={hardware}, requested_precision={cfg.train.get('mixed_precision')}, mixed_precision={cfg.train.effective_precision}",
                f"MDLM data identity: {OmegaConf.to_container(cfg.train.mdlm_identity, resolve=True)}",
            ]
            for line in mdlm_startup[1:]:
                printer(line)

    # data
    if not is_mdlm:
        train_loader, eval_loaders = _build_dataloaders(
            cfg,
            codebook_size=codebook_size,
            pad_id=cfg.model.encoder.pad_id,
            is_mlm=is_mlm,
        )

    # load frozen geometric decoder for FAPE loss and/or eval metrics (optional)
    # Skip decoder setup for MLM objective
    if not is_mdlm:
        decoder = None
    want_fape = False
    want_eval_decode = False

    if objective == "codebook":
        want_fape = bool(getattr(cfg.train, "fape", {}).get("enabled", False))
        # default to False; eval-time decoding is opt-in via config/override
        want_eval_decode = any(
            METRIC_REGISTRY[name].requires_decoder
            for loader in eval_loaders.values()
            for name in getattr(loader, "metric_configs")
        )
        decoder_enabled = bool(getattr(cfg.model, "decoder", {}).get("enabled", False))

        # Auto-enable the decoder when either FAPE or eval-time decoding is requested
        if (want_fape or want_eval_decode) and not decoder_enabled:
            if is_main:
                printer(
                    "train.fape.enabled or train.decoding.eval_enabled is true, but "
                    "model.decoder.enabled=false; enabling decoder automatically."
                )
            try:
                if "decoder" not in cfg.model:
                    cfg.model.decoder = OmegaConf.create({})
                cfg.model.decoder.enabled = True
            except Exception:  # maybe config is immutable?
                pass
            decoder_enabled = True

        if decoder_enabled:
            # resolve preset/path
            dec_preset = getattr(
                cfg.model.decoder, "preset", None
            ) or cfg.model.codebook.get("preset")
            dec_path = getattr(cfg.model.decoder, "path", None)
            device_for_decoder = _get_model_device(model, accelerator)
            decoder = load_pretrained_decoder(
                preset=dec_preset or "base",
                path=dec_path,
                device=device_for_decoder,
                freeze=bool(getattr(cfg.model.decoder, "freeze", True)),
                progress=is_main,
            )
            if accelerator:
                accelerator.wait_for_everyone()
            # check if d_code matches classifier codebook dim
            with torch.no_grad():
                inferred_d_code = int(decoder.projector_in.weight.shape[1])  # type: ignore[attr-defined]
                E = cast(
                    CodebookClassifier, _unwrap_model(model, accelerator).classifier
                ).E
                if inferred_d_code != int(E.shape[1]):
                    raise RuntimeError(
                        f"Decoder d_code={inferred_d_code} does not match codebook dim "
                        f"{int(E.shape[1])}"
                    )

    if not is_mdlm:
        # optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
            eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )

    # determine training steps
    grad_accum_steps: int = cfg.train.get("gradient_accumulation_steps", 1)
    # derive steps_per_epoch when possible (used for both max_steps and logging)
    steps_per_epoch: Optional[int] = None
    try:
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)  # type: ignore[arg-type]
        if steps_per_epoch <= 0:
            steps_per_epoch = None
    except TypeError:
        # len(train_loader) may be undefined for some iterable datasets
        steps_per_epoch = None

    if cfg.train.get("max_epochs") is not None:
        if steps_per_epoch is None:
            raise ValueError(
                "cfg.train.max_epochs is set but steps_per_epoch could not be derived "
                "from the train dataloader."
            )
        max_steps = int(cfg.train.max_epochs) * steps_per_epoch
    else:
        max_steps = int(cfg.train.get("max_steps", 10000))

    # Schedule durations are measured in successful optimizer updates.
    scheduler = _build_scheduler(
        optimizer,
        decay=decay,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        total_steps=max_steps,
    )

    # prepare with Accelerate (if available)
    if accelerator:
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    resume_payload = None
    progress = None
    resume_stream = None
    resume_error = None
    try:
        signature = resume_signature(
            cfg,
            sources=_parse_train_configs(cfg),
            codebook=codebook,
            accelerator=accelerator,
            training_decoder=decoder if want_fape else None,
        )
        if cfg.train.get("resume_from"):
            resume_payload = read_training_checkpoint(Path(cfg.train.resume_from))
            validate_resume_signature(resume_payload, signature)
            progress = restore_training_state(
                resume_payload,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
            )
            if progress["global_step"] > max_steps:
                raise ValueError("Resume counter exceeds original update budget")
            cursor = progress["batches_in_epoch"]
            if cursor > len(train_loader) or (
                cursor != len(train_loader) and cursor % grad_accum_steps
            ):
                raise ValueError(
                    "Resume cursor is not a completed local accumulation boundary"
                )
            resume_stream, _ = epoch_iterator(
                train_loader,
                epoch=progress["epoch"],
                consumed=cursor,
                seed=seed,
                rank=accelerator.process_index if accelerator else 0,
                loader_state=progress["loader_generator_state"],
            )
    except Exception as exc:
        resume_error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(resume_error, accelerator, "Resume preflight failed")

    output_error = None
    if is_main:
        try:
            _ensure_dirs(list(io_dirs.values()))
        except Exception as exc:
            output_error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(output_error, accelerator, "Creating project directories failed")
    wb = None
    logging_error = None
    try:
        wb = _maybe_init_wandb(
            cfg,
            is_main_process=is_main,
            logs_dir=io_dirs["logs"],
            resume_payload=resume_payload,
        )
    except Exception as exc:
        logging_error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(logging_error, accelerator, "W&B initialization failed")
    output_error = None
    if is_main:
        try:
            _save_config_snapshot(cfg, io_dirs["configs"] / "run.yaml")
        except Exception as exc:
            output_error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(output_error, accelerator, "Saving configuration failed")
    wandb_run_id = getattr(getattr(wb, "run", None), "id", None) or (
        resume_payload["wandb_run_id"] if resume_payload else None
    )
    if accelerator:
        wandb_run_id = gather_object([wandb_run_id])[0]

    # train loop
    model.train()
    global_step = progress["global_step"] if progress else 0
    micro_step = progress["micro_step"] if progress else 0
    log_interval = int(cfg.train.get("log_every", 50))
    eval_interval = int(cfg.train.get("eval", {}).get("steps", 1000))
    grad_clip = float(cfg.train.get("max_grad_norm", 1.0))

    # console output (main process only
    console_cfg = cfg.train.get("console")
    console_enabled = True
    if console_cfg is not None:
        console_enabled = bool(console_cfg.get("enabled", True))
    # console progbar renders to stdout only, text lines are also logged separately to file
    log_file_handle = None
    output_error = None
    if is_main:
        try:
            log_file_handle = (io_dirs["logs"] / "train.log").open(
                "a", encoding="utf-8"
            )
            print(
                f"Training started. Objective: {objective}",
                file=log_file_handle,
                flush=True,
            )
            if is_mdlm:
                print("\n".join(mdlm_startup), file=log_file_handle, flush=True)
            if resume_payload:
                print(
                    f"Resume/rollback: checkpoint step {global_step}; remote history watermark {getattr(wb, 'watermark', -1)}",
                    file=log_file_handle,
                    flush=True,
                )
        except Exception as exc:
            output_error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(output_error, accelerator, "Opening training log failed")
    console = ConsoleLogger(
        total_steps=max_steps,
        initial_step=global_step,
        is_main=is_main,
        enabled=console_enabled,
        file=sys.stdout,
    )
    # Initialize modular evaluation system
    evaluator = (
        None
        if is_mdlm
        else Evaluator(
            cfg=cfg,
            model=model,
            accelerator=accelerator,
            decoder=decoder,
        )
    )
    metric_logger = MetricLogger(
        console=console,
        wandb=wb,
        log_file=log_file_handle,
        is_main=is_main,
        objective=objective,
    )

    # FLOPs tracking preserves each task's historical position accounting.
    total_tokens = 0
    total_residues = 0
    task: MDLMTask | ClassificationTask
    if is_mdlm:
        assert codebook_size is not None
        task = MDLMTask(cfg, codebook_size=codebook_size)
    else:
        task = ClassificationTask(
            cfg,
            decoder=decoder,
            codebook=None
            if is_mlm
            else cast(
                CodebookClassifier, _unwrap_model(model, accelerator).classifier
            ).E,
        )

    epoch = progress["epoch"] if progress else 0
    batches_in_pass = progress["batches_in_epoch"] if progress else 0
    loader_generator_state = (
        progress["loader_generator_state"]
        if progress
        else torch.Generator().manual_seed(seed).get_state()
    )
    if progress:
        total_tokens, total_residues = (
            progress["executed_positions"],
            progress["residues_seen"],
        )
        task.restore_logging_state(cast(dict, progress))

    def checkpoint_state():
        return {
            "signature": signature,
            "wandb_run_id": wandb_run_id,
            "local": {
                "epoch": epoch,
                "batches_in_epoch": batches_in_pass,
                "loader_generator_state": loader_generator_state,
                "logging": task.logging_state(),
            },
        }

    epoch_limit = cfg.train.get("max_epochs")
    device = _get_model_device(model, accelerator)
    world_size = accelerator.num_processes if accelerator else 1
    optimizer.zero_grad(set_to_none=True)
    while global_step < max_steps and (epoch_limit is None or epoch < int(epoch_limit)):
        replay_error = None
        try:
            if progress:
                stream = resume_stream
                restore_rng_state(progress["rng"])
                progress = None
                resume_stream = None
            else:
                stream, loader_generator_state = epoch_iterator(
                    train_loader,
                    epoch=epoch,
                    consumed=0,
                    seed=seed,
                    rank=accelerator.process_index if accelerator else 0,
                )
        except Exception as exc:
            replay_error = f"{type(exc).__name__}: {exc}"
        _raise_rank_errors(replay_error, accelerator, "Replaying training data failed")
        resumed_batches = batches_in_pass
        updates_before_pass = global_step
        eligible_windows_in_pass = 0
        for window in iter_windows(stream, grad_accum_steps, accelerator):
            batches_in_pass += len(window)
            current_step = global_step + 1
            current_epoch = epoch + batches_in_pass / max(1, len(train_loader))
            # Tasks prepare CPU payloads/counts; collectives remain in the driver.
            prepare_error = None
            try:
                prepared = task.prepare_window(
                    window,
                    epoch=epoch,
                    micro_step=micro_step,
                    global_step=global_step,
                    rank=accelerator.process_index if accelerator else 0,
                    world_size=world_size,
                )
            except Exception as exc:
                prepare_error = f"{type(exc).__name__}: {exc}"
            _raise_rank_errors(
                prepare_error,
                accelerator,
                f"Preparing {'MDLM' if is_mdlm else 'classification'} window failed",
            )
            window = prepared.batches
            counts = prepared.counts.to(device)
            if accelerator:
                counts = accelerator.reduce(counts, reduction="sum")
            denominators = task.denominators(counts)
            accounting = task.consume_counts(counts)
            total_residues += accounting.residues_seen
            total_tokens += accounting.executed_positions
            micro_step += len(window)
            if not any(denominators.values()):
                continue
            eligible_windows_in_pass += 1
            window_statistics = None
            for micro_index, batch in enumerate(window):
                sync = (
                    accelerator.no_sync(model)
                    if accelerator and micro_index < len(window) - 1
                    else nullcontext()
                )
                with sync:
                    error = None
                    try:
                        result = task.forward(
                            model, batch, device=device, global_step=global_step
                        )
                        # Preserve numerator order: classification CE then weighted FAPE.
                        losses = iter(result.loss_sums.items())
                        name, numerator = next(losses)
                        loss = numerator * (
                            world_size / denominators[name]
                            if denominators[name]
                            else 0.0
                        )
                        for name, numerator in losses:
                            loss = loss + numerator * (
                                world_size / denominators[name]
                                if denominators[name]
                                else 0.0
                            )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite training loss")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    _raise_rank_errors(error, accelerator, "Training forward failed")
                    if accelerator:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    if window_statistics is None:
                        window_statistics = torch.zeros_like(result.statistics)
                    window_statistics += result.statistics
            if grad_clip > 0:
                if accelerator:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            skipped = bool(accelerator and accelerator.optimizer_step_was_skipped)
            optimizer.zero_grad(set_to_none=True)
            if skipped:
                continue
            scheduler.step()
            assert window_statistics is not None
            if accelerator:
                window_statistics = accelerator.reduce(
                    window_statistics, reduction="sum"
                )
            task.record_update(window_statistics, counts)
            # logging
            if current_step % log_interval == 0 and is_main:
                cumulative_flops = compute_flops_6n(num_params, total_tokens)
                msg, payload = task.format_log(
                    step=current_step,
                    max_steps=max_steps,
                    micro_step=micro_step,
                    epoch=current_epoch,
                    lr=scheduler.get_last_lr()[0],
                    flops=cumulative_flops,
                    residues_seen=total_residues,
                    executed_positions=total_tokens,
                )
                console.train(msg)
                if log_file_handle is not None:
                    print(
                        msg + f" (flops_actual={cumulative_flops})",
                        file=log_file_handle,
                        flush=True,
                    )
                if wb is not None:
                    wb.log(payload, step=current_step)
                task.reset_log_window()

            if is_mdlm:
                denoising_due = mdlm_eval.enabled and current_step % eval_interval == 0
                generation_due = (
                    mdlm_eval.generation.enabled
                    and current_step % mdlm_eval.generation.steps == 0
                )
                if denoising_due or generation_due:
                    all_eval_metrics = evaluate_mdlm(
                        model,
                        eval_loaders,
                        cfg,
                        accelerator=accelerator,
                        decoder=decoder,
                        run_denoising=denoising_due,
                        run_generation=generation_due,
                    )
                    metric_logger.log_eval_all(
                        all_eval_metrics,
                        current_step,
                        current_epoch,
                        compute_flops_6n(num_params, total_tokens),
                    )

            # eval across all configured eval loaders (using modular eval system)
            if (
                not is_mdlm
                and current_step % eval_interval == 0
                and len(eval_loaders) > 0
            ):
                assert evaluator is not None
                all_eval_metrics = evaluator.evaluate_all(eval_loaders)

                # Add epoch to metrics if available
                if current_epoch is not None:
                    for metrics in all_eval_metrics.values():
                        metrics["epoch"] = float(current_epoch)

                # Compute cumulative training FLOPs for eval logging
                eval_train_flops = compute_flops_6n(num_params, total_tokens)

                # Log all eval metrics (including training FLOPs at this checkpoint)
                metric_logger.log_eval_all(
                    all_eval_metrics, current_step, current_epoch, eval_train_flops
                )

            _raise_rank_errors(
                getattr(wb, "error", None), accelerator, "W&B logging failed"
            )
            global_step += 1
            console.step(1)
            # checkpointing
            ckpt_steps = cfg.train.get("save_every")
            if (
                ckpt_steps is not None
                and int(ckpt_steps) > 0
                and (global_step % int(ckpt_steps) == 0)
            ):
                step_path = io_dirs["checkpoints"] / f"step_{global_step:08d}.pt"
                _save_checkpoint(
                    step_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    cfg=cfg,
                    accelerator=accelerator,
                    micro_step=micro_step,
                    residues_seen=total_residues,
                    executed_positions=total_tokens,
                    training_state=checkpoint_state(),
                )
                checkpoint_error = None
                if is_main:
                    try:
                        with _atomic_destination(
                            io_dirs["checkpoints"] / "latest.pt"
                        ) as temporary:
                            shutil.copyfile(step_path, temporary)
                    except Exception as exc:
                        checkpoint_error = f"{type(exc).__name__}: {exc}"
                _raise_rank_errors(checkpoint_error, accelerator, "Checkpoint failed")
            if global_step >= max_steps:
                break
        if batches_in_pass == 0:
            raise RuntimeError("Training loader produced no complete batches")
        if (
            not resumed_batches
            and global_step == updates_before_pass
            and (not task.allow_skipped_only_pass or eligible_windows_in_pass == 0)
        ):
            raise RuntimeError("Training pass made no successful optimizer update")
        if global_step < max_steps:
            epoch += 1
            batches_in_pass = 0

    if progress:
        restore_rng_state(progress["rng"])
    _save_checkpoint(
        io_dirs["model"] / "final.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        micro_step=micro_step,
        residues_seen=total_residues,
        executed_positions=total_tokens,
        cfg=cfg,
        accelerator=accelerator,
        training_state=checkpoint_state(),
    )
    if is_main:
        console.close()
        console.print("Training complete.")
        if log_file_handle is not None:
            print("Training complete.", file=log_file_handle, flush=True)
        # close log file if opened
        if log_file_handle is not None:
            try:
                log_file_handle.close()
            except Exception:
                pass


if __name__ == "__main__":
    print(
        "This module is intended to be invoked via the CLI: `stok train ...`",
        file=sys.stderr,
    )
