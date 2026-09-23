from functools import partial
from contextlib import contextmanager, nullcontext
from itertools import islice
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import gather_object, set_seed
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    Sampler,
    DistributedSampler,
)

from stok.data.collate import align_coords, mlm_collate, tokenize_residues
from stok.data.dataset import (
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
from stok.models.stok import STokModel
from stok.utils.codebook import load_codebook
from stok.utils.console import ConsoleLogger
from stok.utils.decoding import decode_token_aligned_coords, logits_to_soft_codes_gumbel
from stok.utils.masking import residue_mask_from_tokens
from stok.utils.flops import compute_flops_6n, count_parameters, format_flops_scientific
from stok.utils.losses import fape_loss, token_ce_loss
from stok.utils.tokenizer import Tokenizer


def _maybe_get_accelerator():
    from accelerate import Accelerator
    accelerator = Accelerator()
    if accelerator.distributed_type.name not in {"NO", "MULTI_CPU", "MULTI_GPU"}:
        raise ValueError(f"Unsupported distributed backend: {accelerator.distributed_type}; use replicated DDP")
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
    root = Path(str(cfg.train.get("project_path") or Path.cwd())).resolve()
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


def _collect_rng_state() -> dict[str, Any]:
    # convert numpy RNG state to only primitives/lists to be loadable with weights_only=True
    np_state = list(np.random.get_state())
    try:
        # element 1 is the key array
        if hasattr(np_state[1], "tolist"):
            np_state[1] = np_state[1].tolist()
    except Exception:
        pass
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np_state,
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            # on some backends/devices this may not be available
            pass
    return state


@contextmanager
def _atomic_destination(path: Path):
    """Replace a destination only after its complete sibling file is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _unwrap_model(model, accelerator).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(global_step),
        "micro_step": int(micro_step),
        "step_unit": "optimizer_update",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "rng_state": _collect_rng_state(),
    }
    with _atomic_destination(path) as temporary:
        torch.save(payload, temporary)


def _load_pretrained_encoder(
    model: STokModel,
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
        tokens, labels = zip(*batch)  # type: ignore[arg-type]
        return torch.stack(tokens, dim=0), torch.stack(labels, dim=0)

    # else TokenizedDataset dicts with 'seq' and optionally 'indices'
    input_ids = []
    label_ids = []
    coords_batch: list[torch.Tensor] = []
    for item in batch:  # type: ignore[assignment]
        seq: str = item["seq"]

        if pad_id != tokenizer.pad_token_id:
            raise ValueError("Model pad_id must match tokenizer padding")
        ids = tokenize_residues(seq, tokenizer, max_len)

        # build labels aligned to tokens: CLS/EOS/PAD -> ignore_index
        L = ids.size(0)
        labels = torch.full((L,), ignore_index, dtype=torch.long)

        # Handle indices if present (may be absent for structure folder datasets)
        indices_raw = item.get("indices")
        if indices_raw is not None:
            indices: torch.Tensor = indices_raw.long()
            if num_classes is not None and (indices >= num_classes).any():
                raise ValueError(f"Sample {item.get('pid', '?')}: class ID out of range")
            copy_len = min(len(seq), int(indices.numel()), L - 2)
            values = indices[:copy_len]
            labels[1:1+copy_len] = values.masked_fill(values < 0, ignore_index)

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
    is_mlm: bool = False,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    rank, world_size = distributed_rank()
    batch_size: int = cfg.data.batch_size
    max_len: int = cfg.data.max_len
    num_workers: int = cfg.data.num_workers
    pin_memory: bool = cfg.data.pin_memory
    ignore_index: int = cfg.model.classifier.ignore_index

    # resolve dataloader buffering
    prefetch_factor: int = int(getattr(cfg.data, "prefetch_factor", 2))

    # resolve whether to load 3D coordinates from disk
    user_load_coords = getattr(cfg.data, "load_coords", None)

    eval_configs = _parse_eval_configs(cfg)
    train_configs = _parse_train_configs(cfg)
    objective = "mlm" if is_mlm else "codebook"
    fape_required = not is_mlm and bool(cfg.train.get("fape", {}).get("enabled", False))
    def coordinate_setting(options, needed, required=False):
        value = options.get("load_coords", user_load_coords)
        alias = options.get("has_coords")
        if alias is not None:
            if "load_coords" in options and value is not None and bool(value) != bool(alias):
                raise ValueError("Conflicting has_coords and load_coords settings")
            value = alias
        if value is False and required:
            raise ValueError("load_coords=false conflicts with requested structure supervision/metrics")
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
        require_indices: bool = True,
        *,
        dataset_format: str | None = None,
        chain_id: str | None = None,
        recursive: bool = False,
    ):
        p = Path(path)

        # Explicit structure folder format
        if dataset_format == "structure":
            from stok.data.structure_dataset import StructureFolderDataset

            ds = StructureFolderDataset(
                folder_path=str(p),
                max_length=max_len,
                chain_id=chain_id,
                recursive=recursive, load_coords=load_coords,
            )
            return ds

        # heuristic: directory containing parquet shards -> Iterable; else map-style
        if p.is_dir():
            has_parquet = (
                any(p.glob("*.parquet")) or any(p.glob("*.parq")) or any(p.glob("*.pq"))
            )
            if has_parquet:
                shuffle_shards = bool(getattr(cfg.data, "shuffle_shards", True))
                shuffle_rows = bool(getattr(cfg.data, "shuffle_rows", True))
                return IterableTokenizedDataset(
                    dataset_path=str(p),
                    max_length=max_len,
                    shuffle_shards=shuffle_shards,
                    shuffle_rows=shuffle_rows,
                    load_coords=bool(load_coords),
                    require_indices=require_indices,
                )

            # Auto-detect structure folder (no parquet, has structure files)
            has_structures = any(
                f.suffix.lower() in structure_exts for f in p.iterdir() if f.is_file()
            )
            if has_structures:
                from stok.data.structure_dataset import StructureFolderDataset

                return StructureFolderDataset(
                    folder_path=str(p),
                    max_length=max_len,
                    chain_id=chain_id,
                    recursive=recursive, load_coords=load_coords,
                )

        return TokenizedDataset(
            dataset_path=str(path),
            max_length=max_len,
            load_coords=bool(load_coords),
            require_indices=require_indices,
        )

    if len(train_configs) > 0:
        # Real dataset(s); tokenize in collate
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

        if len(train_configs) == 1:
            # Single dataset (backwards compatible)
            train_load, force_coords = coordinate_setting(train_configs[0], fape_required, fape_required)
            train_ds = _pick_dataset(
                str(train_configs[0]["path"]),
                train_load,
                require_indices=not is_mlm,
            )
            if force_coords and not train_ds.has_coords:
                raise ValueError("load_coords=true requires a coordinate-capable training source")
        else:
            # Multiple datasets with fractions
            ds_pairs: list[tuple[Dataset | IterableDataset, float]] = []
            for tcfg in train_configs:
                t_load_coords, force_coords = coordinate_setting(tcfg, fape_required, fape_required)
                ds = _pick_dataset(
                    str(tcfg["path"]),
                    t_load_coords,
                    require_indices=not is_mlm,
                )
                if force_coords and not ds.has_coords:
                    raise ValueError(f"load_coords=true requires coordinates: {tcfg['path']}")
                ds_pairs.append((ds, float(tcfg["fraction"])))

            any_iterable = any(isinstance(ds, IterableDataset) for ds, _ in ds_pairs)
            if any_iterable:
                # Convert map-style datasets to iterable wrappers, then interleave
                iterables: list[IterableDataset] = []
                fracs: list[float] = []
                total_samples = 0
                for ds, frac in ds_pairs:
                    if isinstance(ds, IterableDataset):
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
                map_datasets: list[Dataset] = [ds for ds, _ in ds_pairs]  # type: ignore[list-item]
                lengths = [int(len(ds)) for ds in map_datasets]
                fracs = [float(fr) for _, fr in ds_pairs]
                concat = ConcatDataset(map_datasets)
                sampler = MixtureSampler(
                    lengths=lengths,
                    rank=rank, world_size=world_size,
                    fractions=fracs,
                    seed=int(cfg.train.get("seed", 1337)),
                )
                concat.has_coords = any(getattr(ds, "has_coords", False) for ds in map_datasets)
                concat.has_labels = any(getattr(ds, "has_labels", True) for ds in map_datasets)
                train_ds = concat
                train_sampler = sampler
    else:
        # fallback dummy data for quick smoke test
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
        if tokenizer.pad_token_id != pad_id or len(tokenizer) != int(cfg.model.encoder.vocab_size):
            raise ValueError("Model vocabulary and pad_id must match tokenizer")
        for key in ("bos_id", "eos_id"):
            OmegaConf.update(cfg, f"model.encoder.{key}",
                             getattr(tokenizer, key.replace("_id", "_token_id")), force_add=True)

    def _make_dl_kwargs(batch_sz: int):
        kwargs = {
            "batch_size": batch_sz,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn,
            "persistent_workers": (num_workers > 0),
        }
        if num_workers > 0 and prefetch_factor is not None and prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        return kwargs

    if is_iterable:
        train_ds.training_batch_size = batch_size
        train_ds.num_workers = num_workers
        dropped = train_ds.num_samples - _usable_samples(train_ds)
        if rank == 0 and dropped:
            print(f"Training stream drops {dropped} samples per pass for complete rank/worker batches")
    elif train_sampler is None:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=int(cfg.train.get("seed", 1337)), drop_last=True)
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
        resolved = resolve_eval_metrics(cfg, name, objective=objective)
        needs_coords = any(METRIC_REGISTRY[key].requires_coords for key in resolved)
        requires_coords = any(settings["explicit"] and METRIC_REGISTRY[key].requires_coords
                              for key, settings in resolved.items())
        eval_load_coords, force_coords = coordinate_setting(eval_cfg, needs_coords, requires_coords)
        # Extract structure folder format options
        eval_format = eval_cfg.get("format")
        eval_chain_id = eval_cfg.get("chain_id")
        eval_recursive = bool(eval_cfg.get("recursive", False))

        # Structure folders always have coords, don't require indices
        ds = _pick_dataset(
            eval_path,
            eval_load_coords,
            require_indices=False,
            dataset_format=eval_format,
            chain_id=eval_chain_id,
            recursive=eval_recursive,
        )
        if force_coords and not ds.has_coords:
            raise ValueError(f"Dataset {name}: load_coords=true requires a coordinate-capable source")
        for metric_name, settings in resolved.items():
            if not settings["explicit"]:
                continue
            if METRIC_REGISTRY[metric_name].requires_coords and not ds.has_coords:
                raise ValueError(f"Dataset {name}, metric {metric_name}: missing coordinates")
            if not is_mlm and metric_name in {"accuracy", "perplexity"} and not ds.has_labels:
                raise ValueError(f"Dataset {name}, metric {metric_name}: missing labels")
        eval_cfg["load_coords"] = bool(ds.has_coords)
        if isinstance(ds, IterableDataset):
            ds.shuffle_shards = False
            ds.shuffle_rows = False
            eval_sampler = None
        else:
            eval_sampler = range(rank, len(ds), world_size)
        eval_kwargs = _make_dl_kwargs(eval_batch_size)
        eval_seed = int(cfg.train.get("eval", {}).get("seed", cfg.train.get("seed", 1337)))
        eval_kwargs["generator"] = torch.Generator().manual_seed(eval_seed)
        if is_mlm:
            eval_kwargs["collate_fn"] = partial(mlm_collate, tokenizer=tokenizer,
                max_len=max_len, mask_prob=mask_prob, mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob, pad_id=pad_id,
                ignore_index=ignore_index, eval_seed=eval_seed, dataset_name=name)
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


def _maybe_wandb_login(cfg: DictConfig, *, is_main_process: bool):
    if cfg.train.get("wandb") and cfg.train.wandb.get("enabled", True):
        if is_main_process:
            try:
                import wandb

                wandb.login()  # trigger login prompt early; do not create a run yet
            except Exception:
                # proceed without W&B
                pass


def _maybe_init_wandb(
    cfg: DictConfig, *, is_main_process: bool, logs_dir: Optional[Path] = None
):
    wb = None
    if cfg.train.get("wandb") and cfg.train.wandb.get("enabled", True):
        if is_main_process:
            try:
                import wandb

                init_kwargs = dict(
                    project=cfg.train.wandb.get("project", "stok"),
                    entity=cfg.train.wandb.get("entity"),
                    group=cfg.train.wandb.get("group"),
                    name=cfg.train.wandb.get("name"),
                    tags=list(cfg.train.wandb.get("tags", [])),
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
                if logs_dir is not None:
                    os.environ["WANDB_DIR"] = logs_dir.as_posix()
                    init_kwargs["dir"] = logs_dir.as_posix()
                wandb.init(**init_kwargs)
                wb = wandb
            except Exception:
                # proceed without W&B
                wb = None
    return wb


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
                raise RuntimeError(f"Unequal accumulation window lengths across ranks: {sizes}")
        if not window:
            return
        yield window


def _raise_rank_errors(error, accelerator, context: str):
    errors = gather_object([error]) if accelerator else [error]
    if any(errors):
        raise RuntimeError(f"{context}: {errors}")


def run_training(cfg: DictConfig):
    for key, supported in (("model.classifier.tie_to_codebook", True),
                           ("model.codebook.trainable", False),
                           ("model.decoder.freeze", True)):
        if OmegaConf.select(cfg, key, default=supported) != supported:
            raise ValueError(f"{key} only supports {supported} in the training CLI")
    if str(cfg.train.optimizer.get("name", "adamw")).lower() != "adamw":
        raise ValueError("train.optimizer.name only supports adamw")
    if OmegaConf.select(cfg, "model.init.std") is not None:
        raise ValueError("model.init.std is unsupported; initialization follows module defaults")
    for name, value, minimum in (
        ("grad_accum_steps", cfg.train.get("grad_accum_steps", 1), 1),
        ("log_steps", cfg.train.get("log_steps", 1), 1),
        ("eval.steps", cfg.train.eval.get("steps", 1), 1),
        ("num_steps", cfg.train.get("num_steps", 0), 0),
        ("epochs", cfg.train.get("epochs"), 0),
    ):
        if value is not None and int(value) < minimum:
            raise ValueError(f"train.{name} must be >= {minimum}")
    os.environ["DS_LOG_LEVEL"] = "warn"  # set DeepSpeed log level to warn

    # set global seed (BEFORE Accelerator init)
    seed = int(cfg.train.get("seed", cfg.get("seed", 1337)))
    set_seed(seed)

    accelerator = _maybe_get_accelerator()
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

    # Determine training objective
    objective = str(cfg.train.get("objective", "codebook")).lower()
    if objective not in {"codebook", "mlm"}:
        raise ValueError(
            f"Unknown train.objective: {objective}. Expected 'codebook' or 'mlm'."
        )
    is_mlm = objective == "mlm"

    if is_main:
        printer(f"Training objective: {objective}")

    # prompt for W&B login early so the API key prompt happens immediately
    _maybe_wandb_login(cfg, is_main_process=is_main)

    # warn if multiple GPUs are visible but only one process is active
    if accelerator and is_main:
        world_size = getattr(accelerator, "num_processes", 1)
        if world_size == 1 and torch.cuda.device_count() > 1:
            printer(
                "Multiple CUDA devices detected but only one process is active. "
                "Launch multi-GPU with: accelerate launch -m stok.train <overrides>"
            )

    # resolve project directories and save config (main only)
    io_dirs = _resolve_project_dirs(cfg)
    if is_main:
        _ensure_dirs(
            [
                io_dirs["root"],
                io_dirs["model"],
                io_dirs["checkpoints"],
                io_dirs["logs"],
                io_dirs["configs"],
            ]
        )
    if accelerator:
        accelerator.wait_for_everyone()

    # Load codebook only for codebook objective
    codebook = None
    codebook_size = None
    if not is_mlm:
        codebook = load_codebook(
            preset=cfg.model.codebook.get("preset"),
            path=cfg.model.codebook.get("path"),
        )
        codebook_size = codebook.shape[0]

    # Build model with appropriate head type
    model = STokModel(
        vocab_size=cfg.model.encoder.vocab_size,
        pad_id=cfg.model.encoder.pad_id,
        d_model=cfg.model.encoder.d_model,
        n_heads=cfg.model.encoder.n_heads,
        n_layers=cfg.model.encoder.n_layers,
        ffn_mult=cfg.model.encoder.ffn_mult,
        dropout=cfg.model.encoder.dropout,
        attn_dropout=cfg.model.encoder.attn_dropout,
        codebook=codebook,  # None for MLM
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
        norm_type=cfg.model.encoder.norm,
        head_type=objective,
        tie_word_embeddings=(
            cfg.train.mlm.get("tie_word_embeddings", True) if is_mlm else True
        ),
    )

    # Load pre-trained encoder if specified (typically for codebook training after MLM)
    pretrained_encoder_path = cfg.train.get("pretrained_encoder")
    if pretrained_encoder_path is not None and is_main:
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

    # data
    train_loader, eval_loaders = _build_dataloaders(
        cfg,
        codebook_size=codebook_size,
        pad_id=cfg.model.encoder.pad_id,
        is_mlm=is_mlm,
    )

    # load frozen geometric decoder for FAPE loss and/or eval metrics (optional)
    # Skip decoder setup for MLM objective
    decoder = None
    want_fape = False
    want_eval_decode = False
    log_pred_nan_frac = False

    if not is_mlm:
        want_fape = bool(getattr(cfg.train, "fape", {}).get("enabled", False))
        # default to False; eval-time decoding is opt-in via config/override
        want_eval_decode = any(METRIC_REGISTRY[name].requires_decoder
            for loader in eval_loaders.values() for name in loader.metric_configs)
        # FAPE behavior toggles (with safe defaults)
        log_pred_nan_frac = bool(
            getattr(cfg.train, "fape", {}).get("log_pred_nan_frac", True)
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
                E = _unwrap_model(model, accelerator).classifier.E
                if inferred_d_code != int(E.shape[1]):
                    raise RuntimeError(
                        f"Decoder d_code={inferred_d_code} does not match codebook dim "
                        f"{int(E.shape[1])}"
                    )

    if is_main:
        _save_config_snapshot(cfg, io_dirs["configs"] / "run.yaml")

    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=tuple(cfg.train.optimizer.betas),
        weight_decay=cfg.train.optimizer.weight_decay,
    )

    # determine training steps
    grad_accum_steps: int = cfg.train.get("grad_accum_steps", 1)
    # derive steps_per_epoch when possible (used for both max_steps and logging)
    steps_per_epoch: Optional[int] = None
    try:
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)  # type: ignore[arg-type]
        if steps_per_epoch <= 0:
            steps_per_epoch = None
    except TypeError:
        # len(train_loader) may be undefined for some iterable datasets
        steps_per_epoch = None

    if cfg.train.get("epochs") is not None:
        if steps_per_epoch is None:
            raise ValueError(
                "cfg.train.epochs is set but steps_per_epoch could not be derived "
                "from the train dataloader."
            )
        max_steps = int(cfg.train.epochs) * steps_per_epoch
    else:
        max_steps = int(cfg.train.get("num_steps", 10000))

    # build scheduler (WSD with decay selection)
    sched_cfg = cfg.train.scheduler
    if not sched_cfg.get("decay"):
        raise ValueError(
            "Missing required config: train.scheduler.decay (expected 'cosine' or 'linear')"
        )
    decay: str = str(sched_cfg.get("decay")).lower()
    warmup_steps: int = int(sched_cfg.get("warmup_steps", 0))
    stable_steps: int = int(sched_cfg.get("stable_steps", 0))
    # Allow explicit 0; None triggers derivation in _build_scheduler
    decay_steps_raw: Optional[int] = sched_cfg.get("decay_steps")
    decay_steps: Optional[int] = (
        int(decay_steps_raw) if decay_steps_raw is not None else None
    )
    if (
        warmup_steps < 0
        or stable_steps < 0
        or (decay_steps is not None and decay_steps < 0)
    ):
        raise ValueError("scheduler step counts must be non-negative")

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

    # W&B
    wb = _maybe_init_wandb(cfg, is_main_process=is_main, logs_dir=io_dirs["logs"])

    # train loop
    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    log_interval = int(cfg.train.get("log_steps", 50))
    eval_interval = int(cfg.train.get("eval", {}).get("steps", 1000))
    ignore_index = int(cfg.model.classifier.ignore_index)
    grad_clip = float(cfg.train.get("grad_clip_norm", 1.0))

    # console output (main process only
    console_cfg = cfg.train.get("console")
    console_enabled = True
    if console_cfg is not None:
        console_enabled = bool(console_cfg.get("enabled", True))
    # console progbar renders to stdout only, text lines are also logged separately to file
    log_file_handle = None
    if is_main:
        log_file_handle = (io_dirs["logs"] / "train.log").open("a", encoding="utf-8")
    console = ConsoleLogger(
        total_steps=max_steps,
        initial_step=global_step,
        is_main=is_main,
        enabled=console_enabled,
        file=sys.stdout,
    )
    if is_main and log_file_handle is not None:
        print(
            f"Training started. Objective: {objective}",
            file=log_file_handle,
            flush=True,
        )

    # Initialize modular evaluation system
    evaluator = Evaluator(
        cfg=cfg,
        model=model,
        accelerator=accelerator,
        decoder=decoder,
    )
    metric_logger = MetricLogger(
        console=console,
        wandb=wb,
        log_file=log_file_handle,
        is_main=is_main,
        objective=objective,
    )

    # additional training accumulators (over the current log window)
    running_cls_loss = 0.0
    running_cls_count = 0
    running_fape_loss = 0.0
    running_fape_count = 0
    running_pred_nan_frac_sum = 0.0
    running_pred_nan_frac_count = 0
    # MLM-specific accumulators
    running_masked_acc_sum = 0.0
    running_masked_acc_count = 0
    # FLOPs tracking (cumulative tokens for 6N approximation)
    total_tokens = 0

    # Gumbel temperature schedule (only for codebook objective)
    def _anneal_tau(step: int) -> float:
        gcfg = getattr(cfg.train, "gumbel", {})
        t0 = float(gcfg.get("tau_start", 1.0))
        t1 = float(gcfg.get("tau_end", 0.5))
        T = int(gcfg.get("anneal_steps", 20000))
        if T <= 0:
            return t1
        if step >= T:
            return t1
        # linear
        return t0 + (t1 - t0) * (float(step) / float(T))

    epoch = 0
    epoch_limit = cfg.train.get("epochs")
    device = _get_model_device(model, accelerator)
    world_size = accelerator.num_processes if accelerator else 1
    optimizer.zero_grad(set_to_none=True)
    while global_step < max_steps and (epoch_limit is None or epoch < int(epoch_limit)):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        epoch += 1
        batches_in_pass = 0
        updates_before_pass = global_step
        for window in iter_windows(train_loader, grad_accum_steps, accelerator):
            batches_in_pass += len(window)
            current_step = global_step + 1
            current_epoch = (epoch - 1) + batches_in_pass / max(1, len(train_loader))
            active_fape = decoder is not None and want_fape and global_step >= int(cfg.train.fape.start_step)
            # Denominators precede forwards; only input batches are buffered.
            n_tokens = sum(int((batch[1] != ignore_index).sum()) for batch in window)
            n_structures = 0
            processed_tokens = 0
            for batch in window:
                tokens = batch[0]
                processed_tokens += int((tokens != int(cfg.model.encoder.pad_id)).sum())
                if active_fape and len(batch) == 3:
                    mask = residue_mask_from_tokens(tokens, pad_id=int(cfg.model.encoder.pad_id),
                        bos_id=int(cfg.model.encoder.get("bos_id", 0)), eos_id=int(cfg.model.encoder.get("eos_id", 2)))
                    n_structures += int((mask & torch.isfinite(batch[2]).all((-2, -1))).any(1).sum())
            counts = torch.tensor([n_tokens, n_structures, processed_tokens], device=device, dtype=torch.long)
            if accelerator:
                counts = accelerator.reduce(counts, reduction="sum")
            global_tokens, global_structures, processed_tokens = counts.tolist()
            total_tokens += processed_tokens
            micro_step += len(window)
            if global_tokens == 0 and (global_structures == 0 or float(cfg.train.fape.weight) == 0):
                continue
            window_ce = 0.0
            window_fape = 0.0
            window_correct = 0
            for micro_index, batch in enumerate(window):
                sync = accelerator.no_sync(model) if accelerator and micro_index < len(window)-1 else nullcontext()
                with sync:
                    tokens, labels = (t.to(device) for t in batch[:2])
                    coords = batch[2].to(device) if len(batch) == 3 else None
                    error = None
                    try:
                        outputs = model(tokens=tokens)
                        ce_sum = token_ce_loss(outputs["logits"], labels, ignore_index, reduction="sum")
                        fape_sum = ce_sum * 0.0
                        if active_fape and coords is not None:
                            mask = residue_mask_from_tokens(tokens, pad_id=int(cfg.model.encoder.pad_id),
                                bos_id=int(cfg.model.encoder.get("bos_id", 0)), eos_id=int(cfg.model.encoder.get("eos_id", 2)))
                            eligible = (mask & torch.isfinite(coords).all((-2, -1))).any(1)
                            if eligible.any():
                                soft_codes = logits_to_soft_codes_gumbel(outputs["logits"],
                                    _unwrap_model(model, accelerator).classifier.E,
                                    tau=_anneal_tau(global_step), hard=bool(cfg.train.gumbel.get("hard", False)))
                                pred_coords = decode_token_aligned_coords(decoder, soft_codes, mask)
                                fape_sum = fape_loss(pred_coords, coords, mask) * eligible.sum()
                                if log_pred_nan_frac:
                                    running_pred_nan_frac_sum += float(torch.isnan(pred_coords[mask]).float().mean())
                                    running_pred_nan_frac_count += 1
                        loss = ce_sum * (world_size / global_tokens if global_tokens else 0.)
                        loss = loss + float(cfg.train.fape.weight) * fape_sum * (world_size / global_structures if global_structures else 0.)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite training loss")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    _raise_rank_errors(error, accelerator, "Training forward failed")
                    if accelerator:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    window_ce += float(ce_sum.detach())
                    window_fape += float(fape_sum.detach())
                    with torch.no_grad():
                        valid = labels != ignore_index
                        window_correct += int(((outputs["logits"].argmax(-1) == labels) & valid).sum())
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
            sums = torch.tensor([window_ce, window_fape, window_correct], device=device, dtype=torch.float64)
            if accelerator:
                sums = accelerator.reduce(sums, reduction="sum")
            cls_mean = float(sums[0]) / max(1, global_tokens)
            fape_mean = float(sums[1]) / max(1, global_structures)
            running_loss += cls_mean + float(cfg.train.fape.weight) * fape_mean
            if global_tokens:
                running_cls_loss += float(sums[0])
                running_cls_count += global_tokens
                running_masked_acc_sum += float(sums[2])
                running_masked_acc_count += global_tokens
            if global_structures:
                running_fape_loss += float(sums[1])
                running_fape_count += global_structures

            # logging
            if current_step % log_interval == 0 and is_main:
                acc = running_masked_acc_sum / max(1, running_masked_acc_count)
                lr = scheduler.get_last_lr()[0]

                # compute averages over the current log interval
                avg_total_loss = running_loss / max(1, log_interval)
                avg_cls_loss = (
                    running_cls_loss / float(max(1, running_cls_count))
                    if running_cls_count > 0
                    else None
                )
                ppl = math.exp(avg_cls_loss) if avg_cls_loss is not None else None

                # Compute cumulative FLOPs (6N approximation)
                cumulative_flops = compute_flops_6n(num_params, total_tokens)

                # build console log message
                msg = f"step {current_step}/{max_steps} | micro_step {micro_step}"
                if current_epoch is not None:
                    msg += f" | epoch {current_epoch:.3f}"
                # add FLOPs (scientific notation for console)
                msg += f" | flops {format_flops_scientific(cumulative_flops)}"
                # loss
                msg += f" | loss {avg_total_loss:.4f}"

                if is_mlm:
                    # MLM-specific logging
                    avg_masked_acc = (
                        running_masked_acc_sum / float(max(1, running_masked_acc_count))
                        if running_masked_acc_count > 0
                        else acc
                    )
                    msg += f" | acc {avg_masked_acc:.4f}"
                    if ppl is not None:
                        msg += f" | ppl {ppl:.2f}"
                    msg += f" | lr {lr:.2e}"
                else:
                    # Codebook-specific logging
                    msg += f" | acc {acc:.4f} | lr {lr:.2e}"
                    if avg_cls_loss is not None:
                        msg += f" | cls {avg_cls_loss:.4f} | ppl {ppl:.2f}"

                    avg_fape_loss = (
                        running_fape_loss / float(max(1, running_fape_count))
                        if running_fape_count > 0
                        else None
                    )
                    avg_pred_nan_frac = (
                        running_pred_nan_frac_sum
                        / float(max(1, running_pred_nan_frac_count))
                        if running_pred_nan_frac_count > 0
                        else None
                    )
                    if avg_fape_loss is not None:
                        msg += f" | fape {avg_fape_loss:.4f}"
                    if log_pred_nan_frac and (avg_pred_nan_frac is not None):
                        msg += f" | pnan {avg_pred_nan_frac:.3f}"

                console.train(msg)
                if log_file_handle is not None:
                    # Include full FLOPs value in file log
                    file_msg = msg + f" (flops_actual={cumulative_flops})"
                    print(file_msg, file=log_file_handle, flush=True)

                # W&B logging
                if wb is not None:
                    payload: dict[str, float] = {
                        "train/loss": float(avg_total_loss),
                        "lr": float(lr),
                        "train/micro_step": float(micro_step),
                    }

                    if is_mlm:
                        avg_masked_acc = (
                            running_masked_acc_sum
                            / float(max(1, running_masked_acc_count))
                            if running_masked_acc_count > 0
                            else acc
                        )
                        payload["train/mask_acc"] = float(avg_masked_acc)
                        if ppl is not None:
                            payload["train/ppl"] = float(ppl)
                    else:
                        payload["train/acc"] = float(acc)
                        if avg_cls_loss is not None and ppl is not None:
                            payload["train/cls_loss"] = float(avg_cls_loss)
                            payload["train/ppl"] = float(ppl)
                        avg_fape_loss = (
                            running_fape_loss / float(max(1, running_fape_count))
                            if running_fape_count > 0
                            else None
                        )
                        avg_pred_nan_frac = (
                            running_pred_nan_frac_sum
                            / float(max(1, running_pred_nan_frac_count))
                            if running_pred_nan_frac_count > 0
                            else None
                        )
                        if avg_fape_loss is not None:
                            payload["train/fape_loss"] = float(avg_fape_loss)
                        if log_pred_nan_frac and (avg_pred_nan_frac is not None):
                            payload["train/pred_nan_frac"] = float(avg_pred_nan_frac)

                    if current_epoch is not None:
                        payload["train/epoch"] = float(current_epoch)
                    # Add cumulative FLOPs
                    payload["train/flops"] = float(cumulative_flops)
                    payload["train/tokens"] = float(total_tokens)
                    wb.log(payload, step=current_step)

                # reset accumulators for the next log interval
                running_loss = 0.0
                running_cls_loss = 0.0
                running_cls_count = 0
                running_fape_loss = 0.0
                running_fape_count = 0
                running_pred_nan_frac_sum = 0.0
                running_pred_nan_frac_count = 0
                running_masked_acc_sum = 0.0
                running_masked_acc_count = 0

            # eval across all configured eval loaders (using modular eval system)
            if current_step % eval_interval == 0 and len(eval_loaders) > 0:
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

            global_step += 1
            console.step(1)
            # checkpointing
            ckpt_steps = cfg.train.get("checkpoint_steps")
            if (
                ckpt_steps is not None
                and int(ckpt_steps) > 0
                and (global_step % int(ckpt_steps) == 0)
            ):
                step_path = io_dirs["checkpoints"] / f"step_{global_step:08d}.pt"
                checkpoint_error = None
                if is_main:
                    try:
                        _save_checkpoint(
                            step_path, model=model, optimizer=optimizer,
                            scheduler=scheduler, global_step=global_step,
                            cfg=cfg, accelerator=accelerator, micro_step=micro_step,
                        )
                        with _atomic_destination(io_dirs["checkpoints"] / "latest.pt") as temporary:
                            shutil.copyfile(step_path, temporary)
                    except Exception as exc:
                        checkpoint_error = f"{type(exc).__name__}: {exc}"
                errors = gather_object([checkpoint_error]) if accelerator else [checkpoint_error]
                if any(errors):
                    raise RuntimeError(f"Checkpoint failed: {errors}")
            if global_step >= max_steps:
                break
        if batches_in_pass == 0:
            raise RuntimeError("Training loader produced no complete batches")
        if global_step == updates_before_pass:
            raise RuntimeError("Training pass made no successful optimizer update")

    checkpoint_error = None
    if is_main:
        try:
            _save_checkpoint(io_dirs["model"] / "final.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, global_step=global_step, micro_step=micro_step,
                cfg=cfg, accelerator=accelerator)
        except Exception as exc:
            checkpoint_error = f"{type(exc).__name__}: {exc}"
    errors = gather_object([checkpoint_error]) if accelerator else [checkpoint_error]
    if any(errors):
        raise RuntimeError(f"Final checkpoint failed: {errors}")
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
