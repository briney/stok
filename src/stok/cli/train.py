import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    Sampler,
)

from stok.data.collate import mdlm_collate, mlm_collate
from stok.data.dataset import (
    DummyMLMDataset,
    DummySequenceDataset,
    InterleavedIterableDataset,
    IterableTokenizedDataset,
    MapAsIterableDataset,
    TokenizedDataset,
)
from stok.eval import Evaluator, MetricLogger
from stok.models.decoder import load_pretrained_decoder
from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.models.stok import STokModel
from stok.utils.codebook import load_codebook
from stok.utils.console import ConsoleLogger
from stok.utils.decoding import logits_to_soft_codes_gumbel
from stok.utils.flops import compute_flops_6n, count_parameters, format_flops_scientific
from stok.utils.losses import fape_loss
from stok.utils.tokenizer import Tokenizer


def _maybe_get_accelerator():
    try:
        from accelerate import Accelerator

        return Accelerator()
    except Exception:
        return None


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


def _restore_rng_state(state: dict[str, Any]):
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np_state = state["numpy"]
            # accept both raw numpy state and "listified" variant
            if isinstance(np_state, (list, tuple)) and len(np_state) >= 5:
                key = np_state[1]
                if isinstance(key, list):
                    try:
                        key = np.array(key, dtype=np.uint32)
                    except Exception:
                        key = np.array(key)
                np_state = (np_state[0], key, np_state[2], np_state[3], np_state[4])
            np.random.set_state(np_state)
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        # best-effort restore; ignore incompatibilities
        pass


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
    cfg: DictConfig,
    accelerator,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _unwrap_model(model, accelerator).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(global_step),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "rng_state": _collect_rng_state(),
    }
    torch.save(payload, path.as_posix())




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

        enc = tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]

        # build labels aligned to tokens: CLS/EOS/PAD -> ignore_index
        L = ids.size(0)
        labels = torch.full((L,), ignore_index, dtype=torch.long)

        # Handle indices if present (may be absent for structure folder datasets)
        indices_raw = item.get("indices")
        if indices_raw is not None:
            indices: torch.Tensor = indices_raw.long()
            # copy only non-negative indices; positions 1..(1+copy_len) receive labels,
            # respecting truncation before EOS
            valid_indices = indices[indices >= 0]
            copy_len = min(int(valid_indices.numel()), max(0, L - 2))
            if copy_len > 0:
                labels[1 : 1 + copy_len] = valid_indices[:copy_len]

        input_ids.append(ids)
        label_ids.append(labels)
        # optional coords tensor [max_len, 3, 3]
        c = item.get("coords")
        if c is not None and isinstance(c, torch.Tensor):
            coords_batch.append(c)

    tokens = torch.stack(input_ids, dim=0)
    labels = torch.stack(label_ids, dim=0)
    if len(coords_batch) > 0:
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

        self.lengths = [int(L) for L in lengths]
        self.fractions = fr.tolist()
        self.seed = int(seed)
        self._epoch = 0
        self.offsets = np.cumsum([0] + self.lengths[:-1]).tolist()
        self.num_samples = (
            int(num_samples) if num_samples is not None else int(sum(self.lengths))
        )

    def __len__(self) -> int:
        return int(self.num_samples)

    def __iter__(self):
        rng = np.random.RandomState((self.seed + (self._epoch * 1009)) & 0xFFFFFFFF)
        self._epoch += 1
        fr = np.asarray(self.fractions, dtype=np.float64)
        for _ in range(int(self.num_samples)):
            ds_idx = int(rng.choice(len(self.lengths), p=fr))
            j = int(rng.randint(0, self.lengths[ds_idx]))
            yield int(self.offsets[ds_idx] + j)


def _build_dataloaders(
    cfg: DictConfig,
    *,
    codebook_size: int | None,
    pad_id: int,
    is_mlm: bool = False,
    is_mdlm: bool = False,
    noise_schedule_seq: NoiseSchedule | None = None,
    noise_schedule_struct: NoiseSchedule | None = None,
    mdlm_tracks: str | None = None,
) -> tuple[DataLoader, dict[str, DataLoader]]:
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

            return StructureFolderDataset(
                folder_path=str(p),
                max_length=max_len,
                chain_id=chain_id,
                recursive=recursive,
            )

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
                    recursive=recursive,
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
        # Derive token IDs from tokenizer rather than hard-coding
        _mask_id = tokenizer.mask_token_id
        _pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else pad_id

        if is_mdlm:
            _mdlm_cfg = cfg.train.get("mdlm", {})
            _antithetic = bool(_mdlm_cfg.get("antithetic_time_sampling", True))
            _independent = bool(_mdlm_cfg.get("independent_track_times", True))
            _mdlm_tracks_local = mdlm_tracks or str(_mdlm_cfg.get("tracks", "joint"))

            # Convention: must match stok.models.mdlm.MDLMModel.__init__.
            _struct_mask_id: int | None = None
            _struct_pad_id: int | None = None
            if _mdlm_tracks_local == "joint":
                if codebook_size is None:
                    raise ValueError(
                        "MDLM joint mode requires a codebook to derive "
                        "struct_mask_id / struct_pad_id; codebook_size is None."
                    )
                _struct_mask_id = codebook_size
                _struct_pad_id = codebook_size + 1

            def collate(batch):
                return mdlm_collate(
                    batch,
                    tokenizer,
                    noise_schedule_seq=noise_schedule_seq,
                    noise_schedule_struct=noise_schedule_struct,
                    max_len=max_len,
                    seq_mask_id=_mask_id,
                    seq_pad_id=_pad_id,
                    struct_mask_id=_struct_mask_id,
                    struct_pad_id=_struct_pad_id,
                    ignore_index=ignore_index,
                    antithetic_time_sampling=_antithetic,
                    independent_track_times=_independent,
                    tracks=_mdlm_tracks_local,
                )

            collate_fn = collate
        elif is_mlm:

            def collate(batch):
                return mlm_collate(
                    batch,
                    tokenizer,
                    max_len=max_len,
                    mask_prob=mask_prob,
                    mask_token_prob=mask_token_prob,
                    random_token_prob=random_token_prob,
                    pad_id=_pad_id,
                    mask_id=_mask_id,
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
                )

            collate_fn = collate

        _seq_only_mode = is_mlm or (is_mdlm and mdlm_tracks == "seq_only")
        if len(train_configs) == 1:
            # Single dataset (backwards compatible)
            train_ds = _pick_dataset(
                str(train_configs[0]["path"]),
                bool(user_load_coords) if not _seq_only_mode else False,
                require_indices=not _seq_only_mode,
            )
        else:
            # Multiple datasets with fractions
            ds_pairs: list[tuple[Dataset | IterableDataset, float]] = []
            for tcfg in train_configs:
                t_load_coords = tcfg.get("load_coords", user_load_coords)
                ds = _pick_dataset(
                    str(tcfg["path"]),
                    bool(t_load_coords) if not _seq_only_mode else False,
                    require_indices=not _seq_only_mode,
                )
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
                        total_samples += int(len(itds))  # type: ignore[arg-type]
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
                    fractions=fracs,
                    seed=int(cfg.train.get("seed", 1337)),
                )
                train_ds = concat
                train_sampler = sampler
    else:
        # fallback dummy data for quick smoke test
        if is_mdlm or is_mlm:
            train_ds = DummyMLMDataset(
                num_samples=512,
                seq_len=min(max_len, 256) - 2,  # Account for CLS/EOS tokens
            )
            tokenizer = Tokenizer()
            _mask_id = tokenizer.mask_token_id
            _pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else pad_id

            if is_mdlm:
                _mdlm_cfg = cfg.train.get("mdlm", {})
                _antithetic = bool(_mdlm_cfg.get("antithetic_time_sampling", True))
                _independent = bool(_mdlm_cfg.get("independent_track_times", True))
                _mdlm_tracks_local = mdlm_tracks or str(_mdlm_cfg.get("tracks", "joint"))

                def collate(batch):
                    return mdlm_collate(
                        batch,
                        tokenizer,
                        noise_schedule_seq=noise_schedule_seq,
                        noise_schedule_struct=noise_schedule_struct,
                        max_len=max_len,
                        seq_mask_id=_mask_id,
                        seq_pad_id=_pad_id,
                        ignore_index=ignore_index,
                        antithetic_time_sampling=_antithetic,
                        independent_track_times=_independent,
                        tracks=_mdlm_tracks_local,
                    )

                collate_fn = collate
            else:

                def collate(batch):
                    return mlm_collate(
                        batch,
                        tokenizer,
                        max_len=max_len,
                        mask_prob=mask_prob,
                        mask_token_prob=mask_token_prob,
                        random_token_prob=random_token_prob,
                        pad_id=_pad_id,
                        mask_id=_mask_id,
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

    # configure shuffle depending on dataset type / sampler usage
    is_iterable = isinstance(train_ds, IterableDataset)
    # only meaningful for multi-process loading
    if tokenizer is None and len(eval_configs) > 0:
        tokenizer = Tokenizer()
        _mask_id = tokenizer.mask_token_id
        _pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else pad_id

        if is_mlm:

            def collate(batch):
                return mlm_collate(
                    batch,
                    tokenizer,
                    max_len=max_len,
                    mask_prob=mask_prob,
                    mask_token_prob=mask_token_prob,
                    random_token_prob=random_token_prob,
                    pad_id=_pad_id,
                    mask_id=_mask_id,
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
                )

            collate_fn = collate

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
    eval_loaders: dict[str, DataLoader] = {}
    for name, eval_cfg in eval_configs.items():
        eval_path = eval_cfg["path"]
        eval_batch_size = int(eval_cfg.get("batch_size", batch_size))
        eval_load_coords = eval_cfg.get("load_coords", user_load_coords)
        # Extract structure folder format options
        eval_format = eval_cfg.get("format")
        eval_chain_id = eval_cfg.get("chain_id")
        eval_recursive = bool(eval_cfg.get("recursive", False))

        # Structure folders always have coords, don't require indices
        is_structure_format = eval_format == "structure"
        ds = _pick_dataset(
            eval_path,
            bool(eval_load_coords) if not is_mlm and not is_structure_format else True,
            require_indices=not is_mlm and not is_structure_format,
            dataset_format=eval_format,
            chain_id=eval_chain_id,
            recursive=eval_recursive,
        )
        eval_loaders[name] = DataLoader(
            ds,
            shuffle=False,
            drop_last=False,
            **_make_dl_kwargs(eval_batch_size),
        )
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


def run_training(cfg: DictConfig):
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
    if objective not in {"codebook", "mlm", "mdlm"}:
        raise ValueError(
            f"Unknown train.objective: {objective}. Expected 'codebook', 'mlm', or 'mdlm'."
        )
    is_mlm = objective == "mlm"
    is_mdlm = objective == "mdlm"

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
        _save_config_snapshot(cfg, io_dirs["configs"] / "run.yaml")
    if accelerator:
        accelerator.wait_for_everyone()

    # Load codebook only for codebook objective (not MLM, not MDLM seq_only)
    codebook = None
    codebook_size = None
    mdlm_cfg = cfg.train.get("mdlm", {})
    mdlm_tracks = str(mdlm_cfg.get("tracks", "joint")) if is_mdlm else None
    if not is_mlm and not is_mdlm:
        codebook = load_codebook(
            preset=cfg.model.codebook.get("preset"),
            path=cfg.model.codebook.get("path"),
        )
        codebook_size = codebook.shape[0]
    elif is_mdlm and mdlm_tracks == "joint":
        codebook = load_codebook(
            preset=cfg.model.codebook.get("preset"),
            path=cfg.model.codebook.get("path"),
        )
        codebook_size = codebook.shape[0]

    # Build noise schedules for MDLM
    noise_schedule_seq = None
    noise_schedule_struct = None
    if is_mdlm:
        ns_seq_cfg = mdlm_cfg.get("noise_schedule_seq", {})
        noise_schedule_seq = NoiseSchedule(
            schedule_type=str(ns_seq_cfg.get("type", "cosine")),
            sigmoid_k=float(ns_seq_cfg.get("sigmoid_k", 6.0)),
            log_linear_k=float(ns_seq_cfg.get("log_linear_k", 3.0)),
            eps=float(ns_seq_cfg.get("eps", 1e-5)),
        )
        if mdlm_tracks == "joint":
            ns_struct_cfg = mdlm_cfg.get("noise_schedule_struct", {})
            noise_schedule_struct = NoiseSchedule(
                schedule_type=str(ns_struct_cfg.get("type", "cosine")),
                sigmoid_k=float(ns_struct_cfg.get("sigmoid_k", 6.0)),
                log_linear_k=float(ns_struct_cfg.get("log_linear_k", 3.0)),
                eps=float(ns_struct_cfg.get("eps", 1e-5)),
            )

    # Build model
    # For MDLM, we need the tokenizer early for mask_token_id
    _mdlm_tokenizer = Tokenizer() if is_mdlm else None
    if is_mdlm:
        model = MDLMModel(
            tracks=mdlm_tracks,
            seq_vocab_size=cfg.model.encoder.vocab_size,
            seq_pad_id=cfg.model.encoder.pad_id,
            seq_mask_id=_mdlm_tokenizer.mask_token_id,
            codebook=codebook if mdlm_tracks == "joint" else None,
            d_model=cfg.model.encoder.d_model,
            n_heads=cfg.model.encoder.n_heads,
            n_layers=cfg.model.encoder.n_layers,
            ffn_mult=cfg.model.encoder.ffn_mult,
            dropout=cfg.model.encoder.dropout,
            attn_dropout=cfg.model.encoder.attn_dropout,
            norm_type=cfg.model.encoder.norm,
            noise_schedule_seq=noise_schedule_seq,
            noise_schedule_struct=noise_schedule_struct,
            lambda_seq=float(mdlm_cfg.get("lambda_seq", 1.0)),
            lambda_struct=float(mdlm_cfg.get("lambda_struct", 1.0)),
            classifier_kwargs=(
                dict(
                    use_cosine=cfg.model.classifier.use_cosine,
                    learnable_temperature=cfg.model.classifier.learnable_temperature,
                    bias_from_code_norm=cfg.model.classifier.bias_from_code_norm,
                    projector_dim=cfg.model.classifier.projector_dim,
                )
                if mdlm_tracks == "joint"
                else None
            ),
            tie_seq_embeddings=bool(mdlm_cfg.get("tie_seq_embeddings", True)),
            time_conditioning=str(mdlm_cfg.get("time_conditioning", "adaln")),
            time_embed_dim=(
                int(mdlm_cfg.time_embed_dim) if mdlm_cfg.get("time_embed_dim") else None
            ),
            time_combine=str(mdlm_cfg.get("time_combine", "sum")),
        )
    else:
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
    # Skip for MDLM (handled by weight_loading.py in Phase 3)
    pretrained_encoder_path = cfg.train.get("pretrained_encoder")
    if pretrained_encoder_path is not None and not is_mdlm and is_main:
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

    # load frozen geometric decoder for FAPE loss and/or eval metrics (optional)
    # Skip decoder setup for MLM and MDLM objectives
    decoder = None
    want_fape = False
    want_eval_decode = False
    log_pred_nan_frac = False

    if not is_mlm and not is_mdlm:
        want_fape = bool(getattr(cfg.train, "fape", {}).get("enabled", False))
        # default to False; eval-time decoding is opt-in via config/override
        want_eval_decode = bool(
            getattr(cfg.train, "decoding", {}).get("eval_enabled", False)
        )
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

        if decoder_enabled and (want_fape or want_eval_decode):
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

    # data
    train_loader, eval_loaders = _build_dataloaders(
        cfg,
        codebook_size=codebook_size,
        pad_id=cfg.model.encoder.pad_id,
        is_mlm=is_mlm,
        is_mdlm=is_mdlm,
        noise_schedule_seq=noise_schedule_seq,
        noise_schedule_struct=noise_schedule_struct,
        mdlm_tracks=mdlm_tracks,
    )

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
        steps_per_epoch = math.ceil(len(train_loader))  # type: ignore[arg-type]
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
        # max_steps is in optimizer-step units
        max_steps = int(cfg.train.epochs) * steps_per_epoch // grad_accum_steps
    else:
        # num_steps means optimizer steps
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
        to_prepare = [model, optimizer, train_loader]
        to_prepare.extend(eval_loaders.values())
        prepared = accelerator.prepare(*to_prepare)
        # Unpack prepared components in order
        model = prepared[0]
        optimizer = prepared[1]
        train_loader = prepared[2]
        eval_names = list(eval_loaders.keys())
        eval_loaders = {name: prepared[3 + i] for i, name in enumerate(eval_names)}
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # W&B
    wb = _maybe_init_wandb(cfg, is_main_process=is_main, logs_dir=io_dirs["logs"])

    # train loop
    model.train()
    micro_step = 0
    optimizer_step = 0
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
        initial_step=optimizer_step,
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
    # MLM/MDLM-specific accumulators
    running_masked_acc_sum = 0.0
    running_masked_acc_count = 0
    running_struct_loss = 0.0
    running_struct_count = 0
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

    def _do_optimizer_step():
        """Clip gradients, step optimizer/scheduler, zero grads."""
        if grad_clip is not None and grad_clip > 0:
            if accelerator:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    while optimizer_step < max_steps:
        for batch in train_loader:
            # epoch bookkeeping from micro_step
            current_epoch: Optional[float] = None
            if steps_per_epoch is not None:
                current_epoch = float(micro_step + 1) / float(steps_per_epoch)

            # Unpack batch depending on objective
            if is_mdlm:
                # MDLM collate returns a dict
                if accelerator is None:
                    _dev = _get_model_device(model, accelerator)
                    batch = {
                        k: v.to(_dev) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                outputs = model(
                    seq_tokens=batch["seq_tokens"],
                    t_seq=batch["t_seq"],
                    seq_targets=batch["seq_targets"],
                    seq_mask=batch["seq_mask"],
                    key_padding_mask=batch["key_padding_mask"],
                    struct_tokens=batch.get("struct_tokens"),
                    t_struct=batch.get("t_struct"),
                    struct_targets=batch.get("struct_targets"),
                    struct_mask=batch.get("struct_mask"),
                )
                loss: torch.Tensor = outputs["loss"]
                tokens = batch["seq_tokens"]
                labels = batch["seq_targets"]
                coords = None
            else:
                # batch can be (tokens, labels) or (tokens, labels, coords)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    tokens, labels, coords = batch
                else:
                    tokens, labels = batch  # type: ignore[misc]
                    coords = None
                if accelerator is None:
                    _dev = _get_model_device(model, accelerator)
                    tokens = tokens.to(_dev)
                    labels = labels.to(_dev)
                    if coords is not None:
                        coords = coords.to(_dev)

                # base model forward (token classification only)
                outputs = model(tokens=tokens, labels=labels, ignore_index=ignore_index)
                loss = outputs["loss"]

            # optional FAPE loss using frozen decoder (codebook objective only)
            if (
                not is_mlm
                and not is_mdlm
                and decoder is not None
                and want_fape
                and (optimizer_step >= int(cfg.train.fape.start_step))
            ):
                if coords is not None:
                    pad_id = int(cfg.model.encoder.pad_id)
                    mask = tokens != pad_id
                    tau = _anneal_tau(optimizer_step)
                    soft_codes = logits_to_soft_codes_gumbel(
                        outputs["logits"],  # [B, L, C]
                        _unwrap_model(model, accelerator).classifier.E,  # [C, d_code]
                        tau=float(tau),
                        hard=bool(getattr(cfg.train, "gumbel", {}).get("hard", False)),
                    )
                    bb = decoder(soft_codes, mask=mask)  # type: ignore[operator]
                    pred_coords = bb.view(bb.size(0), bb.size(1), 3, 3)
                    # metric: fraction of NaNs in predicted coords
                    # can happen when encoder isn't producing coherent outputs (yet!)
                    if log_pred_nan_frac:
                        pred_nan_frac_t = torch.isnan(pred_coords).float().mean()
                        outputs["pred_nan_frac"] = float(
                            pred_nan_frac_t.detach().item()
                        )
                    # don't bother with FAPE loss if all predicted coords are NaN
                    if torch.isnan(pred_coords).all():
                        outputs["pred_coords"] = pred_coords
                        outputs["structure_loss"] = None
                    else:
                        fape = fape_loss(
                            pred_coords=pred_coords,
                            true_coords=coords,
                            residue_mask=mask,
                        )
                        outputs["pred_coords"] = pred_coords
                        # Always expose FAPE value (even if NaN/Inf) for logging
                        outputs["structure_loss"] = fape
                        # Only add finite FAPE to the optimization loss
                        if torch.isfinite(fape).item():
                            loss = loss + float(cfg.train.fape.weight) * fape
                elif is_main and (micro_step == 0):
                    printer(
                        "FAPE enabled but no coords in dataset; skipping FAPE term."
                    )

            # normalize by grad accumulation
            loss_to_backprop = loss / grad_accum_steps
            if accelerator:
                accelerator.backward(loss_to_backprop)
            else:
                loss_to_backprop.backward()

            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                _do_optimizer_step()
                optimizer_step += 1

            running_loss += float(loss.detach().item())

            # Accumulate tokens for FLOPs tracking (non-padding tokens)
            pad_id_for_count = int(cfg.model.encoder.pad_id)
            batch_tokens = int((tokens != pad_id_for_count).sum().item())
            total_tokens += batch_tokens

            # accumulate loss components
            if is_mdlm:
                _loss_seq = outputs.get("loss_seq")
                if _loss_seq is not None:
                    running_cls_loss += float(_loss_seq.detach().item())
                    running_cls_count += 1
                _loss_struct = outputs.get("loss_struct")
                if _loss_struct is not None:
                    running_struct_loss += float(_loss_struct.detach().item())
                    running_struct_count += 1
                # MDLM masked accuracy (at masked positions)
                with torch.no_grad():
                    masked_acc = _compute_accuracy(
                        outputs["seq_logits"], labels, ignore_index
                    )
                    running_masked_acc_sum += masked_acc
                    running_masked_acc_count += 1
            else:
                cls_loss_tensor = outputs.get("classification_loss")
                if cls_loss_tensor is not None:
                    running_cls_loss += float(cls_loss_tensor.detach().item())
                    running_cls_count += 1

            # For MLM, compute masked token accuracy
            if is_mlm:
                with torch.no_grad():
                    masked_acc = _compute_accuracy(
                        outputs["logits"], labels, ignore_index
                    )
                    running_masked_acc_sum += masked_acc
                    running_masked_acc_count += 1

            # Codebook-specific accumulations
            if not is_mlm and not is_mdlm:
                fape_loss_tensor = outputs.get("structure_loss")
                if fape_loss_tensor is not None:
                    running_fape_loss += float(fape_loss_tensor.detach().item())
                    running_fape_count += 1
                if log_pred_nan_frac:
                    _pnan = outputs.get("pred_nan_frac")
                    if _pnan is not None:
                        running_pred_nan_frac_sum += float(_pnan)
                        running_pred_nan_frac_count += 1

            # logging (gated on optimizer_step)
            if (
                optimizer_step > 0
                and optimizer_step % log_interval == 0
                and micro_step % grad_accum_steps == 0
                and is_main
            ):
                with torch.no_grad():
                    _logits_key = "seq_logits" if is_mdlm else "logits"
                    acc = _compute_accuracy(outputs[_logits_key], labels, ignore_index)
                lr = scheduler.get_last_lr()[0]

                # compute averages over the current log interval
                avg_total_loss = running_loss / max(1, log_interval * grad_accum_steps)
                avg_cls_loss = (
                    running_cls_loss / float(max(1, running_cls_count))
                    if running_cls_count > 0
                    else None
                )
                ppl = math.exp(avg_cls_loss) if avg_cls_loss is not None else None

                # Compute cumulative FLOPs (6N approximation)
                cumulative_flops = compute_flops_6n(num_params, total_tokens)

                # build console log message
                msg = f"step {optimizer_step}/{max_steps}"
                if current_epoch is not None:
                    msg += f" | epoch {current_epoch:.3f}"
                # add FLOPs (scientific notation for console)
                msg += f" | flops {format_flops_scientific(cumulative_flops)}"
                # loss
                msg += f" | loss {avg_total_loss:.4f}"

                if is_mdlm:
                    # MDLM-specific logging
                    avg_masked_acc = (
                        running_masked_acc_sum / float(max(1, running_masked_acc_count))
                        if running_masked_acc_count > 0
                        else acc
                    )
                    msg += f" | mask_acc {avg_masked_acc:.4f}"
                    if avg_cls_loss is not None:
                        msg += f" | loss_seq {avg_cls_loss:.4f}"
                    avg_struct_loss = (
                        running_struct_loss / float(max(1, running_struct_count))
                        if running_struct_count > 0
                        else None
                    )
                    if avg_struct_loss is not None:
                        msg += f" | loss_struct {avg_struct_loss:.4f}"
                    msg += f" | lr {lr:.2e}"
                    # Log mean t and mask rate
                    _t_mean = float(batch["t_seq"].mean().item())
                    _mask_rate = float(batch["seq_mask"].float().mean().item())
                    msg += f" | t_mean {_t_mean:.3f} | mask_rate {_mask_rate:.3f}"
                    # Log struct track stats if present
                    if batch.get("t_struct") is not None:
                        _t_struct_mean = float(batch["t_struct"].mean().item())
                        msg += f" | t_struct {_t_struct_mean:.3f}"
                    if batch.get("struct_mask") is not None:
                        _struct_mask_rate = float(
                            batch["struct_mask"].float().mean().item()
                        )
                        msg += f" | struct_mask {_struct_mask_rate:.3f}"
                elif is_mlm:
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
                        "train/optimizer_step": float(optimizer_step),
                        "train/micro_step": float(micro_step),
                    }

                    if is_mdlm:
                        avg_masked_acc = (
                            running_masked_acc_sum
                            / float(max(1, running_masked_acc_count))
                            if running_masked_acc_count > 0
                            else acc
                        )
                        payload["train/mask_acc"] = float(avg_masked_acc)
                        if avg_cls_loss is not None:
                            payload["train/loss_seq"] = float(avg_cls_loss)
                        _avg_struct_loss = (
                            running_struct_loss / float(max(1, running_struct_count))
                            if running_struct_count > 0
                            else None
                        )
                        if _avg_struct_loss is not None:
                            payload["train/loss_struct"] = float(_avg_struct_loss)
                        payload["train/t_seq_mean"] = float(batch["t_seq"].mean().item())
                        payload["train/mask_rate_seq"] = float(
                            batch["seq_mask"].float().mean().item()
                        )
                        if batch.get("t_struct") is not None:
                            payload["train/t_struct_mean"] = float(
                                batch["t_struct"].mean().item()
                            )
                        if batch.get("struct_mask") is not None:
                            payload["train/mask_rate_struct"] = float(
                                batch["struct_mask"].float().mean().item()
                            )
                    elif is_mlm:
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
                    wb.log(payload, step=optimizer_step)

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
                running_struct_loss = 0.0
                running_struct_count = 0

            # eval across all configured eval loaders (gated on optimizer_step)
            if (
                optimizer_step > 0
                and optimizer_step % eval_interval == 0
                and micro_step % grad_accum_steps == 0
                and len(eval_loaders) > 0
            ):
                all_eval_metrics = evaluator.evaluate_all(eval_loaders)

                # Add epoch to metrics if available
                if current_epoch is not None:
                    for metrics in all_eval_metrics.values():
                        metrics["epoch"] = float(current_epoch)

                # Compute cumulative training FLOPs for eval logging
                eval_train_flops = compute_flops_6n(num_params, total_tokens)

                # Log all eval metrics (including training FLOPs at this checkpoint)
                metric_logger.log_eval_all(
                    all_eval_metrics, optimizer_step, current_epoch, eval_train_flops
                )

            # Console progress and checkpointing only on optimizer steps
            if micro_step % grad_accum_steps == 0:
                console.step(1)
                # checkpointing
                ckpt_steps = cfg.train.get("checkpoint_steps")
                if (
                    is_main
                    and ckpt_steps is not None
                    and int(ckpt_steps) > 0
                    and (optimizer_step % int(ckpt_steps) == 0)
                ):
                    step_path = (
                        io_dirs["checkpoints"] / f"step_{optimizer_step:08d}.pt"
                    )
                    _save_checkpoint(
                        step_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=optimizer_step,
                        cfg=cfg,
                        accelerator=accelerator,
                    )
                    # update latest pointer
                    try:
                        shutil.copyfile(
                            step_path.as_posix(),
                            (io_dirs["checkpoints"] / "latest.pt").as_posix(),
                        )
                    except Exception:
                        pass
                    if accelerator:
                        accelerator.wait_for_everyone()
            if optimizer_step >= max_steps:
                break

        # Flush pending gradients at end of epoch if accumulation window is incomplete
        if micro_step % grad_accum_steps != 0 and optimizer_step < max_steps:
            _do_optimizer_step()
            optimizer_step += 1
            console.step(1)

    if is_main:
        # final checkpoint
        final_path = io_dirs["model"] / "final.pt"
        _save_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=optimizer_step,
            cfg=cfg,
            accelerator=accelerator,
        )
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
