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
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

from stok.data.dataset import (
    DummySequenceDataset,
    IterableTokenizedDataset,
    TokenizedDataset,
)
from stok.models.decoder import load_pretrained_decoder
from stok.models.stok import STokModel
from stok.utils.codebook import load_codebook
from stok.utils.console import ConsoleLogger
from stok.utils.decoding import (
    decode_coords,
    indices_to_codes,
    logits_to_soft_codes_gumbel,
    sample_indices_top_p,
)
from stok.utils.losses import fape_loss
from stok.utils.metrics import lddt_ca, rmsd, tm_score
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


def _try_load_latest_checkpoint(
    ckpt_dir: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator,
) -> int:
    """
    Returns restored global_step if a checkpoint is loaded; otherwise 0.
    """
    latest = ckpt_dir / "latest.pt"
    if not latest.exists():
        return 0
    # all processes load to keep state in sync under DDP
    try:
        ckpt = torch.load(latest.as_posix(), map_location="cpu")
        _unwrap_model(model, accelerator).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "rng_state" in ckpt:
            _restore_rng_state(ckpt["rng_state"])
        return int(ckpt.get("global_step", 0))
    except Exception:
        # if anything goes wrong, start from scratch
        return 0


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

    # else TokenizedDataset dicts with 'seq' and 'indices'
    input_ids = []
    label_ids = []
    coords_batch: list[torch.Tensor] = []
    for item in batch:  # type: ignore[assignment]
        seq: str = item["seq"]
        indices: torch.Tensor = item["indices"].long()

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


def _build_dataloaders(
    cfg: DictConfig, *, codebook_size: int, pad_id: int
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

    tokenizer: Optional[Tokenizer] = None
    collate_fn = None

    # dataset picker usable for train/eval
    def _pick_dataset(path: str, load_coords: bool):
        p = Path(path)
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
                )
        return TokenizedDataset(
            dataset_path=str(path),
            max_length=max_len,
            load_coords=bool(load_coords),
        )

    if cfg.data.get("train"):
        # CSV/Parquet-backed dataset; tokenize in collate
        train_ds = _pick_dataset(str(cfg.data.train), bool(user_load_coords))
        tokenizer = Tokenizer()

        def collate(batch):
            return _tokenize_and_align(
                batch,
                tokenizer,
                max_len=max_len,
                ignore_index=ignore_index,
                pad_id=pad_id,
            )

        collate_fn = collate
    else:
        # fallback dummy data for quick smoke test
        train_ds = DummySequenceDataset(
            num_samples=512,
            seq_len=min(max_len, 256),
            vocab_size=cfg.model.encoder.vocab_size,
            num_classes=codebook_size,
            pad_id=pad_id,
        )

    # configure shuffle depending on dataset type
    is_iterable = isinstance(train_ds, IterableDataset)
    # only meaningful for multi-process loading
    if tokenizer is None and len(eval_configs) > 0:
        tokenizer = Tokenizer()

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

    train_loader = DataLoader(
        train_ds,
        shuffle=not is_iterable,
        drop_last=True,
        **_make_dl_kwargs(batch_size),
    )
    eval_loaders: dict[str, DataLoader] = {}
    for name, eval_cfg in eval_configs.items():
        eval_path = eval_cfg["path"]
        eval_batch_size = int(eval_cfg.get("batch_size", batch_size))
        eval_load_coords = eval_cfg.get("load_coords", user_load_coords)
        ds = _pick_dataset(eval_path, bool(eval_load_coords))
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
    except Exception:
        pass

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

    # load codebook and build model
    codebook = load_codebook(
        preset=cfg.model.codebook.get("preset"),
        path=cfg.model.codebook.get("path"),
    )
    codebook_size = codebook.shape[0]
    model = STokModel(
        vocab_size=cfg.model.encoder.vocab_size,
        pad_id=cfg.model.encoder.pad_id,
        d_model=cfg.model.encoder.d_model,
        n_heads=cfg.model.encoder.n_heads,
        n_layers=cfg.model.encoder.n_layers,
        ffn_mult=cfg.model.encoder.ffn_mult,
        dropout=cfg.model.encoder.dropout,
        attn_dropout=cfg.model.encoder.attn_dropout,
        codebook=codebook,
        classifier_kwargs=dict(
            use_cosine=cfg.model.classifier.use_cosine,
            learnable_temperature=cfg.model.classifier.learnable_temperature,
            bias_from_code_norm=cfg.model.classifier.bias_from_code_norm,
            projector_dim=cfg.model.classifier.projector_dim,
        ),
        norm_type=cfg.model.encoder.norm,
    )

    # load frozen geometric decoder for FAPE loss and/or eval metrics (optional)
    decoder = None
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
        cfg, codebook_size=codebook_size, pad_id=cfg.model.encoder.pad_id
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
    global_step = 0
    running_loss = 0.0
    log_interval = int(cfg.train.get("log_steps", 50))
    eval_interval = int(cfg.train.get("eval_steps", 1000))
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
        print("Training started.", file=log_file_handle, flush=True)

    # additional training accumulators (over the current log window)
    running_cls_loss = 0.0
    running_cls_count = 0
    running_fape_loss = 0.0
    running_fape_count = 0
    running_pred_nan_frac_sum = 0.0
    running_pred_nan_frac_count = 0

    # Gumbel temperature schedule
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

    while global_step < max_steps:
        for batch in train_loader:
            # step/epoch bookkeeping (global_step is zero-based)
            current_step = global_step + 1
            current_epoch: Optional[float] = None
            if steps_per_epoch is not None:
                current_epoch = float(current_step) / float(steps_per_epoch)

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
            loss: torch.Tensor = outputs["loss"]

            # optional FAPE loss using frozen decoder
            if (
                decoder is not None
                and want_fape
                and (global_step >= int(cfg.train.fape.start_step))
            ):
                if coords is not None:
                    pad_id = int(cfg.model.encoder.pad_id)
                    mask = tokens != pad_id
                    tau = _anneal_tau(global_step)
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
                elif is_main and (global_step == 0):
                    printer(
                        "FAPE enabled but no coords in dataset; skipping FAPE term."
                    )

            # normalize by grad accumulation
            loss_to_backprop = loss / grad_accum_steps
            if accelerator:
                accelerator.backward(loss_to_backprop)
            else:
                loss_to_backprop.backward()

            if (global_step + 1) % grad_accum_steps == 0:
                if grad_clip is not None and grad_clip > 0:
                    if accelerator:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().item())

            # accumulate loss components
            cls_loss_tensor = outputs.get("classification_loss")
            if cls_loss_tensor is not None:
                running_cls_loss += float(cls_loss_tensor.detach().item())
                running_cls_count += 1
            fape_loss_tensor = outputs.get("structure_loss")
            if fape_loss_tensor is not None:
                running_fape_loss += float(fape_loss_tensor.detach().item())
                running_fape_count += 1
            if log_pred_nan_frac:
                _pnan = outputs.get("pred_nan_frac")
                if _pnan is not None:
                    running_pred_nan_frac_sum += float(_pnan)
                    running_pred_nan_frac_count += 1

            # logging
            if current_step % log_interval == 0 and is_main:
                with torch.no_grad():
                    acc = _compute_accuracy(outputs["logits"], labels, ignore_index)
                lr = scheduler.get_last_lr()[0]

                # compute averages over the current log interval
                avg_total_loss = running_loss / max(1, log_interval)
                avg_cls_loss = (
                    running_cls_loss / float(max(1, running_cls_count))
                    if running_cls_count > 0
                    else None
                )
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
                ppl = math.exp(avg_cls_loss) if avg_cls_loss is not None else None

                # build console log message
                msg = f"step {current_step}/{max_steps}"
                if current_epoch is not None:
                    msg += f" | epoch {current_epoch:.3f}"
                msg += (
                    f" | loss {avg_total_loss:.4f}"
                    f" | acc {acc:.4f}"
                    f" | lr {lr:.2e}"
                )
                if avg_cls_loss is not None:
                    msg += f" | cls {avg_cls_loss:.4f} | ppl {ppl:.2f}"
                if avg_fape_loss is not None:
                    msg += f" | fape {avg_fape_loss:.4f}"
                if log_pred_nan_frac and (avg_pred_nan_frac is not None):
                    msg += f" | pnan {avg_pred_nan_frac:.3f}"
                console.train(msg)
                if log_file_handle is not None:
                    print(msg, file=log_file_handle, flush=True)

                # W&B logging
                if wb is not None:
                    payload: dict[str, float] = {
                        "train/loss": float(avg_total_loss),
                        "train/acc": float(acc),
                        "lr": float(lr),
                    }
                    if avg_cls_loss is not None and ppl is not None:
                        payload["train/cls_loss"] = float(avg_cls_loss)
                        payload["train/ppl"] = float(ppl)
                    if avg_fape_loss is not None:
                        payload["train/fape_loss"] = float(avg_fape_loss)
                    if log_pred_nan_frac and (avg_pred_nan_frac is not None):
                        payload["train/pred_nan_frac"] = float(avg_pred_nan_frac)
                    if current_epoch is not None:
                        payload["train/epoch"] = float(current_epoch)
                    wb.log(payload, step=current_step)

                # reset accumulators for the next log interval
                running_loss = 0.0
                running_cls_loss = 0.0
                running_cls_count = 0
                running_fape_loss = 0.0
                running_fape_count = 0
                running_pred_nan_frac_sum = 0.0
                running_pred_nan_frac_count = 0

            # eval across all configured eval loaders
            if current_step % eval_interval == 0 and len(eval_loaders) > 0:
                model.eval()
                all_eval_metrics: dict[str, dict[str, float]] = {}

                for eval_name, eval_loader in eval_loaders.items():
                    eval_loss_sum = 0.0
                    eval_acc_sum = 0.0
                    eval_batches = 0.0
                    eval_cls_loss_sum = 0.0
                    eval_cls_batches = 0.0
                    eval_fape_loss_sum = 0.0
                    eval_fape_batches = 0.0
                    eval_lddt_sum = 0.0
                    eval_tm_sum = 0.0
                    eval_rmsd_sum = 0.0
                    eval_struct_count = 0.0
                    eval_pred_nan_frac_sum = 0.0
                    eval_pred_nan_frac_count = 0.0

                    with torch.no_grad():
                        for ev in eval_loader:
                            # eval batch may also include coords (to compute structure-based metrics)
                            if isinstance(ev, (list, tuple)) and len(ev) == 3:
                                etok, elab, ecoords = ev
                            else:
                                etok, elab = ev
                                ecoords = None
                            if accelerator is None:
                                _dev = _get_model_device(model, accelerator)
                                etok = etok.to(_dev)
                                elab = elab.to(_dev)
                                if ecoords is not None:
                                    ecoords = ecoords.to(_dev)
                            out = model(
                                tokens=etok, labels=elab, ignore_index=ignore_index
                            )
                            eval_loss_sum += float(out["loss"].item())
                            eval_acc_sum += _compute_accuracy(
                                out["logits"], elab, ignore_index
                            )
                            eval_batches += 1.0

                            # accumulate loss components
                            cls_loss_tensor = out.get("classification_loss")
                            if cls_loss_tensor is not None:
                                eval_cls_loss_sum += float(cls_loss_tensor.item())
                                eval_cls_batches += 1.0
                            fape_loss_tensor = out.get("structure_loss")
                            if fape_loss_tensor is not None:
                                eval_fape_loss_sum += float(fape_loss_tensor.item())
                                eval_fape_batches += 1.0

                            # optional structure metrics if predicted coords available
                            pred_coords = out.get("pred_coords", None)
                            # if decoder is available and eval decoding is enabled, produce pred_coords
                            if (
                                pred_coords is None
                                and decoder is not None
                                and want_eval_decode
                            ):
                                pad_id = int(cfg.model.encoder.pad_id)
                                res_mask = etok != pad_id
                                method = str(
                                    getattr(cfg.train.decoding, "eval_method", "argmax")
                                )
                                if method == "top_p":
                                    temperature = float(
                                        getattr(cfg.train.decoding, "temperature", 1.0)
                                    )
                                    top_p = float(
                                        getattr(cfg.train.decoding, "top_p", 0.9)
                                    )
                                    probs = torch.softmax(
                                        out["logits"] / max(1e-8, temperature), dim=-1
                                    )
                                    idx = sample_indices_top_p(
                                        probs, top_p=top_p, temperature=1.0
                                    )
                                    codes = indices_to_codes(
                                        _unwrap_model(model, accelerator).classifier.E,
                                        idx,
                                    )
                                else:
                                    idx = out["logits"].argmax(dim=-1)
                                    codes = indices_to_codes(
                                        _unwrap_model(model, accelerator).classifier.E,
                                        idx,
                                    )
                                pred_coords = decode_coords(decoder, codes, res_mask)
                                out["pred_coords"] = pred_coords

                            if pred_coords is not None and ecoords is not None:
                                if log_pred_nan_frac:
                                    _pnan_eval = torch.isnan(pred_coords).float().mean()
                                    eval_pred_nan_frac_sum += float(_pnan_eval.item())
                                    eval_pred_nan_frac_count += 1.0
                                res_mask = etok != cfg.model.encoder.pad_id
                                # compute eval metrics; guard against numeric failures (NaN, etc)
                                try:
                                    lddt_b, _ = lddt_ca(
                                        pred_coords, ecoords, residue_mask=res_mask
                                    )
                                    tm_b, _ = tm_score(
                                        pred_coords, ecoords, residue_mask=res_mask
                                    )
                                    rmsd_b = rmsd(
                                        pred_coords,
                                        ecoords,
                                        residue_mask=res_mask,
                                        align=True,
                                        atom_set="CA",
                                    )
                                    eval_lddt_sum += float(lddt_b.mean().item())
                                    eval_tm_sum += float(tm_b.mean().item())
                                    eval_rmsd_sum += float(rmsd_b.mean().item())
                                    eval_struct_count += 1.0
                                except Exception:
                                    pass
                                # also compute FAPE on eval if predicted coords are present (as a metric)
                                try:
                                    fape_eval = fape_loss(
                                        pred_coords, ecoords, residue_mask=res_mask
                                    )
                                    eval_fape_loss_sum += float(fape_eval.item())
                                    eval_fape_batches += 1.0
                                except Exception:
                                    pass

                    # aggregate across processes if using Accelerate
                    if accelerator:
                        metrics_device = accelerator.device
                        metrics_local = torch.tensor(
                            [
                                eval_loss_sum,
                                eval_acc_sum,
                                eval_batches,
                                eval_cls_loss_sum,
                                eval_cls_batches,
                                eval_fape_loss_sum,
                                eval_fape_batches,
                            ],
                            dtype=torch.float32,
                            device=metrics_device,
                        )
                        gathered = accelerator.gather_for_metrics(metrics_local)
                        # gathered can be shape [7] on single process or [N, 7] on multi
                        if gathered.dim() == 1:
                            eval_loss_sum = float(gathered[0].item())
                            eval_acc_sum = float(gathered[1].item())
                            eval_batches = float(gathered[2].item())
                            eval_cls_loss_sum = float(gathered[3].item())
                            eval_cls_batches = float(gathered[4].item())
                            eval_fape_loss_sum = float(gathered[5].item())
                            eval_fape_batches = float(gathered[6].item())
                        else:
                            eval_loss_sum = float(gathered[:, 0].sum().item())
                            eval_acc_sum = float(gathered[:, 1].sum().item())
                            eval_batches = float(gathered[:, 2].sum().item())
                            eval_cls_loss_sum = float(gathered[:, 3].sum().item())
                            eval_cls_batches = float(gathered[:, 4].sum().item())
                            eval_fape_loss_sum = float(gathered[:, 5].sum().item())
                            eval_fape_batches = float(gathered[:, 6].sum().item())

                    eval_loss = eval_loss_sum / max(1.0, eval_batches)
                    eval_acc = eval_acc_sum / max(1.0, eval_batches)
                    eval_cls_loss = (
                        eval_cls_loss_sum / max(1.0, eval_cls_batches)
                        if eval_cls_batches > 0
                        else None
                    )
                    eval_fape_loss = (
                        eval_fape_loss_sum / max(1.0, eval_fape_batches)
                        if eval_fape_batches > 0
                        else None
                    )
                    eval_ppl = (
                        math.exp(eval_cls_loss) if eval_cls_loss is not None else None
                    )
                    eval_pred_nan_frac = (
                        eval_pred_nan_frac_sum / max(1.0, eval_pred_nan_frac_count)
                        if eval_pred_nan_frac_count > 0
                        else None
                    )

                    metrics: dict[str, float] = {
                        "loss": float(eval_loss),
                        "acc": float(eval_acc),
                    }
                    if eval_cls_loss is not None and eval_ppl is not None:
                        metrics["cls_loss"] = float(eval_cls_loss)
                        metrics["ppl"] = float(eval_ppl)
                    if eval_fape_loss is not None:
                        metrics["fape_loss"] = float(eval_fape_loss)
                    if log_pred_nan_frac and (eval_pred_nan_frac is not None):
                        metrics["pred_nan_frac"] = float(eval_pred_nan_frac)
                    if eval_struct_count > 0:
                        metrics["lddt"] = float(eval_lddt_sum / eval_struct_count)
                        metrics["tm"] = float(eval_tm_sum / eval_struct_count)
                        metrics["rmsd"] = float(eval_rmsd_sum / eval_struct_count)
                    if current_epoch is not None:
                        metrics["epoch"] = float(current_epoch)

                    all_eval_metrics[eval_name] = metrics

                model.train()

                if is_main:
                    for eval_name, metrics in all_eval_metrics.items():
                        msg = f"eval/{eval_name} | step {current_step}"
                        epoch_val = metrics.get("epoch")
                        if epoch_val is not None:
                            msg += f" | epoch {epoch_val:.3f}"
                        msg += (
                            f" | loss {metrics['loss']:.4f}"
                            f" | acc {metrics['acc']:.4f}"
                        )
                        if "cls_loss" in metrics and "ppl" in metrics:
                            msg += (
                                f" | cls {metrics['cls_loss']:.4f}"
                                f" | ppl {metrics['ppl']:.2f}"
                            )
                        if "fape_loss" in metrics:
                            msg += f" | fape {metrics['fape_loss']:.4f}"
                        if log_pred_nan_frac and ("pred_nan_frac" in metrics):
                            msg += f" | pnan {metrics['pred_nan_frac']:.3f}"
                        if "lddt" in metrics:
                            msg += (
                                f" | lDDT {metrics['lddt']:.3f}"
                                f" | TM {metrics['tm']:.3f}"
                                f" | RMSD {metrics['rmsd']:.3f}Å"
                            )
                        console.eval(msg)
                        if log_file_handle is not None:
                            print(msg, file=log_file_handle, flush=True)
                        if wb is not None:
                            payload: dict[str, float] = {
                                f"eval/{eval_name}/loss": float(metrics["loss"]),
                                f"eval/{eval_name}/acc": float(metrics["acc"]),
                            }
                            if "cls_loss" in metrics and "ppl" in metrics:
                                payload[f"eval/{eval_name}/cls_loss"] = float(
                                    metrics["cls_loss"]
                                )
                                payload[f"eval/{eval_name}/ppl"] = float(metrics["ppl"])
                            if "fape_loss" in metrics:
                                payload[f"eval/{eval_name}/fape_loss"] = float(
                                    metrics["fape_loss"]
                                )
                            if log_pred_nan_frac and ("pred_nan_frac" in metrics):
                                payload[f"eval/{eval_name}/pred_nan_frac"] = float(
                                    metrics["pred_nan_frac"]
                                )
                            if "lddt" in metrics:
                                payload[f"eval/{eval_name}/lddt"] = float(
                                    metrics["lddt"]
                                )
                                payload[f"eval/{eval_name}/tm"] = float(metrics["tm"])
                                payload[f"eval/{eval_name}/rmsd"] = float(
                                    metrics["rmsd"]
                                )
                            if epoch_val is not None:
                                payload[f"eval/{eval_name}/epoch"] = float(epoch_val)
                            wb.log(payload, step=current_step)

            global_step += 1
            console.step(1)
            # checkpointing
            ckpt_steps = cfg.train.get("checkpoint_steps")
            if (
                is_main
                and ckpt_steps is not None
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
            if global_step >= max_steps:
                break

    if is_main:
        # final checkpoint
        final_path = io_dirs["model"] / "final.pt"
        _save_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
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
