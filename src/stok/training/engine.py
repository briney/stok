"""Shared training orchestration, optimization, evaluation, and checkpoints."""

from contextlib import nullcontext
from itertools import islice
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, cast

import torch
import torch.nn as nn
from accelerate.utils import gather_object, set_seed
from accelerate.utils.environment import get_cpu_distributed_information
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from stok.data.loaders import (
    _build_dataloaders,
    _parse_eval_configs,
    _parse_train_configs,
)
from stok.config import validate_training_config
from stok.eval import MetricLogger
from stok.models.decoder import load_pretrained_decoder
from stok.models.build import build_model
from stok.data.mdlm import validate_mdlm_sources
from stok.utils.codebook import load_codebook
from stok.utils.console import ConsoleLogger
from stok.utils.flops import compute_flops_6n, count_parameters
from stok.training.tasks import MDLMTask
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
from stok.utils.pretrained import state_sha256, json_sha256


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
    micro_step: int,
    residues_seen: int,
    executed_positions: int,
    training_state: dict,
    runtime: dict,
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
            "config": OmegaConf.to_container(cfg, resolve=True),
            "runtime": runtime,
            "residues_seen": int(residues_seen),
            "executed_positions": int(executed_positions),
        }
        payload["optimizer_initialized"] = [
            key for key, state in payload["optimizer"]["state"].items() if state
        ]
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError(
                "Checkpoint requires a completed accumulation boundary with no pending gradients"
            )
        scaler = getattr(accelerator, "scaler", None)
        rank_state = {
            **training_state["local"],
            "rank": accelerator.process_index if accelerator else 0,
            "rng": _collect_rng_state(
                device=accelerator.device if accelerator else torch.device("cpu")
            ),
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
    except Exception as exc:
        if accelerator is None:
            raise
        error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(error, accelerator, "Collecting checkpoint state failed")
    assert payload is not None
    payload.update(
        format_version=4,
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


def run_training(cfg: DictConfig) -> None:
    validate_training_config(cfg)
    authored = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    from stok.eval.mdlm import (
        evaluate_mdlm,
        mdlm_evaluation_protocol,
        resolve_mdlm_eval_config,
        validate_mdlm_decoder,
    )

    from stok.utils.mdlm import REGIMES

    for regime in REGIMES:
        cfg.train.mdlm.regime_weights.setdefault(regime, 0.0)
    if cfg.train.mdlm.placement == "span":
        cfg.train.mdlm.setdefault("span_mean", 8.0)
    if cfg.train.mdlm.noise.name == "power":
        cfg.train.mdlm.noise.setdefault("power", 2.0)
    cfg.train.eval.mdlm = resolve_mdlm_eval_config(cfg)
    from stok.eval.cases import read_evaluation_cases, publish_evaluation_summary

    case_error = None
    try:
        shared = (
            read_evaluation_cases(cfg.train.eval.mdlm.case_manifest)
            if cfg.train.eval.mdlm.case_manifest
            else None
        )
        resolve_mdlm_eval_config(cfg, identity={"shared_cases": shared})
    except Exception as exc:
        # Single-process case/cap errors stay before device setup. Launched peers
        # must initialize communication to exchange even rank-local read errors.
        distributed = get_cpu_distributed_information().world_size > 1 or (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if not distributed:
            raise
        case_error = f"{type(exc).__name__}: {exc}"
    OmegaConf.set_readonly(cfg, True)
    objective = "mdlm"
    project = cfg.train.output_dir
    if not isinstance(project, str) or not project.strip():
        raise ValueError("MDLM requires a nonempty unique train.output_dir")
    destination = Path(project).resolve()
    if (
        not cfg.train.resume_from
        and destination.exists()
        and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise ValueError(
            "Fresh MDLM output_dir is populated; choose a unique directory or resume"
        )
    train_sources = {item["name"]: item for item in _parse_train_configs(cfg)}
    if not train_sources:
        raise ValueError("MDLM requires a nonempty training source selection")
    schedule = cfg.train.scheduler
    decay = schedule.rsplit("_", 1)[1]
    warmup_steps = cfg.train.warmup_steps
    stable_steps = cfg.train.stable_steps
    decay_steps = cfg.train.decay_steps
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
    _raise_rank_errors(case_error, accelerator, "MDLM preflight failed")
    is_main = accelerator.is_main_process if accelerator else True
    printer = accelerator.print if accelerator else print

    decoder = None
    preflight_error = None
    try:
        device = accelerator.device if accelerator else torch.device("cpu")
        effective_precision = accelerator.mixed_precision if accelerator else "no"
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
        codebook = load_codebook(
            preset=cfg.model.codebook.get("preset"),
            path=cfg.model.codebook.get("path"),
        )
        codebook_size = codebook.shape[0]
        mdlm_identity = validate_mdlm_sources(
            train_sources,
            _parse_eval_configs(cfg),
            codebook=codebook,
            split_manifest=cfg.data.get("split_manifest"),
            case_manifest=cfg.train.eval.mdlm.case_manifest,
        )
        runtime: dict[str, Any] = {
            "effective_precision": effective_precision,
            "mdlm_identity": mdlm_identity,
        }
        mdlm_eval = resolve_mdlm_eval_config(cfg, identity=mdlm_identity)
    except Exception as exc:
        preflight_error = f"{type(exc).__name__}: {exc}"
    _raise_rank_errors(preflight_error, accelerator, "MDLM preflight failed")

    try:
        train_loader, eval_loaders = _build_dataloaders(
            cfg,
            identity=mdlm_identity,
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

    # Count trainable parameters for FLOPs tracking (6N approximation)
    num_params = count_parameters(model, trainable_only=True)
    if is_main:
        printer(f"Trainable parameters: {num_params:,}")
        device = accelerator.device if accelerator else torch.device("cpu")
        hardware = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        )
        mdlm_startup = [
            f"Trainable parameters: {num_params:,}",
            f"MDLM hardware: device={device}, hardware={hardware}, requested_precision={cfg.train.get('mixed_precision')}, mixed_precision={effective_precision}",
            f"MDLM data identity: {mdlm_identity}",
        ]
        for line in mdlm_startup[1:]:
            printer(line)

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
            identity=mdlm_identity,
        )
        runtime.update(
            **{key: signature[key] for key in ("software", "execution", "source")},
            components={
                "objective": "mdlm",
                "model": "stok_mdlm",
                "sequence_tokenizer": "native",
                "structure_representation": "frozen_vq",
                "optimizer": "adamw",
                "scheduler": cfg.train.scheduler,
            },
        )
        if decoder is not None:
            runtime["components"]["decoder"] = "geometric"
            runtime["decoder"] = {
                "sha256": state_sha256(decoder.state_dict()),
                "codebook_sha256": mdlm_identity["codebook_sha256"],
            }
        runtime["evaluation_protocol"] = mdlm_evaluation_protocol(
            cfg,
            identity=mdlm_identity,
            environment={
                key: runtime[key] for key in ("software", "execution", "source")
            },
            decoder=runtime.get("decoder"),
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
            OmegaConf.save(
                authored, io_dirs["configs"] / "authored.yaml", resolve=False
            )
            OmegaConf.save(
                OmegaConf.create(runtime), io_dirs["configs"] / "runtime.yaml"
            )
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
    metric_logger = MetricLogger(
        console=console,
        wandb=wb,
        log_file=log_file_handle,
        is_main=is_main,
    )

    # FLOPs use globally forwarded padded positions.
    total_tokens = 0
    total_residues = 0
    task = MDLMTask(cfg, codebook_size=codebook_size)

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
        task.restore_logging_state(progress["logging"])

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
                "Preparing MDLM window failed",
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
                        # Normalize the weighted diffusion sum over the full global window.
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
            if current_step % log_interval == 0:
                if is_main:
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

            denoising_due = mdlm_eval.enabled and current_step % eval_interval == 0
            generation_due = (
                mdlm_eval.generation.enabled
                and current_step % mdlm_eval.generation.steps == 0
            )
            if denoising_due or generation_due:
                all_eval_metrics, evaluation_summary = evaluate_mdlm(
                    model,
                    eval_loaders,
                    cfg,
                    accelerator=accelerator,
                    identity=mdlm_identity,
                    protocol=runtime["evaluation_protocol"],
                    model_identity={
                        "training_signature": json_sha256(signature),
                        "global_step": current_step,
                    },
                    decoder=decoder,
                    run_denoising=denoising_due,
                    run_generation=generation_due,
                )
                summary_error = None
                if is_main:
                    try:
                        publish_evaluation_summary(
                            io_dirs["logs"]
                            / "evaluations"
                            / f"step-{current_step}-{evaluation_summary['measurement_sha256']}.json",
                            evaluation_summary,
                        )
                    except Exception as exc:
                        summary_error = f"{type(exc).__name__}: {exc}"
                _raise_rank_errors(
                    summary_error, accelerator, "Publishing evaluation summary failed"
                )
                metric_logger.log_eval_all(
                    all_eval_metrics,
                    current_step,
                    current_epoch,
                    compute_flops_6n(num_params, total_tokens),
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
                    runtime=runtime,
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
            and eligible_windows_in_pass == 0
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
        runtime=runtime,
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
