"""Versioned training state and deterministic epoch replay for replicated training."""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import os
import math
import platform
import random
import tempfile

import numpy as np
import torch
from accelerate.utils import gather_object
from omegaconf import OmegaConf

from stok.data.dataset import set_dataset_epoch
from stok.utils.pretrained import file_sha256, state_sha256


def collect_rng_state():
    numpy = list(np.random.get_state())
    numpy[1] = numpy[1].tolist()
    state = {
        "python": random.getstate(),
        "numpy": numpy,
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy = list(state["numpy"])
    numpy[1] = np.asarray(numpy[1], dtype=np.uint32)
    np.random.set_state(tuple(numpy))
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def atomic_destination(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def rank_errors(error, accelerator, context):
    errors = gather_object([error]) if accelerator else [error]
    if any(errors):
        raise RuntimeError(f"{context}: {errors}")


def resume_signature(cfg, *, sources, codebook, accelerator):
    config = OmegaConf.to_container(cfg, resolve=True)
    train, data = config["train"], config["data"]
    for key in (
        "project_path",
        "resume_from",
        "wandb",
        "console",
        "log_steps",
        "eval",
        "checkpoint_steps",
        "mdlm_identity",
        "effective_precision",
        "decoding",
    ):
        train.pop(key, None)
    data.pop("eval", None)
    # Decoder settings affect training only when geometry supervision is active.
    if not train.get("fape", {}).get("enabled"):
        config["model"].pop("decoder", None)
    config.pop("print_model_summary", None)
    identities = []
    for source in sources:
        path = Path(source["path"])
        files = (
            sorted(p for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else [path]
        )
        identities.append(
            [
                (str(p.relative_to(path)) if path.is_dir() else p.name, file_sha256(p))
                for p in files
            ]
        )
    return {
        "config": config,
        "sources": identities,
        "source_order": sources,
        "mdlm_identity": OmegaConf.select(
            cfg, "train.mdlm_identity.training_signature"
        ),
        "codebook": state_sha256({"codebook": codebook})
        if codebook is not None
        else None,
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("torch", "accelerate", "numpy", "pyarrow")
            },
        },
        "execution": {
            "world_size": accelerator.num_processes if accelerator else 1,
            "precision": accelerator.mixed_precision if accelerator else "no",
            "device": accelerator.device.type if accelerator else "cpu",
            "distributed": accelerator.distributed_type.name if accelerator else "NO",
            "threads": torch.get_num_threads(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }


def read_training_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or (
        type(payload.get("format_version")) is not int or payload["format_version"] != 2
    ):
        raise ValueError(
            "Full resume requires a version 2 training checkpoint; weights-only/legacy checkpoints cannot resume"
        )
    required = {
        "model",
        "optimizer",
        "scheduler",
        "signature",
        "rank_states",
        "config",
        "global_step",
        "micro_step",
        "residues_seen",
        "executed_positions",
        "wandb_run_id",
    }
    if required - payload.keys():
        raise ValueError(
            f"Incomplete resume checkpoint: missing {sorted(required - payload.keys())}"
        )
    return payload


def validate_resume_signature(payload: dict, expected: dict) -> None:
    if payload["signature"] != expected:
        changed = [
            key for key in expected if payload["signature"].get(key) != expected[key]
        ]
        raise ValueError(
            f"Resume signature mismatch: {changed}; original data/model/execution/budget must be unchanged"
        )
    ranks = payload["rank_states"]
    size = expected["execution"]["world_size"]
    if (
        not isinstance(ranks, list)
        or len(ranks) != size
        or [r.get("rank") for r in ranks] != list(range(size))
    ):
        raise ValueError("Missing or invalid per-rank resume state")
    for rank in ranks:
        required = {
            "rng",
            "scaler",
            "epoch",
            "batches_in_epoch",
            "loader_generator_state",
            "logging",
        }
        if required - rank.keys():
            raise ValueError("Incomplete per-rank resume state")
        if not all(
            type(rank[key]) is int and rank[key] >= 0
            for key in ("epoch", "batches_in_epoch")
        ):
            raise ValueError("Invalid resume cursor")
        logging = rank["logging"]
        required_logging = {
            "running_loss",
            "running_updates",
            "running_cls_loss",
            "running_cls_count",
            "running_fape_loss",
            "running_fape_count",
            "running_pred_nan_frac_sum",
            "running_pred_nan_frac_count",
            "running_masked_acc_sum",
            "running_masked_acc_count",
            "total_missing_structure",
            "total_noncanonical_sequence",
            "mdlm_running",
        }
        if not isinstance(logging, dict) or required_logging - logging.keys():
            raise ValueError("Incomplete per-rank logging state")
        if not isinstance(logging["mdlm_running"], torch.Tensor) or logging[
            "mdlm_running"
        ].shape != (5, 2):
            raise ValueError("Invalid per-rank MDLM logging state")
        if any(
            not isinstance(logging[key], (int, float))
            or not math.isfinite(logging[key])
            for key in required_logging - {"mdlm_running"}
        ):
            raise ValueError("Invalid numeric logging state")
        if not isinstance(rank["loader_generator_state"], torch.Tensor):
            raise ValueError("Missing epoch-start loader generator state")
    if any(
        type(payload[key]) is not int or payload[key] < 0
        for key in ("global_step", "micro_step", "residues_seen", "executed_positions")
    ):
        raise ValueError("Invalid resume counters")
    if payload["scheduler"].get("last_epoch") != payload["global_step"]:
        raise ValueError("Resume scheduler and successful-update counters disagree")


@dataclass
class TrainingProgress:
    global_step: int
    micro_step: int
    residues_seen: int
    executed_positions: int
    rank_state: dict


def restore_training_state(
    payload: dict, *, model, optimizer, scheduler, accelerator
) -> TrainingProgress:
    plain = accelerator.unwrap_model(model) if accelerator else model
    plain.load_state_dict(payload["model"], strict=True)
    if payload["config"]["train"].get("objective") == "mdlm":
        plain.mdlm_regime_weights = payload["config"]["train"]["mdlm"]["regime_weights"]
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if payload["global_step"] and not optimizer.state:
        raise ValueError(
            "Missing optimizer continuation state after successful updates"
        )
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if state and (
                {"step", "exp_avg", "exp_avg_sq"} - state.keys()
                or any(
                    state[key].shape != parameter.shape
                    for key in ("exp_avg", "exp_avg_sq")
                )
            ):
                raise ValueError("Incomplete or incompatible AdamW continuation state")
    rank = payload["rank_states"][accelerator.process_index if accelerator else 0]
    scaler = getattr(accelerator, "scaler", None)
    if (scaler is None) != (rank["scaler"] is None):
        raise ValueError(
            "Resume AMP scaler does not match this execution configuration"
        )
    if scaler is not None:
        scaler.load_state_dict(rank["scaler"])
    # Validate RNG before any output replacement; restore it again after replay.
    rng = collect_rng_state()
    try:
        restore_rng_state(rank["rng"])
        torch.Generator().set_state(rank["loader_generator_state"])
    finally:
        restore_rng_state(rng)
    return TrainingProgress(
        payload["global_step"],
        payload["micro_step"],
        payload["residues_seen"],
        payload["executed_positions"],
        rank,
    )


def epoch_iterator(loader, *, epoch, consumed, seed, rank, loader_state=None):
    """Construct one deterministic epoch, then discard the consumed local prefix."""
    set_dataset_epoch(loader.dataset, epoch)
    if callable(getattr(loader.sampler, "set_epoch", None)):
        loader.sampler.set_epoch(epoch)
    epoch_seed = (seed + 1000003 * epoch + 1009 * rank) % (2**63 - 1)
    loader.generator = torch.Generator().manual_seed(epoch_seed)
    if loader_state is not None:
        loader.generator.set_state(loader_state)
    start_state = loader.generator.get_state()
    stream = iter(loader)
    # ponytail: O(consumed batches) replay; add indexed seeking only if resume latency warrants it.
    for _ in range(consumed):
        try:
            next(stream)
        except StopIteration as exc:
            raise ValueError(
                "Resume cursor exceeds the reconstructed local epoch"
            ) from exc
    return stream, start_state


class ResumeWandb:
    """Suppress duplicate remote history without changing the training cursor."""

    def __init__(self, wandb, watermark):
        self.wandb = wandb
        self.watermark = watermark
        self.error = None

    @property
    def run(self):
        return self.wandb.run

    def log(self, payload, *, step):
        if step > self.watermark:
            rng = collect_rng_state()
            try:
                self.wandb.log(payload, step=step)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                restore_rng_state(rng)
