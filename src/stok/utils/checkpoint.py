"""Versioned training state and deterministic epoch replay for replicated training."""

from contextlib import contextmanager
from pathlib import Path
from typing import cast
import os
import random
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

from stok.data.dataset import set_dataset_epoch
from stok.utils.pretrained import state_sha256


def collect_rng_state(*, device=None):
    """Scope training snapshots to its device; generic snapshots preserve initialized CUDA."""
    numpy = cast(
        tuple[str, np.ndarray, int, int, float], np.random.get_state(legacy=True)
    )
    state = {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "torch": torch.get_rng_state(),
    }
    if (device is None and torch.cuda.is_initialized()) or (
        device is not None and device.type == "cuda"
    ):
        state["cuda"] = torch.cuda.get_rng_state_all()
        state["cuda_device"] = torch.cuda.current_device()
    return state


def _numpy_rng_state(state):
    numpy = list(state["numpy"])
    numpy[1] = np.asarray(numpy[1], dtype=np.uint32)
    return tuple(numpy)


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(_numpy_rng_state(state))
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


def scientific_config(cfg):
    """Project the training trajectory, permitting evaluation/output overrides."""
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Resume configuration must resolve to a mapping")
    train, data = config["train"], config["data"]
    for key in (
        "output_dir",
        "resume_from",
        "wandb",
        "console",
        "log_every",
        "eval",
        "save_every",
    ):
        train.pop(key, None)
    data.pop("eval", None)
    config["model"].pop("decoder", None)
    return config


def execution_identity(accelerator) -> dict:
    """Execution facts that change the training trajectory or its per-rank state."""
    return {
        "world_size": accelerator.num_processes if accelerator else 1,
        "precision": accelerator.mixed_precision if accelerator else "no",
        "device": accelerator.device.type if accelerator else "cpu",
    }


def resume_signature(cfg, *, codebook, accelerator, identity):
    return {
        "config": scientific_config(cfg),
        "mdlm_identity": identity["training_signature"],
        "codebook": state_sha256({"codebook": codebook})
        if codebook is not None
        else None,
        "execution": execution_identity(accelerator),
    }


def read_training_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 4:
        raise ValueError("STok requires a version 4 training checkpoint")
    return payload


def validate_resume_signature(payload: dict, expected: dict) -> None:
    saved = payload["signature"]
    if saved != expected:
        changed = [key for key in expected if saved.get(key) != expected[key]]
        raise ValueError(
            f"Resume signature mismatch: {changed}; original data/model/execution must be unchanged"
        )
    if len(payload["rank_states"]) != expected["execution"]["world_size"]:
        raise ValueError("Checkpoint per-rank state does not match world size")


def restore_training_state(
    payload: dict, *, model, optimizer, scheduler, accelerator
) -> dict:
    rank = payload["rank_states"][accelerator.process_index if accelerator else 0]
    plain = accelerator.unwrap_model(model) if accelerator else model
    plain.load_state_dict(payload["model"], strict=True)
    plain.mdlm_regime_weights = payload["config"]["train"]["mdlm"]["regime_weights"]
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler = getattr(accelerator, "scaler", None)
    if scaler is not None:
        scaler.load_state_dict(rank["scaler"])
    return {
        "logging": rank["logging"],
        "epoch": rank["epoch"],
        "batches_in_epoch": rank["batches_in_epoch"],
        "global_step": payload["global_step"],
        "micro_step": payload["micro_step"],
        "residues_seen": payload["residues_seen"],
        "executed_positions": payload["executed_positions"],
        "rng": rank["rng"],
        "loader_generator_state": rank["loader_generator_state"],
    }


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
