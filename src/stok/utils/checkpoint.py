"""Versioned training state and deterministic epoch replay for replicated training."""

from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import cast
import os
import platform
import random
import tempfile
import hashlib

import numpy as np
import torch
from accelerate.utils import gather_object
from omegaconf import OmegaConf

from stok.config import validate_training_config
from stok.data.dataset import set_dataset_epoch
from stok.data.mdlm import (
    mdlm_training_signature,
    mdlm_vocabulary,
    validate_mdlm_identity,
)
from stok.training.tasks import validate_logging_state
from stok.utils.pretrained import file_sha256, state_sha256


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


def rank_errors(error, accelerator, context):
    errors = gather_object([error]) if accelerator else [error]
    if any(errors):
        raise RuntimeError(f"{context}: {errors}")


def package_source_sha256(root: Path | None = None) -> str:
    """Hash installed Python/config names and bytes identically in checkout and wheel."""
    root = root or Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.suffix not in {".py", ".yaml"} or any(
            part in {"__pycache__", "build", "dist"} for part in relative.parts
        ):
            continue
        name, content = relative.as_posix().encode(), path.read_bytes()
        for value in (name, content):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


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
    config.pop("print_model_summary", None)
    return config


def execution_identity(accelerator) -> dict:
    """One owner for actual native numerical/distributed execution metadata."""
    execution = {
        "world_size": accelerator.num_processes if accelerator else 1,
        "precision": accelerator.mixed_precision if accelerator else "no",
        "device": accelerator.device.type if accelerator else "cpu",
        "distributed": accelerator.distributed_type.name if accelerator else "NO",
        "threads": torch.get_num_threads(),
        "cuda_rng_state_sizes": [
            state.numel() for state in torch.cuda.get_rng_state_all()
        ]
        if accelerator and accelerator.device.type == "cuda"
        else [],
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version()
        if accelerator and accelerator.device.type == "cuda"
        else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "sdpa": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "math_low_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        },
    }

    return {
        "source": {"sha256": package_source_sha256()},
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "torch",
                    "accelerate",
                    "numpy",
                    "pyarrow",
                    "hydra-core",
                    "omegaconf",
                    "tokenizers",
                    "transformers",
                    "x-transformers",
                    "vector-quantize-pytorch",
                )
            },
        },
        "execution": execution,
    }


def resume_signature(cfg, *, sources, codebook, accelerator, identity):
    config = scientific_config(cfg)
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
        "mdlm_identity": identity["training_signature"],
        "codebook": state_sha256({"codebook": codebook})
        if codebook is not None
        else None,
        **execution_identity(accelerator),
    }


def read_training_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or (
        type(payload.get("format_version")) is not int or payload["format_version"] != 4
    ):
        raise ValueError(
            "STok requires a complete version 4 training checkpoint; unsupported versions and weights-only checkpoints cannot load"
        )
    required = {
        "model",
        "optimizer",
        "optimizer_initialized",
        "scheduler",
        "signature",
        "rank_states",
        "config",
        "runtime",
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
    validate_resume_signature(payload, payload["signature"])
    return payload


def validate_resume_signature(payload: dict, expected: dict) -> None:
    saved = dict(payload["signature"])
    if saved != expected:
        changed = [key for key in expected if saved.get(key) != expected[key]]
        raise ValueError(
            f"Resume signature mismatch: {changed}; original data/model/execution/budget must be unchanged"
        )
    runtime = payload["runtime"]
    required_runtime = {
        "components",
        "effective_precision",
        "mdlm_identity",
        "software",
        "execution",
        "source",
        "evaluation_protocol",
    }
    if not isinstance(runtime, dict) or required_runtime - runtime.keys():
        raise ValueError("Incomplete runtime manifest")
    components = runtime["components"]
    cfg = OmegaConf.create(payload["config"])
    if scientific_config(cfg) != saved["config"]:
        raise ValueError(
            "Checkpoint scientific configuration disagrees with resume identity"
        )
    validate_training_config(cfg)
    required_components = {
        "objective": cfg.train.objective,
        "model": "stok_mdlm",
        "sequence_tokenizer": "native",
        "structure_representation": "frozen_vq",
        "optimizer": cfg.train.optimizer,
        "scheduler": cfg.train.scheduler,
    }
    if (
        not isinstance(components, dict)
        or any(
            components.get(key) != value for key, value in required_components.items()
        )
        or any(not isinstance(value, str) or not value for value in components.values())
    ):
        raise ValueError("Incomplete or inconsistent component/representation manifest")
    identity = runtime["mdlm_identity"]
    codebook = payload["model"].get("structure_codebook")
    if not isinstance(codebook, torch.Tensor) or codebook.ndim != 2:
        raise ValueError("Checkpoint structure codebook is missing or invalid")
    digest = state_sha256({"codebook": codebook})
    if digest != saved["codebook"] or digest != identity.get("codebook_sha256"):
        raise ValueError("Checkpoint structure codebook disagrees with resume identity")
    if identity.get("vocabulary") != mdlm_vocabulary(codebook):
        raise ValueError("Checkpoint vocabulary identity is incompatible")
    validate_mdlm_identity(identity)
    from stok.eval.mdlm import mdlm_evaluation_protocol

    protocol = mdlm_evaluation_protocol(
        cfg,
        identity=identity,
        environment={key: runtime[key] for key in ("software", "execution", "source")},
        decoder=runtime.get("decoder"),
    )
    if runtime["evaluation_protocol"] != protocol:
        raise ValueError(
            "Checkpoint evaluation protocol disagrees with saved configuration/runtime"
        )
    try:
        training_signature = mdlm_training_signature(identity)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Incomplete MDLM training identity") from exc
    if (
        any(
            runtime.get(key) != saved[key]
            for key in ("software", "execution", "source")
        )
        or training_signature != identity.get("training_signature")
        or training_signature != saved["mdlm_identity"]
        or runtime["effective_precision"] != saved["execution"]["precision"]
        or (
            cfg.train.mixed_precision is not None
            and cfg.train.mixed_precision != runtime["effective_precision"]
        )
    ):
        raise ValueError("Checkpoint runtime manifest disagrees with resume identity")
    ranks = payload["rank_states"]
    size = expected["execution"]["world_size"]
    if (
        not isinstance(ranks, list)
        or len(ranks) != size
        or [r.get("rank") for r in ranks] != list(range(size))
    ):
        raise ValueError("Missing or invalid per-rank resume state")
    execution = expected["execution"]
    sdpa = execution.get("sdpa")
    if (
        not isinstance(sdpa, dict)
        or set(sdpa)
        != {
            "flash",
            "math",
            "memory_efficient",
            "cudnn",
            "math_low_precision_reduction",
        }
        or any(type(value) is not bool for value in sdpa.values())
    ):
        raise ValueError("Incomplete SDPA execution policy in resume identity")
    sizes = execution.get("cuda_rng_state_sizes")
    if (
        not isinstance(sizes, list)
        or any(type(size) is not int or size <= 0 for size in sizes)
        or bool(sizes) != (execution["device"] == "cuda")
    ):
        raise ValueError("Missing or invalid CUDA RNG inventory in execution signature")
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
        rng = rank["rng"]
        try:
            # Validate native CPU states on local generators, without changing
            # training streams or touching the saved/current CUDA device.
            random.Random(0).setstate(rng["python"])
            np.random.RandomState(0).set_state(_numpy_rng_state(rng))
            torch.Generator().set_state(rng["torch"])
            torch.Generator().set_state(rank["loader_generator_state"])
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Incomplete or invalid CPU/loader RNG state") from exc
        if sizes:
            states, active = rng.get("cuda"), rng.get("cuda_device")
            if (
                not isinstance(states, list)
                or len(states) != len(sizes)
                or type(active) is not int
                or not 0 <= active < len(sizes)
                or any(
                    not isinstance(state, torch.Tensor)
                    or state.device.type != "cpu"
                    or state.dtype != torch.uint8
                    or state.shape != (size,)
                    or not state.is_contiguous()
                    for state, size in zip(states, sizes)
                )
            ):
                raise ValueError("Incomplete or invalid CUDA RNG state collection")
        elif "cuda" in rng or "cuda_device" in rng:
            raise ValueError("Unexpected CUDA RNG state for CPU execution")
        validate_logging_state(rank["logging"])
    if any(
        type(payload[key]) is not int or payload[key] < 0
        for key in ("global_step", "micro_step", "residues_seen", "executed_positions")
    ):
        raise ValueError("Invalid resume counters")
    if payload["scheduler"].get("last_epoch") != payload["global_step"]:
        raise ValueError("Resume scheduler and successful-update counters disagree")
    # AdamW initializes state lazily: frozen/unused parameters may legitimately lack it.
    # Separate saved coverage detects a deleted/emptied initialized entry without
    # assuming every optimizer parameter has participated in an update.
    initialized = payload["optimizer_initialized"]
    states = payload["optimizer"]["state"]
    actual = {key for key, state in states.items() if state}
    if (
        not isinstance(initialized, list)
        or len(initialized) != len(set(initialized))
        or set(initialized) != actual
    ):
        raise ValueError("Missing or unexpected initialized AdamW parameter state")
    if payload["global_step"] and not states:
        raise ValueError(
            "Missing optimizer continuation state after successful updates"
        )
    for state in states.values():
        if state and (
            {"step", "exp_avg", "exp_avg_sq"} - state.keys()
            or any(
                not isinstance(state[key], torch.Tensor)
                for key in ("exp_avg", "exp_avg_sq")
            )
            or state["exp_avg"].shape != state["exp_avg_sq"].shape
        ):
            raise ValueError("Incomplete or incompatible AdamW continuation state")


def restore_training_state(
    payload: dict, *, model, optimizer, scheduler, accelerator
) -> dict:
    rank = payload["rank_states"][accelerator.process_index if accelerator else 0]
    device = accelerator.device if accelerator else torch.device("cpu")
    if device.type == "cuda":
        states = collect_rng_state(device=device)
        if rank["rng"].get("cuda_device") != states["cuda_device"] or payload[
            "signature"
        ]["execution"].get("cuda_rng_state_sizes") != [
            state.numel() for state in states["cuda"]
        ]:
            raise ValueError(
                "CUDA RNG inventory or active device changed for this rank"
            )
    elif "cuda" in rank["rng"]:
        raise ValueError("CUDA RNG training state requires CUDA execution")
    plain = accelerator.unwrap_model(model) if accelerator else model
    plain.load_state_dict(payload["model"], strict=True)
    plain.mdlm_regime_weights = payload["config"]["train"]["mdlm"]["regime_weights"]
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if state and any(
                state[key].shape != parameter.shape for key in ("exp_avg", "exp_avg_sq")
            ):
                raise ValueError("Incomplete or incompatible AdamW continuation state")
    scaler = getattr(accelerator, "scaler", None)
    if (scaler is None) != (rank["scaler"] is None):
        raise ValueError(
            "Resume AMP scaler does not match this execution configuration"
        )
    if scaler is not None:
        scaler.load_state_dict(rank["scaler"])
    # Apply actual device RNG before output replacement; restore again after replay.
    rng = collect_rng_state(device=device)
    try:
        restore_rng_state(rank["rng"])
    finally:
        restore_rng_state(rng)
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
