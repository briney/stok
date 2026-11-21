"""Seeded, token-weighted MDLM validation diagnostics, independent of training."""

from contextlib import ExitStack
from typing import cast

from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from stok.data.mdlm import CANONICAL_AA, MDLMBatch, prepare_mdlm_batch
from stok.eval.metrics.structure import STRUCTURE_SCORES
from stok.utils.decoding import decode_token_aligned_coords
from stok.utils.mdlm import (
    corrupt_mdlm_batch,
    mask_schedule,
    mdlm_loss_terms,
    stable_seed,
)
from stok.utils.pretrained import state_sha256
from stok.utils.sampling import inference_context, sample_mdlm
from stok.utils.tokenizer import Tokenizer

DEFAULTS = {
    "enabled": False,
    "mask_probabilities": [0.15, 0.5, 0.85],
    "max_samples": None,
    "generation": {
        "enabled": False,
        "steps": 10000,
        "sampling_steps": 64,
        "num_samples": 64,
        "decode": False,
        "schedule": {"name": "linear"},
    },
}
# Explicit regime/probability overrides use none of these values, but the corruption
# signature needs a config; a fixed one keeps evaluation independent of training.
EVAL_CORRUPTION = OmegaConf.create(
    {
        "placement": "token",
        "regime_weights": {"joint_independent": 1.0},
        "noise": {"name": "linear", "min_mask_probability": 1e-4},
    }
)
# Generation tasks: the regime hides one whole track and conditions on the other.
GENERATION_TASKS = (("folding", "structure_only"), ("inverse_folding", "sequence_only"))


def resolve_mdlm_eval_config(cfg: DictConfig) -> DictConfig:
    """Resolve evaluation settings without changing authored choices.

    ``generation.steps`` is the successful-update cadence; ``sampling_steps``
    controls reverse transitions.
    """
    from stok.config import reject_unknown_keys

    if not isinstance(cfg.train.get("eval"), DictConfig):
        raise ValueError("train.eval must be a mapping")
    if "seed" in cfg.train.eval:
        raise ValueError("Unsupported configuration field: train.eval.seed")
    provided = OmegaConf.to_container(cfg.train.eval.get("mdlm", {}), resolve=True)
    if not isinstance(provided, dict):
        raise ValueError("MDLM evaluation config must be a mapping")
    generation = provided.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("train.eval.mdlm.generation must be a mapping")
    schedule = generation.get("schedule", {})
    if not isinstance(schedule, dict):
        raise ValueError("train.eval.mdlm.generation.schedule must be a mapping")
    contract = OmegaConf.create(DEFAULTS)
    if schedule.get("name") == "power":
        contract.generation.schedule.power = 2.0
    reject_unknown_keys(OmegaConf.create(provided), contract, "train.eval.mdlm")
    resolved = cast(DictConfig, OmegaConf.merge(DEFAULTS, provided))
    probabilities = resolved.mask_probabilities
    if (
        not isinstance(probabilities, ListConfig)
        or not probabilities
        or len(set(probabilities)) != len(probabilities)
        or any(type(p) not in (int, float) or not 0 < p <= 1 for p in probabilities)
    ):
        raise ValueError(
            "train.eval.mdlm.mask_probabilities must be unique values in (0, 1]"
        )
    if resolved.max_samples is not None and (
        type(resolved.max_samples) is not int or resolved.max_samples < 1
    ):
        raise ValueError("train.eval.mdlm.max_samples must be a positive integer")
    generation = resolved.generation
    if generation.steps is None:
        generation.enabled = False
    for name in ("steps", "sampling_steps", "num_samples"):
        if name == "steps" and generation.steps is None:
            continue
        if type(generation[name]) is not int or generation[name] < 1:
            raise ValueError(f"MDLM generation {name} must be a positive integer")
    if generation.schedule.name == "power":
        generation.schedule.setdefault("power", 2.0)
    mask_schedule(
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        name=generation.schedule.name,
        power=float(generation.schedule.get("power", 2)),
    )
    OmegaConf.set_readonly(resolved, True)
    return resolved


def validate_mdlm_decoder(decoder, codebook, expected_digest: str) -> None:
    """Dimensions alone cannot establish the meaning of exported code IDs."""
    digest = state_sha256({"codebook": codebook})
    if (
        decoder is None
        or getattr(decoder, "codebook_sha256", None) != digest
        or digest != expected_digest
    ):
        raise ValueError(
            "MDLM decoder requires a verified matching codebook digest from its full checkpoint"
        )
    if any(parameter.requires_grad for parameter in decoder.parameters()):
        raise ValueError("MDLM geometric decoder must be frozen")


def _geometry_scores(state, predicted, batch):
    for index, metric in enumerate(STRUCTURE_SCORES.values()):
        score = (
            None
            if batch["coords"] is None
            else metric(
                predicted,
                batch["coords"],
                batch["residue_mask"] & batch["structure_valid"],
            )
        )
        if score is None:
            state[index, 2] += 1
        elif not torch.isfinite(score):
            raise FloatingPointError("Nonfinite MDLM structure score")
        else:
            state[index, 0] += score.double()
            state[index, 1] += 1


def _decode(decoder, codebook, tokens, batch):
    # Input-only sentinel vectors never enter the residue decoder mask.
    codes = codebook[tokens.clamp(0, len(codebook) - 1)]
    return decode_token_aligned_coords(decoder, codes, batch["residue_mask"])


def _to_device(batch: MDLMBatch, device) -> MDLMBatch:
    return cast(
        MDLMBatch,
        {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in batch.items()
        },
    )


def evaluate_mdlm(
    model,
    loaders: dict[str, DataLoader],
    generation_loaders: dict[str, DataLoader],
    cfg: DictConfig,
    *,
    accelerator,
    decoder=None,
    run_denoising: bool = True,
    run_generation: bool = True,
) -> dict[str, dict[str, float]]:
    """Score each validation source with masks seeded only by sequence ID.

    Center crops, token placement and ``("eval", task, probability, sequence_id)``
    seeds keep every run's masks and draws identical, whatever its training choices.
    """
    settings = resolve_mdlm_eval_config(cfg)
    run_denoising = run_denoising and settings.enabled
    run_generation = run_generation and settings.generation.enabled
    # Tests evaluate plain modules; training always passes its accelerator.
    eval_model = accelerator.unwrap_model(model) if accelerator else model
    device = accelerator.device if accelerator else next(model.parameters()).device
    codebook_size = getattr(eval_model, "codebook_size", None)
    if type(codebook_size) is not int or codebook_size < 1:
        raise ValueError("MDLM evaluation codebook_size must be a positive integer")
    codebook = eval_model.structure_codebook
    decode = run_generation and settings.generation.decode
    if decode:
        validate_mdlm_decoder(
            decoder, codebook, state_sha256({"codebook": codebook.cpu()})
        )
    tokenizer = Tokenizer()
    canonical = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(CANONICAL_AA)), device=device
    )
    probabilities = [float(p) for p in settings.mask_probabilities]

    def batch_for(rows):
        return _to_device(
            prepare_mdlm_batch(
                rows,
                tokenizer,
                max_len=int(cfg.data.max_len),
                codebook_size=codebook_size,
                crop="center",
                seeds=[0] * len(rows),
            ),
            device,
        )

    def reduce(state):
        return accelerator.reduce(state, reduction="sum") if accelerator else state

    results = {}
    with ExitStack() as contexts:
        contexts.enter_context(inference_context(model))
        if decoder is not None:
            contexts.enter_context(inference_context(decoder))
        for dataset in loaders:
            metrics: dict[str, float] = {}
            if run_denoising:
                # [probability, (ce, correct, masked, eligible), track]
                denoising = torch.zeros(
                    (len(probabilities), 4, 2), device=device, dtype=torch.float64
                )
                for rows in loaders[dataset]:
                    batch = batch_for(rows)
                    for index, p in enumerate(probabilities):
                        corruption = corrupt_mdlm_batch(
                            batch,
                            EVAL_CORRUPTION,
                            seeds=[
                                stable_seed(["eval", "denoise", p, row["sequence_id"]])
                                for row in rows
                            ],
                            mask_probability=p,
                            regime="joint_independent",
                            placement="token",
                        )
                        terms = mdlm_loss_terms(
                            eval_model(
                                sequence_tokens=corruption["sequence_tokens"],
                                structure_tokens=corruption["structure_tokens"],
                            ),
                            batch,
                            corruption,
                            canonical_aa_ids=canonical,
                        )
                        denoising[index] += torch.stack(
                            [
                                terms[name].double()
                                for name in (
                                    "ce_sum",
                                    "correct",
                                    "masked_count",
                                    "eligible_count",
                                )
                            ]
                        )
                denoising = reduce(denoising)
                for index, p in enumerate(probabilities):
                    for track, modality in enumerate(("sequence", "structure")):
                        ce, correct, masked, _ = denoising[index, :, track].tolist()
                        if masked:
                            prefix = f"denoise_p{p:g}/{modality}"
                            metrics[f"{prefix}/masked_ce"] = ce / masked
                            metrics[f"{prefix}/acc"] = correct / masked
            if run_generation:
                # [task, (correct, targets)]; geometry [folding|control, score, (sum, n)]
                generation = torch.zeros((2, 2), device=device, dtype=torch.float64)
                geometry = torch.zeros((2, 3, 3), device=device, dtype=torch.float64)
                for rows in generation_loaders[dataset]:
                    for row in rows:
                        batch = batch_for([row])
                        for index, (task, regime) in enumerate(GENERATION_TASKS):
                            seed = stable_seed(
                                ["eval", task, "generation", row["sequence_id"]]
                            )
                            corruption = corrupt_mdlm_batch(
                                batch,
                                EVAL_CORRUPTION,
                                seeds=[seed],
                                mask_probability=1.0,
                                regime=regime,
                                placement="token",
                            )
                            generate = corruption["eligible"]
                            if not generate.any():
                                continue
                            sampled = sample_mdlm(
                                eval_model,
                                batch,
                                generate_mask=generate,
                                group_ids=corruption["group_ids"].masked_fill(
                                    ~generate, -1
                                ),
                                schedule=settings.generation.schedule,
                                steps=settings.generation.sampling_steps,
                                seeds=[seed],
                                canonical_aa_ids=canonical,
                            )
                            track = 1 if task == "folding" else 0
                            name = ("sequence_tokens", "structure_tokens")[track]
                            targets = generate[..., track]
                            generation[index, 0] += (
                                sampled[name][targets] == batch[name][targets]
                            ).sum()
                            generation[index, 1] += targets.sum()
                            if decode and task == "folding":
                                _geometry_scores(
                                    geometry[0],
                                    _decode(
                                        decoder,
                                        codebook,
                                        sampled["structure_tokens"],
                                        batch,
                                    ),
                                    batch,
                                )
                        if (
                            decode
                            and (
                                batch["structure_valid"] | ~batch["residue_mask"]
                            ).all()
                        ):
                            _geometry_scores(
                                geometry[1],
                                _decode(
                                    decoder, codebook, batch["structure_tokens"], batch
                                ),
                                batch,
                            )
                generation, geometry = reduce(generation), reduce(geometry)
                for index, name in enumerate(
                    ("folding/structure_acc", "inverse_folding/sequence_recovery")
                ):
                    correct, targets = generation[index].tolist()
                    if targets:
                        metrics[name] = correct / targets
                for index, task in enumerate(("folding", "true_token_control")):
                    for column, score in enumerate(STRUCTURE_SCORES):
                        total, count, _ = geometry[index, column].tolist()
                        if count:
                            metrics[f"{task}/{score}"] = total / count
            results[dataset] = metrics
    return results
