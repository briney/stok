"""Frozen, token-weighted MDLM diagnostics, independent of training corruption."""

from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
import math

from accelerate.utils import gather_object
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader

from stok.data.mdlm import CANONICAL_AA, _sample_key, prepare_mdlm_batch
from stok.eval.evaluator import _get_model_device, _unwrap_model
from stok.eval.metrics.structure import LDDTMetric, RMSDMetric, TMScoreMetric
from stok.utils.decoding import decode_token_aligned_coords
from stok.utils.mdlm import (
    REGIMES,
    build_mask_groups,
    corrupt_mdlm_batch,
    mask_schedule,
    mdlm_loss_terms,
    stable_seed,
)
from stok.utils.pretrained import state_sha256
from stok.utils.sampling import inference_context, sample_mdlm
from stok.utils.tokenizer import Tokenizer


def resolve_mdlm_eval_config(cfg: DictConfig) -> DictConfig:
    """Resolve only MDLM defaults; write benchmark/policy metadata to run artifacts.

    ``generation.steps`` is the successful-update cadence; ``sampling_steps``
    controls reverse transitions. Explicit case maps replace the default matrix.
    Cohort paths are validated by data preflight; its identities gate scoring.
    """
    defaults = {
        "enabled": False,
        "cases": {
            f"{regime}_{placement}_p{p:g}": {
                "regime": regime,
                "placement": placement,
                "probability": p,
                "span_mean": 8,
            }
            for regime in REGIMES
            for placement in ("token", "span")
            for p in (0.15, 0.5, 0.85, 1.0)
        },
        "generation": {
            "enabled": False,
            "steps": 10000,
            "sampling_steps": 64,
            "max_samples": 16,
            "decode": False,
            "schedule": {"name": "linear", "power": 2},
            "cases": {
                "folding": {
                    "regime": "structure_only",
                    "placement": "token",
                    "span_mean": 8,
                },
                "inverse_folding_like": {
                    "regime": "sequence_only",
                    "placement": "token",
                    "span_mean": 8,
                },
                "joint": {
                    "regime": "joint_independent",
                    "placement": "token",
                    "span_mean": 8,
                },
            },
        },
        "conditioning_policy": "inverse_folding_like_native_sequence_tokenizer",
        "label_context": "native_full_chain",
    }
    raw_config = cfg.train.eval.get("mdlm")
    provided = (
        OmegaConf.to_container(raw_config, resolve=True)
        if raw_config is not None
        else {}
    )
    resolved = OmegaConf.merge(defaults, provided)
    # Case maps are whole benchmark definitions, rather than incremental overrides.
    for section in (resolved, resolved.generation):
        source = provided if section is resolved else provided.get("generation", {})
        if "cases" in source:
            section.cases = source["cases"]
    for section, denoising in ((resolved, True), (resolved.generation, False)):
        if not isinstance(section.cases, DictConfig):
            raise ValueError("MDLM evaluation cases must be a named mapping")
        for name, case in section.cases.items():
            if not name or "/" in name or case.get("regime") not in REGIMES:
                raise ValueError("MDLM evaluation case name/regime is invalid")
            case.setdefault("placement", "token")
            case.setdefault("span_mean", 8)
            if (
                case.placement not in {"token", "span"}
                or not math.isfinite(float(case.span_mean))
                or float(case.span_mean) < 1
            ):
                raise ValueError("MDLM evaluation placement/span_mean is invalid")
            if denoising and (
                "probability" not in case
                or not math.isfinite(float(case.probability))
                or not 0 <= float(case.probability) <= 1
            ):
                raise ValueError(
                    "MDLM evaluation probability must be explicit and in [0,1]"
                )
    generation = resolved.generation
    if generation.steps is None:
        generation.enabled = False
    for name in ("steps", "sampling_steps", "max_samples"):
        if name == "steps" and generation.steps is None:
            continue
        if type(generation[name]) is not int or generation[name] < 1:
            raise ValueError(f"MDLM generation {name} must be a positive integer")
    if generation.max_samples > 16:
        raise ValueError("MDLM frozen generation cohort has a maximum of 16 samples")
    mask_schedule(
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        name=generation.schedule.name,
        power=float(generation.schedule.power),
    )
    seed = cfg.train.eval.get("seed", 1729)
    if type(seed) is not int or seed < 0:
        raise ValueError("MDLM evaluation seed must be a nonnegative integer")
    with open_dict(cfg.train.eval):
        cfg.train.eval.seed = seed
        cfg.train.eval.mdlm = resolved
    if resolved.enabled or generation.enabled:
        identity = cfg.train.get("mdlm_identity", {})
        for name, enabled in (
            ("eval_cohort", resolved.enabled),
            ("generation_cohort", generation.enabled),
        ):
            if not enabled:
                continue
            cohort = identity.get(name)
            if not cohort or not cohort.get("sha256") or not cohort.get("sample_keys"):
                raise ValueError(f"MDLM evaluation requires a nonempty frozen {name}")
            keys = cohort.sample_keys
            if len(set(keys)) != len(keys):
                raise ValueError(f"MDLM {name} has duplicate cohort membership")
            if name == "generation_cohort" and len(keys) > generation.max_samples:
                raise ValueError(
                    "MDLM frozen generation cohort exceeds its maximum (at most 16)"
                )
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


def _raise_errors(error, accelerator, context):
    errors = gather_object([error]) if accelerator else [error]
    if any(errors):
        raise RuntimeError(f"MDLM evaluation {context}: {errors}")


def _geometry_scores(state, predicted, batch):
    for index, metric in enumerate((LDDTMetric(), RMSDMetric(), TMScoreMetric())):
        score = (
            None
            if batch["coords"] is None
            else metric.score(predicted, batch["coords"], batch["residue_mask"])
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


def evaluate_mdlm(
    model,
    loaders: dict[str, DataLoader],
    cfg: DictConfig,
    *,
    accelerator,
    decoder=None,
    run_denoising: bool = True,
    run_generation: bool = True,
) -> dict[str, dict[str, float]]:
    """Filter frozen populations before scoring; uneven ranks never forward DDP.

    Optional internal cadence selectors let training run expensive generation
    independently without mutating the benchmark configuration.
    """
    error = None
    try:
        settings = resolve_mdlm_eval_config(cfg)
        run_denoising = run_denoising and settings.enabled
        run_generation = run_generation and settings.generation.enabled
        eval_model = _unwrap_model(model, accelerator)
        device = _get_model_device(model, accelerator)
        if run_generation and settings.generation.decode:
            validate_mdlm_decoder(
                decoder,
                eval_model.structure_codebook,
                cfg.train.mdlm_identity.codebook_sha256,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_errors(error, accelerator, "configuration")
    if not run_denoising and not run_generation:
        return {}
    identity = cfg.train.mdlm_identity
    eval_keys = set(identity.eval_cohort.sample_keys) if run_denoising else set()
    generation_keys = (
        set(identity.generation_cohort.sample_keys) if run_generation else set()
    )
    requested = eval_keys | generation_keys
    seen = Counter()
    denoising_cases = list(settings.cases.items()) if run_denoising else []
    generation_cases = list(settings.generation.cases.items()) if run_generation else []
    weights = getattr(eval_model, "mdlm_regime_weights", {})
    joint_qualified = isinstance(weights, Mapping) and any(
        float(weights.get(name, 0)) > 0 for name in ("joint_independent", "joint_tied")
    )
    tokenizer = Tokenizer()
    canonical = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(CANONICAL_AA)), device=device
    )
    results = {}
    with ExitStack() as contexts:
        contexts.enter_context(inference_context(model))
        if decoder is not None:
            contexts.enter_context(inference_context(decoder))
        for dataset, loader in loaders.items():
            # Sums are reduced once after loader/forward error agreement, including empty ranks.
            denoising = torch.zeros(
                (len(denoising_cases), 4, 2), device=device, dtype=torch.float64
            )
            generation = torch.zeros(
                (len(generation_cases), 7), device=device, dtype=torch.float64
            )
            geometry = torch.zeros(
                (len(generation_cases) + 1, 3, 3), device=device, dtype=torch.float64
            )
            populations = torch.zeros(2, device=device, dtype=torch.float64)
            error = None
            try:
                for raw in loader:
                    selected = [
                        (
                            row,
                            _sample_key(row["dataset"], row["sequence_id"]),
                        )
                        for row in raw
                    ]
                    selected = [(row, key) for row, key in selected if key in requested]
                    seen.update(key for _, key in selected)
                    eval_rows = [row for row, key in selected if key in eval_keys]
                    if eval_rows:
                        batch = prepare_mdlm_batch(
                            eval_rows,
                            tokenizer,
                            max_len=int(cfg.data.max_len),
                            codebook_size=eval_model.codebook_size,
                            crop="center",
                            seeds=[0] * len(eval_rows),
                        )
                        batch = {
                            key: value.to(device)
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in batch.items()
                        }
                        populations[0] += len(eval_rows)
                        for index, (name, case) in enumerate(denoising_cases):
                            # This diagnostic config is fixed and never reads train.mdlm.
                            corruption_cfg = OmegaConf.create(
                                {
                                    "regime_weights": {case.regime: 1},
                                    "placement": case.placement,
                                    "span_mean": case.span_mean,
                                    "noise": {
                                        "name": "linear",
                                        "power": 2,
                                        "min_mask_probability": 1e-4,
                                    },
                                }
                            )
                            corruption = corrupt_mdlm_batch(
                                batch,
                                corruption_cfg,
                                seeds=[
                                    stable_seed(
                                        [cfg.train.eval.seed, key, name, "denoising"]
                                    )
                                    for key in batch["sample_keys"]
                                ],
                                mask_probability=float(case.probability),
                                regime=case.regime,
                            )
                            outputs = eval_model(
                                sequence_tokens=corruption["sequence_tokens"],
                                structure_tokens=corruption["structure_tokens"],
                            )
                            terms = mdlm_loss_terms(
                                outputs, batch, corruption, canonical_aa_ids=canonical
                            )
                            denoising[index] += torch.stack(
                                [
                                    terms[key].double()
                                    for key in (
                                        "ce_sum",
                                        "correct",
                                        "masked_count",
                                        "eligible_count",
                                    )
                                ]
                            )
                    # At most 16 frozen samples; singleton trajectories make decoder context fixed too.
                    for row, key in selected:
                        if key not in generation_keys:
                            continue
                        populations[1] += 1
                        batch = prepare_mdlm_batch(
                            [row],
                            tokenizer,
                            max_len=int(cfg.data.max_len),
                            codebook_size=eval_model.codebook_size,
                            crop="center",
                            seeds=[0],
                        )
                        batch = {
                            name: value.to(device)
                            if isinstance(value, torch.Tensor)
                            else value
                            for name, value in batch.items()
                        }
                        for index, (name, case) in enumerate(generation_cases):
                            if case.regime.startswith("joint") and not joint_qualified:
                                continue
                            generate = (
                                batch["residue_mask"][..., None]
                                .expand(-1, -1, 2)
                                .clone()
                            )
                            if case.regime == "structure_only":
                                generate[..., 0] = False
                            elif case.regime == "sequence_only":
                                generate[..., 1] = False
                            seed = stable_seed(
                                [cfg.train.eval.seed, key, name, "generation"]
                            )
                            groups = build_mask_groups(
                                generate[0],
                                batch["residue_mask"][0],
                                placement=case.placement,
                                tied=case.regime == "joint_tied",
                                span_mean=float(case.span_mean),
                                generator=torch.Generator().manual_seed(seed),
                            )[None]
                            sampled = sample_mdlm(
                                eval_model,
                                batch,
                                generate_mask=generate,
                                group_ids=groups,
                                schedule=settings.generation.schedule,
                                steps=settings.generation.sampling_steps,
                                seeds=[seed],
                                canonical_aa_ids=canonical,
                            )
                            for track, tokens in enumerate(
                                ("sequence_tokens", "structure_tokens")
                            ):
                                condition = ~generate[..., track]
                                generation[index, 0] += (
                                    sampled[tokens][condition]
                                    == batch[tokens][condition]
                                ).sum()
                                generation[index, 1] += condition.sum()
                                completed = (
                                    torch.isin(sampled[tokens], canonical)
                                    if track == 0
                                    else (
                                        (sampled[tokens] >= 0)
                                        & (sampled[tokens] < eval_model.codebook_size)
                                    )
                                )
                                generation[index, 2] += (
                                    completed & generate[..., track]
                                ).sum()
                                generation[index, 3] += generate[..., track].sum()
                            valid = generate[..., 0] & batch["sequence_valid"]
                            generation[index, 4] += (
                                sampled["sequence_tokens"][valid]
                                == batch["sequence_tokens"][valid]
                            ).sum()
                            generation[index, 5] += valid.sum()
                            generation[index, 6] += 1
                            if settings.generation.decode and generate[..., 1].any():
                                predicted = _decode(
                                    decoder,
                                    eval_model.structure_codebook,
                                    sampled["structure_tokens"],
                                    batch,
                                )
                                _geometry_scores(geometry[index], predicted, batch)
                        if settings.generation.decode:
                            complete = (
                                batch["structure_valid"] | ~batch["residue_mask"]
                            ).all()
                            if complete:
                                predicted = _decode(
                                    decoder,
                                    eval_model.structure_codebook,
                                    batch["structure_tokens"],
                                    batch,
                                )
                                _geometry_scores(geometry[-1], predicted, batch)
                            else:
                                geometry[-1, :, 2] += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            _raise_errors(error, accelerator, f"dataset {dataset}")
            if accelerator:
                denoising, generation, geometry, populations = [
                    accelerator.reduce(state, reduction="sum")
                    for state in (denoising, generation, geometry, populations)
                ]
            if not populations.sum():
                continue
            metrics = {}
            if populations[0]:
                for index, (name, _) in enumerate(denoising_cases):
                    for track, modality in enumerate(("sequence", "structure")):
                        ce, correct, masked, eligible = denoising[
                            index, :, track
                        ].tolist()
                        prefix = f"{name}/{modality}"
                        metrics.update(
                            {
                                f"{prefix}/num_masked": masked,
                                f"{prefix}/num_eligible": eligible,
                                f"{prefix}/unavailable": float(masked == 0),
                            }
                        )
                        if masked:
                            metrics.update(
                                {
                                    f"{prefix}/masked_ce": ce / masked,
                                    f"{prefix}/acc": correct / masked,
                                }
                            )
            if populations[1]:
                for index, (name, case) in enumerate(generation_cases):
                    prefix = f"{name}/generation"
                    if case.regime.startswith("joint"):
                        metrics[f"{prefix}/joint_qualified"] = float(joint_qualified)
                        if not joint_qualified:
                            continue
                    (
                        preserved,
                        clamped,
                        filled,
                        requested_count,
                        correct,
                        valid,
                        samples,
                    ) = generation[index].tolist()
                    metrics.update(
                        {
                            f"{prefix}/condition_preservation": preserved / clamped
                            if clamped
                            else 1.0,
                            f"{prefix}/token_completion": filled / requested_count
                            if requested_count
                            else 1.0,
                            f"{prefix}/num_samples": samples,
                        }
                    )
                    if case.regime == "sequence_only":
                        metrics[f"{prefix}/native_tokenizer_conditioning"] = 1.0
                    if valid:
                        metrics[f"{prefix}/sequence_acc"] = correct / valid
                    metrics[f"{prefix}/sequence_num_valid"] = valid
                if settings.generation.decode:
                    for index, (name, case) in enumerate(
                        [
                            *generation_cases,
                            ("true_token_control", {"regime": "structure_only"}),
                        ]
                    ):
                        regime = case["regime"]
                        if regime == "sequence_only" or (
                            regime.startswith("joint") and not joint_qualified
                        ):
                            continue
                        for column, score in enumerate(("lddt", "rmsd", "tm")):
                            total, count, skipped = geometry[index, column].tolist()
                            prefix = f"{name}/structure/{score}"
                            metrics.update(
                                {
                                    f"{prefix}/num_valid": count,
                                    f"{prefix}/num_skipped": skipped,
                                    f"{prefix}/unavailable": float(count == 0),
                                }
                            )
                            if count:
                                metrics[prefix] = total / count
            results[dataset] = metrics
        combined = Counter()
        for counts in gather_object([dict(seen)]) if accelerator else [seen]:
            combined.update(counts)
        if set(combined) != requested or any(value != 1 for value in combined.values()):
            raise RuntimeError(
                "MDLM frozen cohort has missing or duplicate samples across evaluation loaders/ranks"
            )
    return results
