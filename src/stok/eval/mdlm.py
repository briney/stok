"""Frozen, token-weighted MDLM diagnostics, independent of training corruption."""

from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any, cast

from accelerate.utils import gather_object
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from stok.data.mdlm import CANONICAL_AA, MDLMBatch, prepare_mdlm_batch
from stok.training.engine import _get_model_device, _unwrap_model
from stok.eval.metrics.structure import LDDTMetric, RMSDMetric, TMScoreMetric
from stok.utils.decoding import decode_token_aligned_coords
from stok.utils.mdlm import (
    MDLMCorruption,
    mask_schedule,
    mdlm_loss_terms,
)
from stok.utils.pretrained import state_sha256
from stok.utils.sampling import inference_context, sample_mdlm
from stok.utils.tokenizer import Tokenizer


def resolve_mdlm_eval_config(
    cfg: DictConfig, *, identity: Mapping[str, Any] | None = None
) -> DictConfig:
    """Resolve scientific evaluation settings without changing authored choices.

    ``generation.steps`` is the successful-update cadence; ``sampling_steps``
    controls reverse transitions. Family selections refer to the frozen artifact.
    Data preflight validates its canonical bindings before scoring.
    """
    defaults = {
        "enabled": False,
        "case_manifest": None,
        "families": None,
        "generation": {
            "enabled": False,
            "steps": 10000,
            "sampling_steps": 64,
            "max_cases": 16,
            "families": None,
            "decode": False,
            "schedule": {"name": "linear"},
        },
        "conditioning_policy": "inverse_folding_like_native_sequence_tokenizer",
        "label_context": "native_full_chain",
    }
    if not isinstance(cfg.train.get("eval"), DictConfig):
        raise ValueError("train.eval must be a mapping")
    if "seed" in cfg.train.eval:
        raise ValueError("Unsupported configuration field: train.eval.seed")
    provided = OmegaConf.to_container(cfg.train.eval.get("mdlm", {}), resolve=True)
    if not isinstance(provided, dict):
        raise ValueError("MDLM evaluation config must be a mapping")
    from stok.config import _check_fields

    contract = OmegaConf.create(defaults)
    contract.generation.steps = None
    generation = provided.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("train.eval.mdlm.generation must be a mapping")
    schedule = generation.get("schedule", {})
    if not isinstance(schedule, dict):
        raise ValueError("train.eval.mdlm.generation.schedule must be a mapping")
    if schedule.get("name") == "power":
        contract.generation.schedule.power = 2.0
    _check_fields(OmegaConf.create(provided), contract, "train.eval.mdlm")
    for key in ("conditioning_policy", "label_context"):
        if key in provided and provided[key] != defaults[key]:
            raise ValueError(f"train.eval.mdlm.{key} must declare {defaults[key]}")
    resolved = cast(DictConfig, OmegaConf.merge(defaults, provided))
    path = resolved.case_manifest
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("case_manifest must be a nonempty path or null")
    for section in (resolved, resolved.generation):
        families = section.families
        if families is not None and (
            not isinstance(families, ListConfig)
            or not families
            or any(not isinstance(name, str) or not name for name in families)
            or len(set(families)) != len(families)
        ):
            raise ValueError("families must be a unique nonempty key list or null")
    generation = resolved.generation
    if generation.steps is None:
        generation.enabled = False
    for name in ("steps", "sampling_steps", "max_cases"):
        if name == "steps" and generation.steps is None:
            continue
        if type(generation[name]) is not int or generation[name] < 1:
            raise ValueError(f"MDLM generation {name} must be a positive integer")
    if generation.max_cases > 16:
        raise ValueError("MDLM generation has a maximum of 16 expanded cases")
    if generation.schedule.name == "power":
        generation.schedule.setdefault("power", 2.0)
    mask_schedule(
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        name=generation.schedule.name,
        power=float(generation.schedule.get("power", 2)),
    )
    if identity is not None:
        shared = identity.get("shared_cases")
        for kind, section in (("denoising", resolved), ("generation", generation)):
            if not section.enabled:
                continue
            if not shared:
                raise ValueError("MDLM evaluation requires a frozen case_manifest")
            available = shared["request"][kind]
            selected = (
                list(available) if section.families is None else list(section.families)
            )
            if not selected or set(selected) - available.keys():
                raise ValueError(f"No matching or unknown {kind} families")
            count = sum(
                c["kind"] == kind and c["family_key"] in selected
                for c in shared["cases"]
            )
            if not count:
                raise ValueError(f"No matching {kind} cases")
            if kind == "generation" and count > generation.max_cases:
                raise ValueError("MDLM generation exceeds maximum of 16 expanded cases")
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


def _raise_errors(error, accelerator, context):
    errors = gather_object([error]) if accelerator else [error]
    if any(errors):
        raise RuntimeError(f"MDLM evaluation {context}: {errors}")


def _geometry_scores(state, predicted, batch):
    for index, metric in enumerate((LDDTMetric(), RMSDMetric(), TMScoreMetric())):
        score = (
            None
            if batch["coords"] is None
            else metric.score(
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


def selected_mdlm_cases(settings, shared):
    if shared is None:
        return []
    selected = []
    for case in shared["cases"]:
        section = settings if case["kind"] == "denoising" else settings.generation
        if section.enabled and (
            section.families is None or case["family_key"] in section.families
        ):
            selected.append(case)
    return selected


def mdlm_evaluation_protocol(cfg, *, identity, environment, decoder):
    from stok.eval.cases import evaluation_protocol

    settings = resolve_mdlm_eval_config(cfg, identity=identity)
    shared = identity.get("shared_cases")
    if shared is None:
        return None
    cases = selected_mdlm_cases(settings, shared)
    science = cast(dict[str, Any], OmegaConf.to_container(settings, resolve=True))
    science["families"] = sorted(
        {case["family_key"] for case in cases if case["kind"] == "denoising"}
    )
    science["generation"]["families"] = sorted(
        {case["family_key"] for case in cases if case["kind"] == "generation"}
    )
    if not settings.generation.enabled:
        science["generation"] = {"enabled": False}
    return evaluation_protocol(
        science,
        shared_cases={**shared, "cases": cases},
        representation={
            "sources": sorted(
                source["representation_sha256"]
                for source in identity["sources"].values()
                if source["kind"] == "eval"
            )
        },
        decoder=decoder
        if settings.generation.enabled and settings.generation.decode
        else None,
        environment=environment,
    )


def evaluate_mdlm(
    model,
    loaders: dict[str, DataLoader],
    cfg: DictConfig,
    *,
    accelerator,
    identity: Mapping[str, Any],
    protocol: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    decoder=None,
    run_denoising: bool = True,
    run_generation: bool = True,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Execute frozen cases on their row owner, preserving controls and coverage."""
    from stok.eval.cases import evaluation_measurement, project_case_controls

    error = None
    try:
        settings = resolve_mdlm_eval_config(cfg, identity=identity)
        run_denoising = run_denoising and settings.enabled
        run_generation = run_generation and settings.generation.enabled
        eval_model = _unwrap_model(model, accelerator)
        device = _get_model_device(model, accelerator)
        model_codebook_size = getattr(eval_model, "codebook_size", None)
        if type(model_codebook_size) is not int or model_codebook_size < 1:
            raise ValueError("MDLM evaluation codebook_size must be a positive integer")
        codebook_size = cast(int, model_codebook_size)
        selected = selected_mdlm_cases(settings, identity.get("shared_cases"))
        if protocol is None or protocol["selected_case_ids"] != [
            case["case_id"] for case in selected
        ]:
            raise ValueError("Evaluation protocol disagrees with selected frozen cases")
        evaluation_measurement(
            protocol, model_identity=model_identity, coverage={}, metrics={}
        )
        from stok.utils.checkpoint import execution_identity

        decoder_identity = (
            {
                "sha256": state_sha256(decoder.state_dict()),
                "codebook_sha256": identity["codebook_sha256"],
            }
            if decoder is not None
            else None
        )
        environment = execution_identity(accelerator)
        if environment["execution"]["device"] != device.type:
            raise ValueError("Evaluation accelerator/device identity mismatch")
        actual_protocol = mdlm_evaluation_protocol(
            cfg,
            identity=identity,
            environment=environment,
            decoder=decoder_identity,
        )
        if protocol != actual_protocol:
            raise ValueError(
                "Evaluation protocol disagrees with actual settings/decoder/execution"
            )

        if run_generation and settings.generation.decode:
            validate_mdlm_decoder(
                decoder, eval_model.structure_codebook, identity["codebook_sha256"]
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_errors(error, accelerator, "configuration")
    cases = [
        case
        for case in selected
        if run_denoising
        and case["kind"] == "denoising"
        or run_generation
        and case["kind"] == "generation"
    ]
    by_member = {}
    for case in cases:
        by_member.setdefault(case["canonical_id"], []).append(case)
    requested = set(by_member)
    denoising_cases = [
        (key, definition)
        for key, definition in identity["shared_cases"]["request"]["denoising"].items()
        if any(c["kind"] == "denoising" and c["family_key"] == key for c in cases)
    ]
    generation_cases = [
        (key, definition)
        for key, definition in identity["shared_cases"]["request"]["generation"].items()
        if any(c["kind"] == "generation" and c["family_key"] == key for c in cases)
    ]
    weights = getattr(eval_model, "mdlm_regime_weights", {})
    joint_qualified = isinstance(weights, Mapping) and any(
        float(weights.get(name, 0)) > 0 for name in ("joint_independent", "joint_tied")
    )
    tokenizer = Tokenizer()
    canonical = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(CANONICAL_AA)), device=device
    )
    results, seen, records = {}, Counter(), []
    rank = accelerator.process_index if accelerator else 0
    size = accelerator.num_processes if accelerator else 1
    for case in cases:
        entry = identity["coverage"][case["canonical_id"]]
        if entry["status"] == "rejected" and case["ordinal"] % size == rank:
            records.append(
                {
                    "case_id": case["case_id"],
                    "status": "rejected",
                    "reason": entry["reason"],
                    "unavailable_targets": sum(sum(row) for row in case["eligible"]),
                    "unavailable_conditioning": sum(
                        sum(row) for row in case["conditioning"]
                    ),
                }
            )
    with ExitStack() as contexts:
        contexts.enter_context(inference_context(model))
        if decoder is not None:
            contexts.enter_context(inference_context(decoder))
        for dataset, loader in loaders.items():
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
                    for row in raw:
                        key = row["canonical_id"]
                        if key not in requested:
                            continue
                        entry = identity["coverage"][key]
                        if entry["status"] != "admitted" or entry["source"] != dataset:
                            raise ValueError(
                                "Frozen case row disagrees with admitted source coverage"
                            )
                        seen[key] += 1
                        control_crops = set()
                        for case in by_member[key]:
                            if (
                                row.get("canonical_content_sha256")
                                != case["content_sha256"]
                                or row.get("residue_map_sha256")
                                != case["residue_map_sha256"]
                            ):
                                raise ValueError(
                                    "canonical row binding disagrees with frozen case"
                                )
                            record = {
                                "case_id": case["case_id"],
                                "status": "evaluated",
                                "reason": None,
                                "unavailable_targets": 0,
                                "unavailable_conditioning": 0,
                            }
                            records.append(record)
                            if case["crop"][1] - case["crop"][0] > cfg.data.max_len - 2:
                                record.update(
                                    status="unsupported", reason="crop_capacity"
                                )
                                continue
                            batch = prepare_mdlm_batch(
                                [row],
                                tokenizer,
                                max_len=int(cfg.data.max_len),
                                codebook_size=codebook_size,
                                crop="center",
                                seeds=[case["seed"]],
                                crop_intervals=[tuple(case["crop"])],
                            )
                            batch = cast(
                                MDLMBatch,
                                {
                                    name: value.to(device)
                                    if isinstance(value, torch.Tensor)
                                    else value
                                    for name, value in batch.items()
                                },
                            )
                            corruption = project_case_controls([case], batch)
                            record["unavailable_targets"] = int(
                                (
                                    corruption["requested_eligible"]
                                    & ~corruption["eligible"]
                                ).sum()
                            )
                            record["unavailable_conditioning"] = int(
                                corruption["unavailable_conditioning"].sum()
                            )
                            if not bool(corruption["case_available"].all()):
                                record.update(
                                    status="unavailable", reason="conditioning"
                                )
                                continue
                            if case["kind"] == "denoising":
                                index = [name for name, _ in denoising_cases].index(
                                    case["family_key"]
                                )
                                populations[0] += 1
                                outputs = eval_model(
                                    sequence_tokens=corruption["sequence_tokens"],
                                    structure_tokens=corruption["structure_tokens"],
                                )
                                terms = mdlm_loss_terms(
                                    outputs,
                                    batch,
                                    cast(MDLMCorruption, corruption),
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
                                continue
                            if not bool(corruption["eligible"].any()):
                                record.update(status="unavailable", reason="no_targets")
                                continue
                            populations[1] += 1
                            index = [name for name, _ in generation_cases].index(
                                case["family_key"]
                            )
                            if (
                                case["definition"]["regime"].startswith("joint")
                                and not joint_qualified
                            ):
                                record.update(
                                    status="unsupported", reason="joint_exposure"
                                )
                                continue
                            generate = corruption["eligible"]
                            groups = corruption["group_ids"].masked_fill(~generate, -1)
                            sampled = sample_mdlm(
                                eval_model,
                                batch,
                                generate_mask=generate,
                                group_ids=groups,
                                schedule=settings.generation.schedule,
                                steps=settings.generation.sampling_steps,
                                seeds=[case["seed"]],
                                canonical_aa_ids=canonical,
                            )
                            for track, tokens in enumerate(
                                ("sequence_tokens", "structure_tokens")
                            ):
                                condition = corruption["requested_conditioning"][
                                    ..., track
                                ]
                                generation[index, 0] += (
                                    sampled[tokens][condition]
                                    == (
                                        batch["sequence_tokens"]
                                        if track == 0
                                        else batch["structure_tokens"]
                                    )[condition]
                                ).sum()
                                generation[index, 1] += condition.sum()
                                completed = (
                                    torch.isin(sampled[tokens], canonical)
                                    if track == 0
                                    else (
                                        (sampled[tokens] >= 0)
                                        & (sampled[tokens] < codebook_size)
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
                            crop = tuple(case["crop"])
                            if settings.generation.decode and crop not in control_crops:
                                control_crops.add(crop)
                                if (
                                    batch["structure_valid"] | ~batch["residue_mask"]
                                ).all():
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
                    if case["regime"].startswith("joint"):
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
                    metrics[f"{prefix}/num_cases"] = samples
                    if not samples:
                        continue
                    metrics.update(
                        {
                            f"{prefix}/condition_preservation": preserved / clamped
                            if clamped
                            else 1.0,
                            f"{prefix}/token_completion": filled / requested_count,
                        }
                    )
                    if case["regime"] == "sequence_only":
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
        admitted = {
            key
            for key in requested
            if identity["coverage"][key]["status"] == "admitted"
        }
        if set(combined) != admitted or any(value != 1 for value in combined.values()):
            raise RuntimeError(
                "MDLM frozen cases have missing or duplicate admitted rows across loaders/ranks"
            )
        records = gather_object(records) if accelerator else records
    by_id = {record["case_id"]: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {case["case_id"] for case in cases}:
        raise RuntimeError("Frozen case ownership is missing or duplicated")
    records = [by_id[case["case_id"]] for case in cases]
    coverage = {
        "requested_case_ids": [case["case_id"] for case in cases],
        "canonical_controls": [
            {
                key: case[key]
                for key in (
                    "case_id",
                    "canonical_id",
                    "seed",
                    "crop",
                    "positions",
                    "eligible",
                    "masked",
                    "group_ids",
                    "conditioning",
                )
            }
            for case in cases
        ],
        "requested_cases": len(cases),
        "unique_samples": len(requested),
        "cases": records,
        **{
            status + "_cases": sum(r["status"] == status for r in records)
            for status in ("evaluated", "rejected", "unavailable", "unsupported")
        },
        **{
            key: sum(r[key] for r in records)
            for key in ("unavailable_targets", "unavailable_conditioning")
        },
    }
    return results, evaluation_measurement(
        protocol, model_identity=model_identity, coverage=coverage, metrics=results
    )
