"""Local RNG streams and absorbing corruption for aligned masked diffusion."""

import hashlib
import json
import math
from typing import TYPE_CHECKING, TypedDict

from omegaconf import DictConfig
import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..data.mdlm import MDLMBatch


class MDLMCorruption(TypedDict):
    sequence_tokens: Tensor
    structure_tokens: Tensor
    masked: Tensor
    eligible: Tensor
    group_ids: Tensor
    mask_probability: Tensor
    weight: Tensor
    regimes: list[str]


REGIMES = ("joint_independent", "structure_only", "sequence_only", "joint_tied")


def stable_seed(parts: list) -> int:
    """Canonical primitive JSON, including an explicit caller-supplied purpose."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (
        2**63
    )


def _schedule_input(value: Tensor, name: str, power: float) -> Tensor:
    if name not in {"linear", "cosine", "power"}:
        raise ValueError(f"Unsupported MDLM noise schedule: {name}")
    if not math.isfinite(power) or power <= 0:
        raise ValueError("MDLM noise power must be finite and positive")
    if not value.is_floating_point():
        raise ValueError("MDLM schedule inputs must be floating-point tensors")
    if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("MDLM schedule inputs must be finite and in [0,1]")
    return value if value.dtype == torch.float64 else value.float()


def mask_schedule(t: Tensor, *, name: str, power: float = 2.0) -> tuple[Tensor, Tensor]:
    """Return p(t), p'(t); fractional powers have a singular derivative at t=0."""
    with torch.autocast(t.device.type, enabled=False):
        t = _schedule_input(t, name, power)
        if name == "linear":
            p, derivative = t, torch.ones_like(t)
        elif name == "cosine":
            p = torch.sin(math.pi / 2 * t).square()
            derivative = math.pi / 2 * torch.sin(math.pi * t)
            derivative = derivative.masked_fill((t == 0) | (t == 1), 0)
        else:
            p, derivative = t.pow(power), power * t.pow(power - 1)
        p = torch.where(t == 0, 0.0, torch.where(t == 1, 1.0, p))
        interior = (t > 0) & (t < 1)
        if (
            not torch.isfinite(p).all()
            or not torch.isfinite(derivative[interior]).all()
        ):
            raise ValueError("MDLM schedule exceeds numerical support")
        return p, derivative


def time_from_mask_probability(p: Tensor, *, name: str, power: float = 2.0) -> Tensor:
    with torch.autocast(p.device.type, enabled=False):
        p = _schedule_input(p, name, power)
        if name == "linear":
            t = p
        elif name == "cosine":
            t = 2 / math.pi * torch.asin(p.sqrt())
        else:
            t = p.pow(1 / power)
        if not torch.isfinite(t).all():
            raise ValueError("MDLM inverse schedule exceeds numerical support")
        return torch.where(p == 0, 0.0, torch.where(p == 1, 1.0, t))


def build_mask_groups(
    eligible: Tensor,
    residue_mask: Tensor,
    *,
    placement: str,
    tied: bool,
    span_mean: float,
    generator: torch.Generator,
) -> Tensor:
    """Partition physical slots first; inactive cells never join a group."""
    if placement not in {"token", "span"}:
        raise ValueError(f"Unsupported MDLM placement: {placement}")
    if not math.isfinite(span_mean) or span_mean < 1:
        raise ValueError("MDLM span_mean must be finite and >= 1")
    if (
        eligible.ndim != 2
        or eligible.shape[1] != 2
        or residue_mask.shape != eligible.shape[:1]
        or eligible.dtype != torch.bool
        or residue_mask.dtype != torch.bool
    ):
        raise ValueError("MDLM groups require bool eligible[L,2] and residue_mask[L]")
    result = torch.full_like(eligible, -1, dtype=torch.long)
    previous = torch.cat((residue_mask.new_zeros(1), residue_mask[:-1]))
    offset = 0
    for track in range(1 if tied else 2):
        boundaries = torch.ones_like(residue_mask)
        if placement == "span":
            boundaries = (
                torch.rand(
                    len(residue_mask), generator=generator, device=generator.device
                )
                < 1 / span_mean
            ).to(eligible.device)
        starts = residue_mask & (~previous | boundaries)
        ids = starts.long().cumsum(0) - 1 + offset
        active = eligible & residue_mask[:, None]
        if tied:
            result = ids[:, None].expand(-1, 2).masked_fill(~active, -1)
        else:
            result[:, track] = ids.masked_fill(~active[:, track], -1)
            offset += int(starts.sum())
    return result


def _generator(seed: int, purpose: str) -> torch.Generator:
    return torch.Generator().manual_seed(stable_seed([seed, purpose]))


def validate_mdlm_config(config: DictConfig) -> None:
    """Validate corruption and modality weights before creating run artifacts."""
    weights = config.regime_weights
    raw = [float(weights.get(name, 0)) for name in REGIMES]
    if (
        set(weights) - set(REGIMES)
        or any(not math.isfinite(w) or w < 0 for w in raw)
        or not max(raw)
    ):
        raise ValueError(
            "MDLM regime_weights must name supported regimes and be finite, nonnegative and nonzero"
        )
    if config.placement not in {"token", "span"}:
        raise ValueError("Unsupported MDLM placement")
    if not math.isfinite(float(config.span_mean)) or float(config.span_mean) < 1:
        raise ValueError("MDLM span_mean must be finite and >= 1")
    p_min = float(config.noise.min_mask_probability)
    if not math.isfinite(p_min) or not 0 < p_min < 1:
        raise ValueError("MDLM min_mask_probability must be finite and in (0,1)")
    t_min = time_from_mask_probability(
        torch.tensor(p_min, dtype=torch.float64),
        name=config.noise.name,
        power=float(config.noise.power),
    )
    if (
        not 0
        < float(t_min)
        < float(torch.nextafter(torch.ones_like(t_min), torch.zeros_like(t_min)))
    ):
        raise ValueError("MDLM minimum noise time exceeds numerical support")
    # Corruption-only callers need not specify loss weighting.
    seq, struct = (
        float(config.get("sequence_loss_weight", 1)),
        float(config.get("structure_loss_weight", 1)),
    )
    if any(not math.isfinite(w) or w < 0 for w in (seq, struct)):
        raise ValueError("MDLM loss weights must be finite and nonnegative")
    for regime, enabled in zip(REGIMES, raw):
        active = (
            struct
            if regime == "structure_only"
            else seq
            if regime == "sequence_only"
            else seq + struct
        )
        if enabled and active <= 0:
            raise ValueError(
                f"MDLM enabled regime {regime} needs a positively weighted modality"
            )


def corrupt_mdlm_batch(
    batch: "MDLMBatch",
    config: DictConfig,
    *,
    seeds: list[int],
    mask_probability: float | None = None,
    regime: str | None = None,
    placement: str | None = None,
) -> MDLMCorruption:
    """Training samples time/regime; explicit diagnostics require both overrides."""
    diagnostic = any(
        value is not None for value in (mask_probability, regime, placement)
    )
    if diagnostic and (mask_probability is None or regime is None):
        raise ValueError(
            "MDLM diagnostic overrides require both regime and mask_probability"
        )
    if mask_probability is not None and (
        regime not in REGIMES
        or not math.isfinite(mask_probability)
        or not 0 <= mask_probability <= 1
    ):
        raise ValueError("MDLM diagnostic regime/probability is invalid")
    validate_mdlm_config(config)
    weights = torch.tensor(
        [float(config.regime_weights.get(name, 0)) for name in REGIMES],
        dtype=torch.float64,
    )
    weights = weights / weights.max()
    placement = config.placement if placement is None else placement
    if placement not in {"token", "span"}:
        raise ValueError(f"Unsupported MDLM placement: {placement}")
    span_mean = float(config.span_mean)
    name, power = config.noise.name, float(config.noise.power)
    t_min = time_from_mask_probability(
        torch.tensor(float(config.noise.min_mask_probability), dtype=torch.float64),
        name=name,
        power=power,
    )
    t_upper = torch.nextafter(torch.ones_like(t_min), torch.zeros_like(t_min))
    sequence = batch["sequence_tokens"].clone()
    structure = batch["structure_tokens"].clone()
    if sequence.ndim != 2 or structure.shape != sequence.shape or not sequence.shape[0]:
        raise ValueError("MDLM corruption requires a nonempty aligned batch")
    if len(seeds) != len(sequence) or any(type(seed) is not int for seed in seeds):
        raise ValueError("MDLM corruption requires one integer seed per sample")
    mask_id, codebook_size = batch["sequence_mask_id"], batch["codebook_size"]
    if (
        type(mask_id) is not int
        or mask_id < 0
        or type(codebook_size) is not int
        or codebook_size < 1
    ):
        raise ValueError("MDLM corruption requires valid mask/codebook metadata")
    eligible = torch.stack((batch["sequence_valid"], batch["structure_valid"]), -1)
    eligible = eligible & batch["residue_mask"][..., None]
    masked = torch.zeros_like(eligible)
    groups = torch.full_like(eligible, -1, dtype=torch.long)
    probabilities, loss_weights, regimes = [], [], []
    with torch.autocast(sequence.device.type, enabled=False):
        for i, seed in enumerate(seeds):
            selected = (
                regime
                if diagnostic
                else REGIMES[
                    int(
                        torch.multinomial(
                            weights, 1, generator=_generator(seed, "regime")
                        )
                    )
                ]
            )
            if selected == "structure_only":
                eligible[i, :, 0] = False
            elif selected == "sequence_only":
                eligible[i, :, 1] = False
            if mask_probability is not None:
                p, weight = torch.tensor(float(mask_probability)), torch.tensor(1.0)
            else:
                t = t_min + (1 - t_min) * torch.rand(
                    (), generator=_generator(seed, "time"), dtype=torch.float64
                )
                t = torch.minimum(t, t_upper)
                p, derivative = mask_schedule(t, name=name, power=power)
                weight = (1 - t_min) * derivative / p
            groups[i] = build_mask_groups(
                eligible[i],
                batch["residue_mask"][i],
                placement=placement,
                tied=selected == "joint_tied",
                span_mean=span_mean,
                generator=_generator(seed, "partition"),
            )
            active_ids, inverse = groups[i][eligible[i]].unique(return_inverse=True)
            draws = (
                torch.rand(
                    len(active_ids),
                    generator=_generator(seed, "corruption"),
                    dtype=torch.float64,
                )
                < p
            )
            masked[i][eligible[i]] = draws.to(sequence.device)[inverse]
            probabilities.append(p)
            loss_weights.append(weight)
            regimes.append(selected)
    sequence.masked_fill_(masked[..., 0], mask_id)
    structure.masked_fill_(masked[..., 1], codebook_size + 1)
    return {
        "sequence_tokens": sequence,
        "structure_tokens": structure,
        "masked": masked,
        "eligible": eligible,
        "group_ids": groups,
        "mask_probability": torch.stack(probabilities).to(sequence.device),
        "weight": torch.stack(loss_weights).to(sequence.device),
        "regimes": regimes,
    }


class MDLMLossTerms(TypedDict):
    weighted_sum: Tensor
    ce_sum: Tensor
    correct: Tensor
    masked_count: Tensor
    eligible_count: Tensor


def mdlm_loss_terms(
    outputs: dict[str, Tensor],
    batch: "MDLMBatch",
    corruption: MDLMCorruption,
    *,
    canonical_aa_ids: Tensor,
) -> MDLMLossTerms:
    """Masked CE numerators; normalization uses pre-corruption eligible cells."""
    shape = batch["sequence_tokens"].shape
    eligible, masked = corruption["eligible"], corruption["masked"]
    if (
        eligible.shape != (*shape, 2)
        or masked.shape != eligible.shape
        or eligible.dtype != torch.bool
        or masked.dtype != torch.bool
        or (masked & ~eligible).any()
    ):
        raise ValueError("MDLM loss requires aligned eligible/masked cells")
    available = torch.stack((batch["sequence_valid"], batch["structure_valid"]), -1)
    if (eligible & ~(available & batch["residue_mask"][..., None])).any():
        raise ValueError("MDLM eligible cells must be available residue targets")
    weight = corruption["weight"]
    if (
        weight.shape != shape[:1]
        or not torch.isfinite(weight).all()
        or (weight < 0).any()
    ):
        raise ValueError(
            "MDLM diffusion weights must be finite nonnegative per-sample values"
        )
    sequence = outputs["sequence_logits"]
    structure = outputs["structure_logits"]
    if (
        sequence.shape[:2] != shape
        or sequence.ndim != 3
        or structure.shape != (*shape, batch["codebook_size"])
        or canonical_aa_ids.ndim != 1
        or not canonical_aa_ids.numel()
        or canonical_aa_ids.dtype != torch.long
        or (canonical_aa_ids < 0).any()
        or (canonical_aa_ids >= sequence.shape[-1]).any()
        or canonical_aa_ids.unique().numel() != canonical_aa_ids.numel()
    ):
        raise ValueError("MDLM output/canonical vocabulary shapes are invalid")
    aa = canonical_aa_ids.to(sequence.device)
    remap = torch.full(
        (sequence.shape[-1],), -1, device=sequence.device, dtype=torch.long
    )
    remap[aa] = torch.arange(len(aa), device=aa.device)
    weighted_sums, ce_sums, correct = [], [], []
    with torch.autocast(sequence.device.type, enabled=False):
        for track, (logits, targets) in enumerate(
            (
                (sequence, batch["sequence_tokens"]),
                (structure, batch["structure_tokens"]),
            )
        ):
            valid, selected = eligible[..., track], masked[..., track]
            # Validate unmasked targets/predictions too: masking must not hide bad inputs.
            if not torch.isfinite(
                logits[available[..., track] & batch["residue_mask"]]
            ).all():
                raise FloatingPointError("Nonfinite valid MDLM predictions")
            target = targets[valid]
            if ((target < 0) | (target >= logits.shape[-1])).any():
                raise ValueError("MDLM target IDs are outside the output vocabulary")
            if track == 0:
                if (remap[target] < 0).any():
                    raise ValueError("MDLM sequence target is not canonical")
                predictions = logits[selected].float().index_select(-1, aa)
                target = remap[targets[selected]]
            else:
                predictions, target = logits[selected].float(), targets[selected]
            # Empty reductions connect both heads, including empty local DDP ranks.
            ce = torch.nn.functional.cross_entropy(
                predictions, target, reduction="none"
            )
            ce_sums.append(ce.sum())
            weighted_sums.append((ce * weight[:, None].expand(shape)[selected]).sum())
            correct.append((predictions.argmax(-1) == target).sum())
    return {
        "weighted_sum": torch.stack(weighted_sums),
        "ce_sum": torch.stack(ce_sums),
        "correct": torch.stack(correct),
        "masked_count": masked.sum((0, 1)),
        "eligible_count": eligible.sum((0, 1)),
    }
