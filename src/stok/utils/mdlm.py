"""Local RNG streams and absorbing corruption for aligned masked diffusion."""

import hashlib
import json
import math
import random
from typing import TYPE_CHECKING, TypedDict, cast

from omegaconf import DictConfig, ListConfig, OmegaConf
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
MDLM_MODES = ("regimes", "independent_tracks")
NOISE_NAMES = ("linear", "cosine", "power", "beta_uniform")
WEIGHT_MODES = ("elbo", "uniform", "clipped_elbo")
TRACK_STATE_PAIRS = (
    ("active", "active"),
    ("active", "clean"),
    ("active", "absent"),
    ("clean", "active"),
    ("absent", "active"),
)
_NOISE_SCHEMA = {
    "name": "linear",
    "power": 2.0,
    "beta_alpha": 3.0,
    "beta_beta": 9.0,
    "beta_weight": 0.8,
    "min_mask_probability": 0.0001,
    "weight_mode": "elbo",
    "max_weight": 20.0,
}
_TRACK_SCHEMA = {
    "clean_prob": 0.0,
    "absent_prob": 0.0,
    "span_prob": 0.0,
    "span_mean": 8.0,
    "noise": _NOISE_SCHEMA,
}
_STAGE_SCHEMA = OmegaConf.create(
    {
        "until_step": None,
        "mode": "regimes",
        "placement": "token",
        "span_mean": 8.0,
        "regime_weights": {name: 0.0 for name in REGIMES},
        "noise": _NOISE_SCHEMA,
        "tied_prob": 0.0,
        "sequence": _TRACK_SCHEMA,
        "structure": _TRACK_SCHEMA,
    }
)


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


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"MDLM {label} must be finite")
    return float(value)


def _validate_noise_config(noise: DictConfig) -> None:
    if not isinstance(noise, DictConfig):
        raise ValueError("MDLM noise must be a mapping")
    name = noise.get("name")
    if name not in NOISE_NAMES:
        raise ValueError(f"Unsupported MDLM noise schedule: {name}")
    weight_mode = noise.get("weight_mode", "elbo")
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"Unsupported MDLM noise weight_mode: {weight_mode}")
    p_min = _finite_number(noise.get("min_mask_probability"), "min_mask_probability")
    if not 0 < p_min < 1:
        raise ValueError("MDLM min_mask_probability must be finite and in (0,1)")
    if weight_mode == "clipped_elbo":
        if _finite_number(noise.get("max_weight", 20.0), "max_weight") <= 0:
            raise ValueError("MDLM noise max_weight must be positive")
    if name == "beta_uniform":
        if weight_mode == "elbo":
            raise ValueError(
                "MDLM beta_uniform noise requires weight_mode in {'uniform', 'clipped_elbo'}"
            )
        if _finite_number(noise.get("beta_alpha", 3.0), "beta_alpha") <= 0:
            raise ValueError("MDLM beta_alpha must be positive")
        if _finite_number(noise.get("beta_beta", 9.0), "beta_beta") <= 0:
            raise ValueError("MDLM beta_beta must be positive")
        beta_weight = _finite_number(noise.get("beta_weight", 0.8), "beta_weight")
        if not 0 <= beta_weight <= 1:
            raise ValueError("MDLM beta_weight must be in [0, 1]")
        return
    power = _finite_number(noise.get("power", 2.0), "noise.power")
    t_min = time_from_mask_probability(
        torch.tensor(p_min, dtype=torch.float64),
        name=str(name),
        power=power,
    )
    if (
        not 0
        < float(t_min)
        < float(torch.nextafter(torch.ones_like(t_min), torch.zeros_like(t_min)))
    ):
        raise ValueError("MDLM minimum noise time exceeds numerical support")


def _sample_noise(
    noise: DictConfig, *, seed: int, purpose: str
) -> tuple[Tensor, Tensor]:
    name = str(noise.name)
    p_min = float(noise.min_mask_probability)
    weight_mode = str(noise.get("weight_mode", "elbo"))
    max_weight = float(noise.get("max_weight", 20.0))
    if name == "beta_uniform":
        rng = random.Random(stable_seed([seed, purpose]))
        alpha = float(noise.get("beta_alpha", 3.0))
        beta = float(noise.get("beta_beta", 9.0))
        beta_weight = float(noise.get("beta_weight", 0.8))
        raw = (
            rng.betavariate(alpha, beta)
            if rng.random() < beta_weight
            else p_min + (1.0 - p_min) * rng.random()
        )
        p_val = min(max(raw, p_min), math.nextafter(1.0, 0.0))
        p = torch.tensor(p_val, dtype=torch.float64)
        w_val = 1.0 if weight_mode == "uniform" else min(1.0 / p_val, max_weight)
        return p, torch.tensor(w_val, dtype=torch.float64)
    power = float(noise.get("power", 2.0))
    t_min = time_from_mask_probability(
        torch.tensor(p_min, dtype=torch.float64), name=name, power=power
    )
    t_upper = torch.nextafter(torch.ones_like(t_min), torch.zeros_like(t_min))
    t = t_min + (1 - t_min) * torch.rand(
        (), generator=_generator(seed, purpose), dtype=torch.float64
    )
    t = torch.minimum(t, t_upper)
    p, derivative = mask_schedule(t, name=name, power=power)
    if weight_mode == "uniform":
        return p, torch.tensor(1.0, dtype=torch.float64)
    weight = (1 - t_min) * derivative / p
    if weight_mode == "clipped_elbo":
        weight = weight.clamp_max(max_weight)
    return p, weight


def _partition_track(
    eligible_track: Tensor,
    residue_mask: Tensor,
    *,
    placement: str,
    span_mean: float,
    offset: int,
    generator: torch.Generator,
) -> tuple[Tensor, int]:
    previous = torch.cat((residue_mask.new_zeros(1), residue_mask[:-1]))
    boundaries = torch.ones_like(residue_mask)
    if placement == "span":
        boundaries = (
            torch.rand(len(residue_mask), generator=generator, device=generator.device)
            < 1 / span_mean
        ).to(eligible_track.device)
    starts = residue_mask & (~previous | boundaries)
    ids = starts.long().cumsum(0) - 1 + offset
    active = eligible_track & residue_mask
    return ids.masked_fill(~active, -1), offset + int(starts.sum())


def _track_probs(track: DictConfig) -> tuple[float, float, float]:
    clean = float(track.get("clean_prob", 0.0))
    absent = float(track.get("absent_prob", 0.0))
    return clean, absent, max(0.0, 1.0 - clean - absent)


def _pair_weights(config: DictConfig) -> list[float]:
    c0, a0, m0 = _track_probs(config.sequence)
    c1, a1, m1 = _track_probs(config.structure)
    return [m0 * m1, m0 * c1, m0 * a1, c0 * m1, a0 * m1]


def resolve_mdlm_stages(
    config: DictConfig,
) -> tuple[tuple[int | None, DictConfig], ...]:
    """Expand base config and optional curriculum into cached (until_step, cfg) stages."""
    curriculum = config.get("curriculum")
    if curriculum is None:
        return ((None, config),)
    raw_base = cast(dict, OmegaConf.to_container(config, resolve=True))
    raw_base.pop("curriculum", None)
    base = OmegaConf.create(raw_base)
    stages: list[tuple[int | None, DictConfig]] = []
    for stage in curriculum:
        raw_stage = cast(dict, OmegaConf.to_container(stage, resolve=True))
        until_step = raw_stage.pop("until_step", None)
        merged = cast(DictConfig, OmegaConf.merge(base, raw_stage))
        stages.append((until_step, merged))
    return tuple(stages)


def resolve_mdlm_step_config(
    config: DictConfig, global_step: int
) -> tuple[int, DictConfig]:
    """Select the active curriculum stage for ``global_step``."""
    for index, (until_step, stage_cfg) in enumerate(resolve_mdlm_stages(config)):
        if until_step is None or global_step < until_step:
            return index, stage_cfg
    raise ValueError(f"MDLM curriculum does not cover global_step={global_step}")


def _stage_effective_regime_weights(stage: DictConfig) -> dict[str, float]:
    mode = stage.get("mode", "regimes")
    if mode == "regimes":
        return {
            str(k): float(v) for k, v in (stage.get("regime_weights") or {}).items()
        }
    c0, a0, m0 = _track_probs(stage.sequence)
    c1, a1, m1 = _track_probs(stage.structure)
    tied = float(stage.get("tied_prob", 0.0))
    total = 1.0 - (c0 + a0) * (c1 + a1)
    if total <= 0:
        return {name: 0.0 for name in REGIMES}
    joint = m0 * m1 / total
    return {
        "joint_independent": joint * (1.0 - tied),
        "joint_tied": joint * tied,
        "structure_only": (c0 + a0) * m1 / total,
        "sequence_only": m0 * (c1 + a1) / total,
    }


def effective_regime_weights(config: DictConfig) -> dict[str, float]:
    """Effective regime provenance used for joint-generation qualification."""
    if config.get("curriculum") is None and config.get("mode", "regimes") == "regimes":
        return dict(config.get("regime_weights") or {})
    per_stage = [
        _stage_effective_regime_weights(stage)
        for _, stage in resolve_mdlm_stages(config)
    ]
    return {
        name: max(weights.get(name, 0.0) for weights in per_stage) for name in REGIMES
    }


def _validate_single_mdlm_config(config: DictConfig) -> None:
    mode = config.get("mode", "regimes")
    if mode not in MDLM_MODES:
        raise ValueError(f"Unsupported MDLM mode: {mode}")
    # Corruption-only callers need not specify loss weighting.
    seq_w = _finite_number(
        config.get("sequence_loss_weight", 1.0), "sequence_loss_weight"
    )
    struct_w = _finite_number(
        config.get("structure_loss_weight", 1.0), "structure_loss_weight"
    )
    if seq_w < 0 or struct_w < 0:
        raise ValueError("MDLM loss weights must be finite and nonnegative")
    if mode == "regimes":
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
        if _finite_number(config.get("span_mean", 8.0), "span_mean") < 1:
            raise ValueError("MDLM span_mean must be finite and >= 1")
        _validate_noise_config(config.noise)
        for regime, enabled in zip(REGIMES, raw):
            active = (
                struct_w
                if regime == "structure_only"
                else seq_w
                if regime == "sequence_only"
                else seq_w + struct_w
            )
            if enabled and active <= 0:
                raise ValueError(
                    f"MDLM enabled regime {regime} needs a positively weighted modality"
                )
        return
    tied_prob = _finite_number(config.get("tied_prob", 0.0), "tied_prob")
    if not 0 <= tied_prob <= 1:
        raise ValueError("MDLM tied_prob must be in [0, 1]")
    for track_name in ("sequence", "structure"):
        track = config.get(track_name)
        if not isinstance(track, DictConfig):
            raise ValueError(f"MDLM independent_tracks requires {track_name} mapping")
        clean_p = _finite_number(
            track.get("clean_prob", 0.0), f"{track_name}.clean_prob"
        )
        absent_p = _finite_number(
            track.get("absent_prob", 0.0), f"{track_name}.absent_prob"
        )
        if not (0 <= clean_p <= 1 and 0 <= absent_p <= 1 and clean_p + absent_p <= 1):
            raise ValueError(
                f"MDLM {track_name} clean_prob and absent_prob must be in [0, 1] with sum <= 1"
            )
        span_p = _finite_number(track.get("span_prob", 0.0), f"{track_name}.span_prob")
        if not 0 <= span_p <= 1:
            raise ValueError(f"MDLM {track_name}.span_prob must be in [0, 1]")
        if _finite_number(track.get("span_mean", 8.0), f"{track_name}.span_mean") < 1:
            raise ValueError(f"MDLM {track_name}.span_mean must be finite and >= 1")
        _validate_noise_config(track.noise)
    pair_weights = _pair_weights(config)
    if sum(pair_weights) <= 0:
        raise ValueError(
            "MDLM independent_tracks requires at least one track with positive active probability"
        )
    for (seq_s, struct_s), prob in zip(TRACK_STATE_PAIRS, pair_weights):
        if prob <= 0:
            continue
        active_w = (seq_w if seq_s == "active" else 0.0) + (
            struct_w if struct_s == "active" else 0.0
        )
        if active_w <= 0:
            raise ValueError(
                f"MDLM enabled track state ({seq_s}, {struct_s}) needs a positively weighted modality"
            )


def validate_mdlm_config(config: DictConfig) -> None:
    """Validate corruption, modality weights, and every curriculum stage upfront."""
    from stok.config import reject_unknown_keys

    _validate_single_mdlm_config(config)
    curriculum = config.get("curriculum")
    if curriculum is None:
        return
    if not isinstance(curriculum, (list, ListConfig)) or not curriculum:
        raise ValueError("MDLM curriculum must be null or a nonempty list of stages")
    prev_step = 0
    for idx, stage in enumerate(curriculum):
        if not isinstance(stage, DictConfig):
            raise ValueError(f"MDLM curriculum stage {idx} must be a mapping")
        reject_unknown_keys(stage, _STAGE_SCHEMA, f"train.mdlm.curriculum[{idx}]")
        if "until_step" not in stage:
            raise ValueError(f"MDLM curriculum stage {idx} requires until_step")
        until_step = stage.get("until_step")
        is_last = idx == len(curriculum) - 1
        if is_last:
            if until_step is not None:
                raise ValueError("Final MDLM curriculum stage must set until_step=null")
        else:
            if type(until_step) is not int or until_step <= prev_step:
                raise ValueError(
                    "MDLM curriculum until_step values must be strictly increasing positive integers"
                )
            prev_step = until_step
    for _, stage_cfg in resolve_mdlm_stages(config):
        _validate_single_mdlm_config(stage_cfg)


def _corrupt_independent_tracks(
    batch: "MDLMBatch", config: DictConfig, *, seeds: list[int]
) -> MDLMCorruption:
    sequence = batch["sequence_tokens"].clone()
    structure = batch["structure_tokens"].clone()
    mask_id, codebook_size = batch["sequence_mask_id"], batch["codebook_size"]
    unavailable_id = codebook_size + 2
    eligible = torch.stack((batch["sequence_valid"], batch["structure_valid"]), -1)
    eligible = eligible & batch["residue_mask"][..., None]
    masked = torch.zeros_like(eligible)
    groups = torch.full_like(eligible, -1, dtype=torch.long)
    pair_probs = torch.tensor(_pair_weights(config), dtype=torch.float64)
    tied_prob = float(config.get("tied_prob", 0.0))
    tracks = (config.sequence, config.structure)
    probabilities, loss_weights, regimes = [], [], []
    with torch.autocast(sequence.device.type, enabled=False):
        for i, seed in enumerate(seeds):
            residue = batch["residue_mask"][i]
            has_struct = bool(eligible[i, :, 1].any())
            if not has_struct:
                seq_state, struct_state = "active", "absent"
            else:
                pair_idx = int(
                    torch.multinomial(
                        pair_probs, 1, generator=_generator(seed, "track_state")
                    )
                )
                seq_state, struct_state = TRACK_STATE_PAIRS[pair_idx]
            states = (seq_state, struct_state)
            if seq_state != "active":
                eligible[i, :, 0] = False
                if seq_state == "absent":
                    sequence[i].masked_fill_(residue, mask_id)
            if struct_state != "active":
                eligible[i, :, 1] = False
                if struct_state == "absent" and has_struct:
                    structure[i].masked_fill_(residue, unavailable_id)
            tied = (
                seq_state == "active"
                and struct_state == "active"
                and tied_prob > 0.0
                and float(
                    torch.rand(
                        (), generator=_generator(seed, "tied"), dtype=torch.float64
                    )
                )
                < tied_prob
            )
            p_pair = torch.zeros(2, dtype=torch.float64)
            w_pair = torch.zeros(2, dtype=torch.float64)
            if tied:
                p_shared, w_shared = _sample_noise(
                    tracks[1].noise, seed=seed, purpose="time_tied"
                )
                p_pair[:] = p_shared
                w_pair[:] = w_shared
                use_span = any(
                    float(
                        torch.rand(
                            (),
                            generator=_generator(seed, f"span_{t}"),
                            dtype=torch.float64,
                        )
                    )
                    < float(tracks[t].get("span_prob", 0.0))
                    for t in (0, 1)
                )
                groups[i] = build_mask_groups(
                    eligible[i],
                    residue,
                    placement="span" if use_span else "token",
                    tied=True,
                    span_mean=float(tracks[1].get("span_mean", 8.0)),
                    generator=_generator(seed, "partition"),
                )
                active_ids, inverse = groups[i][eligible[i]].unique(return_inverse=True)
                draws = (
                    torch.rand(
                        len(active_ids),
                        generator=_generator(seed, "corruption"),
                        dtype=torch.float64,
                    )
                    < p_shared
                )
                masked[i][eligible[i]] = draws.to(sequence.device)[inverse]
                regimes.append("tied")
            else:
                offset = 0
                for t in (0, 1):
                    if states[t] != "active":
                        continue
                    p_t, w_t = _sample_noise(
                        tracks[t].noise, seed=seed, purpose=f"time_{t}"
                    )
                    p_pair[t] = p_t
                    w_pair[t] = w_t
                    span_p = float(tracks[t].get("span_prob", 0.0))
                    placement = (
                        "span"
                        if span_p > 0.0
                        and float(
                            torch.rand(
                                (),
                                generator=_generator(seed, f"span_{t}"),
                                dtype=torch.float64,
                            )
                        )
                        < span_p
                        else "token"
                    )
                    track_groups, offset = _partition_track(
                        eligible[i, :, t],
                        residue,
                        placement=placement,
                        span_mean=float(tracks[t].get("span_mean", 8.0)),
                        offset=offset,
                        generator=_generator(seed, f"partition_{t}"),
                    )
                    groups[i, :, t] = track_groups
                    track_elig = eligible[i, :, t]
                    active_ids, inverse = track_groups[track_elig].unique(
                        return_inverse=True
                    )
                    draws = (
                        torch.rand(
                            len(active_ids),
                            generator=_generator(seed, f"corruption_{t}"),
                            dtype=torch.float64,
                        )
                        < p_t
                    )
                    masked[i, :, t][track_elig] = draws.to(sequence.device)[inverse]
                regimes.append(f"{seq_state}_{struct_state}")
            probabilities.append(p_pair)
            loss_weights.append(w_pair)
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
    if not diagnostic and config.get("mode", "regimes") == "independent_tracks":
        return _corrupt_independent_tracks(batch, config, seeds=seeds)
    weights = torch.tensor(
        [float(config.regime_weights.get(name, 0)) for name in REGIMES],
        dtype=torch.float64,
    )
    weights = weights / weights.max()
    placement = config.placement if placement is None else placement
    span_mean = float(config.get("span_mean", 8))
    sequence = batch["sequence_tokens"].clone()
    structure = batch["structure_tokens"].clone()
    mask_id, codebook_size = batch["sequence_mask_id"], batch["codebook_size"]
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
                p, weight = _sample_noise(config.noise, seed=seed, purpose="time")
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
    weight = corruption["weight"]
    sequence = outputs["sequence_logits"]
    structure = outputs["structure_logits"]
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
            selected = masked[..., track]
            if track == 0:
                predictions = logits[selected].float().index_select(-1, aa)
                target = remap[targets[selected]]
            else:
                predictions, target = logits[selected].float(), targets[selected]
            # Empty reductions connect both heads, including empty local DDP ranks.
            ce = torch.nn.functional.cross_entropy(
                predictions, target, reduction="none"
            )
            ce_sums.append(ce.sum())
            track_weight = weight[:, track] if weight.ndim == 2 else weight
            weighted_sums.append(
                (ce * track_weight[:, None].expand(shape)[selected]).sum()
            )
            correct.append((predictions.argmax(-1) == target).sum())
    return {
        "weighted_sum": torch.stack(weighted_sums),
        "ce_sum": torch.stack(ce_sums),
        "correct": torch.stack(correct),
        "masked_count": masked.sum((0, 1)),
        "eligible_count": eligible.sum((0, 1)),
    }
