from collections.abc import Mapping
from contextlib import contextmanager
import math
import random
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from .mdlm import mask_schedule

if TYPE_CHECKING:
    from ..data.mdlm import MDLMBatch

SAMPLING_ORDERS = ("random", "max_prob", "min_entropy")


def _cell_confidence(logits: tuple[Tensor, Tensor], index: int, order: str) -> Tensor:
    """Return [L, 2] per-cell confidence scores (higher is more confident)."""
    scores = []
    for values in logits:
        probs = values[index].float().softmax(-1)
        if order == "max_prob":
            scores.append(probs.max(-1).values)
        else:
            scores.append((probs * probs.clamp_min(1e-12).log()).sum(-1))
    return torch.stack(scores, dim=-1)


@contextmanager
def inference_context(model: nn.Module):
    """Keep inference isolated from training modes and global random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    )
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def sample_mdlm(
    model: nn.Module,
    batch: "MDLMBatch",
    *,
    generate_mask: Tensor,
    group_ids: Tensor,
    schedule: DictConfig,
    steps: int,
    seeds: list[int],
    canonical_aa_ids: Tensor,
    time_grid: Tensor | None = None,
    order: str = "random",
    temperature: float = 1.0,
) -> dict[str, Tensor]:
    """Reveal fixed composite groups, then carry their sampled members forever.

    ``schedule`` is the sampling noise config (name/power). Generation is
    independent of target validity. Joint requests need explicit checkpoint
    metadata at ``model.mdlm_regime_weights`` with a positive joint weight.
    Local CPU streams keep categorical/reveal draws independent of placement.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("MDLM sampling steps must be a positive integer")
    if order not in SAMPLING_ORDERS:
        raise ValueError(f"Unsupported MDLM sampling order: {order}")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("MDLM sampling temperature must be finite and >= 0")
    grid = (
        torch.linspace(1, 0, steps + 1, dtype=torch.float64)
        if time_grid is None
        else time_grid
    )
    probabilities, _ = mask_schedule(
        grid.cpu().double(), name=schedule.name, power=float(schedule.get("power", 2))
    )
    if (probabilities[:-1] <= 0).any():
        raise ValueError("MDLM time grid exceeds schedule numerical support")
    sequence, structure = (
        batch["sequence_tokens"].clone(),
        batch["structure_tokens"].clone(),
    )
    mask_id, size = batch["sequence_mask_id"], batch["codebook_size"]
    joint = generate_mask.any(1).all(-1)
    if joint.any():
        weights = getattr(model, "mdlm_regime_weights", {})
        if not isinstance(weights, Mapping) or not any(
            float(weights.get(name, 0)) > 0
            for name in ("joint_independent", "joint_tied")
        ):
            raise ValueError(
                "MDLM checkpoint is not qualified for joint generation; explicit joint training regime metadata required"
            )
    aa = canonical_aa_ids.to(sequence.device)
    sequence.masked_fill_(generate_mask[..., 0], mask_id)
    structure.masked_fill_(generate_mask[..., 1], size + 1)
    active = generate_mask.clone()
    total_groups = [
        int(group_ids[i][active[i]].unique().numel()) for i in range(len(seeds))
    ]
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    # Group membership is supplied once; active only records unrevealed members.
    num_steps = len(probabilities) - 1
    with inference_context(model):
        for step_index, (p_t, p_s) in enumerate(
            zip(probabilities[:-1], probabilities[1:])
        ):
            outputs = model(sequence_tokens=sequence, structure_tokens=structure)
            logits = (
                outputs["sequence_logits"].index_select(-1, aa),
                outputs["structure_logits"],
            )
            is_last = step_index == num_steps - 1
            for i, generator in enumerate(generators):
                active_mask = active[i]
                hidden_ids, inverse = group_ids[i][active_mask].unique(
                    return_inverse=True
                )
                if not hidden_ids.numel():
                    continue
                if order == "random":
                    reveal = (
                        torch.rand(
                            len(hidden_ids), generator=generator, dtype=torch.float64
                        )
                        < (p_t - p_s) / p_t
                    )
                    chosen_ids = hidden_ids[reveal.to(hidden_ids.device)]
                elif is_last or float(p_s) <= 0.0:
                    chosen_ids = hidden_ids
                else:
                    target_remaining = int(math.floor(total_groups[i] * float(p_s)))
                    reveal_count = max(0, len(hidden_ids) - target_remaining)
                    if reveal_count == 0:
                        continue
                    active_scores = _cell_confidence(logits, i, order)[active_mask]
                    sums = torch.zeros(
                        len(hidden_ids),
                        dtype=active_scores.dtype,
                        device=active_scores.device,
                    )
                    counts = torch.zeros_like(sums)
                    sums.scatter_add_(0, inverse, active_scores)
                    counts.scatter_add_(0, inverse, torch.ones_like(active_scores))
                    group_scores = sums / counts
                    topk = torch.topk(
                        group_scores, min(reveal_count, len(hidden_ids))
                    ).indices
                    chosen_ids = hidden_ids[topk]
                selected = torch.isin(group_ids[i], chosen_ids) & active_mask
                for track, (values, tokens) in enumerate(
                    zip(logits, (sequence, structure))
                ):
                    cells = selected[:, track]
                    if cells.any():
                        cell_logits = values[i, cells].float()
                        if temperature == 0:
                            draws = cell_logits.argmax(-1)
                        else:
                            if temperature != 1.0:
                                cell_logits = cell_logits / float(temperature)
                            # ponytail: CPU draws synchronize per sample; use device streams if throughput requires it.
                            probs = cell_logits.softmax(-1).cpu()
                            draws = (
                                torch.multinomial(probs, 1, generator=generator)
                                .squeeze(-1)
                                .to(tokens.device)
                            )
                        tokens[i, cells] = aa[draws] if track == 0 else draws
                active[i] &= ~selected
    return {"sequence_tokens": sequence, "structure_tokens": structure}
