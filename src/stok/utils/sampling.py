from collections.abc import Mapping
from contextlib import contextmanager
import random
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from .mdlm import mask_schedule

if TYPE_CHECKING:
    from ..data.mdlm import MDLMBatch


def top_p_sample(
    logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0
) -> torch.Tensor:
    # logits: [B, L, C] -> sampled indices: [B, L]
    if temperature != 1.0:
        logits = logits / temperature
    B, L, C = logits.shape
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    to_remove = cumprobs > p
    to_remove[..., 0] = False  # keep at least the top-1
    sorted_probs = torch.where(to_remove, torch.zeros_like(sorted_probs), sorted_probs)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    ranks = torch.multinomial(sorted_probs.reshape(-1, C), 1).view(B, L, 1)
    sampled = torch.gather(sorted_idx, -1, ranks).squeeze(-1)
    return sampled


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
) -> dict[str, Tensor]:
    """Reveal fixed composite groups, then carry their sampled members forever.

    ``schedule`` is the sampling noise config (name/power). Generation is
    independent of target validity. Joint requests need explicit checkpoint
    metadata at ``model.mdlm_regime_weights`` with a positive joint weight.
    Local CPU streams keep categorical/reveal draws independent of placement.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("MDLM sampling steps must be a positive integer")
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
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    # Group membership is supplied once; active only records unrevealed members.
    with inference_context(model):
        for p_t, p_s in zip(probabilities[:-1], probabilities[1:]):
            outputs = model(sequence_tokens=sequence, structure_tokens=structure)
            logits = (
                outputs["sequence_logits"].index_select(-1, aa),
                outputs["structure_logits"],
            )
            for i, generator in enumerate(generators):
                hidden_ids = group_ids[i][active[i]].unique()
                reveal = (
                    torch.rand(
                        len(hidden_ids), generator=generator, dtype=torch.float64
                    )
                    < (p_t - p_s) / p_t
                )
                selected = (
                    torch.isin(group_ids[i], hidden_ids[reveal.to(hidden_ids.device)])
                    & active[i]
                )
                for track, (values, tokens) in enumerate(
                    zip(logits, (sequence, structure))
                ):
                    cells = selected[:, track]
                    if cells.any():
                        # ponytail: CPU draws synchronize per sample; use device streams if throughput requires it.
                        probs = values[i, cells].float().softmax(-1).cpu()
                        draws = (
                            torch.multinomial(probs, 1, generator=generator)
                            .squeeze(-1)
                            .to(tokens.device)
                        )
                        tokens[i, cells] = aa[draws] if track == 0 else draws
                active[i] &= ~selected
    return {"sequence_tokens": sequence, "structure_tokens": structure}
