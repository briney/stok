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
    if (
        not isinstance(grid, Tensor)
        or grid.ndim != 1
        or len(grid) != steps + 1
        or not grid.is_floating_point()
        or not torch.isfinite(grid).all()
        or grid[0] != 1
        or grid[-1] != 0
        or not (grid.diff() < 0).all()
    ):
        raise ValueError(
            "MDLM time grid must strictly descend from 1 to 0 with steps+1 entries"
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
    shape = sequence.shape
    if sequence.ndim != 2 or not shape[0] or structure.shape != shape:
        raise ValueError("MDLM sampling requires a nonempty aligned batch")
    if sequence.device != structure.device:
        raise ValueError("MDLM sampling tensors must share a device")
    if any(tokens.dtype != torch.long for tokens in (sequence, structure)):
        raise ValueError("MDLM sampling tokens must be int64")
    if (
        generate_mask.shape != (*shape, 2)
        or generate_mask.dtype != torch.bool
        or group_ids.shape != generate_mask.shape
        or group_ids.dtype != torch.long
        or generate_mask.device != sequence.device
        or group_ids.device != sequence.device
        or (group_ids < -1).any()
        or not torch.equal(group_ids >= 0, generate_mask)
    ):
        raise ValueError(
            "MDLM groups must exactly cover requested outputs, excluding clamped cells"
        )
    for name in ("residue_mask", "sequence_valid", "structure_valid"):
        mask = batch[name]
        if (
            mask.shape != shape
            or mask.dtype != torch.bool
            or mask.device != sequence.device
        ):
            raise ValueError("MDLM sampling masks must be aligned boolean tensors")
    if (generate_mask & ~batch["residue_mask"][..., None]).any():
        raise ValueError("MDLM generation can only request residue slots")
    if len(seeds) != shape[0] or any(type(seed) is not int for seed in seeds):
        raise ValueError("MDLM sampling requires one integer seed per sample")
    mask_id, size = batch["sequence_mask_id"], batch["codebook_size"]
    embed = getattr(model, "embed", None)
    pad_id = getattr(model, "pad_id", None)
    model_size = getattr(model, "codebook_size", None)
    if (
        not isinstance(embed, nn.Embedding)
        or type(pad_id) is not int
        or type(model_size) is not int
    ):
        raise ValueError("MDLM model embedding or vocabulary metadata is invalid")
    vocab = embed.num_embeddings
    if (
        type(mask_id) is not int
        or not 0 <= mask_id < vocab
        or mask_id == pad_id
        or type(size) is not int
        or size < 1
        or size != model_size
        or ((sequence < 0) | (sequence >= vocab)).any()
        or ((structure < 0) | (structure >= size + 3)).any()
        or (sequence.eq(pad_id) & structure.ne(size)).any()
        or (batch["residue_mask"] & sequence.eq(pad_id)).any()
        or (sequence.eq(mask_id) & ~generate_mask[..., 0]).any()
        or (structure.eq(size + 1) & ~generate_mask[..., 1]).any()
    ):
        raise ValueError("MDLM input tokens or mask/codebook metadata are invalid")
    aa = canonical_aa_ids
    if (
        aa.ndim != 1
        or aa.dtype != torch.long
        or not aa.numel()
        or ((aa < 0) | (aa >= vocab)).any()
        or aa.unique().numel() != aa.numel()
        or ((aa == mask_id) | (aa == pad_id)).any()
        or torch.isin(aa, sequence[~batch["residue_mask"]].to(aa.device)).any()
    ):
        raise ValueError(
            "MDLM canonical output IDs must be distinct clean vocabulary IDs"
        )
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
    aa = aa.to(sequence.device)
    sequence.masked_fill_(generate_mask[..., 0], mask_id)
    structure.masked_fill_(generate_mask[..., 1], size + 1)
    active = generate_mask.clone()
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    # Group membership is supplied once; active only records unrevealed members.
    with inference_context(model):
        for p_t, p_s in zip(probabilities[:-1], probabilities[1:]):
            outputs = model(sequence_tokens=sequence, structure_tokens=structure)
            sequence_logits, structure_logits = (
                outputs["sequence_logits"],
                outputs["structure_logits"],
            )
            if sequence_logits.shape != (*shape, vocab) or structure_logits.shape != (
                *shape,
                size,
            ):
                raise ValueError("MDLM sampling output vocabulary shapes are invalid")
            logits = (sequence_logits.index_select(-1, aa), structure_logits)
            if any(
                not torch.isfinite(values[active[..., track]]).all()
                for track, values in enumerate(logits)
            ):
                raise FloatingPointError("Nonfinite requested MDLM predictions")
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
    if active.any():
        raise RuntimeError("MDLM sampling failed to complete requested outputs")
    return {"sequence_tokens": sequence, "structure_tokens": structure}
