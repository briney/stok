"""Pretrained weight loading and transfer for MDLMModel."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Key mapping tables: (source_prefix, target_prefix)
_MDLM_SEQ_MAP: list[tuple[str, str]] = [
    ("embed_seq.", "embed_seq."),
    ("encoder.", "encoder."),
    ("head_seq.", "head_seq."),
    ("time_embed.", "time_embed."),
    ("loss_fn_seq.", "loss_fn_seq."),
]

_MLM_MAP: list[tuple[str, str]] = [
    ("embed.", "embed_seq."),
    ("encoder.", "encoder."),
    ("lm_head.", "head_seq."),
]

_CODEBOOK_MAP: list[tuple[str, str]] = [
    ("classifier.", "head_struct."),
    ("encoder.", "encoder."),
    ("embed.", "embed_seq."),
]

_MDLM_JOINT_MAP: list[tuple[str, str]] = [
    ("embed_seq.", "embed_seq."),
    ("embed_struct.", "embed_struct."),
    ("track_embed.", "track_embed."),
    ("encoder.", "encoder."),
    ("head_seq.", "head_seq."),
    ("head_struct.", "head_struct."),
    ("time_embed.", "time_embed."),
    ("loss_fn_seq.", "loss_fn_seq."),
    ("loss_fn_struct.", "loss_fn_struct."),
    ("time_combine_proj.", "time_combine_proj."),
]

_SOURCE_MAPS = {
    "mdlm_seq": _MDLM_SEQ_MAP,
    "mlm": _MLM_MAP,
    "codebook": _CODEBOOK_MAP,
    "mdlm_joint": _MDLM_JOINT_MAP,
}


def _map_key(
    source_key: str,
    key_map: list[tuple[str, str]],
) -> str | None:
    """Map a source checkpoint key to a target model key.

    Returns None if the key doesn't match any mapping prefix.
    """
    for src_prefix, tgt_prefix in key_map:
        if source_key.startswith(src_prefix):
            return tgt_prefix + source_key[len(src_prefix):]
    return None


def load_pretrained_weights(
    mdlm_model: nn.Module,
    checkpoint_path: str | Path,
    source_type: str = "mdlm_seq",
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load weights from a pretrained checkpoint into MDLMModel.

    Supports loading from different source checkpoint types with
    appropriate key remapping.

    Args:
        mdlm_model: Target MDLMModel instance.
        checkpoint_path: Path to the checkpoint file (.pt).
        source_type: Type of source checkpoint. One of:
            - ``"mdlm_seq"``: seq_only MDLMModel checkpoint.
            - ``"mlm"``: STokModel with MLM head.
            - ``"codebook"``: STokModel with codebook head.
            - ``"mdlm_joint"``: Full joint MDLMModel checkpoint.
        strict: If True, raise an error for unexpected keys. Default False.

    Returns:
        Tuple of (matched_keys, missing_keys) where matched_keys are
        target model keys that were loaded and missing_keys are target
        model keys that had no match in the source checkpoint.
    """
    if source_type not in _SOURCE_MAPS:
        raise ValueError(
            f"Unknown source_type: {source_type!r}. "
            f"Expected one of {sorted(_SOURCE_MAPS.keys())}"
        )

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract model state dict (handle wrapped checkpoints)
    if "model_state_dict" in checkpoint:
        source_state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        source_state = checkpoint["state_dict"]
    else:
        source_state = checkpoint

    key_map = _SOURCE_MAPS[source_type]
    target_state = mdlm_model.state_dict()

    matched_keys: list[str] = []
    unexpected_keys: list[str] = []
    new_state: dict[str, torch.Tensor] = {}

    for src_key, src_value in source_state.items():
        tgt_key = _map_key(src_key, key_map)
        if tgt_key is None:
            unexpected_keys.append(src_key)
            continue
        if tgt_key not in target_state:
            unexpected_keys.append(src_key)
            continue
        # Check shape compatibility
        if src_value.shape != target_state[tgt_key].shape:
            logger.warning(
                "Shape mismatch for %s -> %s: %s vs %s, skipping",
                src_key,
                tgt_key,
                src_value.shape,
                target_state[tgt_key].shape,
            )
            continue
        new_state[tgt_key] = src_value
        matched_keys.append(tgt_key)

    # Find missing keys (in target but not loaded)
    missing_keys = [k for k in target_state if k not in new_state]

    if unexpected_keys:
        logger.info(
            "Ignored %d unexpected keys from source checkpoint: %s",
            len(unexpected_keys),
            unexpected_keys[:10],
        )

    if missing_keys:
        logger.info(
            "Missing %d keys (will use random init): %s",
            len(missing_keys),
            missing_keys[:10],
        )

    if strict and unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys in checkpoint: {unexpected_keys}"
        )

    # Load matched weights
    mdlm_model.load_state_dict(new_state, strict=False)

    logger.info(
        "Loaded %d/%d parameters from %s (source_type=%s)",
        len(matched_keys),
        len(target_state),
        checkpoint_path.name,
        source_type,
    )

    return matched_keys, missing_keys


class EncoderFreezeHook:
    """Zeros encoder gradients for the first N optimizer steps.

    Register this as a post-backward hook to implement staged unfreezing
    of pretrained encoder weights during joint training.

    Args:
        model: The MDLMModel instance.
        freeze_steps: Number of optimizer steps to freeze encoder for.
    """

    def __init__(self, model: nn.Module, freeze_steps: int):
        self.model = model
        self.freeze_steps = freeze_steps
        self._current_step = 0
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def step(self) -> None:
        """Increment optimizer step counter. Call after each optimizer.step()."""
        self._current_step += 1
        if self._current_step >= self.freeze_steps:
            self.remove()
            logger.info(
                "Encoder unfrozen after %d optimizer steps", self._current_step
            )

    def register(self) -> None:
        """Register gradient zeroing hooks on encoder parameters."""
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            logger.warning("Model has no 'encoder' attribute; freeze hook not registered")
            return
        for param in encoder.parameters():
            hook = param.register_post_accumulate_grad_hook(self._zero_grad)
            self._hooks.append(hook)
        logger.info(
            "Encoder gradients will be zeroed for %d optimizer steps",
            self.freeze_steps,
        )

    def _zero_grad(self, param: torch.Tensor) -> None:
        """Zero gradients for frozen parameters."""
        if self._current_step < self.freeze_steps:
            param.grad = None

    def remove(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
