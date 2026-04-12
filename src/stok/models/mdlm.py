"""Masked Diffusion Language Model (MDLM) for protein sequences and structures."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.losses import MDLMLoss
from .encoder import Encoder
from .head import CodebookClassifier
from .noise_schedule import NoiseSchedule
from .stok import LMHead
from .time_embed import SinusoidalTimeEmbedding


def apply_subs(
    logits: torch.Tensor,
    input_tokens: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Apply SUBS parameterization constraints (dtype-safe).

    At masked positions: set logits for mask_token_id to -inf (the model
    should never predict the mask token itself).

    At unmasked positions: force the model to predict the input token by
    setting its logit to +large and all others to -large.

    Uses dtype-aware min/max values instead of float("-inf") to avoid
    NaN in mixed-precision training.

    Args:
        logits: Raw model logits, shape [B, L, V].
        input_tokens: Input token IDs (possibly noised), shape [B, L].
        mask: Boolean mask, True at masked (noised) positions, shape [B, L].
        mask_token_id: The mask token ID to suppress.

    Returns:
        Constrained logits, shape [B, L, V].
    """
    finfo = torch.finfo(logits.dtype)
    logits = logits.clone()

    # At masked positions: suppress the mask token logit
    # logits[mask, mask_token_id] = -inf
    mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
    mask_token_suppression = torch.zeros_like(logits)
    mask_token_suppression[:, :, mask_token_id] = finfo.min
    logits = logits + mask_expanded.float() * mask_token_suppression

    # At unmasked positions: force prediction to be the input token
    unmasked = ~mask  # [B, L]
    if unmasked.any():
        # Create one-hot for input tokens at unmasked positions
        V = logits.size(-1)
        # Set all logits at unmasked positions to -large
        logits[unmasked] = finfo.min
        # Set logit for the correct token to +large
        token_ids = input_tokens[unmasked]  # [N_unmasked]
        # Clamp to valid range to prevent index errors
        token_ids = token_ids.clamp(0, V - 1)
        logits[unmasked, token_ids] = finfo.max

    return logits


class MDLMModel(nn.Module):
    """Masked Diffusion Language Model for protein sequence (and structure).

    Supports two modes:
    - ``tracks="seq_only"``: Single-track sequence diffusion for stage 1
      pretraining.
    - ``tracks="joint"``: Two-track joint sequence + structure diffusion
      (implemented in Phase 3).

    Args:
        tracks: Operating mode, "seq_only" or "joint".
        seq_vocab_size: Sequence vocabulary size.
        seq_pad_id: Sequence padding token ID.
        seq_mask_id: Sequence mask token ID.
        codebook: Structure codebook tensor [C, d_code] (joint mode only).
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        ffn_mult: FFN hidden dimension multiplier.
        dropout: Residual dropout probability.
        attn_dropout: Attention dropout probability.
        norm_type: Normalization type ("layernorm" or "rmsnorm").
        noise_schedule_seq: Noise schedule for sequence track.
        noise_schedule_struct: Noise schedule for structure track (joint only).
        lambda_seq: Loss weight for sequence track.
        lambda_struct: Loss weight for structure track (joint only).
        classifier_kwargs: Kwargs for CodebookClassifier (joint only).
        tie_seq_embeddings: Tie LM head weights to sequence embedding.
        time_conditioning: Time conditioning mode for encoder ("adaln" or None).
        time_embed_dim: Dimension of time embedding. Defaults to d_model.
        time_combine: How to combine time embeddings ("sum" or "concat_project").
    """

    def __init__(
        self,
        tracks: str = "joint",
        seq_vocab_size: int = 32,
        seq_pad_id: int = 1,
        seq_mask_id: int = 31,
        codebook: torch.Tensor | None = None,
        d_model: int = 1536,
        n_heads: int = 24,
        n_layers: int = 36,
        ffn_mult: float = 2.667,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        norm_type: str = "layernorm",
        noise_schedule_seq: NoiseSchedule | None = None,
        noise_schedule_struct: NoiseSchedule | None = None,
        lambda_seq: float = 1.0,
        lambda_struct: float = 1.0,
        classifier_kwargs: dict | None = None,
        tie_seq_embeddings: bool = True,
        time_conditioning: str = "adaln",
        time_embed_dim: int | None = None,
        time_combine: str = "sum",
    ):
        super().__init__()
        if tracks not in ("seq_only", "joint"):
            raise ValueError(f"Unknown tracks mode: {tracks!r}")
        if noise_schedule_seq is None:
            raise ValueError("noise_schedule_seq is required")
        if tracks == "joint":
            raise NotImplementedError(
                "Joint mode is not yet implemented (Phase 3). Use tracks='seq_only'."
            )

        self.tracks = tracks
        self.seq_pad_id = seq_pad_id
        self.seq_mask_id = seq_mask_id
        self.lambda_seq = lambda_seq
        self.lambda_struct = lambda_struct

        _time_embed_dim = time_embed_dim or d_model

        # -- Sequence embedding --
        self.embed_seq = nn.Embedding(seq_vocab_size, d_model, padding_idx=seq_pad_id)
        nn.init.normal_(self.embed_seq.weight, mean=0.0, std=0.02)

        # -- Time embedding --
        self.time_embed = SinusoidalTimeEmbedding(_time_embed_dim)

        # -- Encoder (with time conditioning) --
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            ffn_mult=ffn_mult,
            norm_type=norm_type,
            time_conditioning=time_conditioning,
            time_embed_dim=_time_embed_dim,
        )

        # -- Sequence head --
        self.head_seq = LMHead(
            d_model=d_model,
            vocab_size=seq_vocab_size,
            tie_weights=self.embed_seq if tie_seq_embeddings else None,
        )

        # -- Loss --
        self.loss_fn_seq = MDLMLoss(noise_schedule_seq)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        t_seq: torch.Tensor,
        seq_targets: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        # Joint-mode args accepted but unused in seq_only:
        struct_tokens: torch.Tensor | None = None,
        t_struct: torch.Tensor | None = None,
        struct_targets: torch.Tensor | None = None,
        struct_mask: torch.Tensor | None = None,
        position_weights_seq: torch.Tensor | None = None,
        position_weights_struct: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass.

        Args:
            seq_tokens: Input sequence token IDs (possibly noised), shape [B, L].
            t_seq: Diffusion time for sequence track, shape [B].
            seq_targets: Ground-truth sequence tokens, shape [B, L].
            seq_mask: Boolean mask, True at masked positions, shape [B, L].
            key_padding_mask: True at padding positions, shape [B, L].
            struct_tokens: Structure tokens (joint mode, ignored in seq_only).
            t_struct: Diffusion time for structure track (joint mode).
            struct_targets: Ground-truth structure tokens (joint mode).
            struct_mask: Structure mask (joint mode).
            position_weights_seq: Per-position weights for seq loss (unused).
            position_weights_struct: Per-position weights for struct loss (unused).

        Returns:
            Dict with keys: "loss", "loss_seq", "seq_logits", and for joint
            mode also "loss_struct", "struct_logits".
        """
        # Infer padding mask from pad_id if not provided
        if key_padding_mask is None:
            key_padding_mask = seq_tokens == self.seq_pad_id  # [B, L]

        # 1. Embed sequence tokens
        h = self.embed_seq(seq_tokens)  # [B, L, d_model]

        # 2. Time embedding
        t_embed = self.time_embed(t_seq)  # [B, d_model]

        # 3. Encode with time conditioning
        h = self.encoder(h, key_padding_mask=key_padding_mask, t_embed=t_embed)

        # 4. Sequence head
        seq_logits = self.head_seq(h)  # [B, L, V]

        # 5. SUBS constraint
        if seq_mask is not None:
            seq_logits = apply_subs(seq_logits, seq_tokens, seq_mask, self.seq_mask_id)

        # 6. Loss
        loss_seq: torch.Tensor | None = None
        if seq_targets is not None and seq_mask is not None:
            loss_seq = self.loss_fn_seq(
                seq_logits, seq_targets, seq_mask, t_seq, key_padding_mask
            )
            loss_seq = self.lambda_seq * loss_seq

        return {
            "loss": loss_seq,
            "loss_seq": loss_seq,
            "seq_logits": seq_logits,
        }
