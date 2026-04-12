"""Masked Diffusion Language Model (MDLM) for protein sequences and structures."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..utils.losses import MDLMLoss
from .encoder import Encoder
from .head import CodebookClassifier
from .noise_schedule import NoiseSchedule
from .stok import LMHead
from .time_embed import SinusoidalTimeEmbedding

logger = logging.getLogger(__name__)


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
    V = logits.size(-1)

    # At masked positions: suppress the mask token logit
    # Only apply if mask_token_id is within the vocab range (struct track's
    # mask token may be outside the codebook classifier's output range)
    if mask_token_id < V:
        mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
        mask_token_suppression = torch.zeros_like(logits)
        mask_token_suppression[:, :, mask_token_id] = finfo.min
        logits = logits + mask_expanded.float() * mask_token_suppression

    # At unmasked positions: force prediction to be the input token
    unmasked = ~mask  # [B, L]
    if unmasked.any():
        # Set all logits at unmasked positions to -large
        logits[unmasked] = finfo.min
        # Set logit for the correct token to +large
        token_ids = input_tokens[unmasked]  # [N_unmasked]
        # Clamp to valid range to prevent index errors
        token_ids = token_ids.clamp(0, V - 1)
        logits[unmasked, token_ids] = finfo.max

    return logits


def _combine_losses(
    loss_seq: torch.Tensor | None,
    loss_struct: torch.Tensor | None,
) -> torch.Tensor | None:
    """Combine sequence and structure losses, handling None values."""
    if loss_seq is not None and loss_struct is not None:
        return loss_seq + loss_struct
    if loss_seq is not None:
        return loss_seq
    return loss_struct


class MDLMModel(nn.Module):
    """Masked Diffusion Language Model for protein sequence (and structure).

    Supports two modes:
    - ``tracks="seq_only"``: Single-track sequence diffusion for stage 1
      pretraining.
    - ``tracks="joint"``: Two-track joint sequence + structure diffusion
      for stage 2 training.

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
        if tracks == "joint" and codebook is None:
            raise ValueError("codebook is required for joint mode")
        if tracks == "joint" and noise_schedule_struct is None:
            raise ValueError("noise_schedule_struct is required for joint mode")

        self.tracks = tracks
        self.seq_pad_id = seq_pad_id
        self.seq_mask_id = seq_mask_id
        self.lambda_seq = lambda_seq
        self.lambda_struct = lambda_struct
        self.time_combine = time_combine

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

        # -- Joint mode components --
        if tracks == "joint":
            codebook_size = codebook.shape[0]  # type: ignore[union-attr]
            # struct_mask_id = codebook_size, struct_pad_id = codebook_size + 1
            self.struct_mask_id = codebook_size
            self.struct_pad_id = codebook_size + 1
            struct_vocab_size = codebook_size + 2  # codes + mask + pad

            # Structure embedding
            self.embed_struct = nn.Embedding(
                struct_vocab_size, d_model, padding_idx=self.struct_pad_id
            )
            embedding_init_std = (classifier_kwargs or {}).get(
                "embedding_init_std", 0.02
            )
            nn.init.normal_(self.embed_struct.weight, mean=0.0, std=embedding_init_std)
            # Zero out padding embedding
            with torch.no_grad():
                self.embed_struct.weight[self.struct_pad_id].zero_()

            # Track embeddings: 0 = seq, 1 = struct
            self.track_embed = nn.Embedding(2, d_model)
            nn.init.normal_(self.track_embed.weight, mean=0.0, std=0.02)

            # Structure head (CodebookClassifier)
            _cls_kwargs = dict(classifier_kwargs or {})
            _cls_kwargs.pop("embedding_init_std", None)
            self.head_struct = CodebookClassifier(
                d_in=d_model, codebook=codebook, **_cls_kwargs  # type: ignore[arg-type]
            )

            # Structure loss
            self.loss_fn_struct = MDLMLoss(noise_schedule_struct)  # type: ignore[arg-type]

            # Time combine projection (if concat_project mode)
            if time_combine == "concat_project":
                self.time_combine_proj = nn.Linear(2 * _time_embed_dim, _time_embed_dim)
        else:
            self.struct_mask_id = -1
            self.struct_pad_id = -1

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

        # 2. Add structure embedding and track embeddings (joint mode)
        if self.tracks == "joint" and struct_tokens is not None:
            h_struct = self.embed_struct(struct_tokens)  # [B, L, d_model]
            h = h + h_struct
            # Add track embeddings: index 0 for seq, index 1 for struct
            h = h + self.track_embed.weight[0] + self.track_embed.weight[1]

        # 3. Time embedding
        t_embed = self.time_embed(t_seq)  # [B, d_model]
        if self.tracks == "joint" and t_struct is not None:
            t_embed_struct = self.time_embed(t_struct)  # [B, d_model]
            if self.time_combine == "concat_project":
                t_embed = self.time_combine_proj(
                    torch.cat([t_embed, t_embed_struct], dim=-1)
                )
            else:
                # Default: sum
                t_embed = t_embed + t_embed_struct

        # 4. Encode with time conditioning
        h = self.encoder(h, key_padding_mask=key_padding_mask, t_embed=t_embed)

        # 5. Sequence head + SUBS
        seq_logits = self.head_seq(h)  # [B, L, V_seq]
        if seq_mask is not None:
            seq_logits = apply_subs(seq_logits, seq_tokens, seq_mask, self.seq_mask_id)

        # 6. Sequence loss
        loss_seq: torch.Tensor | None = None
        if seq_targets is not None and seq_mask is not None:
            loss_seq = self.loss_fn_seq(
                seq_logits, seq_targets, seq_mask, t_seq, key_padding_mask
            )
            loss_seq = self.lambda_seq * loss_seq

        # 7. Structure head + SUBS + loss (joint mode)
        loss_struct: torch.Tensor | None = None
        struct_logits: torch.Tensor | None = None
        if self.tracks == "joint":
            struct_logits = self.head_struct(h)  # [B, L, C]
            if struct_mask is not None and struct_tokens is not None:
                struct_logits = apply_subs(
                    struct_logits, struct_tokens, struct_mask, self.struct_mask_id
                )
            if (
                struct_targets is not None
                and struct_mask is not None
                and t_struct is not None
            ):
                loss_struct = self.loss_fn_struct(
                    struct_logits, struct_targets, struct_mask, t_struct, key_padding_mask
                )
                loss_struct = self.lambda_struct * loss_struct

        # 8. Combined loss
        loss = _combine_losses(loss_seq, loss_struct)

        return {
            "loss": loss,
            "loss_seq": loss_seq,
            "seq_logits": seq_logits,
            "loss_struct": loss_struct,
            "struct_logits": struct_logits,
        }
