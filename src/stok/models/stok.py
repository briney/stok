import torch
import torch.nn as nn

from ..utils.losses import fape_loss, token_ce_loss
from .encoder import Encoder
from .head import CodebookClassifier


class STokModel(nn.Module):
    """Encoder-only STōk model that predicts a structure token per residue."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: float,
        dropout: float,
        attn_dropout: float,
        codebook: torch.Tensor,
        classifier_kwargs: dict | None = None,
        norm_type: str = "layernorm",
    ):
        """Initialize STOK model.

        Args:
            vocab_size: Vocabulary size for input tokens.
            pad_id: Padding token ID.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers.
            ffn_mult: Feedforward multiplier (hidden_dim = d_model * ffn_mult).
            dropout: Dropout probability for residual connections.
            attn_dropout: Dropout probability for attention outputs.
            rope_base: RoPE base frequency.
            codebook: Codebook tensor of shape [C, d_code].
            classifier_kwargs: Optional keyword arguments for classifier.
            norm_type: Normalization type ("layernorm" currently supported).
        """
        super().__init__()
        self.pad_id = pad_id
        self.codebook_size = codebook.shape[0]
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            ffn_mult=ffn_mult,
            norm_type=norm_type,
        )
        self.classifier = CodebookClassifier(
            d_in=d_model,
            codebook=codebook,
            **(classifier_kwargs or {}),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        coords_loss_weight: float = 0.1,
        ignore_index: int = -100,
    ):
        """Forward pass through STOK model.

        Args:
            tokens: Input token IDs of shape [B, L].
            key_padding_mask: Padding mask of shape [B, L] where True marks
                padding positions. If None, inferred from pad_id. Defaults to None.
            labels: Target labels of shape [B, L]. Use ignore_index for
                ignored positions. Defaults to None.
            coords: Coordinates of shape [B, L, 3, 3] for N, CA, C atoms per residue.
                If None, the structure-based FAPE loss is not computed. Defaults to None.
            coords_loss_weight: Weight for the structure-based FAPE loss. Defaults to 0.1.
            ignore_index: Index to ignore in loss computation. Defaults to -100.

        Returns:
            Dictionary containing:
                - logits: Output logits of shape [B, L, C].
                - loss: Cross-entropy loss if labels provided, else None.
        """
        # embedding
        h = self.embed(tokens)  # [B, L, d_model]

        # if mask not provided, build from pad_id
        if key_padding_mask is None:
            key_padding_mask = tokens == self.pad_id  # [B, L], True = pad

        # encode
        h = self.encoder(
            h, key_padding_mask=key_padding_mask, attn_mask=None
        )  # [B, L, d_model]

        # classify
        logits = self.classifier(h)  # [B, L, C]

        loss = None
        classification_loss = None
        # structure_loss = None

        # token classification loss
        if labels is not None:
            classification_loss = token_ce_loss(
                logits=logits,
                labels=labels,
                ignore_index=ignore_index,
            )

        # # structure-based FAPE loss
        # if coords is not None:
        #     structure_loss = fape_loss(
        #         pred_coords=coords,
        #         true_coords=coords,
        #         residue_mask=~key_padding_mask,
        #     )

        # combine losses
        if classification_loss is not None:
            loss = classification_loss
        # if structure_loss is not None:
        #     if loss is None:
        #         loss = structure_loss * coords_loss_weight
        #     else:
        #         loss += structure_loss * coords_loss_weight

        return {
            "logits": logits,
            "loss": loss,
            "classification_loss": classification_loss,
            # "structure_loss": structure_loss,
        }
