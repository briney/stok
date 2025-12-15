import torch
import torch.nn as nn

from ..utils.losses import token_ce_loss
from .encoder import Encoder
from .head import CodebookClassifier


class LMHead(nn.Module):
    """Simple language modeling head for MLM pre-training."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: nn.Embedding | None = None,
    ):
        """Initialize LM head.

        Args:
            d_model: Model dimension.
            vocab_size: Vocabulary size for output logits.
            tie_weights: Optional embedding to tie weights with. If provided,
                the decoder weights will be shared with the embedding weights.
        """
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

        if tie_weights is not None:
            # Weight tying with input embeddings
            self.decoder = nn.Linear(d_model, vocab_size, bias=False)
            self.decoder.weight = tie_weights.weight
        else:
            self.decoder = nn.Linear(d_model, vocab_size)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass through LM head.

        Args:
            h: Hidden states of shape [B, L, d_model].

        Returns:
            Logits tensor of shape [B, L, vocab_size].
        """
        h = self.dense(h)
        h = self.activation(h)
        h = self.layer_norm(h)
        return self.decoder(h) + self.bias


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
        codebook: torch.Tensor | None = None,
        classifier_kwargs: dict | None = None,
        norm_type: str = "layernorm",
        head_type: str = "codebook",
        tie_word_embeddings: bool = True,
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
            codebook: Codebook tensor of shape [C, d_code]. Required for head_type="codebook".
            classifier_kwargs: Optional keyword arguments for classifier.
            norm_type: Normalization type ("layernorm" currently supported).
            head_type: Type of output head ("codebook" or "mlm").
            tie_word_embeddings: Whether to tie LM head weights to input embeddings (MLM only).
        """
        super().__init__()
        self.pad_id = pad_id
        self.head_type = head_type
        self.vocab_size = vocab_size

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

        # Initialize appropriate head based on head_type
        if head_type == "codebook":
            if codebook is None:
                raise ValueError("codebook is required when head_type='codebook'")
            self.codebook_size = codebook.shape[0]
            self.classifier = CodebookClassifier(
                d_in=d_model,
                codebook=codebook,
                **(classifier_kwargs or {}),
            )
            self.lm_head = None
        elif head_type == "mlm":
            self.codebook_size = None
            self.classifier = None
            self.lm_head = LMHead(
                d_model=d_model,
                vocab_size=vocab_size,
                tie_weights=self.embed if tie_word_embeddings else None,
            )
        else:
            raise ValueError(
                f"Unknown head_type: {head_type}. Expected 'codebook' or 'mlm'."
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
                - logits: Output logits of shape [B, L, C] or [B, L, vocab_size].
                - loss: Cross-entropy loss if labels provided, else None.
                - classification_loss: Same as loss (for compatibility).
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

        # classify based on head type
        if self.head_type == "codebook":
            logits = self.classifier(h)  # [B, L, C]
        else:  # mlm
            logits = self.lm_head(h)  # [B, L, vocab_size]

        loss = None
        classification_loss = None

        # token classification/MLM loss
        if labels is not None:
            classification_loss = token_ce_loss(
                logits=logits,
                labels=labels,
                ignore_index=ignore_index,
            )

        # combine losses
        if classification_loss is not None:
            loss = classification_loss

        return {
            "logits": logits,
            "loss": loss,
            "classification_loss": classification_loss,
        }
