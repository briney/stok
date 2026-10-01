"""Time-independent paired sequence/structure masked-diffusion denoiser."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .encoder import Encoder


class STokMDLM(nn.Module):
    """Sum aligned modality embeddings and predict with tied linear heads.

    Structure IDs 0..C-1 are real codes; C, C+1, C+2 are input-only
    PAD, MASK, UNAVAILABLE. Sequence PAD determines attention padding:
    structure PAD at sequence BOS/EOS remains visible.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        codebook: Tensor,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: float,
        dropout: float,
        attn_dropout: float,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        if codebook.ndim != 2 or not codebook.numel():
            raise ValueError("codebook must be a nonempty matrix")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must be within the sequence vocabulary")
        self.pad_id = pad_id
        self.codebook_size = codebook.shape[0]
        self.structure_pad_id = self.codebook_size
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.structure_embed = nn.Embedding(
            self.codebook_size + 3, d_model, padding_idx=self.structure_pad_id
        )
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
            attn_dropout=attn_dropout,
            norm_type=norm_type,
        )
        self.sequence_bias = nn.Parameter(torch.zeros(vocab_size))
        self.structure_bias = nn.Parameter(torch.zeros(self.codebook_size))
        self.register_buffer("structure_codebook", codebook.detach().clone())

    def forward(
        self,
        sequence_tokens: Tensor,
        structure_tokens: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if sequence_tokens.ndim != 2 or structure_tokens.shape != sequence_tokens.shape:
            raise ValueError(
                "sequence and structure tokens must have matching [B,L] shapes"
            )
        for tokens, size in (
            (sequence_tokens, self.embed.num_embeddings),
            (structure_tokens, self.structure_embed.num_embeddings),
        ):
            if tokens.dtype not in (torch.int32, torch.int64):
                raise ValueError("token IDs must be int32 or int64 tensors")
            if ((tokens < 0) | (tokens >= size)).any():
                raise ValueError("token IDs are outside their vocabulary")
        padding = sequence_tokens.eq(self.pad_id)
        if (padding & structure_tokens.ne(self.structure_pad_id)).any():
            raise ValueError("sequence PAD must pair with structure PAD")
        if key_padding_mask is not None and (
            key_padding_mask.shape != padding.shape
            or key_padding_mask.dtype != torch.bool
            or not torch.equal(key_padding_mask, padding)
        ):
            raise ValueError("key_padding_mask must match sequence PAD positions")
        hidden = self.encoder(
            self.embed(sequence_tokens) + self.structure_embed(structure_tokens),
            key_padding_mask=padding,
        )
        return {
            "sequence_logits": F.linear(hidden, self.embed.weight, self.sequence_bias),
            "structure_logits": F.linear(
                hidden,
                self.structure_embed.weight[: self.codebook_size],
                self.structure_bias,
            ),
        }
