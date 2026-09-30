"""Independently callable GCP-VQVAE encoder and complete vector quantizer."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from vector_quantize_pytorch import VectorQuantize
from x_transformers import ContinuousTransformerWrapper, Encoder

from .gcpnet import GCPNetModel
from ..utils.featurizer import ProteinFeaturiser
from ..utils.pretrained import (
    extract_gcp_component,
    load_gcp_config,
    read_gcp_checkpoint,
    resolve_gcp_artifact,
)


class GCPVQEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.featuriser = ProteinFeaturiser(**config["features"])
        self.encoder = GCPNetModel(
            **{
                key: OmegaConf.create(value) if isinstance(value, dict) else value
                for key, value in config["gcp"].items()
            }
        )
        settings = config["encoder"]
        if settings.get("causal", False):
            raise ValueError("Released GCP-VQVAE encoders require noncausal attention")
        width = settings["dimension"]
        self.max_length = config["max_length"]
        self.encoder_tail = nn.Sequential(
            nn.Conv1d(config["gcp"]["model_cfg"]["h_hidden_dim"], width, 1)
        )
        self.encoder_blocks = ContinuousTransformerWrapper(
            dim_in=width,
            dim_out=width,
            max_seq_len=self.max_length,
            num_memory_tokens=settings["num_memory_tokens"],
            attn_layers=Encoder(
                dim=width,
                ff_mult=settings["ff_mult"],
                ff_glu=True,
                ff_swish=True,
                ff_no_bias=True,
                depth=settings["depth"],
                heads=settings["heads"],
                rotary_pos_emb=settings["rotary_pos_emb"],
                attn_flash=settings["attn_flash"],
                attn_kv_heads=settings["attn_kv_heads"],
                attn_qk_norm=settings["qk_norm"],
                pre_norm=settings["pre_norm"],
                residual_attn=settings["residual_attn"],
            ),
        )
        self.encoder_head = nn.Sequential(
            nn.Conv1d(width, config["quantizer"]["dim"], 1)
        )

    def forward(self, graph: Batch, *, token_mask: torch.Tensor) -> torch.Tensor:
        if (
            token_mask.ndim != 2
            or token_mask.dtype != torch.bool
            or token_mask.size(1) > self.max_length
        ):
            raise ValueError(
                "token_mask must be boolean [B,L] within configured max_length"
            )
        if not token_mask.any(dim=1).all():
            raise ValueError("Each structure must have at least one available token")
        batch_size, length = token_mask.shape
        data = cast(Data, graph)
        positions = data.residue_index
        batches = data.batch
        if (
            positions is None
            or batches is None
            or positions.dtype != torch.int64
            or batches.dtype != torch.int64
        ):
            raise ValueError("Graph residue_index and batch must be long tensors")
        if (
            positions.shape != batches.shape
            or positions.ndim != 1
            or not positions.numel()
        ):
            raise ValueError("Graph residue_index must contain one position per node")
        if (
            graph.num_graphs != batch_size
            or (positions < 0).any()
            or (positions >= length).any()
            or (batches < 0).any()
            or (batches >= batch_size).any()
        ):
            raise ValueError(
                "Graph node positions/batches do not match the dense masks"
            )
        if torch.unique(batches * length + positions).numel() != positions.numel():
            raise ValueError("Duplicate graph node-to-residue positions")
        available = torch.zeros_like(token_mask)
        available[batches, positions] = True
        if (token_mask & ~available).any():
            raise ValueError("Token labels require a corresponding graph node")
        # The existing backbone/featurizer mutate graph attributes; callers retain their input.
        features = self.encoder(self.featuriser(data.clone()))["node_embedding"]
        dense = features.new_zeros(batch_size, length, features.size(-1))
        dense[batches, positions] = features
        projected = self.encoder_tail(dense.transpose(1, 2)).transpose(1, 2)
        transformed = self.encoder_blocks(projected, mask=token_mask)
        return self.encoder_head(transformed.transpose(1, 2)).transpose(1, 2)


class GCPVQTokenizer(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.encoder = GCPVQEncoder(config)
        self.quantizer = VectorQuantize(**config["quantizer"])

    def forward(
        self, graph: Batch, *, residue_mask: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            residue_mask.ndim != 2
            or residue_mask.dtype != torch.bool
            or token_mask.dtype != torch.bool
            or token_mask.shape != residue_mask.shape
        ):
            raise ValueError(
                "Residue/token masks must be matching boolean [B,L] tensors"
            )
        if (token_mask & ~residue_mask).any():
            raise ValueError("token_mask must be a subset of residue_mask")
        latents = self.encoder(graph, token_mask=token_mask)
        return self.quantizer(latents, mask=token_mask)

    @torch.inference_mode()
    def encode(
        self, graph: Batch, *, residue_mask: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            raise ValueError("Inference encoding requires evaluation mode")
        if not cast(torch.Tensor, self.quantizer._codebook.initted).all():
            raise ValueError("Inference requires an initialized quantizer")
        return self(graph, residue_mask=residue_mask, token_mask=token_mask)[1]


def load_pretrained_tokenizer(
    preset: str = "lite",
    *,
    path: str | Path | None = None,
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> GCPVQTokenizer:
    model = GCPVQTokenizer(load_gcp_config(preset))
    state = read_gcp_checkpoint(
        path if path is not None else resolve_gcp_artifact(preset)
    )
    model.encoder.load_state_dict(extract_gcp_component(state, "encoder"), strict=True)
    model.quantizer.load_state_dict(
        extract_gcp_component(state, "quantizer"), strict=True
    )
    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model.to(device)
