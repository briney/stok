"""Coordinates-to-tokens structure encoder (GCP-VQVAE parity).

Implements the ``StructureEncoder``: a forward-pass-identical port of the
GCP-VQVAE encoder path (GCPNet → Conv1d → x-transformers Encoder → Conv1d →
VectorQuantize) so pretrained weights from ``Mahdip72/gcp-vqvae-{large,lite}``
load via ``load_state_dict`` after a one-time key remap.

Training-only machinery (VQ-EMA updates, k-means init, commitment loss) is NOT
ported — at inference, the VQ layer is a deterministic argmin lookup into the
frozen codebook buffer shipped with the converted checkpoint.

Per-preset download/cache/DDP logic mirrors :mod:`stok.models.decoder` so the
encoder and decoder behave symmetrically.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from x_transformers import ContinuousTransformerWrapper
from x_transformers import Encoder as XTEncoder

from stok.models.gcpnet import GCPNetModel
from stok.utils.batching import unbatch_and_pad
from stok.utils.featurizer import ProteinFeaturiser

__all__ = [
    "StructureEncoder",
    "load_pretrained_encoder",
    "ENCODER_URLS",
]


# ---------------------------------------------------------------------------
# GCPNet DictConfig (matches upstream config_gcpnet_encoder.yaml exactly)
# ---------------------------------------------------------------------------

_GCPNET_MODULE_CFG = {
    "norm_pos_diff": True,
    "scalar_gate": 0,
    "vector_gate": True,
    "scalar_nonlinearity": "silu",
    "vector_nonlinearity": "silu",
    "nonlinearities": ["silu", "silu"],
    "r_max": 10.0,
    "num_rbf": 8,
    "bottleneck": 4,
    "vector_linear": True,
    "vector_identity": True,
    "default_bottleneck": 4,
    "predict_node_positions": False,
    "predict_node_rep": True,
    "node_positions_weight": 1.0,
    "update_positions_with_vector_sum": False,
    "enable_e3_equivariance": False,
    "pool": "sum",
}

_GCPNET_MODEL_CFG = {
    "h_input_dim": 49,
    "chi_input_dim": 2,
    "e_input_dim": 9,
    "xi_input_dim": 1,
    "h_hidden_dim": 128,
    "chi_hidden_dim": 16,
    "e_hidden_dim": 32,
    "xi_hidden_dim": 4,
    "num_layers": 6,
    "dropout": 0.0,
}

_GCPNET_LAYER_CFG = {
    "pre_norm": False,
    "use_gcp_norm": True,
    "use_gcp_dropout": True,
    "use_scalar_message_attention": True,
    "num_feedforward_layers": 2,
    "dropout": 0.0,
    "nonlinearity_slope": 0.01,
    "mp_cfg": {
        "edge_encoder": False,
        "edge_gate": False,
        "num_message_layers": 4,
        "message_residual": 0,
        "message_ff_multiplier": 1,
        "self_message": True,
    },
}


def _build_gcpnet() -> GCPNetModel:
    """Instantiate :class:`GCPNetModel` with the frozen upstream config."""
    module_cfg = OmegaConf.create(_GCPNET_MODULE_CFG)
    model_cfg = OmegaConf.create(_GCPNET_MODEL_CFG)
    layer_cfg = OmegaConf.create(_GCPNET_LAYER_CFG)
    return GCPNetModel(
        num_layers=_GCPNET_MODEL_CFG["num_layers"],
        node_s_emb_dim=_GCPNET_MODEL_CFG["h_hidden_dim"],
        node_v_emb_dim=_GCPNET_MODEL_CFG["chi_hidden_dim"],
        edge_s_emb_dim=_GCPNET_MODEL_CFG["e_hidden_dim"],
        edge_v_emb_dim=_GCPNET_MODEL_CFG["xi_hidden_dim"],
        r_max=_GCPNET_MODULE_CFG["r_max"],
        num_rbf=_GCPNET_MODULE_CFG["num_rbf"],
        activation=_GCPNET_MODULE_CFG["scalar_nonlinearity"],
        pool=_GCPNET_MODULE_CFG["pool"],
        module_cfg=module_cfg,
        model_cfg=model_cfg,
        layer_cfg=layer_cfg,
    )


# ---------------------------------------------------------------------------
# StructureEncoder
# ---------------------------------------------------------------------------


class StructureEncoder(nn.Module):
    """Forward-pass-identical GCP-VQVAE encoder (coordinates → token indices).

    Attribute layout is chosen so the conversion script only needs to drop
    training-only keys and strip the upstream ``vqvae.`` prefix; the GCPNet
    portion is renamed from the upstream ``encoder.*`` prefix to ``gcpnet.*``
    to avoid the ambiguity of an ``encoder`` attribute inside a class also
    called an encoder.

    Attributes:
        featurizer: Builds node/edge feats on the graph batch in-place
            (zero-param; not present in the upstream checkpoint).
        gcpnet: Geometric message-passing encoder operating on Cα graphs.
        encoder_tail: Conv1d projection ``128 → d_model`` (wrapped in
            ``nn.Sequential`` to match upstream ``encoder_tail.0.*`` keys).
        encoder_blocks: ``x_transformers.ContinuousTransformerWrapper`` stack.
        encoder_head: Conv1d projection ``d_model → vqvae_dim`` (wrapped in
            ``nn.Sequential`` to match upstream ``encoder_head.0.*`` keys).
        vector_quantizer: ``vector_quantize_pytorch.VectorQuantize`` with
            inference-only settings; only ``_codebook.embed`` matters at eval.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        attn_kv_heads: int,
        ffn_mult: float = 4.0,
        max_length: int = 1280,
        vqvae_dim: int,
        codebook_size: int = 4096,
    ):
        super().__init__()
        # Lazy import so `stok` still imports when vector-quantize-pytorch is
        # not installed — only the structure encoder path needs it.
        from vector_quantize_pytorch import VectorQuantize

        self.d_model = d_model
        self.vqvae_dim = vqvae_dim
        self.max_length = max_length
        self.codebook_size = codebook_size

        self.featurizer = ProteinFeaturiser()
        self.gcpnet = _build_gcpnet()

        gcp_scalar_dim = _GCPNET_MODEL_CFG["h_hidden_dim"]
        self.encoder_tail = nn.Sequential(
            nn.Conv1d(gcp_scalar_dim, d_model, kernel_size=1),
        )
        self.encoder_blocks = ContinuousTransformerWrapper(
            dim_in=d_model,
            dim_out=d_model,
            max_seq_len=max_length,
            num_memory_tokens=0,
            attn_layers=XTEncoder(
                dim=d_model,
                ff_mult=ffn_mult,
                ff_glu=True,
                ff_swish=True,
                ff_no_bias=True,
                depth=n_layers,
                heads=n_heads,
                rotary_pos_emb=True,
                attn_flash=True,
                attn_kv_heads=attn_kv_heads,
                attn_qk_norm=True,
                pre_norm=True,
                residual_attn=False,
            ),
        )
        self.encoder_head = nn.Sequential(
            nn.Conv1d(d_model, vqvae_dim, kernel_size=1),
        )
        # kmeans_init=False so ``_codebook.initted`` is True at construction.
        # Upstream checkpoints ship with initted=True (they ran kmeans during
        # training), so stock/fresh instantiation needs the same convention
        # or the very first forward would fire kmeans and overwrite the
        # codebook. This also keeps unit tests (which run without pretrained
        # weights) deterministic across batch sizes.
        self.vector_quantizer = VectorQuantize(
            dim=vqvae_dim,
            codebook_size=codebook_size,
            kmeans_init=False,
        )

    @torch.inference_mode()
    def forward(
        self,
        graph,
        mask: torch.Tensor,
        nan_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode a graph batch to VQ code indices.

        Args:
            graph: PyG batch with ``coords``, ``residue_type``, ``seq_pos``,
                ``edge_index``, and the ``_slice_dict`` entries expected by
                :class:`ProteinFeaturiser`.
            mask: ``(B, L)`` key-padding mask where ``True`` marks real
                residues. ``L`` must equal ``self.max_length``.
            nan_mask: ``(B, L)`` mask where ``True`` marks residues whose
                Cα coordinates are finite.

        Returns:
            Dict with:
                ``indices`` ``(B, L)`` int64: VQ token indices (padded
                    positions hold whatever the codebook's nearest-neighbor
                    lookup yields; caller trims to true length).
                ``embeddings`` ``(B, L, vqvae_dim)``: quantized embeddings.
                ``valid`` ``(B, L)`` bool: ``mask & nan_mask`` combined.
        """
        valid = mask.to(torch.bool) & nan_mask.to(torch.bool)

        graph = self.featurizer(graph)
        gcp_out = self.gcpnet(graph)
        node_emb = gcp_out["node_embedding"]  # (ΣLᵢ, 128)

        # (ΣLᵢ, 128) → (B, max_length, 128)
        x = unbatch_and_pad(node_emb, graph.batch, self.max_length)

        # Conv1d expects (B, C, L) so transpose around each conv.
        x = x.transpose(1, 2)
        x = self.encoder_tail(x)
        x = x.transpose(1, 2)  # (B, L, d_model)

        x = self.encoder_blocks(x, mask=valid)  # (B, L, d_model)

        x = x.transpose(1, 2)
        x = self.encoder_head(x)
        x = x.transpose(1, 2)  # (B, L, vqvae_dim)

        quantized, indices, _ = self.vector_quantizer(x, mask=valid)

        return {"indices": indices, "embeddings": quantized, "valid": valid}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


_ENCODER_ARCH: dict[str, dict] = {
    "base": dict(
        d_model=1024,
        n_layers=12,
        n_heads=12,
        attn_kv_heads=3,
        ffn_mult=4.0,
        max_length=1280,
        vqvae_dim=256,
        codebook_size=4096,
    ),
    "lite": dict(
        d_model=1024,
        n_layers=8,
        n_heads=8,
        attn_kv_heads=2,
        ffn_mult=4.0,
        max_length=1280,
        vqvae_dim=128,
        codebook_size=4096,
    ),
}


ENCODER_URLS: dict[str, str] = {
    "base": os.environ.get(
        "STOK_ENCODER_BASE_URL",
        "https://huggingface.co/brineylab/STok/resolve/main/encoder.base.pth",
    ),
    "lite": os.environ.get(
        "STOK_ENCODER_LITE_URL",
        "https://huggingface.co/brineylab/STok/resolve/main/encoder.lite.pth",
    ),
}


# ---------------------------------------------------------------------------
# Download / cache helpers (mirror stok/models/decoder.py)
# ---------------------------------------------------------------------------


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        with test_file.open("wb") as f:
            f.write(b"ok")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _resolve_cache_dir() -> Path:
    env_dir = os.environ.get("STOK_ENCODER_CACHE")
    if env_dir:
        p = Path(env_dir)
        if _is_writable_dir(p):
            return p

    try:
        stok_pkg = importlib.import_module("stok")
        pkg_dir = Path(stok_pkg.__file__).resolve().parent / "checkpoints" / "encoder"
        if _is_writable_dir(pkg_dir):
            return pkg_dir
    except Exception:
        pass

    user_cache_root = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    user_dir = Path(user_cache_root) / "stok" / "encoder"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def _ensure_downloaded(preset: str, *, progress: bool = True) -> Path:
    cache_dir = _resolve_cache_dir()
    local_name = f"encoder-{preset}.pt"
    local_path = cache_dir / local_name
    url = ENCODER_URLS.get(preset)
    if url is None:
        raise ValueError(f"No URL registered for encoder preset: {preset}")

    if local_path.exists() and local_path.is_file():
        return local_path

    ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if ddp else 0

    def _download_to_final() -> None:
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        try:
            torch.hub.download_url_to_file(url, str(tmp_path), progress=progress)
            if tmp_path.exists():
                tmp_path.replace(local_path)
            elif not local_path.exists():
                raise FileNotFoundError(
                    f"Encoder checkpoint missing after download: {tmp_path} -> {local_path}"
                )
        finally:
            tmp_path.unlink(missing_ok=True)

    if ddp:
        if (rank == 0) and (not local_path.exists()):
            _download_to_final()
        dist.barrier()
        if not local_path.exists():
            raise FileNotFoundError(
                f"Encoder checkpoint not found after distributed download: {local_path}"
            )
        return local_path

    _download_to_final()
    return local_path


def load_pretrained_encoder(
    preset: Literal["base", "lite"] = "base",
    *,
    path: str | None = None,
    device: torch.device | str = "cpu",
    freeze: bool = True,
    progress: bool = True,
) -> StructureEncoder:
    """Instantiate a :class:`StructureEncoder` from pretrained weights.

    Args:
        preset: Built-in preset name (``"base"`` or ``"lite"``).
        path: Local checkpoint path. If set, skips download logic.
        device: Device to move the model onto.
        freeze: If True, sets eval mode and disables gradients.
        progress: Show download progress when fetching weights.

    Returns:
        A :class:`StructureEncoder` with weights loaded and moved to ``device``.
    """
    if preset not in _ENCODER_ARCH:
        raise ValueError(f"Unsupported preset: {preset}. Choose 'base' or 'lite'.")

    if path is not None:
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    else:
        ckpt_path = _ensure_downloaded(preset, progress=progress)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    arch = _ENCODER_ARCH[preset]
    model = StructureEncoder(**arch)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys while loading encoder.{preset}: {unexpected[:8]}..."
        )

    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model.to(device)
