import importlib
import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder

from .head import Dim6RotStructureHead

__all__ = [
    "GeometricDecoder",
    "load_pretrained_decoder",
]


class GeometricDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: float,
        max_length: int,
        d_code: int,
        num_memory_tokens: int = 0,
        attn_kv_heads: int = 2,
    ):
        super(GeometricDecoder, self).__init__()

        self.decoder_output_scaling_factor = 1.0

        # input projection (from codebook space to decoder space)
        self.projector_in = nn.Linear(d_code, d_model, bias=False)

        # decoder stack
        # the original GCP-VQVAE model uses ContinuousTransformerWrapper from the x-transformers library.
        # in order to simplify loading the weights from the decoder of that model, we do the same here.
        self.decoder_stack = ContinuousTransformerWrapper(
            dim_in=d_model,
            dim_out=d_model,
            max_seq_len=max_length,
            num_memory_tokens=num_memory_tokens,
            attn_layers=Encoder(
                dim=d_model,
                ff_mult=ffn_mult,
                ff_glu=True,  # gate-based feed-forward (GLU family)
                ff_swish=True,  # use Swish instead of GELU → SwiGLU
                ff_no_bias=True,  # removes the two Linear biases in SwiGLU / MLP
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

        # output projection
        self.affine_output_projection = Dim6RotStructureHead(
            d_model,
            trans_scale_factor=1.0,
            predict_torsion_angles=False,
        )

    def forward(
        self,
        structure_tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        true_lengths: torch.Tensor | None = None,
    ):
        x = self.projector_in(structure_tokens)

        decoder_mask_bool = mask.to(torch.bool)
        x = self.decoder_stack(x, mask=decoder_mask_bool)

        bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(mask), preds_only=True
        )

        return bb_pred.flatten(-2) * self.decoder_output_scaling_factor


# decoder architecture presets (single source of truth)
_DECODER_ARCH = {
    "base": dict(
        d_model=1024,
        ffn_mult=4.0,
        n_layers=16,
        n_heads=16,
        attn_kv_heads=1,
        num_memory_tokens=0,
        max_length=1280,
    ),
    "lite": dict(
        d_model=1024,
        ffn_mult=4.0,
        n_layers=12,
        n_heads=8,
        attn_kv_heads=2,
        num_memory_tokens=0,
        max_length=1280,
    ),
}

DECODER_URLS: dict[str, str] = {
    "base": os.environ.get(
        "STOK_DECODER_BASE_URL",
        "https://huggingface.co/brineylab/STok/resolve/main/decoder.base.pth",
    ),
    "lite": os.environ.get(
        "STOK_DECODER_LITE_URL",
        "https://huggingface.co/brineylab/STok/resolve/main/decoder.lite.pth",
    ),
}


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
    # env can override the default cache directory
    env_dir = os.environ.get("STOK_DECODER_CACHE")
    if env_dir:
        p = Path(env_dir)
        if _is_writable_dir(p):
            return p

    # use default cache directory (in the stok package)
    try:
        stok_pkg = importlib.import_module("stok")
        pkg_dir = Path(stok_pkg.__file__).resolve().parent / "checkpoints" / "decoder"
        if _is_writable_dir(pkg_dir):
            return pkg_dir
    except Exception:
        pass

    # if the default cache directory is not writable, use the user cache directory
    user_cache_root = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    user_dir = Path(user_cache_root) / "stok" / "decoder"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def _ensure_downloaded(preset: str, *, progress: bool = True) -> Path:
    cache_dir = _resolve_cache_dir()
    local_name = f"decoder-{preset}.pt"
    local_path = cache_dir / local_name
    url = DECODER_URLS.get(preset)

    # if the weights file already exists, return the existing weights file
    if local_path.exists() and local_path.is_file():
        return local_path

    # download to temp then move to the local path
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    try:
        torch.hub.download_url_to_file(
            url,
            str(tmp_path),
            progress=progress,
        )
        tmp_path.replace(local_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return local_path


def load_pretrained_decoder(
    preset: Literal["base", "lite"] = "base",
    *,
    path: str | None = None,
    device: torch.device | str = "cpu",
    freeze: bool = True,
    progress: bool = True,
) -> GeometricDecoder:
    """Instantiate a (optionally frozen) GeometricDecoder from pre-trained weights.

    When a custom ``path`` is provided, it overrides download/caching but the
    ``preset`` still determines the architecture hyperparameters.

    Args:
        preset: Built-in preset name ("base" or "lite").
        path: Local checkpoint path. If set, skips download logic.
        device: Device to move the model to.
        freeze: If True, sets eval mode and disables gradients.
        progress: Show download progress when fetching weights.

    Returns:
        An initialized ``GeometricDecoder`` with weights loaded.
    """
    if preset not in _DECODER_ARCH:
        raise ValueError(f"Unsupported preset: {preset}. Choose 'base' or 'lite'.")

    if path is not None:
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {path}")
    else:
        ckpt_path = _ensure_downloaded(preset, progress=progress)

    # Load state dict on CPU first
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # Infer input dimensionality from projector weights
    proj_w = state_dict.get("projector_in.weight")
    if proj_w is None:
        raise KeyError(
            "Checkpoint missing 'projector_in.weight'; cannot infer d_code/d_model."
        )
    inferred_d_model, inferred_d_code = int(proj_w.shape[0]), int(proj_w.shape[1])

    arch = _DECODER_ARCH[preset]
    if arch["d_model"] != inferred_d_model:
        raise RuntimeError(
            f"Checkpoint d_model={inferred_d_model} does not match preset '{preset}'"
            f" (expected {arch['d_model']})."
        )

    model = GeometricDecoder(
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        ffn_mult=float(arch["ffn_mult"]),
        max_length=int(arch["max_length"]),
        d_code=inferred_d_code,
        num_memory_tokens=int(arch["num_memory_tokens"]),
        attn_kv_heads=int(arch["attn_kv_heads"]),
    )

    # strict=True will raise on any mismatch; no extra checks necessary
    model.load_state_dict(state_dict, strict=True)

    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model.to(device)
