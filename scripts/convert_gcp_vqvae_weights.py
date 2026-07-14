"""Convert upstream GCP-VQVAE checkpoints to ``stok`` StructureEncoder weights.

One-time migration script that downloads an upstream
``Mahdip72/gcp-vqvae-{large,lite}`` checkpoint from HuggingFace Hub, drops
training-only and decoder-side keys, strips the ``vqvae.`` prefix for the
VQ-transformer portion, and renames the GCPNet portion from ``encoder.*`` to
``gcpnet.*`` so the resulting state dict loads strictly into
:class:`stok.models.structure_encoder.StructureEncoder`.

The converted checkpoints are intended to be re-hosted at
``huggingface.co/brineylab/STok/encoder.{base,lite}.pth`` so the runtime
``load_pretrained_encoder`` helper can pull them directly — same pattern
as the existing :class:`stok.models.decoder.GeometricDecoder`.

Usage::

    python scripts/convert_gcp_vqvae_weights.py --preset base --out encoder.base.pth
    python scripts/convert_gcp_vqvae_weights.py --preset lite --out encoder.lite.pth

The script also verifies codebook equality against the bundled
``src/stok/checkpoints/codebook/{base,lite}.pt`` tensors (which should
already match upstream) and runs a strict-load sanity check before writing.
"""

from __future__ import annotations

import argparse
import importlib.resources as ir
import sys
from pathlib import Path

import torch

# Training-only / decoder-only / ESM keys that must be dropped.
_DROP_PREFIXES = (
    "vqvae.decoder.",
    "vqvae.ntp_",
    "vqvae.markov_",
    "vqvae.ti_tok_",
    "protein_encoder.",
)
# Only ``vqvae.vector_quantizer.zero`` is never persistent in stok's VQ.
# The other three ``_codebook.*`` buffers ARE registered by
# ``vector_quantize_pytorch.VectorQuantize`` regardless of ``kmeans_init`` /
# ``ema_update`` settings, so we pass them through — including the
# ``initted`` flag which must remain True so the runtime VQ skips its
# kmeans-init branch on first forward.
_DROP_EXACT = {
    "vqvae.vector_quantizer.zero",
}

# ``(src, dst)`` pairs for strip-and-rename. Processed in order; the first
# match wins. Released checkpoints wrap GCPNet as ``encoder.encoder.*``;
# older/synthetic checkpoints use ``encoder.*`` directly. Both map to the
# single ``gcpnet.*`` module in stok's ``StructureEncoder``.
_RENAME_RULES: list[tuple[str, str]] = [
    ("vqvae.encoder_tail.", "encoder_tail."),
    ("vqvae.encoder_blocks.", "encoder_blocks."),
    ("vqvae.encoder_head.", "encoder_head."),
    ("vqvae.vector_quantizer.", "vector_quantizer."),
    ("encoder.featuriser.", "featurizer."),
    ("encoder.encoder.", "gcpnet."),
    ("encoder.", "gcpnet."),
]


_PRESET_TO_REPO = {
    "base": "Mahdip72/gcp-vqvae-large",
    "lite": "Mahdip72/gcp-vqvae-lite",
}


def _unwrap_state_dict(raw) -> dict:
    """Strip common training-checkpoint envelopes down to a plain state dict."""
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict-like checkpoint, got {type(raw).__name__}")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    return raw


def _rename_key(key: str) -> str | None:
    """Apply the rename rules to ``key`` or return ``None`` to drop it."""
    if any(key.startswith(p) for p in _DROP_PREFIXES):
        return None
    if key in _DROP_EXACT:
        return None
    for src, dst in _RENAME_RULES:
        if key.startswith(src):
            return dst + key[len(src):]
    return key  # unchanged


def remap_state_dict(raw: dict) -> dict:
    """Apply drop + rename rules to an upstream state dict.

    Args:
        raw: Upstream state dict (post-unwrapping).

    Returns:
        A new state dict with keys matching stok's
        :class:`~stok.models.structure_encoder.StructureEncoder` hierarchy.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        new_k = _rename_key(k)
        if new_k is None:
            continue
        out[new_k] = v
    return out


def _check_codebook_equality(remapped: dict, preset: str) -> None:
    """Compare the converted codebook against stok's bundled codebook asset.

    stok ships :file:`src/stok/checkpoints/codebook/{base,lite}.pt` as a
    ``(C, d)`` tensor. If the upstream checkpoint was trained with the same
    VQ parameters, the flattened ``_codebook.embed`` buffer (shape
    ``(1, C, d)``) should match bit-for-bit. A mismatch means the bundled
    codebook is stale relative to the upstream checkpoint.
    """
    vq_key = "vector_quantizer._codebook.embed"
    if vq_key not in remapped:
        print(f"[warn] no {vq_key} in remapped dict; skipping codebook check")
        return
    vq_embed = remapped[vq_key]
    if vq_embed.dim() == 3 and vq_embed.size(0) == 1:
        vq_embed = vq_embed.squeeze(0)

    try:
        with ir.files("stok.checkpoints.codebook").joinpath(
            f"{preset}.pt"
        ).open("rb") as f:
            local = torch.load(f, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"[warn] stok.checkpoints.codebook.{preset}.pt not found; skipping check")
        return

    if local.shape != vq_embed.shape:
        print(
            f"[warn] codebook shape mismatch: bundled {tuple(local.shape)} vs "
            f"upstream {tuple(vq_embed.shape)}"
        )
        return

    if not torch.allclose(local.to(vq_embed.dtype), vq_embed, atol=1e-6):
        max_err = (local.to(vq_embed.dtype) - vq_embed).abs().max().item()
        print(f"[warn] codebook values differ from bundled asset (max abs err {max_err:.2e})")
    else:
        print(f"[ok] bundled codebook matches upstream (preset={preset})")


def _verify_strict_load(remapped: dict, preset: str) -> None:
    """Instantiate a fresh ``StructureEncoder`` and verify ``load_state_dict``.

    Raises on any unexpected keys or any missing key that isn't in the
    known ``expected_missing`` allow-list.
    """
    from stok.models.structure_encoder import _ENCODER_ARCH, StructureEncoder

    arch = _ENCODER_ARCH[preset]
    model = StructureEncoder(**arch)
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # ``featurizer`` lives in stok only (upstream keeps it in the dataset).
    expected_missing = {"featurizer.positional_encoding.frequency"}
    real_missing = set(missing) - expected_missing

    if unexpected:
        raise RuntimeError(
            f"Unexpected keys after conversion ({len(unexpected)}): "
            f"{sorted(unexpected)[:10]}"
        )
    if real_missing:
        raise RuntimeError(
            f"Unexpected missing keys after conversion ({len(real_missing)}): "
            f"{sorted(real_missing)[:10]}"
        )
    print(
        f"[ok] strict-load passed for preset={preset} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def _download_upstream(preset: str, cache_dir: Path | None) -> Path:
    """Download the upstream ``best_valid.pth`` via ``huggingface_hub``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; `pip install huggingface_hub`"
        ) from exc

    repo_id = _PRESET_TO_REPO[preset]
    kwargs = {"repo_id": repo_id, "filename": "checkpoints/best_valid.pth"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(hf_hub_download(**kwargs))


def convert(
    preset: str,
    *,
    input_path: Path | None = None,
    output_path: Path,
    cache_dir: Path | None = None,
) -> None:
    """End-to-end conversion for a single preset.

    Args:
        preset: ``"base"`` or ``"lite"``.
        input_path: Local path to an upstream checkpoint. When ``None``, the
            script downloads ``best_valid.pth`` from the matching HF repo.
        output_path: Where the remapped stok checkpoint is written.
        cache_dir: Optional override for the HuggingFace Hub cache directory.
    """
    if preset not in _PRESET_TO_REPO:
        raise ValueError(f"Unknown preset: {preset}")

    ckpt_path = input_path or _download_upstream(preset, cache_dir)
    print(f"[info] loading upstream checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = _unwrap_state_dict(raw)
    print(f"[info] upstream keys: {len(raw_state)}")

    remapped = remap_state_dict(raw_state)
    print(f"[info] remapped keys: {len(remapped)} ({len(raw_state) - len(remapped)} dropped)")

    _check_codebook_equality(remapped, preset)
    _verify_strict_load(remapped, preset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(remapped, output_path)
    print(f"[done] wrote {output_path}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument(
        "--preset",
        choices=sorted(_PRESET_TO_REPO),
        required=True,
        help="Which upstream variant to convert.",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local upstream checkpoint path. Omit to download from HuggingFace.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the remapped stok checkpoint.",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the HuggingFace Hub cache directory.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    convert(
        preset=args.preset,
        input_path=args.input,
        output_path=args.out,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
