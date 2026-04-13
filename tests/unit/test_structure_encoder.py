"""Unit tests for the ``StructureEncoder`` module and conversion helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("vector_quantize_pytorch")
pytest.importorskip("x_transformers")
pytest.importorskip("graphein")


# ---------------------------------------------------------------------------
# Presets + instantiation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["base", "lite"])
def test_preset_instantiation_matches_expected_dims(preset: str) -> None:
    from stok.models.structure_encoder import _ENCODER_ARCH, StructureEncoder

    model = StructureEncoder(**_ENCODER_ARCH[preset])

    # num_layers=6 is the locked-in GCPNet depth for both presets (regression
    # against the dormant default of 5 in the upstream GCPNet class).
    assert len(model.gcpnet.interaction_layers) == 6

    assert model.max_length == 1280
    if preset == "base":
        assert model.vqvae_dim == 256
        # x_transformers packs attn+ff pairs, so depth=12 -> 24 child layers.
        assert len(model.encoder_blocks.attn_layers.layers) == 12 * 2
    else:
        assert model.vqvae_dim == 128
        assert len(model.encoder_blocks.attn_layers.layers) == 8 * 2

    # Conv1d shapes (Sequential wrapping matches upstream key layout).
    assert model.encoder_tail[0].weight.shape == torch.Size([1024, 128, 1])
    assert model.encoder_head[0].weight.shape == torch.Size([model.vqvae_dim, 1024, 1])

    # Codebook buffer is (1, K, D) per vector_quantize_pytorch layout.
    embed = model.vector_quantizer._codebook.embed
    assert embed.shape == torch.Size([1, 4096, model.vqvae_dim])


# ---------------------------------------------------------------------------
# Forward pass shape/masking sanity
# ---------------------------------------------------------------------------


def _synthetic_batch(max_length: int = 32):
    from stok.utils.structure_loader import structures_to_batch
    from stok.utils.structure_parser import StructureData

    rng = np.random.default_rng(0)
    structures = [
        StructureData(
            pid="s1",
            protein_sequence="ACDEFGHIKL",
            coords=rng.standard_normal((10, 3, 3)).astype(np.float32) * 5,
            chain_id="A",
        ),
        StructureData(
            pid="s2",
            protein_sequence="MNPQRSTVW",  # length 9 to exercise variable length
            coords=rng.standard_normal((9, 3, 3)).astype(np.float32) * 5,
            chain_id="A",
        ),
    ]
    return structures, structures_to_batch(structures, max_length=max_length)


def test_forward_pass_shapes_and_masking():
    from stok.models.structure_encoder import _ENCODER_ARCH, StructureEncoder

    _, loaded = _synthetic_batch(max_length=32)
    model = StructureEncoder(**{**_ENCODER_ARCH["lite"], "max_length": 32}).eval()

    with torch.inference_mode():
        out = model(loaded.graph, loaded.mask, loaded.nan_mask)

    assert out["indices"].shape == torch.Size([2, 32])
    assert out["indices"].dtype == torch.int64
    assert out["embeddings"].shape == torch.Size([2, 32, 128])
    assert out["valid"].shape == torch.Size([2, 32])
    # Per-sample true lengths.
    assert out["valid"].sum(1).tolist() == [10, 9]


def test_encoder_is_deterministic_within_batch():
    """Repeat calls on the same input must produce identical indices.

    Also checks that a homogeneous batch of the same structure copied N
    times yields identical per-row outputs — i.e., nothing in the forward
    path carries hidden per-row state. Note that *mixed-length* batch
    equivalence is intentionally NOT asserted here: graphein's
    ``to_dense_batch``-based angle functions pad per-graph to the largest
    graph in the batch, so a sample encoded alone can differ from the
    same sample encoded alongside a longer one. That quirk is inherited
    from upstream GCP-VQVAE and is not a stok regression.
    """
    from stok.models.structure_encoder import _ENCODER_ARCH, StructureEncoder
    from stok.utils.structure_loader import structures_to_batch
    from stok.utils.structure_parser import StructureData

    rng = np.random.default_rng(1)
    s = StructureData(
        pid="s",
        protein_sequence="ACDEFGHIKL",
        coords=rng.standard_normal((10, 3, 3)).astype(np.float32) * 5,
        chain_id="A",
    )

    model = StructureEncoder(**{**_ENCODER_ARCH["lite"], "max_length": 32}).eval()

    # Same input, same output.
    loaded = structures_to_batch([s], max_length=32)
    with torch.inference_mode():
        a = model(loaded.graph, loaded.mask, loaded.nan_mask)["indices"]
    loaded = structures_to_batch([s], max_length=32)
    with torch.inference_mode():
        b = model(loaded.graph, loaded.mask, loaded.nan_mask)["indices"]
    assert torch.equal(a, b), "encoder forward is non-deterministic"

    # Homogeneous batch: three copies of the same structure must produce
    # identical per-row indices (no cross-row contamination when the dense
    # batch padding is uniform).
    loaded3 = structures_to_batch([s, s, s], max_length=32)
    with torch.inference_mode():
        out3 = model(loaded3.graph, loaded3.mask, loaded3.nan_mask)
    idx = out3["indices"]
    assert torch.equal(idx[0], idx[1]), "homogeneous batch rows 0 vs 1 differ"
    assert torch.equal(idx[0], idx[2]), "homogeneous batch rows 0 vs 2 differ"


# ---------------------------------------------------------------------------
# Conversion-script key transformations
# ---------------------------------------------------------------------------


def _load_conversion_module():
    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("convert_gcp_vqvae_weights")


def _synthesize_upstream_state_dict(preset: str) -> dict:
    """Reverse stok's rename rules to produce a 'fake upstream' state dict.

    This lets us test the conversion script without downloading 2.5 GB.
    """
    from stok.models.structure_encoder import _ENCODER_ARCH, StructureEncoder

    model = StructureEncoder(**_ENCODER_ARCH[preset])
    stok_sd = model.state_dict()

    upstream: dict[str, torch.Tensor] = {}
    for k, v in stok_sd.items():
        if k.startswith("featurizer."):
            # Upstream has no featurizer at the model level.
            continue
        if k.startswith("gcpnet."):
            upstream["encoder." + k[len("gcpnet.") :]] = v
        elif k.startswith(
            ("encoder_tail.", "encoder_blocks.", "encoder_head.", "vector_quantizer.")
        ):
            upstream["vqvae." + k] = v
        else:
            upstream[k] = v

    # Training-only noise that must be dropped by the conversion rules.
    upstream["vqvae.decoder.fake.weight"] = torch.zeros(1)
    upstream["vqvae.ntp_projector_head.weight"] = torch.zeros(1)
    upstream["vqvae.markov_short_head.weight"] = torch.zeros(1)
    upstream["vqvae.ti_tok_latent_tokens"] = torch.zeros(1)
    upstream["protein_encoder.fake"] = torch.zeros(1)
    upstream["vqvae.vector_quantizer.zero"] = torch.zeros(1)
    return upstream


@pytest.mark.parametrize("preset", ["base", "lite"])
def test_conversion_roundtrip_strict_loads(preset: str) -> None:
    mod = _load_conversion_module()
    upstream_sd = _synthesize_upstream_state_dict(preset)
    remapped = mod.remap_state_dict(upstream_sd)

    # All six training-noise keys should be gone.
    assert "vqvae.decoder.fake.weight" not in remapped
    assert "vqvae.ntp_projector_head.weight" not in remapped
    assert "vqvae.markov_short_head.weight" not in remapped
    assert "vqvae.ti_tok_latent_tokens" not in remapped
    assert "protein_encoder.fake" not in remapped
    assert "vqvae.vector_quantizer.zero" not in remapped

    # No key still carries the upstream prefix.
    for k in remapped:
        assert not k.startswith("vqvae.")
        assert not k.startswith("protein_encoder.")
        # ``encoder.gcp_embedding.*`` → ``gcpnet.gcp_embedding.*``
        assert not k.startswith("encoder.")

    # Strict-load sanity (raises on any stray unexpected key).
    mod._verify_strict_load(remapped, preset)


# ---------------------------------------------------------------------------
# load_pretrained_encoder with --path override
# ---------------------------------------------------------------------------


def test_load_pretrained_encoder_with_explicit_path(tmp_path: Path) -> None:
    mod = _load_conversion_module()
    from stok.models.structure_encoder import load_pretrained_encoder

    upstream_sd = _synthesize_upstream_state_dict("lite")
    remapped = mod.remap_state_dict(upstream_sd)
    ckpt_path = tmp_path / "encoder-lite.pt"
    torch.save(remapped, ckpt_path)

    encoder = load_pretrained_encoder(preset="lite", path=str(ckpt_path), device="cpu")
    assert encoder.training is False
    assert all(not p.requires_grad for p in encoder.parameters())

    # Spot-check one GCPNet weight matches the saved upstream tensor.
    from stok.models.structure_encoder import _ENCODER_ARCH, StructureEncoder

    ref = StructureEncoder(**_ENCODER_ARCH["lite"])
    # The parity point: reloading the remapped state dict into a fresh model
    # and comparing to the ``encoder`` we just built via ``load_pretrained``.
    ref.load_state_dict(torch.load(ckpt_path, weights_only=False), strict=False)
    for k, v in ref.state_dict().items():
        enc_v = dict(encoder.state_dict())[k]
        assert torch.equal(v, enc_v), f"parameter {k} drifted during load"
