"""Shared fixtures for Phase-2 CLI subcommand integration tests.

Tiny seq_only and joint MDLM checkpoints that match the shape parameters
the new CLI commands will compose via Hydra overrides. Kept deliberately
small so each test is fast.
"""

from __future__ import annotations

import pytest
import torch

from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule


CODEBOOK_SIZE = 16
CODEBOOK_DIM = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FFN_MULT = 2.0


def _build_seq_model() -> MDLMModel:
    ns = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="seq_only",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_mult=FFN_MULT,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns,
        time_conditioning="adaln",
    )


def _build_joint_model(codebook: torch.Tensor) -> MDLMModel:
    ns_seq = NoiseSchedule(schedule_type="cosine")
    ns_struct = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="joint",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        codebook=codebook,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_mult=FFN_MULT,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns_seq,
        noise_schedule_struct=ns_struct,
        time_conditioning="adaln",
    )


@pytest.fixture
def seq_checkpoint(tmp_path):
    """Path to a tiny seq_only MDLM checkpoint."""
    model = _build_seq_model()
    path = tmp_path / "seq_model.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def joint_checkpoint(tmp_path):
    """Tuple of (checkpoint_path, codebook_path) for a tiny joint MDLM."""
    codebook = torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM) * 0.1
    codebook_path = tmp_path / "codebook.pt"
    torch.save(codebook, codebook_path)

    model = _build_joint_model(codebook)
    ckpt_path = tmp_path / "joint_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path, codebook_path


@pytest.fixture
def patch_load_decoder(monkeypatch):
    """Replace :func:`stok.api.load_decoder` with a tiny in-memory decoder.

    The real ``base``/``lite`` presets download full-scale pretrained weights
    whose ``d_code`` is fixed at the real codebook's dimensionality. Our
    test fixtures use a random codebook with ``d_code=CODEBOOK_DIM`` (8),
    which cannot be fed through the real preset weights. This fixture swaps
    in a freshly-initialized :class:`GeometricDecoder` whose projector
    matches the test codebook so CLI commands that exercise the decoder
    path can run end-to-end without internet or shape mismatches.
    """
    pytest.importorskip("x_transformers")
    from stok.models.decoder import GeometricDecoder

    def _fake_load_decoder(preset="base", *, path=None, device="cpu"):
        dec = GeometricDecoder(
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.0,
            max_length=64,
            d_code=CODEBOOK_DIM,
            num_memory_tokens=0,
            attn_kv_heads=2,
        )
        dec.eval()
        for p in dec.parameters():
            p.requires_grad = False
        return dec.to(device)

    monkeypatch.setattr("stok.api.load_decoder", _fake_load_decoder)
    return _fake_load_decoder


def seq_only_hydra_overrides() -> list[str]:
    """Hydra overrides that pin the tiny model shape used by fixtures."""
    return [
        "train.objective=mdlm",
        "train.mdlm.tracks=seq_only",
        f"model.encoder.d_model={D_MODEL}",
        f"model.encoder.n_layers={N_LAYERS}",
        f"model.encoder.n_heads={N_HEADS}",
        f"model.encoder.ffn_mult={FFN_MULT}",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
    ]


def joint_hydra_overrides(codebook_path) -> list[str]:
    """Hydra overrides for the tiny joint model fixture."""
    return [
        "train.objective=mdlm",
        "train.mdlm.tracks=joint",
        f"model.encoder.d_model={D_MODEL}",
        f"model.encoder.n_layers={N_LAYERS}",
        f"model.encoder.n_heads={N_HEADS}",
        f"model.encoder.ffn_mult={FFN_MULT}",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        f"model.codebook.path={codebook_path}",
    ]
