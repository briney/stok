import pytest
import torch


pytest.importorskip("x_transformers")


def _make_state_dict(preset: str, d_code: int = 64):
    from stok.models.decoder import GeometricDecoder

    if preset == "base":
        arch = dict(
            d_model=1024,
            ffn_mult=4.0,
            n_layers=16,
            n_heads=16,
            attn_kv_heads=1,
            num_memory_tokens=0,
            max_length=1280,
        )
    elif preset == "lite":
        arch = dict(
            d_model=1024,
            ffn_mult=4.0,
            n_layers=12,
            n_heads=8,
            attn_kv_heads=2,
            num_memory_tokens=0,
            max_length=1280,
        )
    else:
        raise ValueError(preset)

    model = GeometricDecoder(
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        ffn_mult=arch["ffn_mult"],
        max_length=arch["max_length"],
        d_code=d_code,
        num_memory_tokens=arch["num_memory_tokens"],
        attn_kv_heads=arch["attn_kv_heads"],
    )
    return model.state_dict()


def test_path_override_and_freeze(tmp_path):
    from stok.models.decoder import load_pretrained_decoder

    ckpt = _make_state_dict("base", d_code=64)
    ckpt_path = tmp_path / "decoder-base.pt"
    torch.save(ckpt, ckpt_path)

    model = load_pretrained_decoder(
        preset="base", path=str(ckpt_path), device="cpu", freeze=True
    )

    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    assert model.projector_in.weight.shape[0] == 1024
    assert model.projector_in.weight.shape[1] == 64
