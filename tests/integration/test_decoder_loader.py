import os

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


def test_download_and_cache_reuse(tmp_path, monkeypatch):
    from stok.models.decoder import load_pretrained_decoder

    # Isolate cache into a temp directory
    cache_dir = tmp_path / "cache"
    os.environ["STOK_DECODER_CACHE"] = str(cache_dir)

    # Prepare a fake checkpoint to "download"
    fake_ckpt = _make_state_dict("lite", d_code=32)

    calls = {"n": 0}

    def _fake_download(url, dst, hash_prefix=None, progress=True):  # noqa: ARG001
        calls["n"] += 1
        torch.save(fake_ckpt, dst)

    monkeypatch.setenv(
        "STOK_DECODER_LITE_URL", "https://example.invalid/decoder-lite.pt"
    )
    monkeypatch.setenv("STOK_DECODER_LITE_SHA256", "")
    monkeypatch.setenv(
        "STOK_DECODER_BASE_URL", "https://example.invalid/decoder-base.pt"
    )
    monkeypatch.setenv("STOK_DECODER_BASE_SHA256", "")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    # Patch the downloader
    from torch.hub import download_url_to_file as _orig

    monkeypatch.setattr("torch.hub.download_url_to_file", _fake_download)

    try:
        model = load_pretrained_decoder(preset="lite", device="cpu", freeze=True)
        assert calls["n"] == 1
        expected = cache_dir / "decoder-lite.pt"
        assert expected.exists()

        # Second call should reuse cache
        _ = load_pretrained_decoder(preset="lite", device="cpu", freeze=True)
        assert calls["n"] == 1
        assert model.projector_in.weight.shape[0] == 1024
        assert model.projector_in.weight.shape[1] == 32
    finally:
        # restore original in case of leakage (defensive; pytest will isolate anyway)
        try:
            import torch.hub as hub

            hub.download_url_to_file = _orig
        except Exception:
            pass
