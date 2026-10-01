"""Checkpoint trust boundaries and strict component extraction."""

import hashlib
import logging
import pickle

import numpy as np
import pytest
import torch


class UnsupportedMetadata:
    pass


def test_reader_normalizes_nested_and_extracted_states(tmp_path):
    from stok.utils.pretrained import read_gcp_checkpoint

    expected = {"encoder.encoder.layer.weight": torch.arange(6).reshape(2, 3)}
    for payload in [
        expected,
        {
            "model_state_dict": {
                "encoder._orig_mod.encoder.layer.weight": expected[
                    "encoder.encoder.layer.weight"
                ]
            },
            "score": np.float64(0.3),
            "dtype": np.dtype("float64"),
        },
    ]:
        path = tmp_path / "model.pt"
        torch.save(payload, path)
        actual = read_gcp_checkpoint(path)
        assert actual.keys() == expected.keys()
        assert torch.equal(
            actual["encoder.encoder.layer.weight"],
            expected["encoder.encoder.layer.weight"],
        )


def test_reader_rejects_collision_invalid_state_and_arbitrary_metadata(tmp_path):
    from stok.utils.pretrained import read_gcp_checkpoint

    path = tmp_path / "model.pt"
    torch.save({"a.weight": torch.ones(1), "a._orig_mod.weight": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="collision"):
        read_gcp_checkpoint(path)
    torch.save({"model_state_dict": {"weight": "not a tensor"}}, path)
    with pytest.raises(ValueError, match="tensor"):
        read_gcp_checkpoint(path)
    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(1)},
            "metadata": UnsupportedMetadata(),
        },
        path,
    )
    with pytest.raises(pickle.UnpicklingError):
        read_gcp_checkpoint(path)


def test_components_are_extracted_with_known_prefixes_and_no_silent_keys():
    from stok.utils.pretrained import extract_gcp_component

    state = {
        "encoder.featuriser.weight": torch.ones(1),
        "encoder.encoder.weight": torch.ones(2),
        "vqvae.encoder_tail.0.weight": torch.ones(3),
        "vqvae.encoder_blocks.weight": torch.ones(4),
        "vqvae.encoder_head.0.weight": torch.ones(5),
        "vqvae.vector_quantizer._codebook.embed": torch.zeros(1, 4096, 128),
        "vqvae.decoder.projector_in.weight": torch.ones(8, 128),
    }
    assert set(extract_gcp_component(state, "encoder")) == {
        "featuriser.weight",
        "encoder.weight",
        "encoder_tail.0.weight",
        "encoder_blocks.weight",
        "encoder_head.0.weight",
    }
    assert set(extract_gcp_component(state, "quantizer")) == {"_codebook.embed"}
    assert set(extract_gcp_component(state, "decoder")) == {"projector_in.weight"}
    with pytest.raises(ValueError, match="unexpected|Unexpected"):
        extract_gcp_component({**state, "unknown.weight": torch.ones(1)}, "decoder")
    with pytest.raises(ValueError, match="collision"):
        extract_gcp_component(
            {**state, "encoder.encoder_tail.0.weight": torch.ones(3)}, "encoder"
        )
    with pytest.raises(ValueError, match="missing|Missing"):
        extract_gcp_component(
            {"vqvae.vector_quantizer._codebook.embed": torch.zeros(1, 2, 3)}, "encoder"
        )


def test_large_auxiliary_exclusion_is_exact_and_reported(caplog):
    from stok.utils.pretrained import extract_gcp_component

    keys = [
        "downproject.weight",
        "linear1.weight",
        "norm.weight",
        "norm.bias",
        "linear2.weight",
    ]
    state = {
        "projector_in.weight": torch.zeros(8, 256),
        **{"pairwise_classification_head." + name: torch.ones(1) for name in keys},
    }
    with caplog.at_level(logging.INFO):
        extracted = extract_gcp_component(state, "decoder")
    assert set(extracted) == {"projector_in.weight"}
    assert all(name in caplog.text for name in keys)
    with pytest.raises(ValueError, match="auxiliary"):
        extract_gcp_component(
            {**state, "pairwise_classification_head.unexpected.weight": torch.ones(1)},
            "decoder",
        )
    with pytest.raises(ValueError, match="Large"):
        extract_gcp_component(
            {**state, "projector_in.weight": torch.zeros(8, 128)}, "decoder"
        )


def test_pinned_download_verifies_cache_and_publishes_atomically(tmp_path, monkeypatch):
    from stok.utils import pretrained

    payload = b"verified fixture checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(
        pretrained.ARTIFACTS,
        "lite",
        {
            "repository": "owner/model",
            "revision": "a" * 40,
            "sha256": digest,
            "filename": "checkpoint.pt",
        },
    )
    calls = []

    def download(url, destination, **kwargs):
        calls.append(url)
        assert "resolve/" + "a" * 40 in url
        with open(destination, "wb") as handle:
            handle.write(payload)

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    path = pretrained.resolve_gcp_artifact("lite", cache_dir=tmp_path)
    assert path.read_bytes() == payload
    assert pretrained.resolve_gcp_artifact("lite", cache_dir=tmp_path) == path
    assert len(calls) == 1
    path.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="digest"):
        pretrained.resolve_gcp_artifact("lite", cache_dir=tmp_path)
    path.unlink()

    def corrupt_download(url, destination, **kwargs):
        with open(destination, "wb") as handle:
            handle.write(b"truncated")

    monkeypatch.setattr(torch.hub, "download_url_to_file", corrupt_download)
    with pytest.raises(ValueError, match="digest"):
        pretrained.resolve_gcp_artifact("lite", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_archive_codebook_and_large_alias_preserve_exact_matrix(tmp_path):
    from stok.utils.codebook import load_codebook

    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    path = tmp_path / "archive.pt"
    torch.save(
        {
            "model_state_dict": {
                "vqvae._orig_mod.vector_quantizer._codebook.embed": matrix[None]
            }
        },
        path,
    )
    assert torch.equal(load_codebook(path=str(path)), matrix)
    assert torch.equal(load_codebook("large"), load_codebook("base"))


def test_decoder_archives_load_strictly_and_preserve_extracted_input(
    tmp_path, monkeypatch
):
    from stok.models import decoder

    arch = {
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "ffn_mult": 1.0,
        "max_length": 32,
        "num_memory_tokens": 0,
        "attn_kv_heads": 1,
    }
    monkeypatch.setitem(decoder._DECODER_ARCH, "lite", arch)
    original = decoder.GeometricDecoder(**arch, d_code=128)
    path = tmp_path / "model.pt"
    state = {
        "vqvae.decoder._orig_mod." + key: value
        for key, value in original.state_dict().items()
    }
    torch.save({"model_state_dict": state}, path)
    loaded = decoder.load_pretrained_decoder("lite", path=str(path))
    assert set(loaded.state_dict()) == set(original.state_dict())
    assert all(
        torch.equal(value, loaded.state_dict()[key])
        for key, value in original.state_dict().items()
    )
    assert not loaded.training
    state.pop("vqvae.decoder._orig_mod.projector_in.weight")
    torch.save({"model_state_dict": state}, path)
    with pytest.raises((KeyError, RuntimeError)):
        decoder.load_pretrained_decoder("lite", path=str(path))
    raw = original.state_dict()
    raw["unexpected.weight"] = torch.ones(1)
    torch.save(raw, path)
    with pytest.raises(RuntimeError, match="Unexpected"):
        decoder.load_pretrained_decoder("lite", path=str(path))
    raw.pop("unexpected.weight")
    raw["projector_in.weight"] = torch.ones(8, 256)
    torch.save(
        {
            "model_state_dict": {
                "vqvae.decoder." + key: value for key, value in raw.items()
            }
        },
        path,
    )
    with pytest.raises(RuntimeError, match="preset"):
        decoder.load_pretrained_decoder("lite", path=str(path))
