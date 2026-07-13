from types import SimpleNamespace

import pytest
import torch

import scripts.gcp_vqvae_parity.weights as weights
from scripts.gcp_vqvae_parity.common import sha256_file
from scripts.gcp_vqvae_parity.weights import audit_weight_parity, resolve_hf_revision


def _upstream_state():
    return {
        "encoder.layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "vqvae.encoder_tail.0.bias": torch.tensor([1.0, 2.0]),
        "vqvae.encoder_blocks.layer.weight": torch.arange(4, dtype=torch.float32),
        "vqvae.encoder_head.0.weight": torch.arange(3, dtype=torch.float32),
        "vqvae.vector_quantizer._codebook.embed": torch.arange(8, dtype=torch.float32).reshape(
            1, 4, 2
        ),
        "shared_scale": torch.tensor(0.5),
        "vqvae.decoder.block.weight": torch.ones(1),
        "vqvae.ntp_projector_head.weight": torch.ones(1),
        "vqvae.markov_transition_head.weight": torch.ones(1),
        "vqvae.ti_tok_latent_tokens": torch.ones(1),
        "protein_encoder.layer.weight": torch.ones(1),
        "vqvae.vector_quantizer.zero": torch.ones(1),
    }


def _converted_state():
    state = _upstream_state()
    return {
        "gcpnet.layer.weight": state["encoder.layer.weight"],
        "encoder_tail.0.bias": state["vqvae.encoder_tail.0.bias"],
        "encoder_blocks.layer.weight": state["vqvae.encoder_blocks.layer.weight"],
        "encoder_head.0.weight": state["vqvae.encoder_head.0.weight"],
        "vector_quantizer._codebook.embed": state[
            "vqvae.vector_quantizer._codebook.embed"
        ],
        "shared_scale": state["shared_scale"],
    }


def _save_checkpoint_pair(tmp_path, *, upstream=None, converted=None):
    upstream_path = tmp_path / "upstream.pt"
    stok_path = tmp_path / "stok.pt"
    torch.save(
        {"model_state_dict": _upstream_state() if upstream is None else upstream},
        upstream_path,
    )
    torch.save(
        {"state_dict": _converted_state() if converted is None else converted},
        stok_path,
    )
    return upstream_path, stok_path


def _patch_hf_response(monkeypatch, response):
    calls = []

    class FakeApi:
        def model_info(self, repo_id, revision):
            calls.append((repo_id, revision))
            return response

    monkeypatch.setattr(weights, "HfApi", FakeApi)
    return calls


def test_resolve_hf_revision_accepts_full_hexadecimal_commit_sha(monkeypatch):
    sha = "A0" * 20
    calls = _patch_hf_response(monkeypatch, SimpleNamespace(sha=sha))

    assert resolve_hf_revision("Mahdip72/gcp-vqvae-large", "main") == sha
    assert calls == [("Mahdip72/gcp-vqvae-large", "main")]


def test_resolve_hf_revision_rejects_empty_sha(monkeypatch):
    _patch_hf_response(monkeypatch, SimpleNamespace(sha=""))

    with pytest.raises(RuntimeError, match="40-character hexadecimal commit SHA"):
        resolve_hf_revision("Mahdip72/gcp-vqvae-large", None)


def test_resolve_hf_revision_rejects_short_sha(monkeypatch):
    _patch_hf_response(monkeypatch, SimpleNamespace(sha="a" * 39))

    with pytest.raises(RuntimeError, match="40-character hexadecimal commit SHA"):
        resolve_hf_revision("Mahdip72/gcp-vqvae-large", None)


def test_resolve_hf_revision_rejects_nonhex_sha(monkeypatch):
    _patch_hf_response(monkeypatch, SimpleNamespace(sha="g" * 40))

    with pytest.raises(RuntimeError, match="40-character hexadecimal commit SHA"):
        resolve_hf_revision("Mahdip72/gcp-vqvae-large", None)


def test_resolve_hf_revision_rejects_malformed_response(monkeypatch):
    _patch_hf_response(monkeypatch, SimpleNamespace(commit="a" * 40))

    with pytest.raises(RuntimeError, match="40-character hexadecimal commit SHA"):
        resolve_hf_revision("Mahdip72/gcp-vqvae-large", None)


def test_audit_accounts_for_every_systematic_drop_and_rename(tmp_path):
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is True
    assert audit.missing == []
    assert audit.unexpected == []
    assert audit.different == []
    assert audit.compared == 6
    assert audit.upstream_sha256 == sha256_file(upstream_path)
    assert audit.stok_sha256 == sha256_file(stok_path)


def test_audit_weight_parity_reports_value_drift(tmp_path):
    converted = _converted_state()
    converted["gcpnet.layer.weight"] = converted["gcpnet.layer.weight"].clone()
    converted["gcpnet.layer.weight"][0, 0] += 1
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path, converted=converted)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is False
    assert audit.different == ["gcpnet.layer.weight"]


def test_audit_weight_parity_reports_sorted_missing_tensors(tmp_path):
    converted = _converted_state()
    del converted["gcpnet.layer.weight"]
    del converted["encoder_head.0.weight"]
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path, converted=converted)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is False
    assert audit.missing == ["encoder_head.0.weight", "gcpnet.layer.weight"]
    assert audit.compared == 4


def test_audit_weight_parity_reports_sorted_unexpected_tensors(tmp_path):
    converted = _converted_state()
    converted["zzz.unexpected"] = torch.ones(1)
    converted["aaa.unexpected"] = torch.ones(1)
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path, converted=converted)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is False
    assert audit.unexpected == ["aaa.unexpected", "zzz.unexpected"]
    assert audit.compared == 6


def test_audit_weight_parity_reports_shape_mismatch(tmp_path):
    converted = _converted_state()
    converted["encoder_tail.0.bias"] = converted["encoder_tail.0.bias"].reshape(1, 2)
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path, converted=converted)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is False
    assert audit.different == ["encoder_tail.0.bias"]


def test_audit_weight_parity_reports_dtype_mismatch(tmp_path):
    converted = _converted_state()
    converted["encoder_tail.0.bias"] = converted["encoder_tail.0.bias"].to(torch.float64)
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path, converted=converted)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is False
    assert audit.different == ["encoder_tail.0.bias"]


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param(["not", "a", "dict"], id="non-dict"),
        pytest.param({"model_state_dict": []}, id="non-dict-envelope"),
        pytest.param(
            {"model_state_dict": {"encoder.layer.weight": [1.0]}},
            id="non-tensor-value",
        ),
    ],
)
def test_audit_rejects_malformed_upstream_checkpoint_structure(tmp_path, malformed):
    upstream_path = tmp_path / "upstream.pt"
    stok_path = tmp_path / "stok.pt"
    torch.save(malformed, upstream_path)
    torch.save(_converted_state(), stok_path)

    with pytest.raises(TypeError, match="checkpoint state dict"):
        audit_weight_parity(upstream_path, stok_path)


def test_audit_rejects_malformed_converted_checkpoint_structure(tmp_path):
    upstream_path = tmp_path / "upstream.pt"
    stok_path = tmp_path / "stok.pt"
    torch.save(_upstream_state(), upstream_path)
    torch.save({"state_dict": []}, stok_path)

    with pytest.raises(TypeError, match="checkpoint state dict"):
        audit_weight_parity(upstream_path, stok_path)


def test_audit_rejects_empty_checkpoint_state(tmp_path):
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path, upstream={}, converted={})

    with pytest.raises(ValueError, match="checkpoint state dict is empty"):
        audit_weight_parity(upstream_path, stok_path)


def test_audit_rejects_uncategorized_upstream_key(tmp_path):
    upstream = _upstream_state()
    upstream["vqvae.unknown.weight"] = torch.ones(1)
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path, upstream=upstream)

    with pytest.raises(ValueError, match=r"Uncategorized upstream key: vqvae\.unknown\.weight"):
        audit_weight_parity(upstream_path, stok_path)


def test_audit_rejects_duplicate_destination_targets(tmp_path):
    upstream = _upstream_state()
    upstream["gcpnet.layer.weight"] = upstream["encoder.layer.weight"].clone()
    upstream_path, stok_path = _save_checkpoint_pair(tmp_path, upstream=upstream)

    with pytest.raises(ValueError, match=r"Duplicate destination target: gcpnet\.layer\.weight"):
        audit_weight_parity(upstream_path, stok_path)
