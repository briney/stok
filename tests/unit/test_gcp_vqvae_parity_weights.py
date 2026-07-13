import torch

from scripts.convert_gcp_vqvae_weights import remap_state_dict
from scripts.gcp_vqvae_parity.weights import audit_weight_parity


def _upstream_state():
    return {
        "encoder.layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "vqvae.encoder_tail.0.bias": torch.tensor([1.0, 2.0]),
        "vqvae.vector_quantizer._codebook.embed": torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
        "vqvae.decoder.block.weight": torch.ones(1),
    }


def test_audit_weight_parity_passes_for_exact_converted_checkpoint(tmp_path):
    upstream_path = tmp_path / "upstream.pt"
    stok_path = tmp_path / "stok.pt"
    state = _upstream_state()
    torch.save({"model_state_dict": state}, upstream_path)
    torch.save(remap_state_dict(state), stok_path)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is True
    assert audit.missing == []
    assert audit.unexpected == []
    assert audit.different == []
    assert audit.compared == 3


def test_audit_weight_parity_reports_value_drift(tmp_path):
    upstream_path = tmp_path / "upstream.pt"
    stok_path = tmp_path / "stok.pt"
    state = _upstream_state()
    converted = remap_state_dict(state)
    converted["gcpnet.layer.weight"] = converted["gcpnet.layer.weight"].clone()
    converted["gcpnet.layer.weight"][0, 0] += 1
    torch.save(state, upstream_path)
    torch.save(converted, stok_path)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is False
    assert audit.different == ["gcpnet.layer.weight"]
