"""Shared construction preserves the existing model and RNG contracts."""

from omegaconf import OmegaConf
import pytest
import torch
import torch.nn.functional as F

from stok.models.mdlm import STokMDLM
from tests.integration.test_mdlm_resume import equal


def test_builder_matches_existing_mdlm_constructor():
    from stok.models.build import build_model

    cfg = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "vocab_size": 32,
                    "pad_id": 1,
                    "d_model": 16,
                    "n_heads": 2,
                    "n_layers": 1,
                    "ffn_mult": 1,
                    "dropout": 0.2,
                    "attn_dropout": 0.1,
                    "norm": "layernorm",
                },
            },
            "train": {
                "objective": "mdlm",
                "mdlm": {"regime_weights": {"joint_independent": 1}},
            },
        }
    )
    saved_cfg = OmegaConf.to_container(cfg, resolve=True)
    codebook = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    codebook.requires_grad_(True)
    kwargs = dict(
        vocab_size=32,
        pad_id=1,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ffn_mult=1,
        dropout=0.2,
        attn_dropout=0.1,
        norm_type="layernorm",
    )
    torch.manual_seed(1729)
    existing = STokMDLM(**kwargs, codebook=codebook)
    existing_rng = torch.get_rng_state().clone()
    torch.manual_seed(1729)
    built = build_model(cfg, codebook=codebook)
    assert type(built) is type(existing)
    assert torch.equal(existing_rng, torch.get_rng_state())
    assert list(existing.state_dict()) == list(built.state_dict())
    assert list(dict(existing.named_parameters())) == list(
        dict(built.named_parameters())
    )
    equal(existing.state_dict(), built.state_dict())
    assert OmegaConf.to_container(cfg, resolve=True) == saved_cfg
    sequence = torch.tensor([[0, 4, 5, 6, 2, 1], [0, 7, 8, 2, 1, 1]])
    structure = torch.tensor([[8, 0, 9, 10, 8, 8], [8, 1, 2, 8, 8, 8]])
    outputs = []
    gradients = []
    for model in (existing, built):
        torch.manual_seed(42)
        result = model(sequence, structure)
        loss = sum(value.square().mean() for value in result.values())
        loss.backward()
        outputs.append(result)
        gradients.append([parameter.grad for parameter in model.parameters()])
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients[-1]
        )
        assert any(gradient.abs().sum() > 0 for gradient in gradients[-1])
    equal(outputs[0], outputs[1])
    equal(gradients[0], gradients[1])
    assert torch.equal(built.structure_codebook, codebook)
    assert (
        not built.structure_codebook.requires_grad
        and built.structure_codebook.grad is None
    )
    assert codebook.grad is None
    assert built.mdlm_regime_weights == {"joint_independent": 1}
    built.eval()
    hidden = built.encoder(
        built.embed(sequence) + built.structure_embed(structure),
        key_padding_mask=sequence.eq(1),
    )
    result = built(sequence, structure)
    equal(
        result["sequence_logits"],
        F.linear(hidden, built.embed.weight, built.sequence_bias),
    )
    equal(
        result["structure_logits"],
        F.linear(hidden, built.structure_embed.weight[:8], built.structure_bias),
    )


@pytest.mark.parametrize("weights", [None, {}])
def test_missing_or_empty_regime_weights_stay_unqualified(weights):
    from stok.models.build import build_model

    cfg = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "vocab_size": 32,
                    "pad_id": 1,
                    "d_model": 16,
                    "n_heads": 2,
                    "n_layers": 1,
                    "ffn_mult": 1,
                    "dropout": 0,
                    "attn_dropout": 0,
                    "norm": "layernorm",
                }
            },
            "train": {"objective": "mdlm", "mdlm": {}},
        }
    )
    if weights is not None:
        cfg.train.mdlm.regime_weights = weights
    model = build_model(cfg, codebook=torch.zeros(8, 4))
    assert model.mdlm_regime_weights == {}


@pytest.mark.parametrize("cosine", [False, True])
def test_frozen_prototype_head_matches_declared_geometry_and_backpropagates(cosine):
    from stok.models.head import CodebookClassifier

    codebook = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]], requires_grad=True)
    head = CodebookClassifier(
        d_in=4, codebook=codebook, use_cosine=cosine, projector_dim=3
    )
    hidden = torch.arange(16.0).reshape(1, 4, 4).requires_grad_()
    codes = head.to_code(head.ln(head.project(hidden)))
    expected = (
        F.normalize(codes, dim=-1) @ F.normalize(codebook.detach(), dim=-1).T
        if cosine
        else 2 * codes @ codebook.detach().T - codebook.detach().square().sum(-1)
    )
    torch.testing.assert_close(head(hidden), expected)
    head(hidden).square().sum().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
    assert head.E.grad is None and codebook.grad is None and not head.E.requires_grad
