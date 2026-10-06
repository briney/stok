"""Shared construction preserves the existing model and RNG contracts."""

from omegaconf import OmegaConf
import pytest
import torch
import torch.nn.functional as F

from stok.models.mdlm import STokMDLM
from stok.models.stok import STokModel
from tests.integration.test_mdlm_resume import equal


@pytest.mark.parametrize(
    "objective,tied,cosine",
    [
        ("mdlm", True, False),
        ("mlm", True, False),
        ("mlm", False, False),
        ("codebook", True, False),
        ("codebook", True, True),
    ],
)
def test_builder_matches_existing_constructors(objective, tied, cosine):
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
                "classifier": {
                    "use_cosine": cosine,
                    "learnable_temperature": True,
                    "bias_from_code_norm": True,
                    "projector_dim": 6,
                },
            },
            "train": {
                "objective": objective,
                "mlm": {"tie_word_embeddings": tied},
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
    if objective == "mdlm":
        existing = STokMDLM(**kwargs, codebook=codebook)
    else:
        existing = STokModel(
            **kwargs,
            codebook=codebook if objective == "codebook" else None,
            classifier_kwargs=dict(cfg.model.classifier)
            if objective == "codebook"
            else None,
            head_type=objective,
            tie_word_embeddings=tied,
        )
    existing_rng = torch.get_rng_state().clone()
    torch.manual_seed(1729)
    built = build_model(cfg, codebook=codebook if objective != "mlm" else None)
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
        result = (
            model(sequence, structure)
            if objective == "mdlm"
            else model(
                sequence, labels=sequence % (8 if objective == "codebook" else 32)
            )
        )
        loss = (
            sum(value.square().mean() for value in result.values())
            if objective == "mdlm"
            else result["loss"]
        )
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
    if objective == "mlm":
        assert (built.lm_head.decoder.weight is built.embed.weight) == tied
    else:
        frozen = built.structure_codebook if objective == "mdlm" else built.classifier.E
        assert torch.equal(frozen, codebook)
        assert not frozen.requires_grad and frozen.grad is None
        assert codebook.grad is None
    if objective == "mdlm":
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
