"""Paired denoiser contracts, tied outputs, and pilot size."""

import pytest
import torch
import torch.nn.functional as F


def make_model(codebook=None):
    from stok.models.mdlm import STokMDLM

    return STokMDLM(
        vocab_size=32,
        pad_id=1,
        codebook=torch.randn(8, 4) if codebook is None else codebook,
        d_model=16,
        n_heads=2,
        n_layers=2,
        ffn_mult=8 / 3,
        dropout=0.0,
        attn_dropout=0.0,
    )


def paired_tokens():
    return (
        torch.tensor([[0, 4, 5, 6, 2, 1], [0, 7, 8, 2, 1, 1]]),
        torch.tensor([[8, 0, 9, 10, 8, 8], [8, 1, 2, 8, 8, 8]]),
    )


def test_paired_forward_and_tied_gradients():
    torch.manual_seed(3)
    model = make_model().eval()
    sequence, structure = paired_tokens()
    outputs = model(sequence, structure)
    assert outputs["sequence_logits"].shape == (2, 6, 32)
    assert outputs["structure_logits"].shape == (2, 6, 8)
    assert all(torch.isfinite(value).all() for value in outputs.values())
    hidden = model.encoder(
        model.embed(sequence) + model.structure_embed(structure),
        key_padding_mask=sequence.eq(1),
    )
    torch.testing.assert_close(
        outputs["sequence_logits"],
        F.linear(hidden, model.embed.weight, model.sequence_bias),
    )
    torch.testing.assert_close(
        outputs["structure_logits"],
        F.linear(hidden, model.structure_embed.weight[:8], model.structure_bias),
    )
    changed_sequence = sequence.clone()
    changed_sequence[0, 1] = 11
    assert not torch.allclose(
        outputs["structure_logits"][0, 2],
        model(changed_sequence, structure)["structure_logits"][0, 2],
    )
    changed_structure = structure.clone()
    changed_structure[0, 1] = 4
    assert not torch.allclose(
        outputs["sequence_logits"][0, 2],
        model(sequence, changed_structure)["sequence_logits"][0, 2],
    )
    # Padded keys must not influence visible predictions.
    with torch.no_grad():
        model.embed.weight[1].add_(torch.arange(16) * 100)
    padded_outputs = model(sequence, structure, key_padding_mask=sequence.eq(1))
    for key in outputs:
        before = outputs[key][~sequence.eq(1)]
        after = padded_outputs[key][~sequence.eq(1)]
        if key == "sequence_logits":
            # Tying changes the PAD output row directly; compare the other rows.
            before = torch.cat((before[:, :1], before[:, 2:]), dim=-1)
            after = torch.cat((after[:, :1], after[:, 2:]), dim=-1)
        torch.testing.assert_close(before, after)
    sum(value.square().mean() for value in padded_outputs.values()).backward()
    for weight in (model.embed.weight, model.structure_embed.weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0
    assert not model.structure_codebook.requires_grad
    assert model.structure_codebook.grad is None
    assert "structure_codebook" not in dict(model.named_parameters())


def test_input_sentinels_are_not_structure_outputs():
    model = make_model()
    sequence, structure = paired_tokens()
    assert model.structure_embed.num_embeddings == 11
    outputs = model(sequence, structure)
    assert outputs["structure_logits"].shape[-1] == 8
    assert torch.isfinite(outputs["structure_logits"]).all()
    # Fully padded batches are legal and finite as well.
    outputs = model(torch.ones((2, 6), dtype=torch.long), torch.full((2, 6), 8))
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_state_dict_roundtrip():
    source = torch.randn(8, 4, requires_grad=True)
    model = make_model(source).eval()
    original = source.detach().clone()
    with torch.no_grad():
        source.add_(1)
    torch.testing.assert_close(model.structure_codebook, original)
    restored = make_model().eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.structure_codebook, original)
    for key, value in model(*paired_tokens()).items():
        torch.testing.assert_close(value, restored(*paired_tokens())[key])


def test_pilot_parameter_count():
    from stok.models.mdlm import STokMDLM

    with torch.device("meta"):
        model = STokMDLM(
            vocab_size=32,
            pad_id=1,
            codebook=torch.empty(4096, 256),
            d_model=768,
            n_heads=12,
            n_layers=20,
            ffn_mult=8 / 3,
            dropout=0.1,
            attn_dropout=0.0,
            norm_type="layernorm",
        )
    assert all(p.device.type == "meta" for p in model.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 144_797_472


@pytest.mark.parametrize(
    "case",
    [
        "sequence_rank",
        "structure_shape",
        "sequence_dtype",
        "structure_dtype",
        "sequence_negative",
        "sequence_overflow",
        "structure_negative",
        "structure_overflow",
        "padding",
        "mask_shape",
        "mask_dtype",
        "mask_padding",
    ],
)
def test_rejects_invalid_paired_inputs(case):
    model = make_model()
    sequence, structure = paired_tokens()
    kwargs = {}
    if case == "sequence_rank":
        sequence = sequence[0]
    elif case == "structure_shape":
        structure = structure[:, :-1]
    elif case == "sequence_dtype":
        sequence = sequence.float()
    elif case == "structure_dtype":
        structure = structure.float()
    elif case == "sequence_negative":
        sequence[0, 1] = -1
    elif case == "sequence_overflow":
        sequence[0, 1] = 32
    elif case == "structure_negative":
        structure[0, 1] = -1
    elif case == "structure_overflow":
        structure[0, 1] = 11
    elif case == "padding":
        structure[0, -1] = 0
    elif case == "mask_shape":
        kwargs["key_padding_mask"] = torch.zeros((2, 5), dtype=torch.bool)
    elif case == "mask_dtype":
        kwargs["key_padding_mask"] = sequence.eq(1).float()
    else:
        kwargs["key_padding_mask"] = structure.eq(8)
    with pytest.raises(ValueError):
        model(sequence, structure, **kwargs)


@pytest.mark.parametrize(
    "codebook", [torch.empty(0, 4), torch.empty(8, 0), torch.empty(8)]
)
def test_rejects_invalid_codebook_shape(codebook):
    with pytest.raises(ValueError, match="codebook"):
        make_model(codebook)
