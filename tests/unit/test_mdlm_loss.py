"""Independent arithmetic references for eligible-normalized diffusion terms."""

import math

import pytest
import torch

from stok.data.mdlm import prepare_mdlm_batch
from stok.utils.mdlm import corrupt_mdlm_batch
from stok.utils.tokenizer import Tokenizer
from omegaconf import OmegaConf


def loss_case():
    batch = prepare_mdlm_batch(
        [
            {
                "dataset": "local",
                "sequence_id": "a",
                "sequence": "AC",
                "structure_tokens": [0, 1],
            }
        ],
        Tokenizer(),
        max_len=4,
        codebook_size=2,
        crop="center",
        seeds=[1],
    )
    cfg = OmegaConf.create(
        {
            "placement": "token",
            "span_mean": 8,
            "noise": {"name": "linear", "power": 2, "min_mask_probability": 1e-4},
            "regime_weights": {"joint_independent": 1},
        }
    )
    corruption = corrupt_mdlm_batch(
        batch, cfg, seeds=[1], regime="joint_independent", mask_probability=0
    )
    corruption["masked"][0, 1, 0] = True
    corruption["weight"][:] = 2
    outputs = {
        "sequence_logits": torch.zeros(1, 4, 32, requires_grad=True),
        "structure_logits": torch.zeros(1, 4, 2, requires_grad=True),
    }
    aa = torch.tensor(Tokenizer().convert_tokens_to_ids(list("AC")))
    return batch, corruption, outputs, aa


def test_mdlm_denominator_is_eligible_not_masked():
    from stok.utils.mdlm import mdlm_loss_terms

    batch, corruption, outputs, aa = loss_case()
    terms = mdlm_loss_terms(outputs, batch, corruption, canonical_aa_ids=aa)
    assert terms["eligible_count"].tolist() == [2, 2]
    assert terms["masked_count"].tolist() == [1, 0]
    assert terms["ce_sum"].tolist() == pytest.approx([math.log(2), 0])
    assert terms["weighted_sum"].tolist() == pytest.approx([2 * math.log(2), 0])
    loss = terms["weighted_sum"].sum() / terms["eligible_count"].sum()
    assert float(loss.detach()) == pytest.approx(2 * math.log(2) / 4)
    loss.backward()
    assert outputs["sequence_logits"].grad.abs().sum() > 0
    assert outputs["structure_logits"].grad is not None
    assert outputs["structure_logits"].grad.abs().sum() == 0


def test_unequal_modality_coverage_and_loss_weights():
    from stok.utils.mdlm import mdlm_loss_terms

    batch, corruption, outputs, aa = loss_case()
    batch["structure_valid"][0, 2] = False
    corruption["eligible"][0, 2, 1] = False
    corruption["masked"][0, 1, 1] = True
    terms = mdlm_loss_terms(outputs, batch, corruption, canonical_aa_ids=aa)
    weights = torch.tensor([3, 1])
    assert terms["eligible_count"].tolist() == [2, 1]
    assert float(
        (terms["weighted_sum"] * weights).sum().detach()
        / (terms["eligible_count"] * weights).sum()
    ) == pytest.approx(8 * math.log(2) / 7)


@pytest.mark.parametrize("track", [0, 1])
@pytest.mark.parametrize("masked", [False, True])
def test_invalid_targets_are_rejected_before_selection(track, masked):
    from stok.utils.mdlm import mdlm_loss_terms

    batch, corruption, outputs, aa = loss_case()
    corruption["masked"].zero_()
    corruption["masked"][0, 2, track] = masked
    batch["sequence_tokens" if track == 0 else "structure_tokens"][0, 2] = (
        31 if track == 0 else 2
    )
    with pytest.raises(ValueError, match="target"):
        mdlm_loss_terms(outputs, batch, corruption, canonical_aa_ids=aa)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("head", ["sequence_logits", "structure_logits"])
def test_nonfinite_valid_predictions_are_not_dropped(head, value):
    from stok.utils.mdlm import mdlm_loss_terms

    batch, corruption, outputs, aa = loss_case()
    with torch.no_grad():
        outputs[head][0, 2, 0] = value  # eligible but unmasked
    with pytest.raises(FloatingPointError):
        mdlm_loss_terms(outputs, batch, corruption, canonical_aa_ids=aa)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_loss_reduction_is_finite_in_reduced_precision(dtype):
    from stok.utils.mdlm import mdlm_loss_terms

    batch, corruption, outputs, aa = loss_case()
    outputs = {
        name: tensor.detach().to(dtype).requires_grad_()
        for name, tensor in outputs.items()
    }
    corruption["weight"] = torch.tensor([1e5], dtype=torch.float64)
    terms = mdlm_loss_terms(outputs, batch, corruption, canonical_aa_ids=aa)
    assert terms["ce_sum"].dtype == torch.float32
    assert torch.isfinite(terms["weighted_sum"]).all()
    assert float(terms["weighted_sum"].sum().detach()) == pytest.approx(
        1e5 * math.log(2)
    )


def test_zero_draw_connects_both_heads():
    from stok.utils.mdlm import mdlm_loss_terms

    batch, corruption, outputs, aa = loss_case()
    corruption["masked"].zero_()
    terms = mdlm_loss_terms(outputs, batch, corruption, canonical_aa_ids=aa)
    terms["weighted_sum"].sum().backward()
    assert terms["eligible_count"].sum() == 4
    for tensor in outputs.values():
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0


def test_nonfinite_predictions_on_available_inactive_modality_are_rejected():
    from stok.utils.mdlm import mdlm_loss_terms

    batch, corruption, outputs, aa = loss_case()
    corruption["eligible"][..., 1] = False
    with torch.no_grad():
        outputs["structure_logits"][0, 2, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        mdlm_loss_terms(outputs, batch, corruption, canonical_aa_ids=aa)
