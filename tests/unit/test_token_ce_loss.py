import pytest
import torch
import torch.nn.functional as F

from stok.utils.losses import token_ce_loss


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_empty_ce_is_connected_zero(reduction):
    logits = torch.randn(2, 3, 5, requires_grad=True)
    loss = token_ce_loss(logits, torch.full((2, 3), -100), reduction=reduction)
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_ce_matches_torch(reduction):
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[0, -100, 2], [4, 1, -100]])
    torch.testing.assert_close(
        token_ce_loss(logits, labels, reduction=reduction),
        F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction=reduction),
    )


@pytest.mark.parametrize("label", [-1, 5])
def test_ce_rejects_out_of_range_labels(label):
    with pytest.raises(ValueError, match="class"):
        token_ce_loss(torch.randn(1, 1, 5), torch.tensor([[label]]))


def test_default_empty_ce_is_finite():
    logits = torch.randn(1, 2, 3, requires_grad=True)
    assert torch.isfinite(token_ce_loss(logits, torch.full((1, 2), -100)))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_empty_fp16_ce_avoids_reduction_overflow(reduction):
    logits = torch.full((8, 16, 5), 1000.0, dtype=torch.float16, requires_grad=True)
    assert torch.isinf(logits.sum())  # Summing before multiplying by zero overflows.
    loss = token_ce_loss(logits, torch.full((8, 16), -100), reduction=reduction)
    assert loss.item() == 0.0 and loss.requires_grad
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
