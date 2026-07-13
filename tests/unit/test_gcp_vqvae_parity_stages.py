import torch
import torch.nn as nn

from scripts.gcp_vqvae_parity.stages import (
    StageCapture,
    capture_module_outputs,
    compare_stage_captures,
)


class TinyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(2))
            self.second.weight.copy_(torch.tensor([[1.0, 2.0]]))

    def forward(self, x):
        return self.second(self.first(x))


def test_capture_module_outputs_removes_hooks_and_clones_tensors():
    model = TinyPipeline()
    captured = capture_module_outputs(
        model,
        {"first": model.first, "second": model.second},
        lambda: model(torch.tensor([[3.0, 4.0]])),
    )
    assert torch.equal(captured["first"], torch.tensor([[3.0, 4.0]]))
    assert torch.equal(captured["second"], torch.tensor([[11.0]]))
    assert len(model.first._forward_hooks) == 0
    assert len(model.second._forward_hooks) == 0


def test_compare_stage_captures_requires_exact_indices_and_tolerant_floats():
    valid = torch.tensor([[True, True]])
    reference = StageCapture(
        tensors={
            "encoder_head": torch.tensor([[[1.0], [2.0]]]),
            "indices": torch.tensor([[5, 6]]),
            "embeddings": torch.tensor([[[1.0], [2.0]]]),
            "valid": valid,
        }
    )
    stok = StageCapture(
        tensors={
            "encoder_head": torch.tensor([[[1.0 + 5e-7], [2.0]]]),
            "indices": torch.tensor([[5, 6]]),
            "embeddings": torch.tensor([[[1.0], [2.0]]]),
            "valid": valid.clone(),
        }
    )
    comparisons = compare_stage_captures(reference, stok, rtol=1e-5, atol=1e-6)
    assert all(comparison.passed for comparison in comparisons)
