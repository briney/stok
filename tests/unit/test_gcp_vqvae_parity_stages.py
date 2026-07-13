from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.gcp_vqvae_parity import stages
from scripts.gcp_vqvae_parity.stages import (
    StageCapture,
    capture_module_outputs,
    capture_reference_stages,
    capture_stok_stages,
    compare_stage_captures,
)

EXPECTED_STAGES = (
    "featurizer.x",
    "featurizer.x_vector_attr",
    "featurizer.edge_attr",
    "featurizer.edge_vector_attr",
    "gcpnet",
    "encoder_tail",
    "encoder_blocks",
    "encoder_head",
    "indices",
    "embeddings",
    "valid",
)
GRAPH_FIELDS = (
    "x",
    "x_vector_attr",
    "edge_attr",
    "edge_vector_attr",
    "batch",
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


class RejectForwardHooks(nn.Module):
    def register_forward_hook(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("registration failed")


class StubGraph:
    def __init__(self):
        self.x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.x_vector_attr = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
        self.edge_attr = torch.tensor([[1.5], [2.5]])
        self.edge_vector_attr = torch.tensor([[[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]])
        self.batch = torch.tensor([0, 0])


def _graph_snapshot(graph: StubGraph) -> dict[str, torch.Tensor]:
    return {field: getattr(graph, field).detach().clone() for field in GRAPH_FIELDS}


def _assert_graph_matches(graph: StubGraph, expected: dict[str, torch.Tensor]) -> None:
    for field, value in expected.items():
        assert torch.equal(getattr(graph, field), value), field


def _assert_snapshot_matches(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    for field, value in expected.items():
        assert torch.equal(actual[field], value), field


class RecordingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_states: list[tuple[bool, bool]] = []

    def _record_call(self) -> None:
        self.call_states.append((torch.is_grad_enabled(), self.training))


class RecordingGCPNet(RecordingModule):
    def __init__(self):
        super().__init__()
        self.graph_inputs: list[dict[str, torch.Tensor]] = []
        self.raise_on_call = False

    def forward(self, graph):
        self._record_call()
        self.graph_inputs.append(_graph_snapshot(graph))
        if self.raise_on_call:
            raise RuntimeError("gcpnet failed")
        return {"node_embedding": graph.x * 2.0}


class RecordingFeaturizer(RecordingModule):
    def __init__(self):
        super().__init__()
        self.graph_inputs: list[dict[str, torch.Tensor]] = []

    def forward(self, graph):
        self._record_call()
        self.graph_inputs.append(_graph_snapshot(graph))
        graph.x = graph.x + 10.0
        graph.x_vector_attr = graph.x_vector_attr + 20.0
        graph.edge_attr = graph.edge_attr + 30.0
        graph.edge_vector_attr = graph.edge_vector_attr + 40.0
        return graph


class RecordingConv(RecordingModule):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, value):
        self._record_call()
        return value * self.scale


class RecordingBlocks(RecordingModule):
    def __init__(self):
        super().__init__()
        self.masks: list[torch.Tensor] = []

    def forward(self, value, *, mask):
        self._record_call()
        self.masks.append(mask.detach().clone())
        return value + 1.0


class RecordingVectorQuantizer(RecordingModule):
    def __init__(self):
        super().__init__()
        self.masks: list[torch.Tensor] = []

    def forward(self, value, *, mask):
        self._record_call()
        self.masks.append(mask.detach().clone())
        quantized = value + 0.25
        indices = value.argmax(dim=-1)
        return quantized, indices, value.new_zeros(())


class StubVQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_tail = RecordingConv(2.0)
        self.encoder_blocks = RecordingBlocks()
        self.encoder_head = RecordingConv(3.0)
        self.vector_quantizer = RecordingVectorQuantizer()


class StubSuperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RecordingGCPNet()
        self.vqvae = StubVQVAE()
        self.fail_after_tail = False

    def forward(self, batch, *, return_vq_layer):
        assert return_vq_layer is True
        graph = batch["graph"]
        valid = batch["masks"].to(torch.bool) & batch["nan_masks"].to(torch.bool)
        node_embedding = self.encoder(graph)["node_embedding"]
        x = node_embedding.reshape(1, batch["masks"].shape[1], -1)
        tail = self.vqvae.encoder_tail(x.transpose(1, 2))
        if self.fail_after_tail:
            raise RuntimeError("reference forward failed")
        blocks = self.vqvae.encoder_blocks(tail.transpose(1, 2), mask=valid)
        head = self.vqvae.encoder_head(blocks.transpose(1, 2)).transpose(1, 2)
        embeddings, indices, _ = self.vqvae.vector_quantizer(head, mask=valid)
        graph.x.add_(1000.0)
        return {"indices": indices, "embeddings": embeddings}


class StubStokEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_length = 2
        self.featurizer = RecordingFeaturizer()
        self.gcpnet = RecordingGCPNet()
        self.encoder_tail = RecordingConv(2.0)
        self.encoder_blocks = RecordingBlocks()
        self.encoder_head = RecordingConv(3.0)
        self.vector_quantizer = RecordingVectorQuantizer()


def _batch() -> dict[str, object]:
    return {
        "graph": StubGraph(),
        "masks": torch.tensor([[True, True]]),
        "nan_masks": torch.tensor([[True, False]]),
        "metadata": {"source": "fixture"},
    }


def _batch_snapshot(batch: dict[str, object]) -> dict[str, object]:
    return {
        "graph": _graph_snapshot(batch["graph"]),
        "masks": batch["masks"].clone(),
        "nan_masks": batch["nan_masks"].clone(),
        "metadata": copy.deepcopy(batch["metadata"]),
    }


def _assert_batch_unchanged(batch: dict[str, object], expected: dict[str, object]) -> None:
    _assert_graph_matches(batch["graph"], expected["graph"])
    assert torch.equal(batch["masks"], expected["masks"])
    assert torch.equal(batch["nan_masks"], expected["nan_masks"])
    assert batch["metadata"] == expected["metadata"]


def _base_capture() -> StageCapture:
    return StageCapture(
        tensors={
            "featurizer.x": torch.tensor([0.0]),
            "featurizer.x_vector_attr": torch.tensor([0.0]),
            "featurizer.edge_attr": torch.tensor([0.0]),
            "featurizer.edge_vector_attr": torch.tensor([0.0]),
            "gcpnet": torch.tensor([0.0]),
            "encoder_tail": torch.tensor([0.0]),
            "encoder_blocks": torch.tensor([0.0]),
            "encoder_head": torch.tensor([0.0]),
            "indices": torch.tensor([[5, 6]]),
            "embeddings": torch.tensor([[[1.0], [2.0]]]),
            "valid": torch.tensor([[True, True]]),
        }
    )


def _capture_pair() -> tuple[StageCapture, StageCapture]:
    return _base_capture(), copy.deepcopy(_base_capture())


def _comparison_map(reference: StageCapture, stok: StageCapture):
    return {
        result.name: result
        for result in compare_stage_captures(reference, stok, rtol=1e-5, atol=1e-6)
    }


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


def test_capture_module_outputs_cleans_up_after_forward_exception():
    model = TinyPipeline()

    def invoke():
        model.first(torch.tensor([[3.0, 4.0]]))
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        capture_module_outputs(
            model,
            {"first": model.first, "second": model.second},
            invoke,
        )
    assert len(model.first._forward_hooks) == 0
    assert len(model.second._forward_hooks) == 0


def test_capture_module_outputs_cleans_up_after_later_registration_exception():
    model = TinyPipeline()
    reject_hooks = RejectForwardHooks()

    with pytest.raises(RuntimeError, match="registration failed"):
        capture_module_outputs(
            model,
            {"first": model.first, "reject": reject_hooks},
            lambda: None,
        )
    assert len(model.first._forward_hooks) == 0


def test_compare_stage_captures_uses_canonical_pipeline_order():
    reference, stok = _capture_pair()

    comparisons = compare_stage_captures(reference, stok, rtol=1e-5, atol=1e-6)

    assert stages.CANONICAL_STAGES == EXPECTED_STAGES
    assert tuple(comparison.name for comparison in comparisons) == EXPECTED_STAGES
    assert all(comparison.passed for comparison in comparisons)


@pytest.mark.parametrize("side", ["reference", "stok"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_compare_stage_captures_validates_each_stage_map_independently(side, mutation):
    reference, stok = _capture_pair()
    target = reference if side == "reference" else stok
    if mutation == "missing":
        target.tensors.pop("gcpnet")
        detail = "missing=['gcpnet']"
    else:
        target.tensors["extra"] = torch.tensor([0.0])
        detail = "unexpected=['extra']"

    with pytest.raises(ValueError, match=rf"{side} stage keys invalid") as exc_info:
        compare_stage_captures(reference, stok, rtol=1e-5, atol=1e-6)
    assert detail in str(exc_info.value)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_compare_stage_captures_rejects_identically_invalid_stage_maps(mutation):
    reference, stok = _capture_pair()
    if mutation == "missing":
        reference.tensors.pop("gcpnet")
        stok.tensors.pop("gcpnet")
    else:
        reference.tensors["extra"] = torch.tensor([0.0])
        stok.tensors["extra"] = torch.tensor([0.0])

    with pytest.raises(ValueError, match="reference stage keys invalid"):
        compare_stage_captures(reference, stok, rtol=1e-5, atol=1e-6)


def test_compare_stage_captures_requires_exact_masks():
    reference, stok = _capture_pair()
    stok.tensors["valid"][0, 0] = False

    comparisons = _comparison_map(reference, stok)

    assert comparisons["valid"].passed is False
    assert comparisons["indices"].passed is False
    assert comparisons["indices"].mask_equal is False


def test_compare_stage_captures_requires_exact_valid_indices():
    reference, stok = _capture_pair()
    stok.tensors["indices"][0, 0] = 99

    comparison = _comparison_map(reference, stok)["indices"]

    assert comparison.passed is False
    assert comparison.mismatched == 1


def test_compare_stage_captures_requires_exact_quantized_embeddings():
    reference, stok = _capture_pair()
    stok.tensors["embeddings"][0, 0, 0] += 5e-7

    comparison = _comparison_map(reference, stok)["embeddings"]

    assert comparison.passed is False
    assert comparison.exact is False


def test_compare_stage_captures_applies_float_tolerance_boundaries():
    reference, stok = _capture_pair()
    stok.tensors["encoder_head"][0] = 5e-7
    within = _comparison_map(reference, stok)["encoder_head"]

    stok.tensors["encoder_head"][0] = 2e-6
    outside = _comparison_map(reference, stok)["encoder_head"]

    assert within.passed is True
    assert outside.passed is False


def test_live_adapters_share_reference_graph_and_capture_complete_normalized_stages():
    batch = _batch()
    original = _batch_snapshot(batch)
    reference_model = StubSuperModel().eval()
    stok_encoder = StubStokEncoder().eval()

    reference = capture_reference_stages(SimpleNamespace(model=reference_model), batch)
    stok = capture_stok_stages(stok_encoder, batch)

    _assert_batch_unchanged(batch, original)
    assert tuple(reference.tensors) == EXPECTED_STAGES
    assert tuple(stok.tensors) == EXPECTED_STAGES
    assert len(reference_model.encoder.graph_inputs) == 1
    assert len(stok_encoder.gcpnet.graph_inputs) == 1
    assert len(stok_encoder.featurizer.graph_inputs) == 1
    _assert_snapshot_matches(
        reference_model.encoder.graph_inputs[0], stok_encoder.gcpnet.graph_inputs[0]
    )
    _assert_snapshot_matches(reference_model.encoder.graph_inputs[0], original["graph"])
    _assert_snapshot_matches(stok_encoder.featurizer.graph_inputs[0], original["graph"])

    assert torch.equal(reference.tensors["featurizer.x"], original["graph"]["x"])
    assert torch.equal(stok.tensors["featurizer.x"], original["graph"]["x"] + 10.0)
    assert torch.equal(reference.tensors["gcpnet"], torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
    assert torch.equal(
        reference.tensors["encoder_tail"],
        torch.tensor([[[4.0, 8.0], [12.0, 16.0]]]),
    )
    assert torch.equal(
        reference.tensors["encoder_blocks"],
        torch.tensor([[[5.0, 9.0], [13.0, 17.0]]]),
    )
    assert torch.equal(
        reference.tensors["encoder_head"],
        torch.tensor([[[15.0, 27.0], [39.0, 51.0]]]),
    )
    for name in EXPECTED_STAGES[4:]:
        assert torch.equal(reference.tensors[name], stok.tensors[name]), name

    valid = torch.tensor([[True, False]])
    modules = (
        reference_model.encoder,
        reference_model.vqvae.encoder_tail,
        reference_model.vqvae.encoder_blocks,
        reference_model.vqvae.encoder_head,
        reference_model.vqvae.vector_quantizer,
        stok_encoder.featurizer,
        stok_encoder.gcpnet,
        stok_encoder.encoder_tail,
        stok_encoder.encoder_blocks,
        stok_encoder.encoder_head,
        stok_encoder.vector_quantizer,
    )
    assert all(module.call_states == [(False, False)] for module in modules)
    assert torch.equal(reference_model.vqvae.encoder_blocks.masks[0], valid)
    assert torch.equal(reference_model.vqvae.vector_quantizer.masks[0], valid)
    assert torch.equal(stok_encoder.encoder_blocks.masks[0], valid)
    assert torch.equal(stok_encoder.vector_quantizer.masks[0], valid)


def test_reference_adapter_propagates_failure_cleans_hooks_and_preserves_batch():
    batch = _batch()
    original = _batch_snapshot(batch)
    reference_model = StubSuperModel().eval()
    reference_model.fail_after_tail = True
    hooked_modules = (
        reference_model.encoder,
        reference_model.vqvae.encoder_tail,
        reference_model.vqvae.encoder_blocks,
        reference_model.vqvae.encoder_head,
        reference_model.vqvae.vector_quantizer,
    )

    with pytest.raises(RuntimeError, match="reference forward failed"):
        capture_reference_stages(SimpleNamespace(model=reference_model), batch)

    _assert_batch_unchanged(batch, original)
    assert all(len(module._forward_hooks) == 0 for module in hooked_modules)


def test_stok_adapter_propagates_failure_and_preserves_batch():
    batch = _batch()
    original = _batch_snapshot(batch)
    stok_encoder = StubStokEncoder().eval()
    stok_encoder.gcpnet.raise_on_call = True

    with pytest.raises(RuntimeError, match="gcpnet failed"):
        capture_stok_stages(stok_encoder, batch)

    _assert_batch_unchanged(batch, original)
    assert stok_encoder.featurizer.call_states == [(False, False)]
    assert stok_encoder.gcpnet.call_states == [(False, False)]
