from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .metrics import compare_floats, compare_indices
from .types import TensorComparison


@dataclass(frozen=True)
class StageCapture:
    tensors: dict[str, torch.Tensor]


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and "node_embedding" in output:
        return output["node_embedding"]
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot extract tensor from hook output {type(output).__name__}")


def capture_module_outputs(
    root: nn.Module,
    modules: Mapping[str, nn.Module],
    invoke: Callable[[], Any],
    transforms: Mapping[str, Callable[[Any], torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    del root
    captured: dict[str, torch.Tensor] = {}
    transforms = transforms or {}
    handles = []
    for name, module in modules.items():

        def hook(_module, _inputs, output, *, stage=name):
            value = transforms.get(stage, _extract_tensor)(output)
            captured[stage] = value.detach().clone().cpu()

        handles.append(module.register_forward_hook(hook))
    try:
        invoke()
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(modules) - set(captured))
    if missing:
        raise RuntimeError(f"Stages did not execute: {missing}")
    return captured


def _conv_output(output: Any) -> torch.Tensor:
    return _extract_tensor(output).transpose(1, 2)


def _vq_embeddings(output: Any) -> torch.Tensor:
    if not isinstance(output, tuple) or len(output) < 2:
        raise TypeError("Vector quantizer hook did not return embeddings and indices")
    return output[0]


def capture_reference_stages(wrapper: Any, batch: Mapping[str, Any]) -> StageCapture:
    super_model = wrapper.model
    if super_model is None:
        raise RuntimeError("Reference wrapper has no loaded model")
    vqvae = super_model.vqvae
    graph = copy.deepcopy(batch["graph"])
    reference_batch = dict(batch)
    reference_batch["graph"] = graph
    valid = batch["masks"].to(torch.bool) & batch["nan_masks"].to(torch.bool)
    tensors = {
        "featurizer.x": graph.x.detach().clone().cpu(),
        "featurizer.x_vector_attr": graph.x_vector_attr.detach().clone().cpu(),
        "featurizer.edge_attr": graph.edge_attr.detach().clone().cpu(),
        "featurizer.edge_vector_attr": graph.edge_vector_attr.detach().clone().cpu(),
        "valid": valid.detach().clone().cpu(),
    }
    output_holder: dict[str, Any] = {}
    captured = capture_module_outputs(
        super_model,
        {
            "gcpnet": super_model.encoder,
            "encoder_tail": vqvae.encoder_tail,
            "encoder_blocks": vqvae.encoder_blocks,
            "encoder_head": vqvae.encoder_head,
            "embeddings": vqvae.vector_quantizer,
        },
        lambda: output_holder.setdefault(
            "output", super_model(reference_batch, return_vq_layer=True)
        ),
        transforms={
            "encoder_tail": _conv_output,
            "encoder_head": _conv_output,
            "embeddings": _vq_embeddings,
        },
    )
    output = output_holder["output"]
    tensors.update(captured)
    tensors["indices"] = output["indices"].detach().clone().cpu()
    tensors["embeddings"] = output["embeddings"].detach().clone().cpu()
    return StageCapture(tensors=tensors)


def capture_stok_stages(encoder: nn.Module, batch: Mapping[str, Any]) -> StageCapture:
    from stok.utils.batching import unbatch_and_pad

    canonical_graph = copy.deepcopy(batch["graph"])
    stok_feature_graph = encoder.featurizer(copy.deepcopy(batch["graph"]))
    valid = batch["masks"].to(torch.bool) & batch["nan_masks"].to(torch.bool)

    tensors = {
        f"featurizer.{field}": getattr(stok_feature_graph, field).detach().clone().cpu()
        for field in ("x", "x_vector_attr", "edge_attr", "edge_vector_attr")
    }

    with torch.inference_mode():
        node_embedding = encoder.gcpnet(canonical_graph)["node_embedding"]
        x = unbatch_and_pad(node_embedding, canonical_graph.batch, encoder.max_length)
        tail = encoder.encoder_tail(x.transpose(1, 2)).transpose(1, 2)
        blocks = encoder.encoder_blocks(tail, mask=valid)
        head = encoder.encoder_head(blocks.transpose(1, 2)).transpose(1, 2)
        embeddings, indices, _ = encoder.vector_quantizer(head, mask=valid)

    tensors.update(
        {
            "gcpnet": node_embedding.detach().clone().cpu(),
            "encoder_tail": tail.detach().clone().cpu(),
            "encoder_blocks": blocks.detach().clone().cpu(),
            "encoder_head": head.detach().clone().cpu(),
            "indices": indices.detach().clone().cpu(),
            "embeddings": embeddings.detach().clone().cpu(),
            "valid": valid.detach().clone().cpu(),
        }
    )
    return StageCapture(tensors=tensors)


def compare_stage_captures(
    reference: StageCapture,
    stok: StageCapture,
    *,
    rtol: float,
    atol: float,
) -> list[TensorComparison]:
    reference_keys = set(reference.tensors)
    stok_keys = set(stok.tensors)
    if reference_keys != stok_keys:
        raise ValueError(
            f"Stage key mismatch: missing={sorted(reference_keys - stok_keys)}, "
            f"unexpected={sorted(stok_keys - reference_keys)}"
        )
    valid_ref = reference.tensors["valid"]
    valid_stok = stok.tensors["valid"]
    comparisons: list[TensorComparison] = []
    for name in sorted(reference_keys - {"valid"}):
        lhs = reference.tensors[name]
        rhs = stok.tensors[name]
        if name == "indices":
            comparisons.append(compare_indices(name, lhs, rhs, valid_ref, valid_stok))
        elif name == "embeddings":
            result = compare_floats(name, lhs, rhs, rtol=0.0, atol=0.0)
            comparisons.append(result)
        else:
            comparisons.append(compare_floats(name, lhs, rhs, rtol=rtol, atol=atol))
    return comparisons
