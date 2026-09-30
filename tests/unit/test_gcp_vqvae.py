import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from stok.utils.pretrained import load_gcp_config


def tiny_config():
    config = load_gcp_config("lite")
    config["max_length"] = 16
    config["gcp"]["num_layers"] = 1
    config["gcp"]["model_cfg"]["num_layers"] = 1
    config["encoder"].update(dimension=32, depth=1, heads=4, attn_kv_heads=2, ff_mult=1)
    config["quantizer"].update(
        dim=16, codebook_size=32, kmeans_init=False, orthogonal_reg_weight=0
    )
    return config


def small_graph():
    torch.manual_seed(4)
    graphs = []
    for length in (8, 5):
        coords = torch.randn(length, 37, 3)
        edges = torch.tensor(
            [[i for i in range(length - 1)], [i + 1 for i in range(length - 1)]]
        )
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        graphs.append(
            Data(
                coords=coords,
                residue_type=torch.zeros(length, dtype=torch.long),
                seq_pos=torch.arange(length)[:, None],
                residue_index=torch.arange(length),
                edge_index=edges,
                edge_type=torch.zeros(edges.size(1), dtype=torch.long),
                num_nodes=length,
            )
        )
    graph = Batch.from_data_list(graphs)
    # PyG offsets attributes containing "index"; these positions are per example.
    graph.residue_index = torch.cat([item.residue_index for item in graphs])
    residue_mask = torch.arange(8)[None] < torch.tensor([8, 5])[:, None]
    token_mask = residue_mask.clone()
    token_mask[0, 3] = False
    return graph, residue_mask, token_mask


def test_modules_round_trip_masks_graph_reuse_and_inference_state():
    from stok.models.gcp_vqvae import GCPVQTokenizer
    from vector_quantize_pytorch import VectorQuantize

    config = tiny_config()
    model = GCPVQTokenizer(config).eval()
    assert isinstance(model.quantizer, VectorQuantize)
    graph, residues, tokens = small_graph()
    original = graph.clone()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        latent = model.encoder(graph, token_mask=tokens)
        codes, indices, loss = model(graph, residue_mask=residues, token_mask=tokens)
        repeated = model.encode(graph, residue_mask=residues, token_mask=tokens)
    assert latent.shape == (2, 8, 16)
    assert codes.shape == (2, 8, 16)
    assert indices.shape == (2, 8) and indices.dtype == torch.int64
    assert (indices[~tokens] == -1).all()
    assert ((indices[tokens] >= 0) & (indices[tokens] < 32)).all()
    assert torch.equal(indices, repeated)
    assert torch.isfinite(loss).all()
    assert set(graph.keys()) == set(original.keys())
    for key in graph.keys():
        if isinstance(graph[key], torch.Tensor):
            assert torch.equal(graph[key], original[key]), key
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in state.items()
    )
    assert {
        "quantizer._codebook.initted",
        "quantizer._codebook.embed_avg",
        "quantizer._codebook.cluster_size",
    } <= set(state)
    restored = GCPVQTokenizer(copy.deepcopy(config)).eval()
    restored.load_state_dict(state, strict=True)
    assert torch.equal(
        restored.encode(graph, residue_mask=residues, token_mask=tokens), indices
    )


@pytest.mark.parametrize(
    "fault",
    [
        "shape",
        "dtype",
        "outside",
        "all_missing",
        "position",
        "duplicate",
        "missing_node",
    ],
)
def test_invalid_masks_or_node_maps_fail_before_attention(fault):
    from stok.models.gcp_vqvae import GCPVQTokenizer

    model = GCPVQTokenizer(tiny_config()).eval()
    graph, residues, tokens = small_graph()
    if fault == "shape":
        tokens = tokens[:, :-1]
    elif fault == "dtype":
        tokens = tokens.float()
    elif fault == "outside":
        tokens[1, 7] = True
    elif fault == "all_missing":
        tokens[0] = False
    elif fault == "position":
        graph.residue_index[0] = -1
    elif fault == "duplicate":
        graph.residue_index[0] = graph.residue_index[1]
    else:
        residues = torch.nn.functional.pad(residues, (0, 1))
        tokens = torch.nn.functional.pad(tokens, (0, 1))
        residues[0, 8] = tokens[0, 8] = True
    with pytest.raises(ValueError):
        model.encode(graph, residue_mask=residues, token_mask=tokens)


def test_encoder_forward_keeps_gradients_available():
    from stok.models.gcp_vqvae import GCPVQEncoder

    model = GCPVQEncoder(tiny_config())
    graph, _, tokens = small_graph()
    latent = model(graph, token_mask=tokens)
    latent[tokens].square().mean().backward()
    assert model.encoder_head[0].weight.grad is not None
    assert torch.isfinite(model.encoder_head[0].weight.grad).all()


def test_empty_scatter_groups_match_reference_zero():
    from stok.utils.gcp import scatter_reduce

    values = torch.tensor([[-3.0], [-2.0]])
    groups = torch.tensor([0, 0])
    assert torch.equal(
        scatter_reduce(values, groups, 0, 3, "max"),
        torch.tensor([[-2.0], [0.0], [0.0]]),
    )
    assert torch.equal(
        scatter_reduce(values, groups, 0, 3, "min"),
        torch.tensor([[-3.0], [0.0], [0.0]]),
    )


def test_featurizer_requires_supported_positional_features():
    from stok.utils.featurizer import ProteinFeaturiser

    with pytest.raises(ValueError, match="positional"):
        ProteinFeaturiser(scalar_node_features=["amino_acid_one_hot"])


@pytest.mark.parametrize("preset", ["lite", "large"])
def test_published_encoder_and_vq_stage_parity(preset):
    from stok.models.gcp_vqvae import load_pretrained_tokenizer
    from tests.reference.generate_gcp_vqvae import verify_fixture_set

    weights = os.environ.get("STOK_GCP_WEIGHTS")
    fixtures = os.environ.get("STOK_GCP_REFERENCE_FIXTURES")
    if not weights or not fixtures:
        pytest.skip(
            "Set STOK_GCP_WEIGHTS and STOK_GCP_REFERENCE_FIXTURES for published-weight stage parity"
        )
    root = Path(fixtures)
    manifest = verify_fixture_set(root, require_models=True)
    device = os.environ.get("STOK_GCP_DEVICE", "cpu")
    assert manifest["environment"]["device"] == device
    model = load_pretrained_tokenizer(
        preset, path=Path(weights) / preset / "best_valid.pth", device=device
    )
    before = {key: value.clone() for key, value in model.quantizer.state_dict().items()}
    for case in manifest["cases"]:
        if case["preset"] != preset or case["kind"] != "full":
            continue
        with np.load(root / case["fixture"]["path"]) as arrays:
            graph = Batch()
            for key in arrays.files:
                if key.startswith("graph_"):
                    graph[key[6:]] = torch.from_numpy(arrays[key].copy())
            graph.num_nodes = graph.coords.size(0)
            graph._num_graphs = len(case["lengths"])
            graph.ptr = torch.tensor([0, *np.cumsum(case["lengths"]).tolist()])
            graph._slice_dict = {"coords": graph.ptr}
            graph.residue_index = graph.seq_pos[:, 0].long()
            graph = graph.to(device)
            masks = {
                key: torch.from_numpy(arrays[key].copy()).to(device)
                for key in ("residue_mask", "token_mask")
            }
            captured = {}
            hooks = []

            def capture(name, transpose=False, gcp=False):
                def hook(module, inputs, outputs):
                    value = outputs["node_embedding"] if gcp else outputs
                    captured[name] = value.transpose(1, 2) if transpose else value

                return hook

            hooks.append(
                model.encoder.encoder.register_forward_hook(
                    capture("gcp_embeddings", gcp=True)
                )
            )
            hooks.append(
                model.encoder.encoder_tail.register_forward_hook(
                    capture("encoder_projection", transpose=True)
                )
            )
            hooks.append(
                model.encoder.encoder_blocks.register_forward_hook(
                    capture("encoder_embeddings")
                )
            )
            hooks.append(
                model.encoder.encoder_head.register_forward_hook(
                    capture("encoder_latents", transpose=True)
                )
            )
            with torch.inference_mode():
                codes, indices, _ = model(graph, **masks)
            for hook in hooks:
                hook.remove()
            captured["vq_codes"] = codes
            for name, actual in captured.items():
                torch.testing.assert_close(
                    actual.cpu(),
                    torch.from_numpy(arrays[name].copy()),
                    rtol=1e-5,
                    # Native ROCm reductions differ from torch-scatter by a few
                    # ulps; the transformer amplifies them. Measured in tests/README.
                    atol=2e-5
                    if device != "cpu" and name == "encoder_embeddings"
                    else 1e-5,
                    msg=lambda msg: f"{case['name']}/{name}: {msg}",
                )
            assert torch.equal(
                indices.cpu(), torch.from_numpy(arrays["vq_indices"].copy())
            ), case["name"]
    assert all(
        torch.equal(value, model.quantizer.state_dict()[key])
        for key, value in before.items()
    )
