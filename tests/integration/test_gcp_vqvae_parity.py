"""Published-weight file-to-coordinate checks; no upstream runtime imports."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from stok.data.structure_encoding import (
    batch_structure_graphs,
    prepare_reference_structure,
)
from stok.models.decoder import load_pretrained_decoder
from stok.models.gcp_vqvae import load_pretrained_tokenizer
from stok.utils.decoding import decode_structure_tokens, indices_to_codes
from stok.utils.featurizer import ProteinFeaturiser
from tests.reference.generate_gcp_vqvae import verify_fixture_set


@pytest.mark.parametrize("preset", ["lite", "large"])
def test_published_file_graph_token_and_decoder_parity(preset):
    weights = os.environ.get("STOK_GCP_WEIGHTS")
    fixtures = os.environ.get("STOK_GCP_REFERENCE_FIXTURES")
    if not weights or not fixtures:
        pytest.skip(
            "Set STOK_GCP_WEIGHTS and STOK_GCP_REFERENCE_FIXTURES for full file parity"
        )
    root = Path(fixtures)
    manifest = verify_fixture_set(root, require_models=True)
    device = os.environ.get("STOK_GCP_DEVICE", "cpu")
    assert manifest["environment"]["device"] == device, (
        "Use a reference capture on the same backend"
    )
    tokenizer = load_pretrained_tokenizer(
        preset, path=Path(weights) / preset / "best_valid.pth", device=device
    )
    decoder = load_pretrained_decoder(
        preset, path=Path(weights) / preset / "best_valid.pth", device=device
    )
    codebook = tokenizer.quantizer._codebook.embed.squeeze(0)
    span = manifest["configuration"]["max_length"]
    singleton_ids = {}
    for case in manifest["cases"]:
        if case["preset"] != preset or case["kind"] not in {"full", "decoder"}:
            continue
        with np.load(root / case["fixture"]["path"]) as arrays:
            if case["kind"] == "full":
                prepared = [
                    prepare_reference_structure(
                        root / source["path"], chain_id=source["chain_id"]
                    )
                    for source in case["sources"]
                ]
                graph = batch_structure_graphs(
                    [item[0].to_data_list()[0] for item in prepared]
                )
                residues = torch.cat([item[1][:, :span] for item in prepared]).to(
                    device
                )
                tokens = torch.cat([item[2][:, :span] for item in prepared]).to(device)
                features = ProteinFeaturiser()(graph.clone())
                for key in (
                    "edge_index",
                    "batch",
                    "seq_pos",
                    "coords",
                    "residue_type",
                    "x",
                    "x_vector_attr",
                    "edge_attr",
                    "edge_vector_attr",
                    "pos",
                ):
                    actual = features[key]
                    expected = torch.from_numpy(arrays[f"graph_{key}"].copy())
                    if actual.is_floating_point():
                        torch.testing.assert_close(
                            actual,
                            expected,
                            rtol=1e-5,
                            atol=1e-5,
                            msg=lambda msg: f"{case['name']}/graph_{key}: {msg}",
                        )
                    else:
                        assert torch.equal(actual, expected), (case["name"], key)
                graph = graph.to(device)
                with torch.inference_mode():
                    codes, indices, _ = tokenizer(
                        graph, residue_mask=residues, token_mask=tokens
                    )
                assert torch.equal(
                    indices.cpu(), torch.from_numpy(arrays["vq_indices"].copy())
                ), case["name"]
                torch.testing.assert_close(
                    codes.cpu(),
                    torch.from_numpy(arrays["vq_codes"].copy()),
                    rtol=1e-5,
                    atol=1e-5,
                )
                if len(prepared) == 1:
                    singleton_ids[case["sources"][0]["path"]] = (
                        indices[0].clone(),
                        torch.from_numpy(arrays["vq_indices"][0].copy()).to(device),
                    )
                else:
                    for i, source in enumerate(case["sources"]):
                        # Capture upstream batch-dependent terminal angles rather than
                        # claiming singleton invariance that the release does not have.
                        single, expected_single = singleton_ids[source["path"]]
                        assert torch.equal(
                            indices[i] != single,
                            torch.from_numpy(arrays["vq_indices"][i].copy()).to(device)
                            != expected_single,
                        )
                with torch.inference_mode():
                    raw = decoder(codes, tokens).reshape(*indices.shape, 3, 3)
                    formatted = decode_structure_tokens(
                        decoder, codebook, indices, residue_mask=residues
                    )
                expected = torch.from_numpy(arrays["decoder_coordinates"].copy()).to(
                    device
                )
                torch.testing.assert_close(raw, expected, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(
                    formatted[tokens], expected[tokens], rtol=1e-5, atol=1e-5
                )
                assert torch.isnan(formatted[~tokens]).all()
            else:
                indices = torch.from_numpy(arrays["vq_indices"].copy()).to(device)
                tokens = torch.from_numpy(arrays["token_mask"].copy()).to(device)
                codes = indices_to_codes(codebook, indices, allow_missing=True)
                reference_codes = torch.from_numpy(arrays["vq_codes"].copy()).to(device)
                torch.testing.assert_close(
                    codes[tokens],
                    reference_codes[tokens],
                    rtol=0,
                    atol=0,
                )
                assert (codes[~tokens] == 0).all()
                kwargs = (
                    {
                        "true_lengths": torch.from_numpy(
                            arrays["true_lengths"].copy()
                        ).to(device)
                    }
                    if "true_lengths" in arrays.files
                    else {}
                )
                with torch.inference_mode():
                    # Pinned VQ lookup uses Python -1 indexing. Low-level parity
                    # retains its captured inputs; public lookup returns zeros.
                    raw = decoder(reference_codes, tokens, **kwargs).reshape(
                        *indices.shape, 3, 3
                    )
                torch.testing.assert_close(
                    raw.cpu(),
                    torch.from_numpy(arrays["decoder_coordinates"].copy()),
                    rtol=1e-5,
                    atol=1e-5,
                )
