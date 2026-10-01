"""Offline checks for the reference oracle's integrity contract."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def write_oracle(directory):
    """A hand-sized preparation case; no reference imports or weights."""
    from tests.reference.generate_gcp_vqvae import RELEASES, SOURCE

    directory.mkdir(exist_ok=True)
    (directory / "input.pdb").write_text("END\n")
    arrays = {
        "parsed_coordinates": np.zeros((25, 4, 3), dtype=np.float32),
        "prepared_coordinates": np.zeros((25, 4, 3), dtype=np.float32),
        "residue_mask": np.ones((1, 32), dtype=bool),
        "token_mask": np.ones((1, 32), dtype=bool),
    }
    arrays["residue_mask"][:, 25:] = False
    arrays["token_mask"][:, 25:] = False
    np.savez_compressed(directory / "case.npz", **arrays)
    manifest = {
        "schema_version": 1,
        "source": SOURCE,
        "artifacts": RELEASES,
        "environment": {
            "dependencies": {
                "x-transformers": "2.8.0",
                "vector-quantize-pytorch": "1.25.2",
            },
            "python": "3.12.14",
            "device": "cpu",
            "dtype": "float32",
            "attention": "torch SDPA",
        },
        "configuration": {"max_length": 32},
        "generation_command": ["generate_gcp_vqvae.py", "--preparation-only"],
        "inputs": {"input.pdb": hashlib.sha256(b"END\n").hexdigest()},
        "cases": [
            {
                "name": "complete",
                "preset": "lite",
                "kind": "preparation",
                "lengths": [25],
                "sequences": ["A" * 25],
                "stages": sorted(arrays),
                "fixture": {
                    "path": "case.npz",
                    "sha256": hashlib.sha256(
                        (directory / "case.npz").read_bytes()
                    ).hexdigest(),
                    "arrays": {
                        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for key, value in arrays.items()
                    },
                },
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_fixture_integrity_rejects_corruption_and_invalid_provenance(tmp_path):
    from tests.reference.generate_gcp_vqvae import verify_fixture_set

    manifest = write_oracle(tmp_path)
    assert verify_fixture_set(tmp_path)["cases"][0]["lengths"] == [25]
    manifest["cases"][0]["fixture"]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        verify_fixture_set(tmp_path)


@pytest.mark.parametrize(
    "fault", ["duplicate", "shape", "provenance", "mask", "input", "escape", "empty"]
)
def test_fixture_integrity_rejects_malformed_cases(tmp_path, fault):
    from tests.reference.generate_gcp_vqvae import verify_fixture_set

    manifest = write_oracle(tmp_path)
    if fault == "duplicate":
        manifest["cases"].append(manifest["cases"][0])
    elif fault == "shape":
        manifest["cases"][0]["fixture"]["arrays"]["prepared_coordinates"]["shape"] = [
            24,
            4,
            3,
        ]
    elif fault == "provenance":
        del manifest["environment"]["dependencies"]["x-transformers"]
    elif fault == "mask":
        manifest["cases"][0]["lengths"] = [24]
        manifest["cases"][0]["sequences"] = ["A" * 24]
    elif fault == "input":
        (tmp_path / "input.pdb").write_text("corrupt\n")
    elif fault == "escape":
        manifest["cases"][0]["fixture"]["path"] = "../case.npz"
    else:
        manifest["cases"] = []
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_fixture_set(tmp_path)


def test_full_weight_check_cannot_pass_preparation_only(tmp_path):
    from tests.reference.generate_gcp_vqvae import verify_fixture_set

    write_oracle(tmp_path)
    with pytest.raises(ValueError, match="published-weight"):
        verify_fixture_set(tmp_path, require_models=True)


def test_checked_in_preparation_oracle_is_valid():
    from tests.reference.generate_gcp_vqvae import verify_fixture_set

    root = Path(__file__).parents[1] / "test_data" / "gcp_vqvae"
    manifest = verify_fixture_set(root)
    assert {case["preset"] for case in manifest["cases"]} == {"lite", "large"}
    assert any(case["kind"] == "rejected" for case in manifest["cases"])


def write_model_oracle(directory):
    """Tiny literal stage arrays exercise inventory validation without weights."""
    from tests.reference.generate_gcp_vqvae import _write_case

    manifest = write_oracle(directory)
    manifest["cases"] = []
    file_cases = [
        "complete_pdb",
        "complete_cif",
        "incomplete_pdb",
        "incomplete_cif",
        "renumbered",
        "insertion_codes",
    ]
    for preset, dimension in [("lite", 128), ("large", 256)]:
        for name in [*file_cases, "unequal_batch", "decoder_holes", "decoder_prefix"]:
            lengths = [25, 26] if name == "unequal_batch" else [25]
            batch, nodes = len(lengths), sum(lengths)
            mask = np.arange(32)[None, :] < np.array(lengths)[:, None]
            indices = np.zeros((batch, 32), dtype=np.int64)
            indices[~mask] = -1
            arrays = {
                "parsed_coordinates": np.zeros((nodes, 4, 3), dtype=np.float32),
                "prepared_coordinates": np.zeros((nodes, 4, 3), dtype=np.float32),
                "residue_mask": mask,
                "token_mask": mask,
                "graph_edge_index": np.array([[0], [1]], dtype=np.int64),
                "graph_batch": np.repeat(np.arange(batch), lengths),
                "graph_seq_pos": np.zeros((nodes, 1), dtype=np.int64),
                "graph_x": np.zeros((nodes, 49), dtype=np.float32),
                "graph_x_vector_attr": np.zeros((nodes, 2, 3), dtype=np.float32),
                "graph_edge_attr": np.zeros((1, 9), dtype=np.float32),
                "graph_edge_vector_attr": np.zeros((1, 1, 3), dtype=np.float32),
                "graph_pos": np.zeros((nodes, 3), dtype=np.float32),
                "gcp_embeddings": np.zeros((nodes, 128), dtype=np.float32),
                "encoder_projection": np.zeros((batch, 32, 1024), dtype=np.float32),
                "encoder_embeddings": np.zeros((batch, 32, 1024), dtype=np.float32),
                "encoder_latents": np.zeros((batch, 32, dimension), dtype=np.float32),
                "vq_codes": np.zeros((batch, 32, dimension), dtype=np.float32),
                "vq_indices": indices,
                "decoder_coordinates": np.zeros((batch, 32, 3, 3), dtype=np.float32),
            }
            kind = "decoder" if name.startswith("decoder_") else "full"
            if name == "decoder_prefix":
                arrays["true_lengths"] = np.array([17], dtype=np.int64)
            case = _write_case(
                directory,
                f"{preset}_{name}",
                preset,
                kind,
                lengths,
                ["A" * n for n in lengths],
                arrays,
            )
            manifest["cases"].append(case)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize(
    "fault",
    [
        "missing_prefix",
        "prefix_shape",
        "prefix_float",
        "prefix_negative",
        "prefix_too_long",
        "wrong_kind",
    ],
)
def test_full_inventory_requires_real_prefix_and_file_stages(tmp_path, fault):
    from tests.reference.generate_gcp_vqvae import verify_fixture_set

    manifest = write_model_oracle(tmp_path)
    verify_fixture_set(tmp_path, require_models=True)
    prefix = next(
        case for case in manifest["cases"] if case["name"] == "lite_decoder_prefix"
    )
    if fault == "missing_prefix":
        holes = next(
            case for case in manifest["cases"] if case["name"] == "lite_decoder_holes"
        )
        prefix["fixture"], prefix["stages"] = holes["fixture"], holes["stages"]
    elif fault == "wrong_kind":
        manifest["cases"][0]["kind"] = "decoder"
    else:
        path = tmp_path / prefix["fixture"]["path"]
        with np.load(path) as stored:
            arrays = {key: stored[key] for key in stored.files}
        values = {
            "prefix_shape": [[17]],
            "prefix_float": [17.5],
            "prefix_negative": [-1],
            "prefix_too_long": [33],
        }
        arrays["true_lengths"] = np.array(values[fault])
        np.savez_compressed(path, **arrays)
        prefix["fixture"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        prefix["fixture"]["arrays"]["true_lengths"] = {
            "shape": list(arrays["true_lengths"].shape),
            "dtype": str(arrays["true_lengths"].dtype),
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_fixture_set(tmp_path, require_models=True)
