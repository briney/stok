"""Generate an independent oracle from the pinned upstream checkout.

Upstream imports occur only during generation. Verification needs NumPy alone.
No weights are downloaded, and no STok model is used to generate expectations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np

SOURCE = {
    "url": "https://github.com/mahdip72/vq_encoder_decoder",
    "commit": "68c4c284fe204de27fdf61db27fcc01136ea9f28",
    "package_metadata_sha256": "9a18e9540bf6f7801a09196323c0ae184e04fd5849289f792243d782c5491a98",
}
RELEASES = {
    "lite": {
        "repository": "Mahdip72/gcp-vqvae-lite",
        "revision": "d67ca3dfc1ecd6c6a1c3841f584e82f2f9803447",
        "files": {
            "best_valid.pth": "0a212868f83ba1cc426404ec5d09c0d83012c8e34715483a7107091ffcf16e30",
            "config_gcpnet_encoder.yaml": "54dfa49d2113994415456efb9babb2d6abd25c0a73436ac73d5cd51a68d90dd8",
            "config_geometric_decoder.yaml": "5913125ce9d861debbd93599282a4d2003c42684a16dc8c1bd87557a1dedbf44",
            "config_vqvae.yaml": "c08736c872568758c0ab5bbfaa4cd384851c8d3c9c0405c4669b9fadc71e3e1f",
        },
    },
    "large": {
        "repository": "Mahdip72/gcp-vqvae-large",
        "revision": "64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1",
        "files": {
            "best_valid.pth": "7d1d43950a29834e7f702409bf957e9ffb75eb3cb3952074ba4c545bb4130eaf",
            "config_gcpnet_encoder.yaml": "be1662f7f37a79f8b91810eaa1e50ff2f1294b750370ad6e4d46e320c65db633",
            "config_geometric_decoder.yaml": "d195ee74478ca6f73001c4d80beb68f7ebf45201793147564d776313baaa9992",
            "config_vqvae.yaml": "f3d5cc99a15c550ac6af253214e94c7f7cb583b846d45e40ad3ba2dc7d18f96e",
        },
    },
}
PINNED_DEPENDENCIES = {"x-transformers": "2.8.0", "vector-quantize-pytorch": "1.25.2"}
PREPARATION_STAGES = {
    "parsed_coordinates",
    "prepared_coordinates",
    "residue_mask",
    "token_mask",
}
MODEL_STAGES = {
    "graph_edge_index",
    "graph_batch",
    "graph_seq_pos",
    "graph_x",
    "graph_x_vector_attr",
    "graph_edge_attr",
    "graph_edge_vector_attr",
    "graph_pos",
    "gcp_embeddings",
    "encoder_projection",
    "encoder_embeddings",
    "encoder_latents",
    "vq_indices",
    "vq_codes",
    "decoder_coordinates",
}
LARGE_UNUSED_HEAD = {
    "vqvae.decoder.pairwise_classification_head." + name
    for name in (
        "downproject.weight",
        "linear1.weight",
        "norm.weight",
        "norm.bias",
        "linear2.weight",
    )
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Fixture path escapes directory: {relative}")
    return path


def verify_fixture_set(directory, *, require_models=False):
    """Fail closed on missing/corrupt fixtures and malformed provenance/shapes."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("source") != SOURCE:
        raise ValueError("Missing or unsupported reference provenance")
    if manifest.get("artifacts") != RELEASES:
        raise ValueError("Missing immutable artifact provenance")
    environment = manifest.get("environment", {})
    if any(
        environment.get("dependencies", {}).get(k) != v
        for k, v in PINNED_DEPENDENCIES.items()
    ):
        raise ValueError("Missing or incorrect dependency provenance")
    if not all(environment.get(k) for k in ("python", "device", "dtype", "attention")):
        raise ValueError("Missing execution provenance")
    if environment["dtype"] != "float32":
        raise ValueError("The initial reference oracle requires FP32")
    max_length = manifest.get("configuration", {}).get("max_length")
    if not isinstance(max_length, int) or max_length < 25 or max_length > 1280:
        raise ValueError("Invalid reference length configuration")
    if not manifest.get("generation_command") or not manifest.get("inputs"):
        raise ValueError("Missing generation/input provenance")
    for name, expected in manifest["inputs"].items():
        if sha256(_inside(directory, name)) != expected:
            raise ValueError(f"Input digest mismatch: {name}")
    cases = manifest.get("cases", [])
    names = [case["name"] for case in cases]
    if not names or len(set(names)) != len(names):
        raise ValueError("Empty or duplicate fixture case names")
    for case in cases:
        if case.get("preset") not in RELEASES:
            raise ValueError("Unknown fixture preset")
        kind = case.get("kind")
        if kind == "rejected":
            if not case.get("rejections"):
                raise ValueError("Rejected case lacks rejection accounting")
            continue
        if kind not in {"preparation", "full", "decoder"}:
            raise ValueError("Unknown fixture kind")
        fixture = case["fixture"]
        path = _inside(directory, fixture["path"])
        if sha256(path) != fixture["sha256"]:
            raise ValueError(f"Fixture digest mismatch: {case['name']}")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(fixture["arrays"]) or set(arrays.files) != set(
                case["stages"]
            ):
                raise ValueError("Fixture stage inventory mismatch")
            for key, description in fixture["arrays"].items():
                value = arrays[key]
                if (
                    list(value.shape) != description["shape"]
                    or str(value.dtype) != description["dtype"]
                ):
                    raise ValueError(f"Fixture dimensions/dtype mismatch: {key}")
            required = (
                {"vq_codes", "vq_indices", "token_mask", "decoder_coordinates"}
                if kind == "decoder"
                else PREPARATION_STAGES
            )
            if kind == "full":
                required = required | MODEL_STAGES
            if not required <= set(arrays.files):
                raise ValueError("Missing required fixture stages")
            lengths = case["lengths"]
            if not lengths or any(
                type(n) is not int or not 0 < n <= max_length for n in lengths
            ):
                raise ValueError("Invalid fixture lengths")
            batch = len(lengths)
            token_mask = arrays["token_mask"]
            if token_mask.shape != (batch, max_length) or token_mask.dtype != bool:
                raise ValueError("Invalid token mask dimensions/dtype")
            if kind != "decoder":
                residue_mask = arrays["residue_mask"]
                expected_mask = (
                    np.arange(max_length)[None, :] < np.array(lengths)[:, None]
                )
                if residue_mask.dtype != bool or not np.array_equal(
                    residue_mask, expected_mask
                ):
                    raise ValueError("Residue mask disagrees with source lengths")
                if (token_mask & ~residue_mask).any():
                    raise ValueError("Token mask includes padding")
                if [len(seq) for seq in case["sequences"]] != lengths:
                    raise ValueError("Sequence/length mismatch")
                for key in ("parsed_coordinates", "prepared_coordinates"):
                    if (
                        arrays[key].shape != (sum(lengths), 4, 3)
                        or arrays[key].dtype != np.float32
                    ):
                        raise ValueError(f"Invalid coordinate dimensions/dtype: {key}")
            if kind in {"full", "decoder"}:
                dimension = 128 if case["preset"] == "lite" else 256
                indices = arrays["vq_indices"]
                if indices.shape != token_mask.shape or indices.dtype != np.int64:
                    raise ValueError("Invalid VQ index dimensions/dtype")
                if (
                    ((indices < -1) | (indices >= 4096)).any()
                    or (indices[~token_mask] != -1).any()
                    or (indices[token_mask] < 0).any()
                ):
                    raise ValueError("Invalid VQ indices/missing slots")
                if arrays["vq_codes"].shape != (batch, max_length, dimension):
                    raise ValueError("Invalid VQ code dimensions")
                if arrays["decoder_coordinates"].shape != (batch, max_length, 3, 3):
                    raise ValueError("Invalid decoder dimensions")
            if case["name"].endswith("decoder_prefix"):
                if "true_lengths" not in arrays.files:
                    raise ValueError("Missing decoder prefix lengths")
                prefix_lengths = arrays["true_lengths"]
                if (
                    prefix_lengths.shape != (batch,)
                    or not np.issubdtype(prefix_lengths.dtype, np.integer)
                    or ((prefix_lengths < 0) | (prefix_lengths > max_length)).any()
                ):
                    raise ValueError("Invalid decoder prefix lengths")
            if kind == "full":
                nodes = sum(lengths)
                edges = arrays["graph_edge_index"]
                if (
                    edges.ndim != 2
                    or edges.shape[0] != 2
                    or edges.dtype != np.int64
                    or ((edges < 0) | (edges >= nodes)).any()
                ):
                    raise ValueError("Invalid graph edges")
                if arrays["gcp_embeddings"].shape != (nodes, 128) or arrays[
                    "graph_batch"
                ].shape != (nodes,):
                    raise ValueError("Invalid graph/node dimensions")
                if arrays["encoder_latents"].shape != (batch, max_length, dimension):
                    raise ValueError("Invalid encoder latent dimensions")
    if require_models:
        for preset in RELEASES:
            model_cases = [case for case in cases if case["preset"] == preset]
            if not any(
                case["kind"] == "full"
                and len(case["lengths"]) > 1
                and len(set(case["lengths"])) > 1
                for case in model_cases
            ):
                raise ValueError(
                    f"Missing published-weight unequal-batch cases: {preset}"
                )
            for name in (
                "complete_pdb",
                "complete_cif",
                "incomplete_pdb",
                "incomplete_cif",
                "renumbered",
                "insertion_codes",
                "decoder_holes",
                "decoder_prefix",
            ):
                expected_kind = "decoder" if name.startswith("decoder_") else "full"
                if not any(
                    case["name"] == f"{preset}_{name}" and case["kind"] == expected_kind
                    for case in model_cases
                ):
                    raise ValueError(f"Missing published-weight case: {preset}_{name}")
    return manifest


def _write_case(output, name, preset, kind, lengths, sequences, arrays):
    path = output / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return {
        "name": name,
        "preset": preset,
        "kind": kind,
        "lengths": lengths,
        "sequences": sequences,
        "stages": sorted(arrays),
        "fixture": {
            "path": path.name,
            "sha256": sha256(path),
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
        },
    }


def _load_model(folder, max_length):
    """Load the trusted release strictly, without upstream's permissive loader."""
    import logging
    import torch
    import yaml
    from box import Box
    from gcp_vqvae._internal.models.decoders import GeometricDecoder
    from gcp_vqvae._internal.models.gcpnet.models.base import (
        instantiate_encoder,
        PretrainedEncoder,
    )
    from gcp_vqvae._internal.models.super_model import SuperModel
    from gcp_vqvae._internal.models.vqvae import VQVAETransformer
    from gcp_vqvae._internal.utils.utils import load_configs

    configs = load_configs(yaml.safe_load((folder / "config_vqvae.yaml").read_text()))
    configs.model.max_length = max_length
    if configs.train_settings.losses.get("esm"):
        configs.train_settings.losses.esm.enabled = False
    logger = logging.getLogger("reference")
    components, _ = instantiate_encoder(str(folder / "config_gcpnet_encoder.yaml"))
    decoder_config = Box(
        yaml.safe_load((folder / "config_geometric_decoder.yaml").read_text())
    )
    decoder = GeometricDecoder(configs, decoder_config)
    vqvae = VQVAETransformer(configs, decoder, logger)
    model = SuperModel(PretrainedEncoder(components), vqvae, configs, logger=logger)
    with torch.serialization.safe_globals(
        [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype("float64")),
        ]
    ):
        archive = torch.load(
            folder / "best_valid.pth", weights_only=True, map_location="cpu", mmap=True
        )
    state = {}
    for key, value in archive["model_state_dict"].items():
        normalized = ".".join(part for part in key.split(".") if part != "_orig_mod")
        if normalized in state:
            raise ValueError(f"Checkpoint normalization collision: {key}")
        state[normalized] = value
    excluded = []
    if configs.model.vqvae.vector_quantization.dim == 256:
        excluded = sorted(LARGE_UNUSED_HEAD & state.keys())
        for key in excluded:
            del state[key]
    model.load_state_dict(state, strict=True)
    return model.eval(), excluded


def generate(args):
    reference = args.reference.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        commit != SOURCE["commit"]
        or sha256(reference / "gcp-vqvae/pyproject.toml")
        != SOURCE["package_metadata_sha256"]
    ):
        raise ValueError("Reference checkout does not match the pinned source/metadata")
    if subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("Reference checkout has modified tracked files")
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "numpy",
            "biopython",
            "graphein",
            "torch-geometric",
            "torch-scatter",
            "torch-cluster",
            "x-transformers",
            "vector-quantize-pytorch",
            "python-box",
            "pyyaml",
            "ndlinear",
        )
    }
    if any(
        versions[name] != expected for name, expected in PINNED_DEPENDENCIES.items()
    ):
        raise ValueError("Reference-sensitive dependency versions do not match")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 25 <= args.max_length <= 1280:
        raise ValueError("max_length must be in [25, 1280]")
    for preset in RELEASES:
        for filename, digest in RELEASES[preset]["files"].items():
            if sha256(args.weights / preset / filename) != digest:
                raise ValueError(f"Artifact digest mismatch: {preset}/{filename}")
    sys.path.insert(0, str(reference / "gcp-vqvae"))
    import torch
    from gcp_vqvae._internal.demo.dataset import DemoStructureDataset
    from gcp_vqvae._internal.data.dataset import custom_collate_pretrained_gcp

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=args.device != "cpu")
    args.output.mkdir(parents=True)
    inputs = args.output / "inputs"
    shutil.copytree(args.inputs, inputs)
    manifest = {
        "schema_version": 1,
        "source": SOURCE,
        "artifacts": RELEASES,
        "environment": {
            "dependencies": versions,
            "python": platform.python_version(),
            "device": args.device,
            "dtype": "float32",
            "attention": "torch SDPA",
            "machine": platform.machine(),
            "threads": 1,
            "seed": 0,
            **(
                {
                    "accelerator": torch.cuda.get_device_name(),
                    "runtime": torch.version.hip or torch.version.cuda,
                }
                if args.device != "cpu"
                else {}
            ),
        },
        "configuration": {
            "max_length": args.max_length,
            "released_max_length": 1280,
            "mode": "preparation" if args.preparation_only else "full",
        },
        "generation_command": [
            "python",
            "tests/reference/generate_gcp_vqvae.py",
            "--reference",
            "REFERENCE_CHECKOUT",
            "--weights",
            "WEIGHTS_CACHE",
            "--inputs",
            "INPUTS_DIR",
            "--output",
            "OUTPUT_DIR",
            "--max-length",
            str(args.max_length),
            *(["--device", args.device] if args.device != "cpu" else []),
            *(["--preparation-only"] if args.preparation_only else []),
        ],
        "inputs": {
            str(path.relative_to(args.output)): sha256(path)
            for path in sorted(inputs.rglob("*"))
            if path.is_file()
        },
        "cases": [],
        "excluded_checkpoint_keys": {},
    }

    def numpy(tensor):
        return tensor.detach().cpu().numpy().copy()

    for preset in RELEASES:
        folder = args.weights / preset
        dataset = DemoStructureDataset(
            str(inputs),
            max_length=args.max_length,
            encoder_config_path=str(folder / "config_gcpnet_encoder.yaml"),
            progress=False,
        )
        model = None
        if not args.preparation_only:
            model, excluded = _load_model(folder, args.max_length)
            model = model.to(args.device)
            manifest["excluded_checkpoint_keys"][preset] = excluded
        items = [dataset[i] for i in range(len(dataset))]
        groups = [
            (Path(sample["source_path"]).stem, [i])
            for i, sample in enumerate(dataset.samples)
        ]
        complete = [
            i
            for i, sample in enumerate(dataset.samples)
            if Path(sample["source_path"]).stem in {"complete_pdb", "shorter"}
        ]
        if len(complete) == 2:
            groups.append(("unequal_batch", complete))
        for group_name, positions in groups:
            selected = [items[i] for i in positions]
            samples = [dataset.samples[i] for i in positions]
            arrays = {
                "parsed_coordinates": np.concatenate(
                    [
                        np.asarray(sample["coords"], dtype=np.float32)
                        for sample in samples
                    ]
                ),
                "prepared_coordinates": np.concatenate(
                    [
                        numpy(item[6]).reshape(args.max_length, 4, 3)[: len(item[1])]
                        for item in selected
                    ]
                ),
                "residue_mask": np.stack([numpy(item[5]) for item in selected]),
                "token_mask": np.stack([numpy(item[5] & item[8]) for item in selected]),
            }
            lengths = [len(item[1]) for item in selected]
            sequences = [item[1] for item in selected]
            from gcp_vqvae._internal.demo.data_utils import _select_parser

            observed = []
            observed_batch = []
            sources = []
            for batch_index, sample in enumerate(samples):
                path = Path(sample["source_path"])
                structure = _select_parser(str(path)).get_structure(
                    "reference", str(path)
                )
                chains = [
                    chain
                    for chain in structure[0]
                    if any(residue.id[0] == " " for residue in chain)
                ]
                if len(chains) != 1:
                    raise ValueError(
                        "Oracle inputs must contain exactly one coordinate protein chain"
                    )
                chain = chains[0]
                rows = [residue for residue in chain if residue.id[0] == " "]
                observed.extend(rows)
                observed_batch.extend([batch_index] * len(rows))
                sources.append(
                    {
                        "path": str(path.relative_to(args.output)),
                        "chain_id": chain.id,
                        "model_index": 0,
                        "coordinate_residues": len(rows),
                    }
                )
            original = np.array(
                [
                    [
                        residue[atom].coord if atom in residue else [np.nan] * 3
                        for atom in ("N", "CA", "C", "O")
                    ]
                    for residue in observed
                ],
                dtype=np.float32,
            )
            arrays.update(
                {
                    "observed_coordinates": original,
                    "observed_atom_mask": np.isfinite(original).all(axis=-1),
                    "observed_author_ids": np.array(
                        [residue.id[1] for residue in observed], dtype=np.int64
                    ),
                    "observed_insertion_codes": np.array(
                        [residue.id[2] for residue in observed]
                    ),
                    "observed_residue_names": np.array(
                        [residue.resname for residue in observed]
                    ),
                    "observed_batch": np.array(observed_batch, dtype=np.int64),
                }
            )
            if model is not None:
                batch = custom_collate_pretrained_gcp(
                    selected, featuriser=dataset.pretrained_featuriser
                )
                if args.device != "cpu":
                    batch = {
                        key: value.to(args.device) if hasattr(value, "to") else value
                        for key, value in batch.items()
                    }
                graph = batch["graph"]
                for key in (
                    "edge_index",
                    "batch",
                    "seq_pos",
                    "x",
                    "x_vector_attr",
                    "edge_attr",
                    "edge_vector_attr",
                    "pos",
                    "coords",
                    "residue_type",
                ):
                    arrays[f"graph_{key}"] = numpy(graph[key])
                hooks = []

                def capture(name, transform=lambda value: value):
                    def hook(module, inputs, output):
                        arrays[name] = numpy(transform(output))

                    return hook

                hooks.append(
                    model.encoder.encoder.register_forward_hook(
                        capture("gcp_embeddings", lambda value: value["node_embedding"])
                    )
                )
                for name, module, transpose in (
                    ("encoder_projection", model.vqvae.encoder_tail, True),
                    ("encoder_embeddings", model.vqvae.encoder_blocks, False),
                    ("encoder_latents", model.vqvae.encoder_head, True),
                ):
                    transform = (
                        (lambda value: value.transpose(1, 2))
                        if transpose
                        else (lambda value: value)
                    )
                    hooks.append(module.register_forward_hook(capture(name, transform)))
                hooks.append(
                    model.vqvae.vector_quantizer.register_forward_hook(
                        capture("vq_codes", lambda value: value[0])
                    )
                )
                hooks.append(
                    model.vqvae.vector_quantizer.register_forward_hook(
                        capture("vq_indices", lambda value: value[1])
                    )
                )
                hooks.append(
                    model.vqvae.decoder.register_forward_hook(
                        capture(
                            "decoder_coordinates",
                            lambda value: value["outputs"].reshape(
                                len(selected), args.max_length, 3, 3
                            ),
                        )
                    )
                )
                with torch.inference_mode():
                    model(batch)
                for hook in hooks:
                    hook.remove()
            case = _write_case(
                args.output,
                f"{preset}_{group_name}",
                preset,
                "preparation" if model is None else "full",
                lengths,
                sequences,
                arrays,
            )
            manifest["cases"].append(case)
            case["sources"] = sources
        # Preserve per-file upstream filtering counts instead of losing rejected inputs.
        from gcp_vqvae._internal.demo.data_utils import process_structure_file

        for index, path in enumerate(sorted(inputs.iterdir())):
            if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
                continue
            samples, stats = process_structure_file(
                index,
                str(path),
                max_len=args.max_length,
                min_len=25,
                similarity_threshold=0.90,
                gap_threshold=5,
                use_gap_estimation=True,
                max_missing_ratio=0.2,
                max_consecutive_missing=15,
                include_file_index=True,
            )
            if not samples:
                manifest["cases"].append(
                    {
                        "name": f"{preset}_{path.stem}",
                        "preset": preset,
                        "kind": "rejected",
                        "rejections": dict(stats),
                    }
                )
        if model is not None:
            for mode in ("holes", "prefix"):
                indices = (
                    torch.arange(args.max_length, device=args.device).repeat(2, 1)
                    % 4096
                )
                mask = torch.ones_like(indices, dtype=torch.bool)
                mask[1, args.max_length // 2 :] = False
                mask[0, 5:8] = False
                indices[~mask] = -1
                with torch.inference_mode():
                    codes = model.vqvae.vector_quantizer.get_output_from_indices(
                        indices
                    )
                    kwargs = (
                        {"true_lengths": torch.tensor([25, 17], device=args.device)}
                        if mode == "prefix"
                        else {}
                    )
                    outputs = model.vqvae.decoder(codes, mask, **kwargs)["outputs"]
                arrays = {
                    "vq_indices": numpy(indices),
                    "vq_codes": numpy(codes),
                    "token_mask": numpy(mask),
                    "decoder_coordinates": numpy(outputs).reshape(
                        2, args.max_length, 3, 3
                    ),
                }
                if mode == "prefix":
                    arrays["true_lengths"] = np.array([25, 17], dtype=np.int64)
                manifest["cases"].append(
                    _write_case(
                        args.output,
                        f"{preset}_decoder_{mode}",
                        preset,
                        "decoder",
                        [args.max_length, args.max_length // 2],
                        [],
                        arrays,
                    )
                )
            del model
        print(f"Captured {preset} reference stages", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "invocation": [sys.executable, *sys.argv],
            },
            indent=2,
        )
        + "\n"
    )
    verify_fixture_set(args.output, require_models=not args.preparation_only)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="Verify an existing oracle offline")
    parser.add_argument(
        "--require-models",
        action="store_true",
        help="Fail if full published-weight cases are absent",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-length", type=int, default=1280)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Backend for a separate FP32 reference capture (cuda also selects ROCm)",
    )
    parser.add_argument("--preparation-only", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_fixture_set(args.verify, require_models=args.require_models)
        print(f"Verified {len(manifest['cases'])} reference cases")
    else:
        if not all((args.reference, args.weights, args.inputs, args.output)):
            parser.error(
                "generation requires --reference, --weights, --inputs and --output"
            )
        generate(args)


if __name__ == "__main__":
    main()
