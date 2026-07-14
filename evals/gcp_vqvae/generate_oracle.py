"""Generate the cached GCP-VQVAE base-encoder oracle used by the slow eval."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from gcp_vqvae import GCPVQVAE

UPSTREAM_REPO = "Mahdip72/gcp-vqvae-large"
UPSTREAM_REVISION = "64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1"
DEFAULT_INPUT_DIR = Path("~/datasets/structure/cif_500").expanduser()
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("oracle_base.pt")
STRUCTURE_SUFFIXES = {".cif", ".mmcif", ".pdb", ".ent"}
DECODER_SAMPLE_COUNT = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["graph"] = batch["graph"].to(device)
    moved["masks"] = batch["masks"].to(device)
    moved["nan_masks"] = batch["nan_masks"].to(device)
    return moved


def _select_decoder_indices(samples: list[dict[str, Any]], count: int) -> set[int]:
    if not 2 <= count <= len(samples):
        raise ValueError(f"Decoder sample count must be in [2, {len(samples)}], got {count}")
    ranked = sorted(
        range(len(samples)),
        key=lambda index: (
            len(str(samples[index]["seq"])),
            Path(samples[index]["source_path"]).name,
        ),
    )
    selected = {
        ranked[round(rank * (len(ranked) - 1) / (count - 1))] for rank in range(count)
    }
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} distinct decoder samples, selected {len(selected)}")
    return selected


@torch.inference_mode()
def generate_oracle(input_dir: Path, output: Path) -> None:
    input_dir = input_dir.expanduser().resolve()
    paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES
    )
    if len(paths) != 500:
        raise ValueError(f"Expected 500 structure fixtures under {input_dir}, found {len(paths)}")
    if not torch.cuda.is_available():
        raise RuntimeError("Oracle generation requires a CUDA GPU")

    _configure_determinism()
    device = torch.device("cuda:0")
    wrapper = GCPVQVAE(
        mode="embed",
        hf_model_id=UPSTREAM_REPO,
        hf_revision=UPSTREAM_REVISION,
        device=str(device),
        mixed_precision="no",
        deterministic=True,
        seed=0,
    )
    model = wrapper.model
    if model is None:
        raise RuntimeError("GCP-VQVAE did not load its model")

    dataset, collate_fn = wrapper._build_dataset(
        pdb_dir=str(input_dir),
        max_task_samples=None,
        progress=True,
    )
    decoder_indices = _select_decoder_indices(dataset.samples, DECODER_SAMPLE_COUNT)
    accepted: dict[Path, dict[str, Any]] = {}
    for index, sample in enumerate(dataset.samples):
        source = Path(sample["source_path"]).resolve()
        if source in accepted:
            raise ValueError(f"Expected at most one accepted sample per fixture: {source}")
        batch = _move_batch(collate_fn([dataset[index]]), device)
        include_decoder = index in decoder_indices
        output_values = model(batch, return_vq_layer=not include_decoder)
        sequence = str(batch["seq"][0])
        length = len(sequence)
        valid = (batch["masks"] & batch["nan_masks"])[0, :length].detach().cpu().bool()
        indices = output_values["indices"][0, :length].detach().cpu().long()
        decoder_coords = torch.empty((0, 3, 3), dtype=torch.float32)
        if include_decoder:
            decoder_coords = (
                output_values["outputs"]
                .view(-1, wrapper.max_length, 3, 3)[0, :length]
                .detach()
                .cpu()
                .float()
            )
        accepted[source] = {
            "sequence": sequence,
            "valid": valid,
            "indices": indices,
            "decoder_coords": decoder_coords,
        }

    samples = []
    for path in paths:
        sample = accepted.get(path)
        samples.append(
            {
                "filename": path.relative_to(input_dir).as_posix(),
                "sha256": _sha256(path),
                "accepted": sample is not None,
                "sequence": "" if sample is None else sample["sequence"],
                "valid": torch.empty(0, dtype=torch.bool) if sample is None else sample["valid"],
                "indices": (
                    torch.empty(0, dtype=torch.long) if sample is None else sample["indices"]
                ),
                "decoder_coords": (
                    torch.empty((0, 3, 3), dtype=torch.float32)
                    if sample is None
                    else sample["decoder_coords"]
                ),
            }
        )

    checkpoint_path = Path(wrapper.checkpoint_path)
    payload = {
        "schema_version": 2,
        "preset": "base",
        "decoder_sample_count": DECODER_SAMPLE_COUNT,
        "decoder_max_length": wrapper.max_length,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_checkpoint_sha256": _sha256(checkpoint_path),
        "codebook": model.vqvae.vector_quantizer._codebook.embed.detach().cpu().clone(),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    accepted_count = sum(record["accepted"] for record in samples)
    print(f"wrote {output} ({accepted_count} accepted, {len(samples) - accepted_count} rejected)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_oracle(args.input_dir, args.output)


if __name__ == "__main__":
    main()
