"""Measure STok reconstruction of public chains outside upstream coverage limits.

Only the preprocessing admission check is bypassed, locally in this experiment.
Production preparation, filling, masks, encoder and decoder remain unchanged.
"""

import argparse
import csv
from dataclasses import replace
from itertools import groupby
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.gcp_vqvae_roundtrip import backbone_rmsd
from stok.data.structure_encoding import iter_structure_manifest, prepare_structure
from stok.models.decoder import load_pretrained_decoder
from stok.models.gcp_vqvae import load_pretrained_tokenizer
from stok.utils.pretrained import file_sha256, inference_metadata
from stok.utils.structure_parser import parse_polymer_structure


def run(args):
    previous_path = args.docs / "public-roundtrip-chains.csv"
    with previous_path.open() as handle:
        previous = list(csv.DictReader(handle))
    excluded = {
        row["sequence_id"]: row
        for row in previous
        if row["preset"] == "lite" and row["status"] == "excluded"
    }
    entries = [
        entry
        for split in ("selection", "heldout")
        for entry in iter_structure_manifest(args.docs / f"public-{split}.jsonl")
        if entry["sequence_id"] in excluded
    ]
    assert len(entries) == len(excluded) == 10
    releases = json.loads((args.docs / "public-roundtrip-results.json").read_text())[
        "metadata"
    ]["releases"]
    for preset, release in releases.items():
        for name, digest in release["files"].items():
            assert file_sha256(args.weights / preset / name) == digest
    torch.set_num_threads(1)
    torch.manual_seed(20260930)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arrays").mkdir()
    metadata = {
        "environment": inference_metadata(torch.device(args.device)),
        "releases": releases,
        "script_sha256": file_sha256(__file__),
        "previous_results_sha256": file_sha256(previous_path),
        "input_contract": "native/reference; singleton FP32; encoder and decoder1280; only _check_reference_coverage bypassed",
        "metric": "original observed N/CA/C at originally complete N/CA/C/O rows; proper all-backbone Kabsch RMSD in angstroms; no imputed targets",
    }
    rows = []
    with (args.output / "results.jsonl").open("w") as handle:
        for preset in releases:
            checkpoint = args.weights / preset / "best_valid.pth"
            encoder = load_pretrained_tokenizer(
                preset, path=checkpoint, device=args.device
            )
            decoder = load_pretrained_decoder(
                preset, path=checkpoint, device=args.device
            )
            for entry in entries:
                source = replace(
                    parse_polymer_structure(
                        **{k: v for k, v in entry.items() if k != "sequence_id"}
                    ),
                    sequence_id=entry["sequence_id"],
                )
                prior = excluded[source.sequence_id]
                assert source.source["sha256"] == prior["source_sha256"]
                length = len(source.sequence)
                assert 25 <= length <= 1280
                available = source.atom_mask.all(-1)
                with patch("stok.data.structure_encoding._check_reference_coverage"):
                    graph, residues, tokens = prepare_structure(
                        source, sequence_mode="native", imputation="reference"
                    )
                assert np.array_equal(tokens[0, :length].numpy(), available)
                with (
                    torch.inference_mode(),
                    torch.autocast(torch.device(args.device).type, enabled=False),
                ):
                    codes, ids, _ = encoder(
                        graph.to(args.device),
                        residue_mask=residues.to(args.device),
                        token_mask=tokens.to(args.device),
                    )
                    predicted = (
                        decoder(codes, tokens.to(args.device))
                        .reshape(1, 1280, 3, 3)[0, :length]
                        .cpu()
                        .numpy()
                    )
                original = source.coordinates[:, :3].copy()
                predicted[~available] = np.nan
                name = f"{preset}-{source.sequence_id}.npz"
                np.savez_compressed(
                    args.output / "arrays" / name,
                    original_coordinates=original,
                    original_atom_mask=source.atom_mask,
                    score_mask=available,
                    stok_coordinates=predicted,
                    stok_indices=ids[0, :length].cpu().numpy(),
                )
                row = {
                    "preset": preset,
                    "sequence_id": source.sequence_id,
                    "split": prior["split"],
                    "original_exclusion": prior["reason"],
                    "length": length,
                    "scored_residues": int(available.sum()),
                    "missing_fraction": float((~available).mean()),
                    "longest_missing_run": max(
                        (
                            sum(1 for _ in group)
                            for flag, group in groupby(~available)
                            if flag
                        ),
                        default=0,
                    ),
                    "stok_backbone_rmsd": backbone_rmsd(predicted, original, available),
                    "stok_ca_rmsd": backbone_rmsd(
                        predicted[:, 1:2], original[:, 1:2], available
                    ),
                    "source_sha256": source.source["sha256"],
                    "arrays": f"arrays/{name}",
                    "arrays_sha256": file_sha256(args.output / "arrays" / name),
                }
                rows.append(row)
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                print(preset, source.sequence_id, row["stok_backbone_rmsd"], flush=True)
            del encoder, decoder
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
    metadata["results_sha256"] = file_sha256(args.output / "results.jsonl")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--docs", type=Path, default=Path("docs/experiments/gcp-vqvae"))
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
