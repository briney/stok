"""Compare complete released encoder/VQ/decoder stacks on the frozen public cohort.

Run from a repository checkout with the optional upstream reference environment.
The correspondence adapter supplies identical deposited residue slots to both
implementations; upstream filling, graph construction and models remain upstream.
No agreement threshold is imposed.
"""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer


def backbone_rmsd(predicted, original, mask):
    """Optimal proper rigid alignment of the specified original atoms, in Å."""
    moving = np.asarray(predicted, dtype=np.float64)[mask].reshape(-1, 3)
    target = np.asarray(original, dtype=np.float64)[mask].reshape(-1, 3)
    if not np.isfinite(moving).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite coordinates at scored original positions")
    fit = SVDSuperimposer()
    fit.set(target, moving)
    fit.run()
    return float(fit.get_rms())


def run(args):
    import torch
    from stok.data.structure_encoding import (
        StructureExclusion,
        iter_structure_manifest,
        prepare_structure,
    )
    from stok.models.gcp_vqvae import load_pretrained_tokenizer
    from stok.models.decoder import load_pretrained_decoder
    from stok.utils.pretrained import file_sha256, inference_metadata
    from stok.utils.structure_parser import parse_polymer_structure
    from tests.reference.generate_gcp_vqvae import SOURCE, RELEASES, _load_model

    revision = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        revision != SOURCE["commit"]
        or subprocess.check_output(
            [
                "git",
                "-C",
                str(args.reference),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        ).strip()
    ):
        raise ValueError("Use the clean pinned upstream reference checkout")
    for preset, release in RELEASES.items():
        for name, digest in release["files"].items():
            if file_sha256(args.weights / preset / name) != digest:
                raise ValueError(f"Release file digest mismatch: {preset}/{name}")
    sys.path.insert(0, str(args.reference / "gcp-vqvae"))
    from gcp_vqvae._internal.demo.dataset import (
        DemoStructureDataset,
        amino_acid_to_tensor,
    )
    from gcp_vqvae._internal.data.dataset import custom_collate_pretrained_gcp

    torch.set_num_threads(1)
    torch.manual_seed(20260930)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arrays").mkdir()
    cohort = json.loads((args.docs / "public-cohort.json").read_text())
    records = {row["sequence_id"]: row for row in cohort["chains"]}
    manifests = [
        args.docs / f"public-{split}.jsonl" for split in ("selection", "heldout")
    ]
    entries = [
        entry for manifest in manifests for entry in iter_structure_manifest(manifest)
    ]
    metadata = {
        "reference": SOURCE,
        "releases": RELEASES,
        "environment": inference_metadata(torch.device(args.device)),
        "script_sha256": file_sha256(__file__),
        "cohort_sha256": file_sha256(args.docs / "public-cohort.json"),
        "manifest_sha256": {path.name: file_sha256(path) for path in manifests},
        "input_contract": "shared deposited correspondence; native identities; independent reference filling/graphs/models; singleton FP32; encoder and decoder1280",
        "metric": "original observed N/CA/C; optimal proper all-backbone Kabsch alignment; per-atom RMSD in angstroms; no imputed targets",
        "excluded_upstream_checkpoint_keys": {},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with (args.output / "results.jsonl").open("w") as output:
        for preset in RELEASES:
            folder = args.weights / preset
            ours = load_pretrained_tokenizer(
                preset, path=folder / "best_valid.pth", device=args.device
            )
            decoder = load_pretrained_decoder(
                preset, path=folder / "best_valid.pth", device=args.device
            )
            upstream, excluded = _load_model(folder, 1280)
            upstream = upstream.to(args.device)
            collate_featuriser = deepcopy(upstream.encoder.featuriser).cpu()
            metadata["excluded_upstream_checkpoint_keys"][preset] = excluded
            # Bypass upstream's author-number-gap parser, supplying the same
            # deposited polymer slots. Its __getitem__/collate/model code is intact.
            dataset = DemoStructureDataset.__new__(DemoStructureDataset)
            dataset.max_length = 1280
            dataset.letter_to_num = {
                aa: int(amino_acid_to_tensor(aa, 1)[0])
                for aa in "ACDEFGHIKLMNPQRSTVWYX"
            }
            for entry in entries:
                record = records[entry["sequence_id"]]
                source = replace(
                    parse_polymer_structure(
                        **{
                            key: value
                            for key, value in entry.items()
                            if key != "sequence_id"
                        }
                    ),
                    sequence_id=entry["sequence_id"],
                )
                if source.source["sha256"] != record["source_sha256"]:
                    raise ValueError("Frozen source digest mismatch")
                row = {
                    "preset": preset,
                    "sequence_id": source.sequence_id,
                    "split": record["split"],
                    "length": len(source.sequence),
                    "complete": record["complete"],
                    "source_sha256": source.source["sha256"],
                }
                try:
                    graph, residues, tokens = prepare_structure(
                        source, sequence_mode="native", imputation="reference"
                    )
                except StructureExclusion as error:
                    row.update(status="excluded", reason=error.reason)
                else:
                    raw = source.coordinates.copy()
                    available = source.atom_mask.all(-1)
                    raw[~available] = np.nan
                    sequence = "".join(
                        item.get("observed_one_letter") or "X"
                        for item in source.residue_map
                    )
                    dataset.samples = [
                        {
                            "pid": source.sequence_id,
                            "seq": sequence,
                            "coords": raw,
                            "plddt_scores": [0] * len(sequence),
                        }
                    ]
                    item = dataset[0]
                    batch = custom_collate_pretrained_gcp(
                        [item], featuriser=collate_featuriser
                    )
                    if not torch.equal(batch["masks"] & batch["nan_masks"], tokens):
                        raise ValueError(
                            "Implementations disagree on original token availability"
                        )
                    prepared_delta = float(
                        (
                            graph.prepared_coordinates
                            - item[6].reshape(1280, 4, 3)[: len(sequence)]
                        )
                        .abs()
                        .max()
                    )
                    batch = {
                        key: value.to(args.device) if hasattr(value, "to") else value
                        for key, value in batch.items()
                    }
                    with (
                        torch.inference_mode(),
                        torch.autocast(torch.device(args.device).type, enabled=False),
                    ):
                        ref_output = upstream(batch)
                        codes, ids, _ = ours(
                            graph.to(args.device),
                            residue_mask=residues.to(args.device),
                            token_mask=tokens.to(args.device),
                        )
                        predicted = decoder(codes, tokens.to(args.device)).reshape(
                            1, 1280, 3, 3
                        )
                    length = len(sequence)
                    ours_xyz = predicted[0, :length].cpu().numpy()
                    ref_xyz = (
                        ref_output["outputs"]
                        .reshape(1, 1280, 3, 3)[0, :length]
                        .cpu()
                        .numpy()
                    )
                    ours_ids = ids[0, :length].cpu().numpy()
                    ref_ids = ref_output["indices"][0, :length].cpu().numpy()
                    original = source.coordinates[:, :3].copy()
                    ours_xyz[~available] = np.nan
                    ref_xyz[~available] = np.nan
                    distance = np.linalg.norm(
                        ours_xyz[available] - ref_xyz[available], axis=-1
                    )
                    row.update(
                        status="accepted",
                        scored_residues=int(available.sum()),
                        prepared_max_atom_component_delta=prepared_delta,
                        stok_backbone_rmsd=backbone_rmsd(ours_xyz, original, available),
                        upstream_backbone_rmsd=backbone_rmsd(
                            ref_xyz, original, available
                        ),
                        stok_ca_rmsd=backbone_rmsd(
                            ours_xyz[:, 1:2], original[:, 1:2], available
                        ),
                        upstream_ca_rmsd=backbone_rmsd(
                            ref_xyz[:, 1:2], original[:, 1:2], available
                        ),
                        implementation_aligned_backbone_rmsd=backbone_rmsd(
                            ours_xyz, ref_xyz, available
                        ),
                        implementation_unaligned_backbone_rmsd=float(
                            np.sqrt(np.mean(distance**2))
                        ),
                        implementation_max_atom_distance=float(distance.max()),
                        changed_token_ids=int(
                            (ours_ids[available] != ref_ids[available]).sum()
                        ),
                    )
                    name = f"{preset}-{source.sequence_id}.npz"
                    np.savez_compressed(
                        args.output / "arrays" / name,
                        original_coordinates=original,
                        original_atom_mask=source.atom_mask,
                        score_mask=available,
                        stok_coordinates=ours_xyz,
                        upstream_coordinates=ref_xyz,
                        stok_indices=ours_ids,
                        upstream_indices=ref_ids,
                    )
                    row.update(
                        arrays=f"arrays/{name}",
                        arrays_sha256=file_sha256(args.output / "arrays" / name),
                    )
                rows.append(row)
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                print(
                    preset,
                    source.sequence_id,
                    row["status"],
                    row.get("stok_backbone_rmsd", row.get("reason")),
                    flush=True,
                )
            del ours, decoder, upstream
            torch.cuda.empty_cache() if torch.device(
                args.device
            ).type == "cuda" else None
    metadata["results_sha256"] = file_sha256(args.output / "results.jsonl")
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    summary = {}
    for preset in RELEASES:
        selected = [row for row in rows if row["preset"] == preset]
        accepted = [row for row in selected if row["status"] == "accepted"]
        groups = {
            "all": accepted,
            "complete": [row for row in accepted if row["complete"]],
            "incomplete": [row for row in accepted if not row["complete"]],
            **{
                split: [row for row in accepted if row["split"] == split]
                for split in ("selection", "heldout")
            },
        }
        metrics = (
            "stok_backbone_rmsd",
            "upstream_backbone_rmsd",
            "stok_ca_rmsd",
            "upstream_ca_rmsd",
            "implementation_aligned_backbone_rmsd",
            "implementation_unaligned_backbone_rmsd",
        )
        summary[preset] = {
            "attempted": len(selected),
            "accepted": len(accepted),
            "exclusions": dict(
                Counter(
                    row["reason"] for row in selected if row["status"] == "excluded"
                )
            ),
            "scored_residues": sum(row["scored_residues"] for row in accepted),
            "changed_token_ids": sum(row["changed_token_ids"] for row in accepted),
            "max_atom_distance": max(
                row["implementation_max_atom_distance"] for row in accepted
            ),
            "groups": {
                name: {
                    "chains": len(group),
                    "metrics": {
                        metric: {
                            "mean": float(np.mean([row[metric] for row in group])),
                            "median": float(np.median([row[metric] for row in group])),
                            "min": min(row[metric] for row in group),
                            "max": max(row[metric] for row in group),
                        }
                        for metric in metrics
                    },
                }
                for name, group in groups.items()
            },
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--docs", type=Path, default=Path("docs/experiments/gcp-vqvae"))
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
