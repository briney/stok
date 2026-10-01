"""Paired GCP-VQVAE policy measurements; no automatic production-policy selection.

Run with ``python -m experiments.gcp_vqvae_policies --help`` from the checkout.
"""

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import resource
import time
from typing import Any, cast

import numpy as np
import torch
from torch_geometric.data import Data

from stok.data.structure_encoding import (
    StructureExclusion,
    iter_structure_manifest,
    prepare_structure,
    tokenize_structures,
)
from stok.models.decoder import load_pretrained_decoder
from stok.models.gcp_vqvae import load_pretrained_tokenizer
from stok.utils.decoding import decode_structure_tokens
from stok.utils.metrics import lddt_ca, rmsd, tm_score
from stok.utils.pretrained import (
    SOURCE_COMMIT,
    canonical_preset,
    file_sha256,
    inference_metadata,
    json_sha256,
    state_sha256,
)
from stok.utils.structure_parser import (
    StructureMappingError,
    parse_polymer_structure,
)

CONDITIONS = [
    (sequence, fill)
    for sequence in ("native", "unknown")
    for fill in ("reference", "linear", "observed_only")
]
CONDITIONS += [("polymer", "reference")]


def policy_settings(sequence_mode, imputation, *, allow_observed_sequence, device):
    return {
        "schema_version": 1,
        "sequence_source": "deposited_or_supplied",
        "allow_observed_sequence": allow_observed_sequence,
        "sequence_mode": sequence_mode,
        "required_atoms": ["N", "CA", "C", "O"],
        "imputation": imputation,
        "graph_context": "independent_singleton",
        "context_scope": "full_chain",
        "min_length": 25,
        "max_length": 1280,
        "max_missing_ratio": 0.2,
        "max_missing_block": 15,
        "cropping": "none",
        "dtype": "float32",
        "device": str(device),
    }


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


@torch.inference_mode()
def _encode(model, structure, sequence_mode, imputation, *, padded=True):
    graph, residues, tokens = prepare_structure(
        structure, sequence_mode=sequence_mode, imputation=imputation
    )
    displacement = float(cast(Data, graph).max_observed_atom_displacement[0])
    device = next(model.parameters()).device
    cast(Data, graph).to(device)
    if not padded:
        residues, tokens = (
            residues[:, : len(structure.sequence)],
            tokens[:, : len(structure.sequence)],
        )
    latents = []
    hook = model.encoder.register_forward_hook(
        lambda module, inputs, output: latents.append(output.detach())
    )
    try:
        indices = model.encode(
            graph, residue_mask=residues.to(device), token_mask=tokens.to(device)
        )
    finally:
        hook.remove()
    length = len(structure.sequence)
    ids, latent = indices[0, :length].cpu(), latents[0][0, :length].cpu()
    if (
        not torch.equal(ids >= 0, tokens[0, :length])
        or (ids < -1).any()
        or (ids >= model.quantizer.codebook.size(0)).any()
        or not torch.isfinite(latent).all()
    ):
        raise RuntimeError("Unexpected model IDs/masks or nonfinite latents")
    return ids, latent, displacement


def _agreement(first, second, mask):
    if first.shape != second.shape or not torch.equal(first >= 0, second >= 0):
        raise RuntimeError("Paired token availability changed")
    count = int(mask.sum())
    changed = int((first[mask] != second[mask]).sum())
    return {
        "compared_tokens": count,
        "changed_tokens": changed,
        "change_rate": changed / count if count else None,
    }


def _diagnostics(model, structure, sequence, fill, ids, rotation, translation):
    transformed = replace(
        structure,
        coordinates=(structure.coordinates @ rotation + translation).astype(np.float32),
    )
    rotated, _, _ = _encode(model, transformed, sequence, fill)
    short, _, _ = _encode(model, structure, sequence, fill, padded=False)
    result = {
        "rigid_transform": _agreement(ids, rotated, ids >= 0),
        "padding": _agreement(ids, short, ids >= 0),
    }
    stop = len(structure.sequence) - 5
    if stop >= 25:
        cropped = replace(
            structure,
            sequence=structure.sequence[:stop],
            coordinates=structure.coordinates[:stop],
            atom_mask=structure.atom_mask[:stop],
            residue_map=structure.residue_map[:stop],
        )
        try:
            crop_ids, _, _ = _encode(model, cropped, sequence, fill)
            result["crop_context"] = _agreement(ids[:stop], crop_ids, crop_ids >= 0)
            result["crop_context"]["span"] = [0, stop]
        except StructureExclusion as error:
            result["crop_context"] = {"excluded": error.reason, "span": [0, stop]}
    else:
        result["crop_context"] = {"excluded": "crop_too_short"}
    return result


def _variants(structure, gaps):
    yield "source", structure
    # Synthetic perturbations require a complete original target, kept separately.
    if not structure.atom_mask.all():
        return
    length = len(structure.sequence)
    for gap in gaps:
        if gap >= length:
            continue
        for location, start in (("internal", (length - gap) // 2), ("terminal", 0)):
            coordinates = structure.coordinates.copy()
            coordinates[start : start + gap] = np.nan
            mapping = [dict(row) for row in structure.residue_map]
            for row in mapping[start : start + gap]:
                row.update(observed_one_letter=None, observed_monomer_id=None)
            yield (
                f"{location}_gap_{gap}",
                replace(
                    structure,
                    coordinates=coordinates,
                    atom_mask=np.isfinite(coordinates).all(-1),
                    residue_map=tuple(mapping),
                ),
            )
    if gaps:
        coordinates = structure.coordinates.copy()
        coordinates[length // 2, 3] = np.nan
        yield (
            "oxygen_only",
            replace(
                structure,
                coordinates=coordinates,
                atom_mask=np.isfinite(coordinates).all(-1),
            ),
        )


def _paired(records, arrays):
    pairs = []
    for i, first in enumerate(records):
        for second in records[i + 1 :]:
            a, b = arrays[first["condition"]], arrays[second["condition"]]
            if not np.array_equal(a["score_mask"], b["score_mask"]):
                raise RuntimeError("Paired metric denominators changed")
            mask = torch.from_numpy(a["score_mask"])
            comparison = _agreement(
                torch.from_numpy(a["indices"]), torch.from_numpy(b["indices"]), mask
            )
            delta = b["latents"][mask.numpy()] - a["latents"][mask.numpy()]
            gap_distance = a["distance_to_gap"]
            by_distance = []
            for distance in np.unique(gap_distance[mask.numpy()]):
                nearby = mask & torch.from_numpy(gap_distance == distance)
                by_distance.append(
                    {
                        "distance": int(distance),
                        **_agreement(
                            torch.from_numpy(a["indices"]),
                            torch.from_numpy(b["indices"]),
                            nearby,
                        ),
                    }
                )
            pairs.append(
                {
                    "sequence_id": first["sequence_id"],
                    "variant": first["variant"],
                    "first": first["condition"],
                    "second": second["condition"],
                    **comparison,
                    "latent_rms_change": _number(np.sqrt(np.mean(delta**2))),
                    "metric_differences": {
                        name: second[name] - first[name]
                        if first[name] is not None and second[name] is not None
                        else None
                        for name in ("rmsd_backbone", "lddt_ca", "tm_kabsch")
                    },
                    "changes_by_gap_distance": by_distance,
                }
            )
    return pairs


def _aggregate_pairs(pairs, seed):
    """Bootstrap independent source chains separately for each perturbation/pair."""
    grouped = {}
    for pair in pairs:
        grouped.setdefault((pair["variant"], pair["first"], pair["second"]), []).append(
            pair
        )
    rng = np.random.default_rng(seed)
    result = []
    for (variant, first, second), rows in sorted(grouped.items()):
        metrics = {}
        for name in ("rmsd_backbone", "lddt_ca", "tm_kabsch"):
            values = np.array(
                [
                    row["metric_differences"][name]
                    for row in rows
                    if row["metric_differences"][name] is not None
                ]
            )
            metrics[name] = {
                "chain_count": len(values),
                "mean_difference": float(values.mean()) if len(values) else None,
                "bootstrap_95_ci": np.quantile(
                    rng.choice(values, size=(1000, len(values)), replace=True).mean(1),
                    [0.025, 0.975],
                ).tolist()
                if len(values) >= 2
                else None,
            }
        count = sum(row["compared_tokens"] for row in rows)
        changed = sum(row["changed_tokens"] for row in rows)
        result.append(
            {
                "variant": variant,
                "first": first,
                "second": second,
                "chain_count": len(rows),
                "compared_tokens": count,
                "changed_tokens": changed,
                "change_rate": changed / count if count else None,
                "metric_differences": metrics,
                "latent_rms_change_chain_mean": float(
                    np.mean([row["latent_rms_change"] for row in rows])
                ),
            }
        )
    return result


def _aggregate(rows, code_counts):
    accepted = [row for row in rows if row["status"] == "accepted"]
    rejected = [row for row in rows if row["status"] == "rejected"]
    total = sum(code_counts.values())
    probabilities = [count / total for count in code_counts.values()] if total else []
    metrics = ("rmsd_backbone", "lddt_ca", "tm_kabsch")
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "exclusions": dict(Counter(row["reason"] for row in rejected)),
        "source_exclusions": dict(
            Counter(row["reason"] for row in rejected if row["variant"] == "source")
        ),
        "perturbation_exclusions": dict(
            Counter(row["reason"] for row in rejected if row["variant"] != "source")
        ),
        **{
            name: sum(row[name] for row in accepted)
            for name in (
                "residue_count",
                "token_count",
                "scored_residue_count",
                "geometry_count",
                "oxygen_only_token_omissions",
                "runtime_seconds",
            )
        },
        "coverage": sum(row["token_count"] for row in accepted)
        / sum(row["residue_count"] for row in accepted)
        if accepted
        else None,
        "metric_example_counts": {
            name: sum(row[name] is not None for row in accepted) for name in metrics
        },
        "metrics_chain_mean": {
            name: _number(
                np.mean([row[name] for row in accepted if row[name] is not None])
            )
            if any(row[name] is not None for row in accepted)
            else None
            for name in metrics
        },
        "used_codes": len(code_counts),
        "code_counts": {str(key): count for key, count in sorted(code_counts.items())},
        "code_perplexity": math.exp(-sum(p * math.log(p) for p in probabilities))
        if total
        else None,
        "peak_process_rss_bytes": max(
            (row["peak_process_rss_bytes"] for row in accepted), default=0
        ),
        "peak_device_allocated_bytes": max(
            (row["peak_device_allocated_bytes"] for row in accepted), default=0
        ),
    }


@torch.inference_mode()
def run_experiments(
    manifest: Path,
    *,
    preset: str,
    checkpoint: Path,
    output_dir: Path,
    seed: int = 0,
    device: str = "cpu",
    allow_observed_sequence: bool = False,
    synthetic_gaps: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Measure a fixed paired matrix, retaining original observations and all exclusions."""
    preset = canonical_preset(preset)
    if any(type(gap) is not int or gap < 1 for gap in synthetic_gaps):
        raise ValueError("Synthetic gap lengths must be positive integers")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "arrays").mkdir()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotation = rotation.astype(np.float32)
    translation = rng.normal(size=3).astype(np.float32) * 10
    model = load_pretrained_tokenizer(preset, path=checkpoint, device=device)
    decoder = load_pretrained_decoder(preset, path=checkpoint, device=device)
    actual_device = next(model.parameters()).device
    identity = {
        "encoder_sha256": state_sha256(model.encoder.state_dict()),
        "quantizer_sha256": state_sha256(model.quantizer.state_dict()),
        "config_sha256": json_sha256(model.config),
        "codebook_sha256": state_sha256({"codebook": model.quantizer.codebook}),
        "source_revision": SOURCE_COMMIT,
        "decoder_sha256": state_sha256(decoder.state_dict()),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    rows, pairs, grouping_sources, grouping_ids = [], [], [], []
    counts = {f"{sequence}/{fill}": Counter() for sequence, fill in CONDITIONS}
    input_count = 0
    with (output_dir / "results.jsonl").open("w") as results:
        for entry in iter_structure_manifest(manifest):
            input_count += 1
            try:
                source = replace(
                    parse_polymer_structure(
                        **{
                            key: value
                            for key, value in entry.items()
                            if key != "sequence_id"
                        },
                        allow_observed_sequence=allow_observed_sequence,
                    ),
                    sequence_id=entry["sequence_id"],
                )
            except StructureMappingError as error:
                for sequence, fill in CONDITIONS:
                    row = {
                        "sequence_id": entry["sequence_id"],
                        "variant": "source",
                        "condition": f"{sequence}/{fill}",
                        "status": "rejected",
                        "reason": error.reason,
                    }
                    rows.append(row)
                    results.write(json.dumps(row, allow_nan=False) + "\n")
                continue
            original = source.coordinates.copy()
            for variant, structure in _variants(source, synthetic_gaps):
                records, arrays = [], {}
                for sequence, fill in CONDITIONS:
                    condition = f"{sequence}/{fill}"
                    row = {
                        "sequence_id": source.sequence_id,
                        "variant": variant,
                        "condition": condition,
                        "sequence_mode": sequence,
                        "imputation": fill,
                    }
                    try:
                        if actual_device.type == "cuda":
                            torch.cuda.synchronize(actual_device)
                            torch.cuda.reset_peak_memory_stats(actual_device)
                        start = time.perf_counter()
                        ids, latents, displacement = _encode(
                            model, structure, sequence, fill
                        )
                        available = ids >= 0
                        length = len(ids)
                        predicted = decode_structure_tokens(
                            decoder,
                            model.quantizer.codebook,
                            ids[None].to(actual_device),
                            residue_mask=torch.ones(
                                1, length, dtype=torch.bool, device=actual_device
                            ),
                        )[0].cpu()
                        if not torch.isfinite(predicted[available]).all():
                            raise RuntimeError(
                                "Nonfinite predictions at available labels"
                            )
                        if actual_device.type == "cuda":
                            torch.cuda.synchronize(actual_device)
                        elapsed = time.perf_counter() - start
                        truth = torch.tensor(structure.coordinates[:, :3])
                        geometry = torch.tensor(structure.atom_mask[:, :3].all(-1))
                        score = geometry & available
                        lddt, per_residue = lddt_ca(
                            predicted, truth, score[None], return_per_residue=True
                        )
                        missing = torch.where(~available)[0]
                        distances = (
                            (torch.arange(length)[:, None] - missing[None])
                            .abs()
                            .min(-1)
                            .values
                            if missing.numel()
                            else torch.full((length,), -1)
                        )
                        per_residue[0, ~score] = float("nan")
                        payload = {
                            "original_coordinates": structure.coordinates,
                            "held_out_original_coordinates": original,
                            "atom_mask": structure.atom_mask,
                            "geometry_mask": geometry.numpy(),
                            "token_mask": available.numpy(),
                            "score_mask": score.numpy(),
                            "indices": ids.numpy(),
                            "latents": latents.numpy(),
                            "decoded_coordinates": predicted.numpy(),
                            "lddt_per_residue": per_residue[0].numpy(),
                            "distance_to_gap": distances.numpy(),
                        }
                        filename = (
                            f"arrays/{input_count:06d}-{variant}-{sequence}-{fill}.npz"
                        )
                        np.savez_compressed(output_dir / filename, **payload)
                        arrays[condition] = payload
                        row.update(
                            status="accepted",
                            arrays=filename,
                            source=dict(source.source),
                            residue_count=length,
                            token_count=int(available.sum()),
                            geometry_count=int(geometry.sum()),
                            scored_residue_count=int(score.sum()),
                            oxygen_only_token_omissions=int(
                                (geometry & ~available).sum()
                            ),
                            rmsd_backbone=_number(
                                rmsd(
                                    predicted, truth, score[None], atom_set="backbone"
                                )[0]
                            ),
                            lddt_ca=_number(lddt[0]),
                            tm_kabsch=_number(
                                tm_score(predicted, truth, score[None])[0][0]
                            ),
                            max_observed_atom_displacement=displacement,
                            runtime_seconds=elapsed,
                            peak_process_rss_bytes=resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            * 1024,
                            peak_device_allocated_bytes=torch.cuda.max_memory_allocated(
                                actual_device
                            )
                            if actual_device.type == "cuda"
                            else 0,
                            diagnostics=_diagnostics(
                                model,
                                structure,
                                sequence,
                                fill,
                                ids,
                                rotation,
                                translation,
                            ),
                        )
                        counts[condition].update(ids[available].tolist())
                        records.append(row)
                        if (
                            condition == "native/reference"
                            and variant == "source"
                            and len(grouping_sources) < 2
                        ):
                            grouping_sources.append(source)
                            grouping_ids.append(ids)
                    except StructureExclusion as error:
                        row.update(status="rejected", reason=error.reason)
                    rows.append(row)
                    results.write(json.dumps(row, allow_nan=False) + "\n")
                pairs.extend(_paired(records, arrays))
            if not np.array_equal(source.coordinates, original, equal_nan=True):
                raise RuntimeError("Source coordinates were mutated")
    grouped = tokenize_structures(
        model, grouping_sources, sequence_mode="native", imputation="reference"
    )
    grouping_check = {"compared_tokens": 0, "changed_tokens": 0}
    for first, second in zip(grouping_ids, grouped):
        comparison = _agreement(first, second, first >= 0)
        for field in grouping_check:
            grouping_check[field] += comparison[field]
    if (
        identity["encoder_sha256"] != state_sha256(model.encoder.state_dict())
        or identity["quantizer_sha256"] != state_sha256(model.quantizer.state_dict())
        or identity["decoder_sha256"] != state_sha256(decoder.state_dict())
    ):
        raise RuntimeError("Inference mutated model state")
    with (output_dir / "pairs.jsonl").open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, allow_nan=False) + "\n")
    environment = inference_metadata(actual_device)
    report = {
        "schema_version": 1,
        "preset": preset,
        "seed": seed,
        "input_count": input_count,
        "synthetic_gaps": list(synthetic_gaps),
        "identity": identity,
        "environment": environment,
        "input_manifest_sha256": file_sha256(manifest),
        "experiment_script_sha256": file_sha256(__file__),
        "selected_policy": None,
        "grouping_context": "independent_singleton",
        "grouping_check": grouping_check,
        "paired_summary": _aggregate_pairs(pairs, seed),
        "rigid_transform": {
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "conditions": {
            name: _aggregate([row for row in rows if row["condition"] == name], counter)
            for name, counter in counts.items()
        },
        "policy_settings": {
            f"{sequence}/{fill}": policy_settings(
                sequence,
                fill,
                allow_observed_sequence=allow_observed_sequence,
                device=actual_device,
            )
            | {
                "implementation_sha256": environment["implementation_sha256"],
                "stok_revision": environment["stok_revision"],
            }
            for sequence, fill in CONDITIONS
        },
        "limitations": [
            "TM uses current Kabsch alignment, not optimized TM-align",
            "Metrics use original observed N/CA/C intersected with token availability",
            "Held-out synthetic-gap coordinates are diagnostic targets; missing outputs remain NaN",
            "CPU RSS is the process high-water mark; device memory includes loaded models",
            "Grouping uses independent singleton forwards; no mixed-length tensor-batch throughput claim",
            "No production policy has been selected",
        ],
    }
    report["output_hashes"] = {
        str(path.relative_to(output_dir)): file_sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--preset", choices=["lite", "large"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-observed-sequence", action="store_true")
    parser.add_argument("--synthetic-gaps", type=int, nargs="*", default=[])
    args = vars(parser.parse_args())
    args["synthetic_gaps"] = tuple(args["synthetic_gaps"])
    run_experiments(**args)


if __name__ == "__main__":
    main()
