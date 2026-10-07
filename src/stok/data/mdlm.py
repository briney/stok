"""Clean aligned pairs and read-only semantic preflight for masked diffusion."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

import pyarrow.parquet as pq
import torch

from .collate import tokenize_residues
from .structure_export import validate_structure_dataset
from ..utils.pretrained import file_sha256, json_sha256, state_sha256
from ..utils.tokenizer import Tokenizer

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


class MDLMBatch(TypedDict):
    sequence_mask_id: int
    codebook_size: int
    sequence_tokens: torch.Tensor
    structure_tokens: torch.Tensor
    residue_mask: torch.Tensor
    sequence_valid: torch.Tensor
    structure_valid: torch.Tensor
    sample_keys: list[str]
    crop_offsets: torch.Tensor
    coords: torch.Tensor | None
    missing_structure_count: int
    noncanonical_sequence_count: int


class MDLMRunIdentity(TypedDict):
    sources: dict[str, dict[str, Any]]
    sample_key_namespaces: dict[str, str]
    split_sha256: str | None
    tokenizer_sha256: str
    policy_sha256: str
    codebook_sha256: str
    vocabulary: dict[str, int | str]
    training_signature: str
    eval_cohort: dict[str, Any] | None
    generation_cohort: dict[str, Any] | None


def _sample_key(namespace: str, sequence_id: str) -> str:
    return json.dumps([namespace, sequence_id], separators=(",", ":"))


def prepare_mdlm_batch(
    rows: list[dict],
    tokenizer,
    *,
    max_len: int,
    codebook_size: int,
    crop: Literal["random", "center"],
    seeds: list[int],
) -> MDLMBatch:
    """Validate full rows, crop once, then add boundary and padding slots.

    Set each reader's dataset_name to its preflight sample_key_namespaces entry.
    Coordinates must be read with max_length=None so crop offsets remain valid.
    """
    if not rows or len(seeds) != len(rows):
        raise ValueError("rows must be nonempty with one seed per row")
    if type(max_len) is not int or max_len < 3:
        raise ValueError("max_len must be an integer >= 3")
    if type(codebook_size) is not int or codebook_size < 1:
        raise ValueError("codebook_size must be a positive integer")
    if crop not in {"random", "center"}:
        raise ValueError("crop must be random or center")
    mask_id = tokenizer.mask_token_id
    if (
        type(mask_id) is not int
        or not 0 <= mask_id < len(tokenizer)
        or mask_id not in tokenizer.all_special_ids
        or mask_id
        in {
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
        }
    ):
        raise ValueError(
            "Tokenizer must provide a distinct valid special mask token ID"
        )
    canonical_ids = tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
    if len(set(canonical_ids)) != 20 or any(
        token is None or token in tokenizer.all_special_ids for token in canonical_ids
    ):
        raise ValueError("Tokenizer must represent all 20 canonical amino acids")
    shape = (len(rows), max_len)
    sequence_tokens = torch.full(shape, tokenizer.pad_token_id, dtype=torch.long)
    structure_tokens = torch.full(shape, codebook_size, dtype=torch.long)
    residue_mask = torch.zeros(shape, dtype=torch.bool)
    sequence_valid = torch.zeros_like(residue_mask)
    structure_valid = torch.zeros_like(residue_mask)
    coords = (
        torch.full((*shape, 3, 3), float("nan"))
        if any(row.get("coords", row.get("coordinates")) is not None for row in rows)
        else None
    )
    offsets, sample_keys = [], []
    for i, (row, seed) in enumerate(zip(rows, seeds)):
        context = f"Sample {row.get('dataset', '?')}/{row.get('sequence_id', '?')}"
        try:
            namespace, sequence_id = row["dataset"], row["sequence_id"]
            if (
                not isinstance(namespace, str)
                or not namespace
                or not isinstance(sequence_id, str)
                or not sequence_id
            ):
                raise ValueError(
                    "dataset namespace and sequence_id must be nonempty strings"
                )
            seq = row["sequence"]
            if not isinstance(seq, str) or not seq:
                raise ValueError("sequence must be a nonempty string")
            # The shared encoder checks every uncropped residue, including control tokens.
            ids = tokenize_residues(seq, tokenizer, len(seq) + 2)
            raw = row["structure_tokens"]
            if isinstance(raw, torch.Tensor):
                if raw.dtype not in (
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                ):
                    raise ValueError("structure_tokens must be integers")
                labels = raw.cpu()
            else:
                if not isinstance(raw, list) or any(
                    value is not None and (type(value) is not int or value < 0)
                    for value in raw
                ):
                    raise ValueError(
                        "structure_tokens must be nullable nonnegative integers"
                    )
                labels = torch.tensor(
                    [-1 if value is None else value for value in raw], dtype=torch.long
                )
            if labels.shape != (len(seq),):
                raise ValueError("structure_tokens length must match sequence length")
            if ((labels < -1) | (labels >= codebook_size)).any():
                raise ValueError("structure_tokens contain an out-of-codebook label")
            raw_coords = row.get("coords", row.get("coordinates"))
            observed = (
                None
                if raw_coords is None
                else torch.as_tensor(raw_coords, dtype=torch.float32)
            )
            if observed is not None and observed.shape != (len(seq), 3, 3):
                raise ValueError(
                    "coordinates must have shape [sequence_length,3,3]; read with max_length=None"
                )
            length = min(len(seq), max_len - 2)
            slack = len(seq) - length
            start = (
                slack // 2
                if crop == "center"
                else int(
                    torch.randint(
                        slack + 1, (), generator=torch.Generator().manual_seed(seed)
                    )
                )
            )
            positions = slice(1, length + 1)
            sequence_tokens[i, 0] = tokenizer.bos_token_id
            sequence_tokens[i, length + 1] = tokenizer.eos_token_id
            sequence_tokens[i, positions] = ids[1 + start : 1 + start + length]
            selected = labels[start : start + length].long()
            available = selected >= 0
            structure_tokens[i, positions] = selected.masked_fill(
                ~available, codebook_size + 2
            )
            residue_mask[i, positions] = True
            sequence_valid[i, positions] = torch.isin(
                sequence_tokens[i, positions], torch.tensor(canonical_ids)
            )
            structure_valid[i, positions] = available
            if coords is not None and observed is not None:
                coords[i, positions] = observed[start : start + length]
            offsets.append(start)
            sample_keys.append(_sample_key(namespace, sequence_id))
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError(f"{context}: {error}") from error
    return {
        "sequence_mask_id": mask_id,
        "codebook_size": codebook_size,
        "sequence_tokens": sequence_tokens,
        "structure_tokens": structure_tokens,
        "residue_mask": residue_mask,
        "sequence_valid": sequence_valid,
        "structure_valid": structure_valid,
        "sample_keys": sample_keys,
        "crop_offsets": torch.tensor(offsets, dtype=torch.long),
        "coords": coords,
        "missing_structure_count": int((residue_mask & ~structure_valid).sum()),
        "noncanonical_sequence_count": int((residue_mask & ~sequence_valid).sum()),
    }


def _jsonl(path: Path) -> list[dict]:
    try:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("must contain nonempty JSON objects")
        return rows
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: invalid JSONL: {error}") from error


def _key(row: dict, context: str) -> tuple[str, str]:
    if any(
        not isinstance(row.get(name), str) or not row[name]
        for name in ("dataset", "sequence_id")
    ):
        raise ValueError(f"{context}: dataset and sequence_id must be nonempty strings")
    return row["dataset"], row["sequence_id"]


def validate_mdlm_sources(
    train_sources: dict,
    eval_sources: dict,
    *,
    codebook: torch.Tensor,
    split_manifest: Path | None,
    eval_cohort: Path | None = None,
    generation_cohort: Path | None = None,
) -> MDLMRunIdentity:
    """Audit completed sources, exact code identity, frozen splits and cohorts.

    eval_sources may include test datasets for split auditing; tuning cohorts
    must select validation samples. Every split assignment needs source coverage.
    training_signature excludes cohort/evaluation source settings. All recorded
    fields are JSON primitives; this function never creates run/W&B artifacts.
    """
    if not train_sources:
        raise ValueError("train_sources must contain a completed paired dataset")
    if codebook.ndim != 2 or not codebook.numel() or not torch.isfinite(codebook).all():
        raise ValueError("train codebook must be a nonempty finite matrix")
    if set(train_sources) & set(eval_sources):
        raise ValueError("train/eval sources must have distinct dataset names")
    if split_manifest is None and (
        eval_sources
        or len(train_sources) != 1
        or eval_cohort is not None
        or generation_cohort is not None
    ):
        raise ValueError(
            "train/eval sources and cohorts require a split manifest; only one-source overfit may omit splits"
        )
    digest = state_sha256({"codebook": codebook})
    # ponytail: sample metadata stays in RAM; use sqlite if corpus metadata outgrows memory.
    sources, namespaces, samples = {}, {}, {}
    compatible = None
    for kind, configured in (("train", train_sources), ("eval", eval_sources)):
        for name, options in configured.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{kind} source name must be a nonempty string")
            context = f"{kind} source {name}"
            try:
                path = Path(
                    options["path"] if isinstance(options, Mapping) else options
                )
                context += f" ({path})"
                summary = validate_structure_dataset(path)
                tokenizer = summary["provenance"]["tokenizer"]
                if (
                    tokenizer["codebook_sha256"] != digest
                    or tokenizer["codebook_size"] != codebook.shape[0]
                ):
                    raise ValueError(
                        "selected codebook does not match dataset codebook digest/size"
                    )
                semantics = (summary["tokenizer_sha256"], summary["policy_sha256"])
                if compatible is not None and semantics != compatible:
                    raise ValueError(
                        "incompatible tokenizer/policy identity across train/eval sources"
                    )
                compatible = semantics
                namespace = json_sha256(
                    {"provenance": summary["provenance"], "shards": summary["shards"]}
                )
                namespaces[name] = namespace
                sources[name] = {
                    "path": str(path.resolve()),
                    "sha256": namespace,
                    "manifest_sha256": file_sha256(path / "manifest.json"),
                    "kind": kind,
                    "row_count": summary["row_count"],
                    "tokenizer_sha256": semantics[0],
                    "policy_sha256": semantics[1],
                    "policy": summary["provenance"]["policy"],
                    "tokenizer_context": "full_chain",
                }
                ids = set()
                for shard in summary["shards"]:
                    for row in pq.read_table(
                        path / shard["path"], columns=["sequence_id", "source"]
                    ).to_pylist():
                        sequence_id = row["sequence_id"]
                        if (
                            not isinstance(sequence_id, str)
                            or not sequence_id
                            or sequence_id in ids
                        ):
                            raise ValueError(
                                f"duplicate/invalid sequence_id {sequence_id!r}"
                            )
                        ids.add(sequence_id)
                        source = row["source"]
                        if (
                            not isinstance(source, dict)
                            or not isinstance(source.get("sha256"), str)
                            or not source["sha256"]
                        ):
                            raise ValueError(
                                f"sample {sequence_id}: missing source-file identity"
                            )
                        samples[name, sequence_id] = {
                            "kind": kind,
                            "source": source["sha256"],
                            "sample_key": _sample_key(namespace, sequence_id),
                        }
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{context}: {error}") from error
    assert compatible is not None
    assignments = {}
    split_sha256 = None
    if split_manifest is not None:
        split_manifest = Path(split_manifest)
        groups = {}
        for row in _jsonl(split_manifest):
            key = _key(row, str(split_manifest))
            context = f"{split_manifest}: source {key[0]} sample {key[1]}"
            split, cluster = row.get("split"), row.get("cluster_id")
            if (
                not isinstance(split, str)
                or split not in {"train", "validation", "test"}
                or not isinstance(cluster, str)
                or not cluster
            ):
                raise ValueError(f"{context}: invalid split/cluster_id")
            if key in assignments:
                raise ValueError(f"{context}: duplicate/conflicting split assignment")
            if key not in samples:
                raise ValueError(
                    f"{context}: assigned sample missing from supplied sources"
                )
            sample = samples[key]
            if (sample["kind"] == "train") != (split == "train"):
                raise ValueError(f"{context}: split disagrees with train/eval source")
            assignments[key] = split
            for group in (("cluster", cluster), ("source", sample["source"])):
                previous = groups.setdefault(group, split)
                if previous != split:
                    raise ValueError(
                        f"{context}: cross-split {group[0]} overlap {group[1]}"
                    )
        unassigned = samples.keys() - assignments.keys()
        if unassigned:
            key = sorted(unassigned)[0]
            raise ValueError(
                f"{split_manifest}: source {key[0]} sample {key[1]} has no split assignment"
            )
        split_sha256 = file_sha256(split_manifest)

    def cohort_identity(path):
        if path is None:
            return None
        path = Path(path)
        keys, seen = [], set()
        for row in _jsonl(path):
            key = _key(row, f"cohort {path}")
            if key in seen:
                raise ValueError(f"cohort {path}: duplicate sample {key}")
            if key not in samples or assignments.get(key) != "validation":
                raise ValueError(
                    f"cohort {path}: sample {key} must exist in the validation split"
                )
            seen.add(key)
            keys.append(samples[key]["sample_key"])
        return {"sha256": file_sha256(path), "sample_keys": keys}

    vocabulary = {
        "codebook_size": codebook.shape[0],
        "structure_pad": codebook.shape[0],
        "structure_mask": codebook.shape[0] + 1,
        "structure_unavailable": codebook.shape[0] + 2,
        "sequence_targets": CANONICAL_AA,
        "sequence_vocab_sha256": json_sha256(Tokenizer().get_vocab()),
    }
    training_identity = {
        "sources": [
            {"dataset": name, **identity}
            for name, identity in sources.items()
            if identity["kind"] == "train"
        ],
        "split_sha256": split_sha256,
        "codebook_sha256": digest,
        "tokenizer_sha256": compatible[0],
        "policy_sha256": compatible[1],
        "vocabulary": vocabulary,
    }
    return {
        "sources": sources,
        "sample_key_namespaces": namespaces,
        "split_sha256": split_sha256,
        "tokenizer_sha256": compatible[0],
        "policy_sha256": compatible[1],
        "codebook_sha256": digest,
        "vocabulary": vocabulary,
        "training_signature": json_sha256(training_identity),
        "eval_cohort": cohort_identity(eval_cohort),
        "generation_cohort": cohort_identity(generation_cohort),
    }
