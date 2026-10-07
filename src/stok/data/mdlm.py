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
    canonical_population_sha256: str
    population: dict[str, dict[str, Any]]
    splits: dict[str, dict[str, Any]]
    split_sha256: str | None
    split_manifest_sha256: str | None
    tokenizer_sha256: str
    policy_sha256: str
    codebook_sha256: str
    vocabulary: dict[str, int | str]
    training_signature: str
    shared_cases: dict[str, Any] | None
    coverage: dict[str, dict[str, Any]]


def mdlm_vocabulary(codebook: torch.Tensor) -> dict[str, int | str]:
    return {
        "codebook_size": codebook.shape[0],
        "structure_pad": codebook.shape[0],
        "structure_mask": codebook.shape[0] + 1,
        "structure_unavailable": codebook.shape[0] + 2,
        "sequence_targets": CANONICAL_AA,
        "sequence_vocab_sha256": json_sha256(Tokenizer().get_vocab()),
    }


def mdlm_training_signature(identity) -> str:
    """Bind training replay and its canonical/split projection, without artifacts."""
    sources = [
        {"dataset": name, **source}
        for name, source in identity["sources"].items()
        if source["kind"] == "train"
    ]
    members = sorted({key for source in sources for key in source["member_ids"]})
    return json_sha256(
        {
            "sources": sources,
            "population": [identity["population"][key] for key in members],
            "splits": [identity["splits"][key] for key in members]
            if identity["splits"]
            else [],
            **{
                key: identity[key]
                for key in (
                    "codebook_sha256",
                    "tokenizer_sha256",
                    "policy_sha256",
                    "vocabulary",
                )
            },
        }
    )


def prepare_mdlm_batch(
    rows: list[dict],
    tokenizer,
    *,
    max_len: int,
    codebook_size: int,
    crop: Literal["random", "center"],
    seeds: list[int],
    crop_intervals: list[tuple[int, int]] | None = None,
) -> MDLMBatch:
    """Validate full rows, crop once, then add boundary and padding slots.

    Biological keys are canonical IDs; dataset names retain replay identity.
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
    if crop_intervals is not None and len(crop_intervals) != len(rows):
        raise ValueError("crop_intervals requires one crop per row")
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
            canonical_id = row["canonical_id"]
            if not isinstance(canonical_id, str) or len(canonical_id) != 64:
                raise ValueError("canonical_id must be a SHA-256 digest")
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
            if crop_intervals is not None:
                start, stop = crop_intervals[i]
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or not 0 <= start < stop <= len(seq)
                    or stop - start > max_len - 2
                ):
                    raise ValueError("unsupported or invalid recorded crop")
                length = stop - start
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
            sample_keys.append(canonical_id)
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


def validate_mdlm_sources(
    train_sources: Mapping,
    eval_sources: Mapping,
    *,
    codebook: torch.Tensor,
    split_manifest: str | Path | None,
    case_manifest: str | Path | None = None,
) -> MDLMRunIdentity:
    """Audit complete inventory unions before admitting native loader rows."""
    from .canonical import iter_canonical_records
    from ..eval.cases import audit_canonical_splits, read_evaluation_cases

    if not train_sources:
        raise ValueError("train_sources must contain a completed paired dataset")
    if codebook.ndim != 2 or not codebook.numel() or not torch.isfinite(codebook).all():
        raise ValueError("train codebook must be a nonempty finite matrix")
    if set(train_sources) & set(eval_sources):
        raise ValueError("train/eval sources must have distinct dataset names")
    if split_manifest is None and (
        eval_sources or len(train_sources) != 1 or case_manifest
    ):
        raise ValueError(
            "train/eval sources and cases require a split manifest; only one-source overfit may omit splits"
        )
    shared = read_evaluation_cases(case_manifest) if case_manifest is not None else None
    directories = (
        {Path(ref["directory"]).resolve() for ref in shared["canonical_inventories"]}
        if shared
        else set()
    )
    sources, coverage = {}, {}
    compatible = None
    digest = state_sha256({"codebook": codebook})
    for kind, configured in (("train", train_sources), ("eval", eval_sources)):
        for name, options in configured.items():
            context = f"{kind} source {name}"
            try:
                if not isinstance(name, str) or not name:
                    raise ValueError("source name must be a nonempty string")
                path = Path(
                    options["path"] if isinstance(options, Mapping) else options
                ).resolve()
                summary = validate_structure_dataset(path)
                tokenizer = summary["provenance"]["tokenizer"]
                if (
                    tokenizer["codebook_sha256"] != digest
                    or tokenizer["codebook_size"] != codebook.shape[0]
                ):
                    raise ValueError(
                        "selected codebook does not match dataset codebook digest/size"
                    )
                semantics = summary["tokenizer_sha256"], summary["policy_sha256"]
                if compatible is not None and semantics != compatible:
                    raise ValueError(
                        "incompatible tokenizer/policy identity across train/eval sources"
                    )
                compatible = semantics
                directory = Path(summary["canonical_directory"])
                directory = (
                    (path / directory).resolve()
                    if not directory.is_absolute()
                    else directory.resolve()
                )
                directories.add(directory)
                members = [
                    record["canonical_id"]
                    for record in iter_canonical_records(directory)
                ]
                admitted = [
                    row["canonical_id"]
                    for shard in summary["shards"]
                    for row in pq.read_table(
                        path / shard["path"], columns=["canonical_id"]
                    ).to_pylist()
                ]
                rejected = [
                    json.loads(line)
                    for line in (path / "rejections.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                rejected_by_id = {row["canonical_id"]: row for row in rejected}
                for key in members:
                    if key in coverage:
                        raise ValueError(f"duplicate canonical source membership {key}")
                    rejection = rejected_by_id.get(key)
                    coverage[key] = {
                        "source": name,
                        "status": "rejected" if rejection else "admitted",
                        "reason": rejection["reason"] if rejection else None,
                    }
                replay = {
                    "shards": summary["shards"],
                    "row_order": summary["row_order"],
                    "admitted_ids": admitted,
                    "rejections": rejected,
                    "manifest_sha256": file_sha256(path / "manifest.json"),
                }
                sources[name] = {
                    "path": str(path),
                    "kind": kind,
                    "row_count": summary["row_count"],
                    "member_ids": sorted(members),
                    "canonical_population_sha256": summary[
                        "canonical_population_sha256"
                    ],
                    "representation_sha256": summary["representation_sha256"],
                    "provenance": summary["provenance"],
                    "replay": replay,
                    "replay_sha256": json_sha256(replay),
                    "tokenizer_sha256": semantics[0],
                    "policy_sha256": semantics[1],
                    "policy": summary["provenance"]["policy"],
                    "tokenizer_context": "full_chain",
                }
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{context}: {error}") from error
    assert compatible is not None

    # ponytail: compact population metadata stays in RAM; use indexed storage if it outgrows memory.
    def records():
        for directory in sorted(directories):
            yield from iter_canonical_records(directory)

    audit: dict[str, Any]
    if split_manifest is not None:
        audit = audit_canonical_splits(records(), split_manifest)
        population, splits = audit["members"], audit["assignments"]
        for key, entry in coverage.items():
            if (sources[entry["source"]]["kind"] == "train") != (
                splits[key]["split"] == "train"
            ):
                raise ValueError(
                    f"split disagrees with train/eval source: {entry['source']} {key}"
                )
        if shared and any(
            shared[key] != audit[key]
            for key in ("population_sha256", "split_sha256", "split_manifest_sha256")
        ):
            raise ValueError("Frozen cases disagree with canonical population/splits")
    else:
        population = {
            r["canonical_id"]: {
                key: r[key]
                for key in (
                    "canonical_id",
                    "content_sha256",
                    "residue_map_sha256",
                    "parent_ids",
                )
            }
            | {"full_length": len(r["sequence"])}
            for r in records()
        }
        splits = {}
        audit = {
            "population_sha256": json_sha256(
                sorted(
                    (key, r["content_sha256"], r["residue_map_sha256"], r["parent_ids"])
                    for key, r in population.items()
                )
            ),
            "split_sha256": None,
            "split_manifest_sha256": None,
        }
    if shared:
        for case in shared["cases"]:
            entry = coverage.get(case["canonical_id"])
            if not entry or sources[entry["source"]]["kind"] != "eval":
                raise ValueError(
                    f"Frozen case missing admitted/rejected evaluation source: {case['canonical_id']}"
                )
    identity: MDLMRunIdentity = {
        "sources": sources,
        "population": population,
        "splits": splits,
        "canonical_population_sha256": audit["population_sha256"],
        "split_sha256": audit["split_sha256"],
        "split_manifest_sha256": audit["split_manifest_sha256"],
        "tokenizer_sha256": compatible[0],
        "policy_sha256": compatible[1],
        "codebook_sha256": digest,
        "vocabulary": mdlm_vocabulary(codebook),
        "training_signature": "",
        "shared_cases": shared,
        "coverage": coverage,
    }
    identity["training_signature"] = mdlm_training_signature(identity)
    return identity


def validate_mdlm_identity(identity: Mapping[str, Any]) -> None:
    """Recompute saved normalized bindings without reopening original artifacts."""
    from .structure_export import representation_sha256
    from ..eval.cases import _counts, _request, _validate_case
    from ..utils.mdlm import stable_seed

    try:
        population, splits = identity["population"], identity["splits"]
        digest = json_sha256(
            sorted(
                (
                    key,
                    row["content_sha256"],
                    row["residue_map_sha256"],
                    row["parent_ids"],
                )
                for key, row in population.items()
            )
        )
        if digest != identity["canonical_population_sha256"]:
            raise ValueError("Canonical population digest mismatch")
        split_digest = (
            json_sha256([splits[key] for key in sorted(splits)]) if splits else None
        )
        if (
            split_digest != identity["split_sha256"]
            or splits
            and splits.keys() != population.keys()
        ):
            raise ValueError("Canonical split digest/membership mismatch")
        expected_coverage = set()
        for name, source in identity["sources"].items():
            provenance = source["provenance"]
            members = source["member_ids"]
            if (
                len(set(members)) != len(members)
                or expected_coverage.intersection(members)
                or set(members) - population.keys()
            ):
                raise ValueError("Duplicate or unknown canonical source members")
            expected_coverage.update(members)
            source_population = json_sha256(
                sorted(
                    (
                        key,
                        population[key]["content_sha256"],
                        population[key]["residue_map_sha256"],
                        population[key]["parent_ids"],
                    )
                    for key in members
                )
            )
            if (
                source_population != source["canonical_population_sha256"]
                or source_population != provenance["canonical_population_sha256"]
                or representation_sha256(provenance) != source["representation_sha256"]
                or json_sha256(source["replay"]) != source["replay_sha256"]
                or json_sha256(provenance["tokenizer"]) != source["tokenizer_sha256"]
                or json_sha256(provenance["policy"]) != source["policy_sha256"]
                or source["policy"] != provenance["policy"]
                or source["tokenizer_context"] != "full_chain"
                or source["tokenizer_sha256"] != identity["tokenizer_sha256"]
                or source["policy_sha256"] != identity["policy_sha256"]
                or provenance["tokenizer"]["codebook_sha256"]
                != identity["codebook_sha256"]
            ):
                raise ValueError("Representation/replay identity binding mismatch")
            admitted = source["replay"]["admitted_ids"]
            if (
                len(set(admitted)) != len(admitted)
                or len(admitted) != source["row_count"]
                or set(admitted) - set(members)
            ):
                raise ValueError("Replay admitted row coverage mismatch")
            rejected = {
                row["canonical_id"]: row["reason"]
                for row in source["replay"]["rejections"]
            }
            if set(admitted) & rejected.keys() or set(
                admitted
            ) | rejected.keys() != set(members):
                raise ValueError("Saved rejection coverage mismatch")
            for key in members:
                entry = identity["coverage"][key]
                if entry["reason"] != rejected.get(key):
                    raise ValueError("Saved rejection reason mismatch")
                if entry["source"] != name or entry["status"] != (
                    "admitted" if key in admitted else "rejected"
                ):
                    raise ValueError("Saved representation coverage mismatch")
                if splits and (source["kind"] == "train") != (
                    splits[key]["split"] == "train"
                ):
                    raise ValueError("Saved source split mismatch")
        if expected_coverage != identity["coverage"].keys():
            raise ValueError("Saved source coverage membership mismatch")
        shared = identity["shared_cases"]
        if shared is not None:
            if (
                not isinstance(shared, Mapping)
                or type(shared.get("schema_version")) is not int
                or shared["schema_version"] != 1
            ):
                raise ValueError("Unsupported saved case artifact wrapper")
            request, cases = _request(shared["request"]), shared["cases"]
            if (
                shared["population_sha256"] != digest
                or shared["split_sha256"] != split_digest
                or shared["split_manifest_sha256"] != identity["split_manifest_sha256"]
                or shared["shared_cases_sha256"] != json_sha256(cases)
                or any(shared[key] != value for key, value in _counts(cases).items())
            ):
                raise ValueError("Frozen case population/split/control digest mismatch")
            expected = [
                (member, kind, family, replicate)
                for member in request["members"]
                for kind in ("denoising", "generation")
                for family in request[kind]
                for replicate in range(request["replicates"])
            ]
            if [
                (c["canonical_id"], c["kind"], c["family_key"], c["replicate"])
                for c in cases
            ] != expected:
                raise ValueError("Frozen case request expansion mismatch")
            for ordinal, case in enumerate(cases):
                _validate_case(case)
                member = population[case["canonical_id"]]
                size = min(member["full_length"], request["crop_residues"])
                start = (member["full_length"] - size) // 2
                if (
                    case["ordinal"] != ordinal
                    or case["crop"] != [start, start + size]
                    or any(
                        case[key] != member[key]
                        for key in ("content_sha256", "residue_map_sha256")
                    )
                    or case["definition"] != request[case["kind"]][case["family_key"]]
                    or case["seed"]
                    != stable_seed(
                        [
                            "c1-case-v1",
                            request["seed"],
                            case["canonical_id"],
                            case["family_key"],
                            case["replicate"],
                        ]
                    )
                    or splits[case["canonical_id"]]["split"] != "validation"
                    or case["canonical_id"] not in expected_coverage
                ):
                    raise ValueError("Frozen case canonical/request binding mismatch")
        if mdlm_training_signature(identity) != identity["training_signature"]:
            raise ValueError("MDLM training identity mismatch")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Incomplete canonical MDLM identity") from exc
