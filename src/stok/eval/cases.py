"""Canonical split audits and immutable residue-level evaluation controls."""

from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from stok.data.canonical import (
    iter_canonical_records,
    publish_directory,
    validate_canonical_record,
)
from stok.data.mdlm import CANONICAL_AA, MDLMBatch
from stok.utils.mdlm import REGIMES, build_mask_groups, stable_seed
from stok.utils.pretrained import file_sha256, json_sha256


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _read_jsonl(path: Path):
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected an object")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: malformed JSONL object") from error
            yield row


def audit_canonical_splits(
    records: Iterable[Mapping[str, Any]], split_manifest: str | Path
) -> dict[str, Any]:
    """Audit every record, retaining identity/digests/lineage rather than coordinates."""
    path = Path(split_manifest)
    assignments = {}
    for row in _read_jsonl(path):
        if (
            set(row) != {"canonical_id", "split", "cluster_id"}
            or not _digest(row.get("canonical_id"))
            or not isinstance(row.get("split"), str)
            or row["split"] not in {"train", "validation", "test"}
            or not _name(row.get("cluster_id"))
        ):
            raise ValueError("Invalid explicit canonical split assignment")
        if row["canonical_id"] in assignments:
            raise ValueError("Duplicate canonical split assignment")
        assignments[row["canonical_id"]] = row
    # ponytail: compact metadata in RAM; a disk index can replace it for larger populations.
    members, selections, groups = {}, set(), {}
    for record in records:
        validate_canonical_record(record)
        key = record["canonical_id"]
        if key in members:
            raise ValueError(
                "Duplicate or conflicting canonical membership across inventories"
            )
        if key not in assignments:
            raise ValueError("Unassigned canonical member")
        identity, assignment = record["identity"], assignments[key]
        if identity["record_kind"] == "structure":
            selection = tuple(
                identity[field]
                for field in (
                    "source_revision_sha256",
                    "model_index",
                    "model_serial_id",
                    "chain_namespace",
                    "chain_id",
                )
            )
            if selection in selections:
                raise ValueError(
                    "Duplicate biological raw-revision/model/chain selection"
                )
            selections.add(selection)
        labels = (
            ("cluster", assignment["cluster_id"]),
            ("source", identity["source_namespace"], identity["source_accession"]),
            ("revision", identity["source_revision_sha256"]),
        )
        for label in labels:
            if groups.setdefault(label, assignment["split"]) != assignment["split"]:
                raise ValueError("Cross-split cluster/source/revision leakage")
        members[key] = {
            field: record[field]
            for field in (
                "canonical_id",
                "content_sha256",
                "residue_map_sha256",
                "parent_ids",
            )
        }
        members[key]["full_length"] = len(record["sequence"])
    if assignments.keys() != members.keys():
        raise ValueError("Split manifest has unknown canonical assignments")
    # Kahn's traversal accepts arbitrary physical order and avoids recursion limits.
    pending, children = {}, {key: [] for key in members}
    for key, member in members.items():
        parents = member["parent_ids"]
        pending[key] = len(parents)
        for parent in parents:
            if parent not in members:
                raise ValueError("Unknown canonical parent assignment")
            if assignments[parent]["split"] != assignments[key]["split"]:
                raise ValueError("Cross-split inherited parent lineage")
            children[parent].append(key)
    ready = [key for key, count in pending.items() if not count]
    visited = 0
    while ready:
        key = ready.pop()
        visited += 1
        for child in children[key]:
            pending[child] -= 1
            if not pending[child]:
                ready.append(child)
    if visited != len(members):
        raise ValueError("Canonical parent cycle")
    return {
        "assignments": assignments,
        "members": members,
        "population_sha256": json_sha256(
            sorted(
                (
                    key,
                    row["content_sha256"],
                    row["residue_map_sha256"],
                    row["parent_ids"],
                )
                for key, row in members.items()
            )
        ),
        "split_sha256": json_sha256([assignments[key] for key in sorted(assignments)]),
        "split_manifest_sha256": file_sha256(path),
    }


def _family(definition: Any, kind: str) -> dict[str, Any]:
    allowed = {"regime", "placement", "span_mean"} | (
        {"probability"} if kind == "denoising" else set()
    )
    if not isinstance(definition, Mapping) or set(definition) - allowed:
        raise ValueError("Invalid evaluation family fields")
    result = dict(definition)
    result.setdefault("placement", "token")
    if (
        result.get("regime") not in REGIMES
        or not isinstance(result["placement"], str)
        or result["placement"] not in {"token", "span"}
    ):
        raise ValueError("Unsupported native family regime/placement")
    if result["placement"] == "span":
        result.setdefault("span_mean", 8.0)
    elif "span_mean" in result:
        raise ValueError("Token families do not have span_mean")
    if "span_mean" in result and (
        type(result["span_mean"]) not in {float, int}
        or not math.isfinite(result["span_mean"])
        or result["span_mean"] < 1
    ):
        raise ValueError("span_mean must be finite and >= 1")
    if kind == "denoising" and (
        type(result.get("probability")) not in {float, int}
        or not math.isfinite(result["probability"])
        or not 0 <= result["probability"] <= 1
    ):
        raise ValueError("Denoising probability must be explicit, finite and in [0,1]")
    for key in ("span_mean", "probability"):
        if key in result:
            result[key] = float(result[key])
    return result


def _request(request: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "seed",
        "crop_residues",
        "members",
        "replicates",
        "denoising",
        "generation",
    }
    if not isinstance(request, Mapping) or set(request) - allowed:
        raise ValueError("Invalid case request fields")
    resolved = dict(request)
    if (
        type(resolved.get("schema_version")) is not int
        or resolved["schema_version"] != 1
    ):
        raise ValueError("Unsupported case schema version")
    if type(resolved.get("seed")) is not int or resolved["seed"] < 0:
        raise ValueError("Case seed must be a nonnegative integer")
    resolved.setdefault("replicates", 1)
    for key in ("crop_residues", "replicates"):
        if type(resolved.get(key)) is not int or resolved[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    members = resolved.get("members")
    if (
        not isinstance(members, list)
        or not members
        or any(not _digest(key) for key in members)
        or len(set(members)) != len(members)
    ):
        raise ValueError("Case members must be unique explicit canonical IDs")
    resolved.setdefault(
        "denoising",
        {
            f"{regime}_{placement}_p{p:g}": {
                "regime": regime,
                "placement": placement,
                "probability": p,
                **({"span_mean": 8.0} if placement == "span" else {}),
            }
            for regime in REGIMES
            for placement in ("token", "span")
            for p in (0.15, 0.5, 0.85, 1.0)
        },
    )
    resolved.setdefault("generation", {})
    keys = set()
    for kind in ("denoising", "generation"):
        families = resolved[kind]
        if not isinstance(families, Mapping):
            raise ValueError("Evaluation families must be a mapping")
        normalized = {}
        for key, definition in families.items():
            if not _name(key) or "/" in key or key in keys:
                raise ValueError("Family keys must be globally unique nonempty names")
            keys.add(key)
            normalized[key] = _family(definition, kind)
        resolved[kind] = {key: normalized[key] for key in sorted(normalized)}
    if not keys:
        raise ValueError("At least one evaluation family must be enabled")
    return resolved


def _eligibility(record, crop, regime):
    start, stop = crop
    original = torch.tensor(
        [
            [aa in CANONICAL_AA, all(mask)]
            for aa, mask in zip(
                record["sequence"][start:stop], record["atom_mask"][start:stop]
            )
        ],
        dtype=torch.bool,
    )
    eligible = original.clone()
    if regime == "structure_only":
        eligible[:, 0] = False
    elif regime == "sequence_only":
        eligible[:, 1] = False
    return original, eligible


def _case(record, request, kind, family_key, definition, replicate):
    length = len(record["sequence"])
    size = min(length, request["crop_residues"])
    start = (length - size) // 2
    crop = [start, start + size]
    seed = stable_seed(
        ["c1-case-v1", request["seed"], record["canonical_id"], family_key, replicate]
    )
    original, eligible = _eligibility(record, crop, definition["regime"])
    groups = build_mask_groups(
        eligible,
        torch.ones(size, dtype=torch.bool),
        placement=definition["placement"],
        tied=definition["regime"] == "joint_tied",
        span_mean=definition.get("span_mean", 8.0),
        generator=torch.Generator().manual_seed(stable_seed([seed, "partition"])),
    )
    probability = definition.get("probability", 1.0)
    masked = torch.zeros_like(eligible)
    ids, inverse = groups[eligible].unique(return_inverse=True)
    draws = torch.rand(
        len(ids),
        generator=torch.Generator().manual_seed(stable_seed([seed, "corruption"])),
        dtype=torch.float64,
    ) < torch.tensor(probability, dtype=torch.float32)
    masked[eligible] = draws[inverse]
    result = {
        "schema_version": 1,
        "canonical_id": record["canonical_id"],
        "content_sha256": record["content_sha256"],
        "residue_map_sha256": record["residue_map_sha256"],
        "kind": kind,
        "family_key": family_key,
        "replicate": replicate,
        "seed": seed,
        "crop": crop,
        "positions": list(range(start, start + size)),
        "definition": definition,
        "eligible": eligible.tolist(),
        "group_ids": groups.tolist(),
        "masked": masked.tolist(),
        "conditioning": (
            original & ~eligible if kind == "generation" else torch.zeros_like(eligible)
        ).tolist(),
    }
    result["case_id"] = json_sha256(result)
    return result


def _inventory_records(directories, references):
    paths = list(dict.fromkeys(Path(directory).resolve() for directory in directories))
    if not paths:
        raise ValueError("At least one completed canonical inventory is required")
    for path in paths:
        yield from iter_canonical_records(path)
        summary = json.loads((path / "manifest.json").read_text())
        references.append(
            {
                "directory": str(path),
                "manifest_sha256": file_sha256(path / "manifest.json"),
                "population_sha256": summary["population_sha256"],
            }
        )


def _counts(cases):
    return {
        "case_count": len(cases),
        "denoising_case_count": sum(case["kind"] == "denoising" for case in cases),
        "generation_case_count": sum(case["kind"] == "generation" for case in cases),
        "unique_sample_count": len({case["canonical_id"] for case in cases}),
    }


def freeze_evaluation_cases(
    canonical_dirs: Sequence[str | Path],
    split_manifest: str | Path,
    request: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze science before publication, streaming originals and retaining selected controls."""
    resolved = _request(request)
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    members, references, selected = set(resolved["members"]), [], {}

    def records():
        for record in _inventory_records(canonical_dirs, references):
            if record["canonical_id"] in members:
                selected[record["canonical_id"]] = [
                    _case(record, resolved, kind, key, definition, replicate)
                    for kind in ("denoising", "generation")
                    for key, definition in resolved[kind].items()
                    for replicate in range(resolved["replicates"])
                ]
            yield record

    audit = audit_canonical_splits(records(), split_manifest)
    if members - audit["assignments"].keys():
        raise ValueError("Unknown requested canonical case members")
    if any(audit["assignments"][key]["split"] != "validation" for key in members):
        raise ValueError("Monitoring cases must be validation-only")
    cases = [case for key in resolved["members"] for case in selected[key]]
    for ordinal, case in enumerate(cases):
        case["ordinal"] = ordinal
    summary = {
        "schema_version": 1,
        "status": "complete",
        "request": resolved,
        **{
            key: audit[key]
            for key in ("population_sha256", "split_sha256", "split_manifest_sha256")
        },
        "split_manifest": str(Path(split_manifest).resolve()),
        "canonical_inventories": references,
        "shared_cases_sha256": json_sha256(cases),
        **_counts(cases),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        (staging / "cases.jsonl").write_text(
            "".join(
                json.dumps(case, sort_keys=True, allow_nan=False) + "\n"
                for case in cases
            )
        )
        summary["cases_sha256"] = file_sha256(staging / "cases.jsonl")
        (staging / "manifest.json").write_text(
            json.dumps(summary, sort_keys=True, allow_nan=False) + "\n"
        )
        publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {**summary, "cases": cases}


def _validate_case(case: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "canonical_id",
        "content_sha256",
        "residue_map_sha256",
        "kind",
        "family_key",
        "replicate",
        "seed",
        "crop",
        "positions",
        "definition",
        "eligible",
        "group_ids",
        "masked",
        "conditioning",
        "case_id",
        "ordinal",
    }
    try:
        if (
            set(case) != fields
            or type(case["schema_version"]) is not int
            or case["schema_version"] != 1
        ):
            raise ValueError("Invalid frozen case schema/fields")
        for key in ("canonical_id", "content_sha256", "residue_map_sha256", "case_id"):
            if not _digest(case[key]):
                raise ValueError("Invalid frozen case digest")
        if case["kind"] not in {"denoising", "generation"} or not _name(
            case["family_key"]
        ):
            raise ValueError("Invalid frozen family/kind")
        if _family(case["definition"], case["kind"]) != case["definition"]:
            raise ValueError("Frozen definition is not normalized")
        for key in ("ordinal", "replicate", "seed"):
            if type(case[key]) is not int or case[key] < 0:
                raise ValueError(
                    "Frozen ordinals/replicates/seeds must be nonnegative integers"
                )
        crop = case["crop"]
        if (
            not isinstance(crop, list)
            or len(crop) != 2
            or any(type(value) is not int for value in crop)
            or not 0 <= crop[0] < crop[1]
        ):
            raise ValueError("Invalid frozen crop")
        size = crop[1] - crop[0]
        positions = case["positions"]
        if (
            not isinstance(positions, list)
            or any(type(value) is not int for value in positions)
            or positions != list(range(*crop))
        ):
            raise ValueError("Frozen positions must map every full-residue position")
        for key in ("eligible", "masked", "conditioning", "group_ids"):
            rows = case[key]
            if (
                not isinstance(rows, list)
                or len(rows) != size
                or any(not isinstance(row, list) or len(row) != 2 for row in rows)
            ):
                raise ValueError("Frozen controls must have shape [crop_length,2]")
            if any(
                type(value) is not (int if key == "group_ids" else bool)
                for row in rows
                for value in row
            ):
                raise ValueError(
                    "Frozen masks must be booleans; groups must be integers"
                )
        eligible, masked, condition = (
            torch.tensor(case[key], dtype=torch.bool)
            for key in ("eligible", "masked", "conditioning")
        )
        groups = torch.tensor(case["group_ids"], dtype=torch.long)
        if (
            (masked & ~eligible).any()
            or (condition & eligible).any()
            or not torch.equal(groups >= 0, eligible)
            or (groups < -1).any()
            or (groups >= 2 * size).any()
        ):
            raise ValueError("Frozen masks/groups exceed eligible universe")
        definition = case["definition"]
        tied = definition["regime"] == "joint_tied"
        for group in groups[eligible].unique():
            values = masked[groups == group]
            if (values != values[0]).any():
                raise ValueError("Frozen group members must share their draw")
        if tied:
            both = eligible.all(-1)
            if (groups[both, 0] != groups[both, 1]).any():
                raise ValueError("Tied controls must share groups")
        if definition["placement"] == "token" or definition["span_mean"] == 1:
            expected = torch.arange(size)[:, None].expand(-1, 2).clone()
            if not tied:
                expected[:, 1] += size
            if not torch.equal(groups, expected.masked_fill(~eligible, -1)):
                raise ValueError("Invalid frozen deterministic groups")
        else:
            # Validate possible Bernoulli partitions, never replay their random stream.
            offset_min, offset_max = 0, 0
            tracks = [groups.max(-1).values] if tied else list(groups.unbind(-1))
            for active_groups in tracks:
                positions = torch.nonzero(active_groups >= 0).flatten()
                active = active_groups[positions]
                if len(active):
                    delta = active[1:] - active[:-1]
                    if (delta < 0).any() or (delta > positions.diff()).any():
                        raise ValueError("Invalid frozen span group ordering/jump")
                    base_min = max(offset_min, int(active[0] - positions[0]))
                    base_max = min(offset_max, int(active[0]))
                    if base_min > base_max:
                        raise ValueError("Invalid frozen span track offset")
                    offset_min = int(active[-1]) + 1
                    offset_max = offset_min + size - int(positions[-1]) - 1
                else:
                    offset_min, offset_max = 1, size
        probability = definition.get("probability", 1.0)
        if (
            probability == 0
            and masked.any()
            or probability == 1
            and not torch.equal(masked, eligible)
        ):
            raise ValueError("Frozen draw contradicts endpoint mask probability")
        if case["kind"] == "generation" and not torch.equal(masked, eligible):
            raise ValueError("Generation must mask full target modalities")
        if case["kind"] == "denoising" and condition.any():
            raise ValueError("Denoising cases cannot add generation conditioning")
        if case["case_id"] != json_sha256(
            {
                key: value
                for key, value in case.items()
                if key not in {"case_id", "ordinal"}
            }
        ):
            raise ValueError("Frozen case ID mismatch")
    except (KeyError, TypeError, RuntimeError, OverflowError) as error:
        raise ValueError("Malformed frozen case") from error


def _validate_record_cases(record, cases, request):
    for case in cases:
        key, kind = case["family_key"], case["kind"]
        length = len(record["sequence"])
        size = min(length, request["crop_residues"])
        start = (length - size) // 2
        if (
            case["canonical_id"] not in request["members"]
            or case["content_sha256"] != record["content_sha256"]
            or case["residue_map_sha256"] != record["residue_map_sha256"]
            or case["crop"] != [start, start + size]
            or key not in request[kind]
            or case["definition"] != request[kind][key]
            or case["replicate"] >= request["replicates"]
            or case["seed"]
            != stable_seed(
                [
                    "c1-case-v1",
                    request["seed"],
                    record["canonical_id"],
                    key,
                    case["replicate"],
                ]
            )
        ):
            raise ValueError(
                "Case content/map/crop/family/seed disagrees with canonical request"
            )
        original, eligible = _eligibility(
            record, case["crop"], case["definition"]["regime"]
        )
        condition = (
            original & ~eligible if kind == "generation" else torch.zeros_like(eligible)
        )
        if (
            case["eligible"] != eligible.tolist()
            or case["conditioning"] != condition.tolist()
        ):
            raise ValueError(
                "Frozen eligibility/conditioning disagrees with original observations"
            )


def read_evaluation_cases(directory: str | Path) -> dict[str, Any]:
    """Verify references and stored draws without regenerating random controls."""
    directory = Path(directory)
    summary = json.loads((directory / "manifest.json").read_text())
    if (
        not isinstance(summary, dict)
        or type(summary.get("schema_version")) is not int
        or summary["schema_version"] != 1
        or summary.get("status") != "complete"
    ):
        raise ValueError("Invalid frozen case completion manifest")
    request = _request(summary.get("request"))
    if file_sha256(directory / "cases.jsonl") != summary.get("cases_sha256"):
        raise ValueError("Corrupt frozen case file")
    cases, ids, ordinals, tuples, by_member = (
        list(_read_jsonl(directory / "cases.jsonl")),
        set(),
        set(),
        set(),
        {},
    )
    for case in cases:
        _validate_case(case)
        assignment = (case["canonical_id"], case["family_key"], case["replicate"])
        if (
            case["case_id"] in ids
            or case["ordinal"] in ordinals
            or assignment in tuples
        ):
            raise ValueError("Duplicate case ID/ordinal/member-family-replicate")
        ids.add(case["case_id"])
        ordinals.add(case["ordinal"])
        tuples.add(assignment)
        by_member.setdefault(case["canonical_id"], []).append(case)
    cases.sort(key=lambda case: case["ordinal"])
    if (
        ordinals != set(range(len(cases)))
        or json_sha256(cases) != summary.get("shared_cases_sha256")
        or any(
            type(summary.get(key)) is not int or summary[key] != count
            for key, count in _counts(cases).items()
        )
    ):
        raise ValueError("Frozen case order/digest/counts disagree")
    expected = [
        (member, key, replicate)
        for member in request["members"]
        for kind in ("denoising", "generation")
        for key in request[kind]
        for replicate in range(request["replicates"])
    ]
    if [
        (c["canonical_id"], c["family_key"], c["replicate"]) for c in cases
    ] != expected:
        raise ValueError("Frozen cases do not exactly expand the request")
    declared = summary.get("canonical_inventories")
    if (
        not isinstance(declared, list)
        or not declared
        or any(
            not isinstance(ref, dict)
            or set(ref) != {"directory", "manifest_sha256", "population_sha256"}
            or not _name(ref["directory"])
            or not _digest(ref["manifest_sha256"])
            or not _digest(ref["population_sha256"])
            for ref in declared
        )
    ):
        raise ValueError("Invalid canonical inventory references")
    references = []

    def records():
        for record in _inventory_records(
            [ref["directory"] for ref in declared], references
        ):
            _validate_record_cases(
                record, by_member.get(record["canonical_id"], []), request
            )
            yield record

    if not _name(summary.get("split_manifest")):
        raise ValueError("Missing split manifest reference")
    audit = audit_canonical_splits(records(), summary["split_manifest"])
    if references != declared or any(
        audit[key] != summary.get(key)
        for key in ("population_sha256", "split_sha256", "split_manifest_sha256")
    ):
        raise ValueError("Frozen inventory/split integrity reference mismatch")
    if any(
        key not in audit["assignments"]
        or audit["assignments"][key]["split"] != "validation"
        for key in by_member
    ):
        raise ValueError("Cases require known validation-only membership")
    return {**summary, "cases": cases}


def project_case_controls(
    cases: Sequence[Mapping[str, Any]], batch: MDLMBatch
) -> dict[str, Any]:
    """Pair by row, project at BOS offset one, and intersect with arm-valid targets."""
    sequence, structure = (
        batch["sequence_tokens"].clone(),
        batch["structure_tokens"].clone(),
    )
    if (
        sequence.ndim != 2
        or sequence.shape != structure.shape
        or not len(cases)
        or len(cases) != sequence.shape[0]
        or len(batch["sample_keys"]) != len(cases)
    ):
        raise ValueError("Projection requires one ordered case per aligned batch row")
    mask_id, codebook_size = batch["sequence_mask_id"], batch["codebook_size"]
    if (
        type(mask_id) is not int
        or mask_id < 0
        or type(codebook_size) is not int
        or codebook_size < 1
    ):
        raise ValueError("Invalid batch mask/codebook metadata")
    for key in ("residue_mask", "sequence_valid", "structure_valid"):
        if batch[key].shape != sequence.shape or batch[key].dtype != torch.bool:
            raise ValueError("Projection requires aligned boolean batch masks")
    if (
        batch["crop_offsets"].shape != (len(cases),)
        or batch["crop_offsets"].dtype != torch.long
    ):
        raise ValueError("Projection requires integer crop offsets")
    requested = torch.zeros(
        (*sequence.shape, 2), device=sequence.device, dtype=torch.bool
    )
    masked, conditioning = torch.zeros_like(requested), torch.zeros_like(requested)
    groups = torch.full_like(requested, -1, dtype=torch.long)
    ids, probabilities, regimes = [], [], []
    for i, case in enumerate(cases):
        _validate_case(case)
        start, stop = case["crop"]
        size = stop - start
        expected_slots = torch.zeros(
            sequence.shape[1], dtype=torch.bool, device=sequence.device
        )
        if size + 2 > sequence.shape[1]:
            raise ValueError("Unsupported crop exceeds arm capacity")
        expected_slots[1 : size + 1] = True
        if (
            batch["sample_keys"][i] != case["canonical_id"]
            or int(batch["crop_offsets"][i]) != start
            or not torch.equal(expected_slots, batch["residue_mask"][i])
        ):
            raise ValueError("Canonical row/crop/alignment does not match ordered case")
        for target, key in (
            (requested, "eligible"),
            (masked, "masked"),
            (conditioning, "conditioning"),
            (groups, "group_ids"),
        ):
            target[i, 1 : size + 1] = torch.tensor(
                case[key], dtype=target.dtype, device=sequence.device
            )
        ids.append(case["case_id"])
        probabilities.append(case["definition"].get("probability", 1.0))
        regimes.append(case["definition"]["regime"])
    assignments = {
        (case["canonical_id"], case["family_key"], case["replicate"]) for case in cases
    }
    if (
        len(set(ids)) != len(ids)
        or len(assignments) != len(cases)
        or len({case["ordinal"] for case in cases}) != len(cases)
    ):
        raise ValueError("Duplicate projected case ID/ordinal/member-family-replicate")
    available = (
        torch.stack((batch["sequence_valid"], batch["structure_valid"]), -1)
        & batch["residue_mask"][..., None]
    )
    eligible, effective_masked = requested & available, masked & available
    unavailable_conditioning = conditioning & ~available
    sequence.masked_fill_(effective_masked[..., 0], mask_id)
    structure.masked_fill_(effective_masked[..., 1], codebook_size + 1)
    return {
        "sequence_tokens": sequence,
        "structure_tokens": structure,
        "eligible": eligible,
        "masked": effective_masked,
        "group_ids": groups,
        "mask_probability": torch.tensor(
            probabilities, dtype=torch.float32, device=sequence.device
        ),
        "weight": torch.ones(len(cases), dtype=torch.float32, device=sequence.device),
        "regimes": regimes,
        "case_ids": ids,
        "requested_eligible": requested,
        "requested_masked": masked,
        "requested_conditioning": conditioning,
        "unavailable_conditioning": unavailable_conditioning,
        "case_available": ~unavailable_conditioning.any(dim=(1, 2)),
    }


# Only operational fields are omitted, at their owning settings level. Scientific
# sampler/evaluator/decoder objects are hashed intact, including their step counts.
_OPERATIONAL = {
    "steps",
    "output_dir",
    "output_path",
    "run_name",
    "display_name",
    "cohort",
    "generation_cohort",
    "case_manifest",
    "max_samples",
}


def evaluation_protocol(
    settings: Mapping[str, Any],
    *,
    shared_cases: Mapping[str, Any],
    representation: Mapping[str, Any],
    decoder: Mapping[str, Any] | None,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        any(
            not isinstance(value, Mapping)
            for value in (settings, shared_cases, representation, environment)
        )
        or decoder is not None
        and not isinstance(decoder, Mapping)
    ):
        raise ValueError("Protocol inputs must be mappings")
    digest = shared_cases.get("shared_cases_sha256")
    if not _digest(digest):
        raise ValueError("Protocol requires a shared case digest")
    selected = shared_cases.get("cases", [])
    if not isinstance(selected, list) or any(
        not isinstance(case, Mapping) for case in selected
    ):
        raise ValueError("Protocol selected cases must be a list of frozen objects")
    for case in selected:
        _validate_case(case)
    selected = sorted(selected, key=lambda case: case["ordinal"])
    science = {key: value for key, value in settings.items() if key not in _OPERATIONAL}
    for kind in ("denoising", "generation"):
        if kind in science:
            if not isinstance(science[kind], Mapping):
                raise ValueError("Protocol evaluator settings must be mappings")
            science[kind] = {
                key: value
                for key, value in science[kind].items()
                if key not in _OPERATIONAL
            }
    result = {
        "schema_version": 1,
        "shared_cases_sha256": digest,
        "selected_case_ids": [case["case_id"] for case in selected],
        "settings": science,
        "representation": dict(representation),
        "decoder": dict(decoder) if decoder is not None else None,
        "environment": dict(environment),
    }
    result["protocol_sha256"] = json_sha256(result)
    return result


def evaluation_measurement(
    protocol: Mapping[str, Any],
    *,
    model_identity: Mapping[str, Any],
    coverage: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if any(
        not isinstance(value, Mapping)
        for value in (protocol, model_identity, coverage, metrics)
    ):
        raise ValueError("Measurement inputs must be mappings")
    if protocol.get("protocol_sha256") != json_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    ):
        raise ValueError("Measurement requires a valid frozen protocol")
    if not (
        set(model_identity) == {"checkpoint_sha256"}
        and _digest(model_identity["checkpoint_sha256"])
        or set(model_identity) == {"training_signature", "global_step"}
        and _digest(model_identity["training_signature"])
        and type(model_identity["global_step"]) is int
        and model_identity["global_step"] >= 0
    ):
        raise ValueError(
            "Model identity requires checkpoint digest or training signature/global_step"
        )
    result = {
        "schema_version": 1,
        "shared_cases_sha256": protocol["shared_cases_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "model_identity": dict(model_identity),
        "coverage": dict(coverage),
        "metrics": dict(metrics),
    }
    result["measurement_sha256"] = json_sha256(result)
    return result
