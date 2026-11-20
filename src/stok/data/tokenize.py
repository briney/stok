"""Tokenize structures into plain Parquet shards, one job's share at a time."""

from collections.abc import Iterator, Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .format import TOKENIZER_IDENTITY, parquet_shards, read_source, shard_schema
from .structure_encoding import StructureExclusion, tokenize_structures
from ..utils.structure_parser import (
    PolymerStructure,
    StructureMappingError,
    parse_polymer_structure,
)

STRUCTURE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}


def _structures_from_text(
    path: Path,
    *,
    text_column: str,
    id_column: str,
    keep_columns: Sequence[str],
    chain_id,
) -> Iterator[tuple[str, Any, dict]]:
    """Yield (sequence_id, parsed structure or error, kept values) per row."""
    table = pq.read_table(path, columns=[id_column, text_column, *keep_columns])
    with tempfile.TemporaryDirectory() as scratch:
        # ponytail: the parser reads files; a temp file per row keeps it unchanged.
        file = Path(scratch) / "structure.cif"
        for row in table.to_pylist():
            file.write_text(row[text_column])
            kept = {name: row[name] for name in keep_columns}
            yield str(row[id_column]), _parse(file, chain_id), kept


def _parse(path: Path, chain_id):
    try:
        return parse_polymer_structure(path, chain_id=chain_id)
    except StructureMappingError as error:
        return error


def _coordinates(structure: PolymerStructure) -> list:
    return [
        [xyz.tolist() if present else [None] * 3 for xyz, present in zip(residue, mask)]
        for residue, mask in zip(
            structure.coordinates[:, :3], structure.atom_mask[:, :3]
        )
    ]


@torch.inference_mode()
def tokenize_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    tokenizer,
    metadata: dict,
    text_column: str | None = None,
    id_column: str = "id",
    keep_columns: Sequence[str] = (),
    chain_id: str | None = None,
    rows_per_shard: int = 1000,
    include_coordinates: bool = False,
    num_jobs: int = 1,
    job_index: int = 0,
) -> dict[str, int]:
    """Tokenize this job's shards with the fixed training-native-reference policy.

    Shard k belongs to job k % num_jobs. Finished output shards are skipped, so a
    rerun resumes; each shard is written to a temp name and renamed when done.
    """
    if not 0 <= job_index < num_jobs or rows_per_shard < 1:
        raise ValueError("Invalid job split or rows_per_shard")
    if (
        tokenizer.training
        or tokenizer.encoder.max_length != 1280
        or any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in tokenizer.state_dict().values()
        )
    ):
        raise ValueError(
            "Tokenization requires an eval-mode FP32 tokenizer with max_length=1280"
        )
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if parquet_shards(output_dir):
        existing = read_source(output_dir)["tokenizer"] or {}
        for field in TOKENIZER_IDENTITY:
            if existing.get(field) != metadata[field]:
                raise ValueError(f"{output_dir}: existing shards differ on {field}")

    extra = []
    if text_column is not None:
        inputs = [[path] for path in parquet_shards(input_dir)]
        names = [paths[0].stem for paths in inputs]
        if inputs:
            schema = pq.read_schema(inputs[0][0])
            extra = [schema.field(name) for name in keep_columns]
    else:
        files = sorted(
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES
        )
        inputs = [
            files[start : start + rows_per_shard]
            for start in range(0, len(files), rows_per_shard)
        ]
        names = [f"part-{k:06d}" for k in range(len(inputs))]
    schema = shard_schema(
        structure=True,
        coordinates=include_coordinates,
        extra=pa.schema(extra) if extra else None,
        tokenizer=metadata,
    )

    summary = dict.fromkeys(
        (
            "shards_written",
            "shards_skipped",
            "rows",
            "rejected",
            "residues",
            "null_tokens",
        ),
        0,
    )
    for k, (name, source) in enumerate(zip(names, inputs)):
        if k % num_jobs != job_index:
            continue
        destination = output_dir / f"{name}.parquet"
        if destination.exists():
            summary["shards_skipped"] += 1
            continue
        if text_column is not None:
            structures = _structures_from_text(
                source[0],
                text_column=text_column,
                id_column=id_column,
                keep_columns=keep_columns,
                chain_id=chain_id,
            )
        else:
            structures = ((path.stem, _parse(path, chain_id), {}) for path in source)
        rows, rejected = [], []
        for sequence_id, structure, kept in structures:
            if isinstance(structure, StructureMappingError):
                rejected.append(
                    {"sequence_id": sequence_id, "reason": structure.reason}
                )
                continue
            try:
                (indices,) = tokenize_structures(
                    tokenizer,
                    [structure],
                    sequence_mode="native",
                    imputation="reference",
                    min_length=25,
                    max_length=1280,
                    max_missing_ratio=None,
                    max_missing_block=None,
                )
            except StructureExclusion as error:
                rejected.append({"sequence_id": sequence_id, "reason": error.reason})
                continue
            length = len(structure.sequence)
            if (
                indices.shape != (length,)
                or indices.dtype != torch.int64
                or (indices < -1).any()
                or (indices >= metadata["codebook_size"]).any()
                or not torch.equal(
                    indices >= 0,
                    torch.from_numpy(np.asarray(structure.atom_mask.all(-1))),
                )
            ):
                raise ValueError(
                    f"{sequence_id}: invalid model token IDs or availability"
                )
            tokens = [None if index == -1 else index for index in indices.tolist()]
            row = {
                "sequence_id": sequence_id,
                "sequence": structure.sequence,
                "structure_tokens": tokens,
                **kept,
            }
            if include_coordinates:
                row["coordinates"] = _coordinates(structure)
            rows.append(row)
            summary["residues"] += length
            summary["null_tokens"] += tokens.count(None)
        if rejected:
            (output_dir / "rejected").mkdir(exist_ok=True)
            (output_dir / "rejected" / f"{name}.jsonl").write_text(
                "".join(json.dumps(entry) + "\n" for entry in rejected)
            )
        temporary = output_dir / f".{name}.parquet.tmp"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
        )
        os.replace(temporary, destination)
        summary["shards_written"] += 1
        summary["rows"] += len(rows)
        summary["rejected"] += len(rejected)
    return summary
