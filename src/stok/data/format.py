"""Plain Parquet datasets: shard schema, tokenizer metadata and footer summaries."""

from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..utils.pretrained import ARTIFACTS, canonical_preset, file_sha256, state_sha256

TOKENIZER_KEY = b"stok.tokenizer"
# Fields that change token values; stok_version is informational only.
TOKENIZER_IDENTITY = ("preset", "checkpoint_sha256", "codebook_sha256", "codebook_size")


def tokenizer_metadata(tokenizer, *, preset: str, checkpoint: Path | None) -> dict:
    codebook = tokenizer.quantizer.codebook
    return {
        "preset": canonical_preset(preset),
        "checkpoint_sha256": file_sha256(checkpoint)
        if checkpoint is not None
        else ARTIFACTS[canonical_preset(preset)]["sha256"],
        "codebook_sha256": state_sha256({"codebook": codebook}),
        "codebook_size": codebook.size(0),
        "stok_version": version("stok"),
    }


def shard_schema(
    *,
    structure: bool,
    coordinates: bool,
    extra: pa.Schema | None,
    tokenizer: dict | None,
) -> pa.Schema:
    fields = [
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
    ]
    if structure:
        fields.append(pa.field("structure_tokens", pa.list_(pa.int16())))
    if coordinates:
        fields.append(
            pa.field("coordinates", pa.list_(pa.list_(pa.list_(pa.float32()))))
        )
    fields.extend(extra or [])
    metadata = (
        {TOKENIZER_KEY: json.dumps(tokenizer, sort_keys=True).encode()}
        if tokenizer is not None
        else None
    )
    return pa.schema(fields, metadata=metadata)


def parquet_shards(directory: Path) -> list[Path]:
    """Top-level shards only; subdirectories such as rejected/ are ignored."""
    return sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix == ".parquet"
    )


def read_source(directory: str | Path) -> dict[str, Any]:
    """Summarize a dataset directory from its shard footers, without reading rows."""
    directory = Path(directory)
    shards, tokenizer, columns = [], None, None
    for path in parquet_shards(directory):
        footer = pq.ParquetFile(path)
        schema = footer.schema_arrow
        context = f"{directory}: {path.name}"
        if {"sequence_id", "sequence"} - set(schema.names):
            raise ValueError(f"{context} lacks sequence_id/sequence columns")
        if columns is None:
            columns = schema.names
        elif set(columns) != set(schema.names):
            raise ValueError(f"{context} columns disagree with other shards")
        if "structure_tokens" in schema.names:
            raw = (schema.metadata or {}).get(TOKENIZER_KEY)
            if raw is None:
                raise ValueError(
                    f"{context} has structure_tokens but no stok.tokenizer"
                )
            current = json.loads(raw)
            if tokenizer is not None:
                for field in TOKENIZER_IDENTITY:
                    if current.get(field) != tokenizer.get(field):
                        raise ValueError(f"{context} disagrees on tokenizer {field}")
            tokenizer = current
        shards.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "rows": footer.metadata.num_rows,
            }
        )
    if not shards or columns is None:
        raise ValueError(f"{directory}: no Parquet shards")
    return {
        "shards": shards,
        "rows": sum(shard["rows"] for shard in shards),
        "tokenizer": tokenizer,
        "structure": "structure_tokens" in columns,
        "coordinates": "coordinates" in columns,
        "columns": columns,
    }
