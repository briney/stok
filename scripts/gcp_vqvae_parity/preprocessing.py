from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import InputFile


@dataclass(frozen=True)
class ReferenceSample:
    pid: str
    sequence: str
    length: int


@dataclass(frozen=True)
class PreprocessingRecord:
    source_path: str
    source_name: str
    source_sha256: str
    reference_samples: tuple[ReferenceSample, ...]
    stok_sequence: str | None
    stok_length: int | None
    stok_chain_id: str | None
    stok_error_type: str | None
    stok_error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_preprocessing_audit(
    inputs: Sequence[InputFile],
    reference_samples: Sequence[Mapping[str, Any]],
    *,
    stok_parser: Callable[[Path], Any],
) -> list[PreprocessingRecord]:
    grouped: dict[str, list[ReferenceSample]] = {}
    for sample in reference_samples:
        source = str(Path(sample["source_path"]).resolve())
        grouped.setdefault(source, []).append(
            ReferenceSample(
                pid=str(sample["pid"]),
                sequence=str(sample["seq"]),
                length=len(sample["seq"]),
            )
        )
    records: list[PreprocessingRecord] = []
    for input_file in inputs:
        path = Path(input_file.path)
        try:
            parsed = stok_parser(path)
            sequence = str(parsed.protein_sequence)
            record = PreprocessingRecord(
                source_path=input_file.path,
                source_name=input_file.name,
                source_sha256=input_file.sha256,
                reference_samples=tuple(grouped.get(input_file.path, [])),
                stok_sequence=sequence,
                stok_length=len(sequence),
                stok_chain_id=None if parsed.chain_id is None else str(parsed.chain_id),
                stok_error_type=None,
                stok_error_message=None,
            )
        except Exception as exc:
            record = PreprocessingRecord(
                source_path=input_file.path,
                source_name=input_file.name,
                source_sha256=input_file.sha256,
                reference_samples=tuple(grouped.get(input_file.path, [])),
                stok_sequence=None,
                stok_length=None,
                stok_chain_id=None,
                stok_error_type=type(exc).__name__,
                stok_error_message=str(exc),
            )
        records.append(record)
    return records
