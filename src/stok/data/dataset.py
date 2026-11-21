from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .format import parquet_shards


def distributed_rank() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _usable_samples(dataset) -> int:
    total = dataset.num_samples
    batch_size = getattr(dataset, "training_batch_size", None)
    if batch_size is not None:
        quantum = (
            dataset.world_size * batch_size * max(1, getattr(dataset, "num_workers", 0))
        )
        total = total // quantum * quantum
    return total


def _partition_length(dataset) -> int:
    return len(range(dataset.rank, _usable_samples(dataset), dataset.world_size))


def _partition_stream(dataset):
    """Assign positions once, outside any nested mixture streams."""
    if not getattr(dataset, "_explicit_epoch", False):
        dataset._epoch += 1
    worker = get_worker_info()
    worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
    usable = _usable_samples(dataset)
    # Every owner walks the same cheap reference stream; only owned rows are built.
    for position, ref in enumerate(dataset._iter_unsharded(dataset._epoch)):
        if position >= usable:
            break
        if (
            position % dataset.world_size == dataset.rank
            and (position // dataset.world_size) % workers == worker_id
        ):
            yield _materialize(ref)


def _materialize(ref) -> dict[str, Any]:
    """Build one row from a (dataset, shard path, table, index) reference."""
    dataset, path, table, index = ref
    try:
        out = _build_output_from_row(
            table.slice(index, 1).to_pylist()[0],
            max_length=dataset.max_length,
            has_coords=dataset.has_coords,
        )
    except ValueError as exc:
        raise ValueError(f"Shard {path}: {exc}") from exc
    if dataset.dataset_name is not None:
        out["dataset"] = dataset.dataset_name
    return out


def set_dataset_epoch(dataset, epoch: int):
    """Opt in to explicit epochs recursively, preserving standalone implicit iteration."""
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(epoch)
    for child in getattr(dataset, "datasets", ()):
        set_dataset_epoch(child, epoch)


def _parquet_columns(
    path: Path,
    schema: pa.Schema,
    *,
    load_coords: bool,
) -> list[str]:
    """Validate a file's schema and select the columns consumed by the dataset."""
    missing = {"sequence_id", "sequence"} - set(schema.names)
    if missing:
        raise ValueError(f"{path}: Missing required columns: {sorted(missing)}")

    columns = ["sequence_id", "sequence"]
    for name in columns:
        dtype = schema.field(name).type
        if not (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)):
            raise ValueError(f"{path}: {name} must be a string, got {dtype}")
    if "structure_tokens" in schema.names:
        dtype = schema.field("structure_tokens").type
        if not (
            (pa.types.is_list(dtype) or pa.types.is_large_list(dtype))
            and pa.types.is_integer(dtype.value_type)
        ):
            raise ValueError(
                f"{path}: structure_tokens must be a list of integers, got {dtype}"
            )
        columns.append("structure_tokens")
    if load_coords and "coordinates" in schema.names:
        dtype = schema.field("coordinates").type
        element_type, depth = dtype, 0
        while (
            pa.types.is_list(element_type)
            or pa.types.is_large_list(element_type)
            or pa.types.is_fixed_size_list(element_type)
        ):
            element_type = element_type.value_type
            depth += 1
        if depth != 3 or not (
            pa.types.is_integer(element_type) or pa.types.is_floating(element_type)
        ):
            raise ValueError(
                f"{path}: coordinates must be three nested lists of integers or floats, got {dtype}"
            )
        columns.append("coordinates")
    return columns


def _build_output_from_row(
    row: dict[str, Any],
    *,
    max_length: int | None,
    has_coords: bool,
) -> dict[str, Any]:
    """Decode a typed Parquet row; null tokens retain their residue positions."""
    sequence_id, sequence = row["sequence_id"], row["sequence"]
    if sequence_id is None or sequence is None:
        raise ValueError("sequence_id and sequence must not be null")
    out: dict[str, Any] = {
        "sequence_id": sequence_id,
        "sequence": sequence,
    }
    if "structure_tokens" in row:
        tokens = row["structure_tokens"]
        if tokens is None:
            raise ValueError(
                f"{sequence_id}: structure_tokens must not be a null list; use null elements"
            )
        if len(tokens) != len(sequence):
            raise ValueError(
                f"{sequence_id}: structure_tokens length must match sequence length"
            )
        if any(t is not None and t < 0 for t in tokens):
            raise ValueError(
                f"{sequence_id}: structure_tokens must not contain negative values; use null"
            )
        if any(t is not None and t > torch.iinfo(torch.long).max for t in tokens):
            raise ValueError(f"{sequence_id}: structure_tokens exceed the int64 range")
        out["structure_tokens"] = torch.tensor(
            [-1 if t is None else t for t in tokens],
            dtype=torch.long,
        )
    else:
        # Sequence-only sources (e.g. UniRef): every structure position is unavailable.
        out["structure_tokens"] = torch.full((len(sequence),), -1, dtype=torch.long)

    if has_coords:
        coords = row.get("coordinates")
        coordinate_length = len(sequence) if max_length is None else max_length
        padded = np.full((coordinate_length, 3, 3), np.nan, dtype=np.float32)
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            if coords.shape != (len(sequence), 3, 3):
                raise ValueError(
                    f"{sequence_id}: coordinates must have shape [sequence_length, 3, 3]"
                )
            copy_len = min(len(sequence), coordinate_length)
            padded[:copy_len] = coords[:copy_len]
        out["coords"] = torch.from_numpy(padded)
    return out


class IterableTokenizedDataset(IterableDataset):
    """Shard-wise iterable dataset over a directory of Parquet files.

    Loads a single Parquet shard at a time to bound memory use, applies
    deterministic per-epoch shuffling of shards and rows, and partitions
    samples across distributed ranks and dataloader workers.

    Args:
        dataset_path: Path to a directory containing Parquet shard files.
        max_length: Coordinate padding/truncation length; None retains all residues.
        shuffle_shards: Whether to shuffle shard order per epoch.
        shuffle_rows: Whether to shuffle selected row indices per shard per epoch.
        seed: Optional base seed for deterministic epoch shuffles.
        load_coords: Whether to load 3D coordinates.
        min_plddt: Keep only rows whose mean_plddt column is at least this value.
    """

    training_batch_size: int
    num_workers: int

    def __init__(
        self,
        dataset_path: str,
        max_length: int | None,
        *,
        shuffle_shards: bool = True,
        shuffle_rows: bool = True,
        seed: int = 0,
        load_coords: bool = True,
        dataset_name: str | None = None,
        min_plddt: float | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_dir():
            raise RuntimeError(
                "IterableTokenizedDataset expects a directory of Parquet files."
            )
        self.max_length = None if max_length is None else int(max_length)
        self.dataset_name = dataset_name
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_rows = bool(shuffle_rows)
        self.seed = int(seed)
        self._epoch = -1
        self.rank, self.world_size = distributed_rank()

        # enumerate shard files and stats
        shard_paths = parquet_shards(self.dataset_path)
        if len(shard_paths) == 0:
            raise RuntimeError("No Parquet shards found in directory.")

        self._shards: list[Path] = []
        self._rows_per_shard: list[int] = []
        self._columns: list[list[str]] = []
        for sp in shard_paths:
            pf = pq.ParquetFile(sp)
            self._columns.append(
                _parquet_columns(
                    sp,
                    pf.schema_arrow,
                    load_coords=load_coords,
                )
            )
            if ("structure_tokens" in self._columns[0]) != (
                "structure_tokens" in self._columns[-1]
            ):
                raise ValueError(
                    f"{sp}: structure_tokens presence disagrees with other shards"
                )
            self._shards.append(sp)
            self._rows_per_shard.append(int(pf.metadata.num_rows))
        # Row indices kept per shard by min_plddt; None streams every row.
        self._kept: list[np.ndarray] | None = None
        if min_plddt is not None:
            self._kept = []
            for sp in self._shards:
                if "mean_plddt" not in pq.read_schema(sp).names:
                    raise ValueError(f"{sp}: min_plddt needs a mean_plddt column")
                scores = pq.read_table(sp, columns=["mean_plddt"]).column(0)
                try:
                    scores = scores.cast(pa.float64())
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
                    raise ValueError(f"{sp}: mean_plddt must be numeric") from error
                keep = np.asarray(scores.fill_null(-np.inf), dtype=np.float64)
                self._kept.append(np.flatnonzero(keep >= min_plddt))
            if not sum(len(kept) for kept in self._kept):
                raise ValueError(f"{self.dataset_path}: min_plddt keeps no rows")
        self._total_rows = int(
            sum(self._rows_per_shard)
            if self._kept is None
            else sum(len(kept) for kept in self._kept)
        )
        self.num_samples = self._total_rows
        self.has_labels = any("structure_tokens" in cols for cols in self._columns)
        self.has_coords = any("coordinates" in cols for cols in self._columns)

    def __len__(self) -> int:
        return _partition_length(self)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._explicit_epoch = True

    def __iter__(self):
        return _partition_stream(self)

    def _iter_unsharded(self, epoch: int):
        seed_base = (0x9E3779B97F4A7C15 ^ self.seed) + epoch * 0x1000003
        shard_indices = list(range(len(self._shards)))
        if self.shuffle_shards:
            np.random.RandomState(seed_base & 0xFFFFFFFF).shuffle(shard_indices)
        for s_idx in shard_indices:
            path = self._shards[s_idx]
            table = pq.ParquetFile(path).read(columns=self._columns[s_idx])
            rows = (
                list(range(table.num_rows))
                if self._kept is None
                else self._kept[s_idx].tolist()
            )
            if self.shuffle_rows:
                np.random.RandomState((seed_base + 1009 + s_idx) & 0xFFFFFFFF).shuffle(
                    rows
                )
            for i in rows:
                yield self, path, table, i


class InterleavedIterableDataset(IterableDataset):
    """
    Interleave samples from multiple IterableDatasets according to fractions.

    Each epoch yields `num_samples` items (default: the sources' total).

    When a sub-dataset iterator is exhausted, it is re-initialized so mixing
    continues without prematurely stopping.

    Worker IDs stripe emitted positions so that total yielded items across all
    workers is approximately `num_samples` (not multiplied by num_workers).
    """

    training_batch_size: int
    num_workers: int

    def __init__(
        self,
        datasets: Sequence[IterableTokenizedDataset | InterleavedIterableDataset],
        fractions: list[float],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ):
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("InterleavedIterableDataset requires at least 1 dataset")
        if len(datasets) != len(fractions):
            raise ValueError("datasets and fractions must have the same length")

        fr = np.asarray([float(f) for f in fractions], dtype=np.float64)
        if np.any(fr < 0):
            raise ValueError("fractions must be non-negative")
        if float(fr.sum()) <= 0:
            raise ValueError("fractions must sum to a positive value")
        fr = fr / float(fr.sum())

        self.datasets = datasets
        self.has_labels = any(getattr(ds, "has_labels", True) for ds in datasets)
        self.has_coords = any(getattr(ds, "has_coords", False) for ds in datasets)
        self.fractions = fr.tolist()
        self.seed = int(seed)
        self._epoch = -1
        self.rank, self.world_size = distributed_rank()

        self.num_samples = (
            sum(ds.num_samples for ds in datasets)
            if num_samples is None
            else int(num_samples)
        )

    def __len__(self) -> int:
        return _partition_length(self)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._explicit_epoch = True

    def __iter__(self):
        return _partition_stream(self)

    def _iter_unsharded(self, epoch: int):
        seed_base = (0x9E3779B97F4A7C15 ^ self.seed) + epoch * 0x1000003
        rng = np.random.RandomState(seed_base & 0xFFFFFFFF)
        cycles = [epoch] * len(self.datasets)
        iters = [ds._iter_unsharded(epoch) for ds in self.datasets]
        for _ in range(self.num_samples):
            ds_idx = int(rng.choice(len(iters), p=self.fractions))
            try:
                yield next(iters[ds_idx])
            except StopIteration:
                cycles[ds_idx] += 1
                iters[ds_idx] = self.datasets[ds_idx]._iter_unsharded(cycles[ds_idx])
                try:
                    yield next(iters[ds_idx])
                except StopIteration as exc:
                    raise ValueError("Cannot sample an empty mixture source") from exc
