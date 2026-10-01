from __future__ import annotations

from pathlib import Path
import json
from collections.abc import Sequence, Sized
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from ..utils.pretrained import json_sha256

PARQUET_EXTENSIONS = {".parquet", ".parq", ".pq"}


def _structure_provenance(path: Path, schema: pa.Schema):
    """Legacy files have no provenance; generated files must have consistent digests."""
    metadata = schema.metadata or {}
    keys = (b"stok.provenance", b"stok.tokenizer_sha256", b"stok.policy_sha256")
    if not any(key in metadata for key in keys):
        return None
    try:
        provenance = json.loads(metadata[keys[0]])
        if (
            provenance["schema_version"] != 1
            or json_sha256(provenance["tokenizer"]) != metadata[keys[1]].decode()
            or json_sha256(provenance["policy"]) != metadata[keys[2]].decode()
        ):
            raise ValueError("digest mismatch")
        return (
            metadata[keys[1]],
            metadata[keys[2]],
            json_sha256(provenance["execution"]),
        )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"{path}: invalid structure-token provenance") from error


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
    dataset._epoch += 1
    worker = get_worker_info()
    worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
    usable = _usable_samples(dataset)
    # ponytail: each rank scans the deterministic stream; index shards if parsing becomes a bottleneck.
    for position, item in enumerate(dataset._iter_unsharded(dataset._epoch)):
        if position >= usable:
            break
        if (
            position % dataset.world_size == dataset.rank
            and (position // dataset.world_size) % workers == worker_id
        ):
            yield item


def _parquet_columns(
    path: Path,
    schema: pa.Schema,
    *,
    require_structure_tokens: bool,
    load_coords: bool,
) -> list[str]:
    """Validate a file's schema and select the columns consumed by the dataset."""
    _structure_provenance(path, schema)
    required = {"sequence_id", "sequence"}
    if require_structure_tokens:
        required.add("structure_tokens")
    missing = required - set(schema.names)
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
    columns.extend(name for name in ("source", "residue_map") if name in schema.names)
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
    for name in ("source", "residue_map"):
        if name in row:
            out[name] = row[name]
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


class TokenizedDataset(Dataset):
    """Map-style dataset for a single typed Parquet file.

    Required columns: sequence_id (string), sequence (string), and, for
    codebook training, structure_tokens (list of integers). Null token
    elements mark unlabeled residues. Optional coordinates are [L, 3, 3].
    max_length=None keeps full coordinates for downstream aligned cropping.
    dataset_name optionally attaches a stable sample namespace.
    """

    def __init__(
        self,
        dataset_path: str,
        max_length: int | None,
        *,
        load_coords: bool = True,
        require_structure_tokens: bool = True,
        dataset_name: str | None = None,
    ):
        path = Path(dataset_path)
        if path.suffix.lower() not in PARQUET_EXTENSIONS:
            raise ValueError("Provide a Parquet file (.parquet, .parq, or .pq)")
        parquet = pq.ParquetFile(path)
        columns = _parquet_columns(
            path,
            parquet.schema_arrow,
            require_structure_tokens=require_structure_tokens,
            load_coords=load_coords,
        )
        self.data = parquet.read(columns=columns)
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.has_coords = "coordinates" in columns
        self.has_labels = "structure_tokens" in columns

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.slice(index, 1).to_pylist()[0]
        out = _build_output_from_row(
            row, max_length=self.max_length, has_coords=self.has_coords
        )
        if self.dataset_name is not None:
            out["dataset"] = self.dataset_name
        return out


class DummySequenceDataset(Dataset):
    """Placeholder dataset producing random token/label pairs for smoke tests."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        num_classes: int,
        pad_id: int = 0,
    ):
        """Initialize dummy dataset.

        Args:
            num_samples: Number of samples in dataset.
            seq_len: Sequence length for each sample.
            vocab_size: Vocabulary size for token generation.
            num_classes: Number of classes for label generation.
            pad_id: Padding token ID.
        """
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.has_labels = True
        self.has_coords = False
        self.pad_id = pad_id

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
            Number of samples in dataset.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            index: Sample index.

        Returns:
            Tuple of (tokens, labels) with shapes [seq_len] and [seq_len].
        """
        tokens = torch.randint(low=1, high=self.vocab_size, size=(self.seq_len,))
        labels = torch.randint(low=0, high=self.num_classes, size=(self.seq_len,))
        # randomly pad a couple at end
        tokens[-2:] = self.pad_id
        labels[-2:] = -100
        return tokens.long(), labels.long()


class DummyMLMDataset(Dataset):
    """Placeholder dataset producing random sequences for MLM smoke tests."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int = 32,
    ):
        """Initialize dummy MLM dataset.

        Args:
            num_samples: Number of samples in dataset.
            seq_len: Sequence length for each sample.
            vocab_size: Vocabulary size (default 32 for amino acids).
        """
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Amino acid characters (indices 4-23 in DEFAULT_VOCAB)
        self._aa_chars = "LAGVSERTIPDKQNFYMHWC"

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, str]:
        """Get a sample from the dataset.

        Args:
            index: Sample index.

        Returns:
            Dict with 'sequence_id' and 'sequence' keys.
        """
        # Generate random amino acid sequence
        seq = "".join(
            self._aa_chars[i] for i in torch.randint(0, 20, (self.seq_len,)).tolist()
        )
        return {"sequence_id": f"dummy_{index}", "sequence": seq}


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
        require_structure_tokens: Whether structure_tokens is required. Set to False for MLM.
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
        require_structure_tokens: bool = True,
        dataset_name: str | None = None,
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
        shard_paths = sorted(
            [
                p
                for p in self.dataset_path.iterdir()
                if p.is_file() and p.suffix.lower() in PARQUET_EXTENSIONS
            ]
        )
        if len(shard_paths) == 0:
            raise RuntimeError("No Parquet shards found in directory.")

        self._shards: list[Path] = []
        self._rows_per_shard: list[int] = []
        self._columns: list[list[str]] = []
        provenances = set()
        for sp in shard_paths:
            pf = pq.ParquetFile(sp)
            provenances.add(_structure_provenance(sp, pf.schema_arrow))
            self._columns.append(
                _parquet_columns(
                    sp,
                    pf.schema_arrow,
                    require_structure_tokens=require_structure_tokens,
                    load_coords=load_coords,
                )
            )
            self._shards.append(sp)
            self._rows_per_shard.append(int(pf.metadata.num_rows))
        if len(provenances) != 1:
            raise ValueError("Shards have incompatible structure-token provenance")
        self._total_rows = int(sum(self._rows_per_shard))
        self.num_samples = self._total_rows
        self.has_labels = any("structure_tokens" in cols for cols in self._columns)
        self.has_coords = any("coordinates" in cols for cols in self._columns)

    def __len__(self) -> int:
        return _partition_length(self)

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
            rows = list(range(table.num_rows))
            if self.shuffle_rows:
                np.random.RandomState((seed_base + 1009 + s_idx) & 0xFFFFFFFF).shuffle(
                    rows
                )
            for i in rows:
                try:
                    out = _build_output_from_row(
                        table.slice(i, 1).to_pylist()[0],
                        max_length=self.max_length,
                        has_coords=self.has_coords,
                    )
                    if self.dataset_name is not None:
                        out["dataset"] = self.dataset_name
                    yield out
                except ValueError as exc:
                    raise ValueError(f"Shard {path}: {exc}") from exc


class MapAsIterableDataset(IterableDataset):
    """
    Wrap a map-style Dataset to behave like an IterableDataset.

    This is useful when mixing map-style and iterable datasets, because PyTorch
    DataLoader does not allow mixing samplers/shuffle with IterableDatasets.

    Behavior:
      - Each epoch yields `num_samples` samples (default: len(dataset))
      - Sampling is with replacement, uniformly over indices [0, len(dataset))
      - Dataloader worker IDs stripe the sample positions to avoid multiplying
        total emitted samples by `num_workers`.
    """

    training_batch_size: int
    num_workers: int

    def __init__(
        self, dataset: Dataset, *, num_samples: int | None = None, seed: int = 0
    ):
        super().__init__()
        self.dataset = dataset
        self.has_labels = getattr(dataset, "has_labels", True)
        self.has_coords = getattr(dataset, "has_coords", False)
        self.num_samples = (
            int(num_samples) if num_samples is not None else len(cast(Sized, dataset))
        )
        self.seed = int(seed)
        self._epoch = -1
        self.rank, self.world_size = distributed_rank()

    def __len__(self) -> int:
        return _partition_length(self)

    def __iter__(self):
        return _partition_stream(self)

    def _iter_unsharded(self, epoch: int):
        rng = np.random.RandomState(
            ((0x9E3779B97F4A7C15 ^ self.seed) + epoch * 0x1000003) & 0xFFFFFFFF
        )
        if len(cast(Sized, self.dataset)) == 0:
            return
        for _ in range(self.num_samples):
            yield self.dataset[int(rng.randint(0, len(cast(Sized, self.dataset))))]


class InterleavedIterableDataset(IterableDataset):
    """
    Interleave samples from multiple IterableDatasets according to fractions.

    Each epoch yields `num_samples` items (default: sum(len(ds)) when available,
    otherwise falls back to 0 which effectively yields nothing).

    When a sub-dataset iterator is exhausted, it is re-initialized so mixing
    continues without prematurely stopping.

    Worker IDs stripe emitted positions so that total yielded items across all
    workers is approximately `num_samples` (not multiplied by num_workers).
    """

    training_batch_size: int
    num_workers: int

    def __init__(
        self,
        datasets: Sequence[
            IterableTokenizedDataset | MapAsIterableDataset | InterleavedIterableDataset
        ],
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

        if num_samples is None:
            # best-effort: sum dataset lengths if available
            total = 0
            for ds in datasets:
                try:
                    total += ds.num_samples
                except Exception:
                    total = 0
                    break
            num_samples = total
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return _partition_length(self)

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
