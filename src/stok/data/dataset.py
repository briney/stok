from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

PARQUET_EXTENSIONS = {".parquet", ".parq", ".pq"}


def _parquet_columns(
    path: Path,
    schema: pa.Schema,
    *,
    require_structure_tokens: bool,
    load_coords: bool,
) -> list[str]:
    """Validate a file's schema and select the columns consumed by the dataset."""
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
        columns.append("coordinates")
    return columns


def _build_output_from_row(
    row: dict[str, Any],
    *,
    max_length: int,
    has_coords: bool,
) -> dict[str, torch.Tensor | str]:
    """Decode a typed Parquet row; null tokens retain their residue positions."""
    sequence_id, sequence = row["sequence_id"], row["sequence"]
    if sequence_id is None or sequence is None:
        raise ValueError("sequence_id and sequence must not be null")
    out: dict[str, torch.Tensor | str] = {
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

    if has_coords:
        coords = row.get("coordinates")
        padded = np.full((max_length, 3, 3), np.nan, dtype=np.float32)
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            if coords.shape != (len(sequence), 3, 3):
                raise ValueError(
                    f"{sequence_id}: coordinates must have shape [sequence_length, 3, 3]"
                )
            copy_len = min(len(sequence), max_length)
            padded[:copy_len] = coords[:copy_len]
        out["coords"] = torch.from_numpy(padded)
    return out


class TokenizedDataset(Dataset):
    """Map-style dataset for a single typed Parquet file.

    Required columns: sequence_id (string), sequence (string), and, for
    codebook training, structure_tokens (list of integers). Null token
    elements mark unlabeled residues. Optional coordinates are [L, 3, 3].
    """

    def __init__(
        self,
        dataset_path: str,
        max_length: int,
        *,
        load_coords: bool = True,
        require_structure_tokens: bool = True,
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
        self.has_coords = "coordinates" in columns

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.data.slice(idx, 1).to_pylist()[0]
        return _build_output_from_row(
            row,
            max_length=self.max_length,
            has_coords=self.has_coords,
        )


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
        self.pad_id = pad_id

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
            Number of samples in dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

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

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dict with 'sequence_id' and 'sequence' keys.
        """
        # Generate random amino acid sequence
        seq = "".join(
            self._aa_chars[i] for i in torch.randint(0, 20, (self.seq_len,)).tolist()
        )
        return {"sequence_id": f"dummy_{idx}", "sequence": seq}


class IterableTokenizedDataset(IterableDataset):
    """Shard-wise iterable dataset over a directory of Parquet files.

    Loads a single Parquet shard at a time to bound memory use, applies
    deterministic per-epoch shuffling of shards and rows, and partitions
    samples across distributed ranks and dataloader workers.

    Args:
        dataset_path: Path to a directory containing Parquet shard files.
        max_length: Maximum coordinate length for padding/truncation.
        shuffle_shards: Whether to shuffle shard order per epoch.
        shuffle_rows: Whether to shuffle selected row indices per shard per epoch.
        seed: Optional base seed for deterministic epoch shuffles.
        load_coords: Whether to load 3D coordinates.
        require_structure_tokens: Whether structure_tokens is required. Set to False for MLM.
    """

    def __init__(
        self,
        dataset_path: str,
        max_length: int,
        *,
        shuffle_shards: bool = True,
        shuffle_rows: bool = True,
        seed: int = 0,
        load_coords: bool = True,
        require_structure_tokens: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_dir():
            raise RuntimeError(
                "IterableTokenizedDataset expects a directory of Parquet files."
            )
        self.max_length = int(max_length)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_rows = bool(shuffle_rows)
        self.seed = int(seed)
        self._epoch = -1

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
        for sp in shard_paths:
            pf = pq.ParquetFile(sp)
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
        self._offsets = np.cumsum([0] + self._rows_per_shard[:-1]).tolist()
        self._total_rows = int(sum(self._rows_per_shard))
        self.has_coords = any("coordinates" in cols for cols in self._columns)

    def __len__(self) -> int:
        # Per-rank sample cap to keep equal sample counts across ranks
        world_size = 1
        try:
            import torch.distributed as dist  # local import to avoid hard dep at import time

            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
        except Exception:
            world_size = 1
        return self._total_rows // max(1, int(world_size))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # epoch counter for deterministic shuffles
        self._epoch += 1
        seed_base = (0x9E3779B97F4A7C15 ^ self.seed) + (self._epoch * 0x1000003)

        # rank/world from torch.distributed if available
        rank = 0
        world_size = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
        except Exception:
            rank, world_size = 0, 1

        # dataloader workers
        wi = get_worker_info()
        if wi is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = wi.num_workers, wi.id

        # per-epoch shard order
        shard_indices = list(range(len(self._shards)))
        if self.shuffle_shards:
            rng = np.random.RandomState(seed_base & 0xFFFFFFFF)
            rng.shuffle(shard_indices)

        # equalize per-rank sample counts (drop global remainder)
        per_rank_cap = self._total_rows // max(1, world_size)
        emitted = 0

        for s_idx in shard_indices:
            if emitted >= per_rank_cap:
                break
            spath = self._shards[s_idx]
            nrows = int(self._rows_per_shard[s_idx])
            start = int(self._offsets[s_idx])

            # rows assigned to this rank (global striping)
            rank_rows = [
                i for i in range(nrows) if ((start + i) % max(1, world_size)) == rank
            ]
            if not rank_rows:
                continue

            if self.shuffle_rows:
                rng_rows = np.random.RandomState(
                    (seed_base + 1009 + s_idx) & 0xFFFFFFFF
                )
                rng_rows.shuffle(rank_rows)

            # within-rank worker striping
            rank_rows = rank_rows[worker_id :: max(1, num_workers)]
            if not rank_rows:
                continue

            table = pq.ParquetFile(spath).read(columns=self._columns[s_idx])
            for i in rank_rows:
                if emitted >= per_rank_cap:
                    break
                row = table.slice(i, 1).to_pylist()[0]
                yield _build_output_from_row(
                    row,
                    max_length=self.max_length,
                    has_coords=self.has_coords,
                )
                emitted += 1


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

    def __init__(
        self, dataset: Dataset, *, num_samples: int | None = None, seed: int = 0
    ):
        super().__init__()
        self.dataset = dataset
        self.num_samples = int(num_samples) if num_samples is not None else len(dataset)
        self.seed = int(seed)
        self._epoch = -1

    def __len__(self) -> int:
        return int(self.num_samples)

    def __iter__(self):
        self._epoch += 1
        wi = get_worker_info()
        if wi is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = wi.num_workers, wi.id

        # Deterministic per-epoch, per-worker RNG
        seed_base = (0x9E3779B97F4A7C15 ^ self.seed) + (self._epoch * 0x1000003)
        rng = np.random.RandomState((seed_base + worker_id) & 0xFFFFFFFF)

        n = int(self.num_samples)
        L = int(len(self.dataset))
        if L <= 0 or n <= 0:
            return iter(())

        # worker striping over sample positions (keeps global sample count ~num_samples)
        for _pos in range(worker_id, n, max(1, num_workers)):
            j = int(rng.randint(0, L))
            yield self.dataset[j]


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

    def __init__(
        self,
        datasets: list[IterableDataset],
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
        self.fractions = fr.tolist()
        self.seed = int(seed)
        self._epoch = -1

        if num_samples is None:
            # best-effort: sum dataset lengths if available
            total = 0
            for ds in datasets:
                try:
                    total += int(len(ds))  # type: ignore[arg-type]
                except Exception:
                    total = 0
                    break
            num_samples = total
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return int(self.num_samples)

    def __iter__(self):
        self._epoch += 1
        wi = get_worker_info()
        if wi is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = wi.num_workers, wi.id

        n = int(self.num_samples)
        if n <= 0:
            return iter(())

        # Deterministic per-epoch, per-worker RNG
        seed_base = (0x9E3779B97F4A7C15 ^ self.seed) + (self._epoch * 0x1000003)
        rng = np.random.RandomState((seed_base + worker_id) & 0xFFFFFFFF)

        # Create iterators; we re-create an iterator when it is exhausted.
        iters = [iter(ds) for ds in self.datasets]
        fr = np.asarray(self.fractions, dtype=np.float64)

        for _pos in range(worker_id, n, max(1, num_workers)):
            # Choose dataset id according to fractions
            ds_idx = int(rng.choice(len(iters), p=fr))
            try:
                yield next(iters[ds_idx])
            except StopIteration:
                iters[ds_idx] = iter(self.datasets[ds_idx])
                yield next(iters[ds_idx])
