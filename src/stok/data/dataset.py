from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class BaseTokenizedDataset:
    """
    Mixin providing shared row parsing, padding, and optional coordinates handling.
    """

    @staticmethod
    def _build_output_from_row(
        row: pd.Series,
        *,
        max_length: int,
        has_coords: bool,
        require_indices: bool = True,
    ) -> dict[str, torch.Tensor | str]:
        """Build output dictionary from a dataset row.

        Args:
            row: DataFrame row containing protein data.
            max_length: Maximum sequence length for padding.
            has_coords: Whether to parse coordinates from the row.
            require_indices: Whether indices column is required. Set to False for MLM.

        Returns:
            Dictionary with pid, seq, and optionally indices, masks, coords.
        """
        pid = row["pid"]
        seq = row["protein_sequence"]

        out: dict[str, torch.Tensor | str] = {
            "pid": pid,
            "seq": seq,
        }

        # Parse indices if present and required
        if require_indices or "indices" in row.index:
            raw = row.get("indices")
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                indices = []
            elif isinstance(raw, (list, tuple, np.ndarray)):
                indices = [int(i) for i in list(raw) if i is not None]
            elif isinstance(raw, str):
                s = raw.strip()
                indices = [int(i) for i in s.split()] if s else []
            else:
                # fallback: try casting to string then parse; if it fails, empty
                try:
                    s = str(raw).strip()
                    indices = [int(i) for i in s.split()] if s else []
                except Exception:
                    indices = []

            idx_length = len(indices)
            pad_length = max(0, max_length - idx_length)

            # pad indices with -1 and create a mask
            padded_indices = indices + [-1] * pad_length
            mask = [True] * idx_length + [False] * pad_length

            # make tensors
            indices_tensor = torch.tensor(padded_indices, dtype=torch.long)
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            nan_mask = indices_tensor != -1

            out["indices"] = indices_tensor
            out["masks"] = mask_tensor
            out["nan_masks"] = nan_mask

        # parse optional 3D-coordinates only from parquet inputs
        if has_coords:
            raw_coords = row.get("coordinates")
            coords_arr: Optional[np.ndarray]
            # Lazy import to avoid unnecessary dependency at module import time
            import ast as _ast  # type: ignore

            if isinstance(raw_coords, np.ndarray):
                # parquet nested lists can round-trip as object arrays
                if raw_coords.dtype == object:
                    # Deep-normalize two levels to handle arrays-of-arrays objects
                    try:
                        outer = raw_coords.tolist()
                        rows = []
                        for elem in outer:
                            sub = elem.tolist() if hasattr(elem, "tolist") else elem
                            rows.append(
                                [
                                    list(x)
                                    if hasattr(x, "__iter__")
                                    and not isinstance(x, (str, bytes))
                                    else x
                                    for x in sub
                                ]
                            )
                        coords_arr = np.asarray(rows, dtype=np.float32)
                    except Exception:
                        coords_arr = None
                else:
                    coords_arr = raw_coords.astype(np.float32, copy=False)
            elif isinstance(raw_coords, (list, tuple)):
                # Handle list-of-arrays or list-of-lists
                try:
                    rows = []
                    for elem in list(raw_coords):
                        sub = elem.tolist() if hasattr(elem, "tolist") else elem
                        rows.append(
                            [
                                list(x)
                                if hasattr(x, "__iter__")
                                and not isinstance(x, (str, bytes))
                                else x
                                for x in sub
                            ]
                        )
                    coords_arr = np.asarray(rows, dtype=np.float32)
                except Exception:
                    try:
                        coords_arr = np.asarray(raw_coords, dtype=np.float32)
                    except Exception:
                        coords_arr = None
            elif isinstance(raw_coords, float) and pd.isna(raw_coords):
                coords_arr = None
            else:
                coords_arr = None

            # Final generic fallback conversion
            if coords_arr is None and raw_coords is not None:
                obj = (
                    raw_coords.tolist() if hasattr(raw_coords, "tolist") else raw_coords
                )
                try:
                    coords_arr = np.asarray(obj, dtype=np.float32)
                except Exception:
                    try:
                        coords_arr = np.stack(
                            [np.asarray(x, dtype=np.float32) for x in list(obj)],
                            axis=0,
                        )
                    except Exception:
                        coords_arr = None
                # Stringified list fallback (e.g., if serialized as text)
                if (
                    coords_arr is None
                    and isinstance(obj, str)
                    and obj.strip().startswith("[")
                ):
                    try:
                        parsed = _ast.literal_eval(obj)
                        coords_arr = np.asarray(parsed, dtype=np.float32)
                    except Exception:
                        coords_arr = None

            # Coerce to shape [L, 3, 3] if possible
            if coords_arr is not None:
                if coords_arr.ndim == 3 and coords_arr.shape[-2:] == (3, 3):
                    pass
                elif coords_arr.ndim == 3 and coords_arr.shape[:2] == (3, 3):
                    coords_arr = np.transpose(coords_arr, (2, 0, 1))
                elif coords_arr.ndim == 2 and coords_arr.shape == (3, 3):
                    coords_arr = coords_arr[None, ...]
                elif coords_arr.ndim == 2 and (coords_arr.size % 9 == 0):
                    coords_arr = coords_arr.reshape(-1, 3, 3)
                else:
                    coords_arr = None

            if coords_arr is None:
                coords_arr = np.empty((0, 3, 3), dtype=np.float32)

            Lc = int(coords_arr.shape[0])
            copy_len = min(Lc, max_length)
            coords_padded = np.full((max_length, 3, 3), np.nan, dtype=np.float32)
            if copy_len > 0:
                coords_padded[:copy_len] = coords_arr[:copy_len]
            out["coords"] = torch.tensor(coords_padded, dtype=torch.float32)

        return out


class TokenizedDataset(Dataset, BaseTokenizedDataset):
    """Dataset for loading VQ indices from CSV or Parquet files.

    Indices are parsed from either a space-delimited string (CSV) or a
    list/array of integers (Parquet).

    Args:
        dataset_path: Path to CSV/TSV or Parquet file (or Parquet directory).
        max_length: Maximum number of indices to keep (padding with -1).
        load_coords: Whether to load 3D coordinates (Parquet only).
        require_indices: Whether the indices column is required. Set to False for MLM.
    """

    def __init__(
        self,
        dataset_path: str,
        max_length: int,
        *,
        load_coords: bool = True,
        require_indices: bool = True,
    ):
        p = Path(dataset_path)
        suffix = p.suffix.lower()

        self._is_parquet = False
        if p.is_dir() or suffix in {".parquet", ".parq", ".pq"}:
            self.data = pd.read_parquet(dataset_path)
            self._is_parquet = True
        elif suffix in {".csv"}:
            self.data = pd.read_csv(dataset_path)
        elif suffix in {".tsv", ".tab"}:
            self.data = pd.read_csv(dataset_path, sep="\t")
        else:
            # default to parquet for unknown suffixes/directories
            try:
                self.data = pd.read_parquet(dataset_path)
                self._is_parquet = True
            except Exception as e:
                raise RuntimeError(
                    "Unsupported file format. Provide a CSV/TSV or Parquet file."
                ) from e

        self.max_length = max_length
        self.require_indices = require_indices

        # Validate required columns
        required_cols = {"pid", "protein_sequence"}
        if require_indices:
            required_cols.add("indices")
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Coordinates are only supported from Parquet-backed datasets
        self.has_coords = (
            bool(load_coords)
            and self._is_parquet
            and ("coordinates" in self.data.columns)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        return BaseTokenizedDataset._build_output_from_row(
            row,
            max_length=self.max_length,
            has_coords=self.has_coords,
            require_indices=self.require_indices,
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
            Dict with 'pid' and 'seq' keys.
        """
        # Generate random amino acid sequence
        seq = "".join(
            self._aa_chars[i] for i in torch.randint(0, 20, (self.seq_len,)).tolist()
        )
        return {"pid": f"dummy_{idx}", "seq": seq}


class IterableTokenizedDataset(IterableDataset, BaseTokenizedDataset):
    """Shard-wise iterable dataset over a directory of Parquet files.

    Loads a single Parquet shard at a time to bound memory use, applies
    deterministic per-epoch shuffling of shards and rows, and partitions
    samples across distributed ranks and dataloader workers.

    Args:
        dataset_path: Path to a directory containing Parquet shard files.
        max_length: Maximum number of indices to keep (padding with -1).
        shuffle_shards: Whether to shuffle shard order per epoch.
        shuffle_rows: Whether to shuffle selected row indices per shard per epoch.
        seed: Optional base seed for deterministic epoch shuffles.
        load_coords: Whether to load 3D coordinates.
        require_indices: Whether the indices column is required. Set to False for MLM.
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
        require_indices: bool = True,
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
        # user-intent flag for whether to load coordinates at all
        self._load_coords = bool(load_coords)
        self._require_indices = bool(require_indices)

        # enumerate shard files and stats
        shard_paths = sorted(
            [
                p
                for p in self.dataset_path.iterdir()
                if p.suffix.lower() in {".parquet", ".parq", ".pq"}
            ]
        )
        if len(shard_paths) == 0:
            raise RuntimeError("No Parquet shards found in directory.")

        self._shards: list[Path] = []
        self._rows_per_shard: list[int] = []
        cols_union: set[str] = set()
        for sp in shard_paths:
            pf = pq.ParquetFile(sp.as_posix())
            self._shards.append(sp)
            self._rows_per_shard.append(int(pf.metadata.num_rows))
            try:
                schema = pf.schema_arrow
                cols_union.update([f.name for f in schema])
            except Exception:
                # best-effort
                pass
        self._offsets = np.cumsum([0] + self._rows_per_shard[:-1]).tolist()
        self._total_rows = int(sum(self._rows_per_shard))

        # Track whether the directory has coordinates and indices columns
        self.has_coords = ("coordinates" in cols_union) if len(cols_union) > 0 else True
        self._has_indices_col = (
            ("indices" in cols_union) if len(cols_union) > 0 else True
        )

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

            # read only required columns
            want_cols = ["pid", "protein_sequence"]
            # Only include indices if required and present
            if self._require_indices and self._has_indices_col:
                want_cols.append("indices")
            elif self._has_indices_col:
                # Include indices if present even if not required
                want_cols.append("indices")

            use_coords = self.has_coords and self._load_coords
            if use_coords:
                want_cols.append("coordinates")
            df = pd.read_parquet(spath.as_posix(), columns=want_cols)

            for i in rank_rows:
                if emitted >= per_rank_cap:
                    break
                row = df.iloc[i]
                yield BaseTokenizedDataset._build_output_from_row(
                    row,
                    max_length=self.max_length,
                    has_coords=use_coords,
                    require_indices=self._require_indices,
                )
                emitted += 1
