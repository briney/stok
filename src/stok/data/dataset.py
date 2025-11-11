from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VQIndicesDataset(Dataset):
    """Dataset for loading VQ indices from CSV or Parquet files.

    Indices are parsed from either a space-delimited string (CSV) or a
    list/array of integers (Parquet).

    Args:
        dataset_path: Path to CSV/TSV or Parquet file (or Parquet directory).
        max_length: Maximum number of indices to keep (padding with -1).
    """

    def __init__(self, dataset_path: str, max_length: int):
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
        # Coordinates are only supported from Parquet-backed datasets
        self.has_coords = self._is_parquet and ("coordinates" in self.data.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        pid = row["pid"]
        seq = row["protein_sequence"]
        # handle empty/NaN indices cells -> treat as empty list
        raw = row["indices"]
        if isinstance(raw, (list, tuple, np.ndarray)):
            indices = [int(i) for i in list(raw) if i is not None]
        elif isinstance(raw, float) and pd.isna(raw):
            indices = []
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
        pad_length = max(0, self.max_length - idx_length)

        # pad indices with -1 and create a mask
        padded_indices = indices + [-1] * pad_length
        mask = [True] * idx_length + [False] * pad_length

        # make tensors
        indices_tensor = torch.tensor(padded_indices, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        nan_mask = indices_tensor != -1

        out: dict[str, torch.Tensor | str] = {
            "pid": pid,
            "indices": indices_tensor,
            "seq": seq,
            "masks": mask_tensor,
            "nan_masks": nan_mask,
        }

        # parse optional 3D-coordinates only from parquet inputs
        if self.has_coords:
            raw_coords = row["coordinates"]
            coords_arr = None
            if isinstance(raw_coords, np.ndarray):
                # parquet nested lists can round-trip as object arrays
                if raw_coords.dtype == object:
                    coords_arr = np.asarray(raw_coords.tolist(), dtype=np.float32)
                else:
                    coords_arr = raw_coords.astype(np.float32, copy=False)
            elif isinstance(raw_coords, (list, tuple)):
                coords_arr = np.asarray(raw_coords, dtype=np.float32)
            elif isinstance(raw_coords, float) and pd.isna(raw_coords):
                coords_arr = None
            else:
                coords_arr = None

            # Ccerce to shape [L, 3, 3] if possible
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
            copy_len = min(Lc, self.max_length)
            coords_padded = np.full((self.max_length, 3, 3), np.nan, dtype=np.float32)
            if copy_len > 0:
                coords_padded[:copy_len] = coords_arr[:copy_len]
            out["coords"] = torch.tensor(coords_padded, dtype=torch.float32)

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
