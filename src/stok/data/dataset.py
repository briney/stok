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

        if p.is_dir() or suffix in {".parquet", ".parq", ".pq"}:
            try:
                self.data = pd.read_parquet(dataset_path)
            except Exception as e:
                raise RuntimeError(
                    "Reading Parquet requires 'pyarrow' or 'fastparquet'."
                ) from e
        elif suffix in {".csv"}:
            self.data = pd.read_csv(dataset_path)
        elif suffix in {".tsv", ".tab"}:
            self.data = pd.read_csv(dataset_path, sep="\t")
        else:
            # Default to Parquet for unknown suffixes/directories
            try:
                self.data = pd.read_parquet(dataset_path)
            except Exception as e:
                raise RuntimeError(
                    "Unsupported file format. Provide a CSV/TSV or Parquet file."
                ) from e
        self.max_length = max_length

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

        return {
            "pid": pid,
            "indices": indices_tensor,
            "seq": seq,
            "masks": mask_tensor,
            "nan_masks": nan_mask,
        }


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
