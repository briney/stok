"""Dataset for PDB/mmCIF structure folders.

Provides a map-style dataset for loading protein structures from a folder
containing PDB and/or mmCIF files. Designed primarily for evaluation with
structure-based metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stok.utils.structure_parser import parse_structure

__all__ = ["StructureFolderDataset"]

# Supported structure file extensions
STRUCTURE_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}


class StructureFolderDataset(Dataset):
    """Dataset for a folder of PDB/mmCIF structure files.

    This dataset is designed for evaluation with structure-based metrics.
    Each structure file produces one sample with:
      - pid: structure identifier (filename stem)
      - seq: amino acid sequence extracted from structure
      - coords: backbone coordinates [max_length, 3, 3] for N, CA, C atoms
      - masks: boolean mask [max_length] for valid positions
      - nan_masks: same as masks (all backbone atoms present or NaN)

    Note: No VQ indices are provided since these are raw structures without
    pre-computed structure tokens.

    Args:
        folder_path: Path to folder containing PDB/mmCIF files.
        max_length: Maximum sequence length (for padding/truncation).
        chain_id: Specific chain to extract from each file. If None, uses
            first polymer chain found in each structure.
        strict: If True, raise errors on missing backbone atoms.
            If False (default), fill missing atoms with NaN.
        recursive: If True, search subdirectories recursively.

    Raises:
        ValueError: If folder_path is not a directory or contains no
            structure files.

    Example config:
        .. code-block:: yaml

            eval:
              pdb_benchmark:
                path: /path/to/pdb_folder
                format: structure
                chain_id: A
                metrics:
                  only: [lddt, tm_score, rmsd]
    """

    def __init__(
        self,
        folder_path: str | Path,
        max_length: int,
        *,
        chain_id: str | None = None,
        strict: bool = False,
        recursive: bool = False,
    ):
        self.folder_path = Path(folder_path)
        if not self.folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        self.max_length = int(max_length)
        self.chain_id = chain_id
        self.strict = bool(strict)
        self.recursive = bool(recursive)

        # Discover structure files
        if recursive:
            self._files = sorted(
                [
                    p
                    for p in self.folder_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS
                ]
            )
        else:
            self._files = sorted(
                [
                    p
                    for p in self.folder_path.iterdir()
                    if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS
                ]
            )

        if len(self._files) == 0:
            raise ValueError(
                f"No structure files found in {folder_path}. "
                f"Supported extensions: {STRUCTURE_EXTENSIONS}"
            )

        # Flags for compatibility with existing dataset code
        self.has_coords = True
        self._is_parquet = False

    def __len__(self) -> int:
        """Return number of structure files in the dataset."""
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Load and parse a structure file.

        Args:
            idx: Index of the structure file to load.

        Returns:
            Dict with keys:
                - pid (str): Structure identifier
                - seq (str): Amino acid sequence
                - coords (Tensor): [max_length, 3, 3] backbone coordinates
                - masks (Tensor): [max_length] boolean mask for valid positions
                - nan_masks (Tensor): [max_length] same as masks

            Note: 'indices' key is NOT included (not available for raw structures).
        """
        path = self._files[idx]

        # Parse structure
        data = parse_structure(
            path,
            chain_id=self.chain_id,
            strict=self.strict,
        )

        seq_len = len(data.protein_sequence)
        coords = data.coords  # [L, 3, 3]

        # Truncate if needed
        if seq_len > self.max_length:
            seq = data.protein_sequence[: self.max_length]
            coords = coords[: self.max_length]
            seq_len = self.max_length
        else:
            seq = data.protein_sequence

        # Pad coordinates to max_length with NaN
        pad_len = self.max_length - seq_len
        coords_padded = np.full((self.max_length, 3, 3), np.nan, dtype=np.float32)
        coords_padded[:seq_len] = coords

        # Build mask (True for valid positions)
        mask = [True] * seq_len + [False] * pad_len

        out: dict[str, torch.Tensor | str] = {
            "pid": data.pid,
            "seq": seq,
            "coords": torch.tensor(coords_padded, dtype=torch.float32),
            "masks": torch.tensor(mask, dtype=torch.bool),
            "nan_masks": torch.tensor(mask, dtype=torch.bool),
        }

        return out

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"StructureFolderDataset("
            f"folder={self.folder_path}, "
            f"num_files={len(self._files)}, "
            f"max_length={self.max_length})"
        )
