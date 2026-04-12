"""Unit tests for :mod:`stok.utils.mmcif` and the shared structure dispatcher.

Confirms a round-trip through ``build_mmcif`` → ``parse_structure`` returns
the same coordinates and residue names within numerical tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stok.api import _write_structure
from stok.utils.mmcif import build_mmcif
from stok.utils.pdb import build_pdb
from stok.utils.structure_parser import parse_structure


def _random_backbone(length: int, seed: int = 0) -> torch.Tensor:
    """Deterministic backbone coordinates of shape (L, 3, 3)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(length, 3, 3, generator=g) * 5.0


class TestBuildMmcif:
    def test_writes_file_without_sequence(self, tmp_path):
        """No-sequence mode stamps every residue as UNK (``build_pdb``
        parity). ``parse_structure`` cannot read UNK-only structures —
        that's a pre-existing limitation of the parser — so we only
        confirm the file is written and non-empty.
        """
        coords = _random_backbone(6, seed=1)
        cif_path = tmp_path / "out.cif"
        build_mmcif(coords, str(cif_path))
        assert cif_path.exists()
        assert cif_path.stat().st_size > 0

    def test_roundtrip_with_sequence(self, tmp_path):
        coords = _random_backbone(8, seed=2)
        sequence = "ACDEFGHI"
        cif_path = tmp_path / "out.cif"
        build_mmcif(coords, str(cif_path), sequence=sequence)

        parsed = parse_structure(cif_path)
        assert parsed.protein_sequence == sequence
        np.testing.assert_allclose(
            parsed.coords, coords.numpy(), atol=1e-3
        )

    def test_sequence_length_mismatch_raises(self, tmp_path):
        coords = _random_backbone(4, seed=3)
        with pytest.raises(ValueError, match="Sequence length"):
            build_mmcif(coords, str(tmp_path / "bad.cif"), sequence="ACD")

    def test_numpy_coordinates_accepted(self, tmp_path):
        coords = _random_backbone(5, seed=4).numpy()
        cif_path = tmp_path / "np.cif"
        build_mmcif(coords, str(cif_path), sequence="ACDEF")
        parsed = parse_structure(cif_path)
        np.testing.assert_allclose(parsed.coords, coords, atol=1e-3)


class TestPdbVsMmcifRoundTrip:
    def test_same_coords_via_both_formats(self, tmp_path):
        """PDB and mmCIF writers produce parse-equivalent coordinates."""
        coords = _random_backbone(10, seed=5)
        sequence = "ACDEFGHIKL"

        pdb_path = tmp_path / "out.pdb"
        cif_path = tmp_path / "out.cif"
        build_pdb(coords, str(pdb_path), sequence=sequence)
        build_mmcif(coords, str(cif_path), sequence=sequence)

        pdb_parsed = parse_structure(pdb_path)
        cif_parsed = parse_structure(cif_path)

        assert pdb_parsed.protein_sequence == cif_parsed.protein_sequence == sequence
        np.testing.assert_allclose(pdb_parsed.coords, cif_parsed.coords, atol=1e-3)
        np.testing.assert_allclose(cif_parsed.coords, coords.numpy(), atol=1e-3)


class TestWriteStructureDispatcher:
    def test_pdb_extension(self, tmp_path):
        coords = _random_backbone(4, seed=6)
        path = tmp_path / "x.pdb"
        _write_structure(coords, path, sequence="ACDE")
        assert path.exists()
        assert path.read_text().lstrip().startswith(
            ("HEADER", "ATOM", "MODEL", "TITLE")
        )

    def test_ent_extension_treated_as_pdb(self, tmp_path):
        coords = _random_backbone(4, seed=7)
        path = tmp_path / "x.ent"
        _write_structure(coords, path, sequence="ACDE")
        assert path.exists()

    def test_cif_extension(self, tmp_path):
        coords = _random_backbone(4, seed=8)
        path = tmp_path / "x.cif"
        _write_structure(coords, path, sequence="ACDE")
        assert path.exists()
        parsed = parse_structure(path)
        assert parsed.protein_sequence == "ACDE"

    def test_mmcif_extension(self, tmp_path):
        coords = _random_backbone(4, seed=9)
        path = tmp_path / "x.mmcif"
        _write_structure(coords, path, sequence="ACDE")
        assert path.exists()

    def test_unknown_extension_raises(self, tmp_path):
        coords = _random_backbone(4, seed=10)
        with pytest.raises(ValueError, match="Unknown structure format"):
            _write_structure(coords, tmp_path / "x.xyz", sequence="ACDE")
