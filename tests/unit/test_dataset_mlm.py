"""Tests for dataset support for MLM training (without indices column)."""

import pandas as pd
import pytest

from stok.data.dataset import DummyMLMDataset, TokenizedDataset


class TestTokenizedDatasetMLM:
    """Tests for TokenizedDataset with require_structure_tokens=False."""

    def test_dataset_without_indices_column(self, tmp_path):
        """Test that dataset loads correctly without indices column."""
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame(
            {
                "sequence_id": ["p1", "p2", "p3"],
                "sequence": ["MVLSPADKTNVKA", "MNIFEMLRIDKGLQVVA", "ACDEFGHIK"],
            }
        )
        df.to_parquet(parquet_path, index=False)

        ds = TokenizedDataset(
            str(parquet_path), max_length=32, require_structure_tokens=False
        )

        assert len(ds) == 3

        item = ds[0]
        assert "sequence_id" in item
        assert "sequence" in item
        assert "structure_tokens" not in item  # Should not have indices

    def test_dataset_with_indices_column_when_not_required(self, tmp_path):
        """Test that dataset includes indices when present even if not required."""
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame(
            {
                "sequence_id": ["p1", "p2"],
                "sequence": ["MVLSPADKTNVKA", "MNIFEMLRIDKGL"],
                "structure_tokens": [list(range(1, 14)), list(range(1, 13))],
            }
        )
        df.to_parquet(parquet_path, index=False)

        ds = TokenizedDataset(
            str(parquet_path), max_length=32, require_structure_tokens=False
        )

        item = ds[0]
        assert (
            "structure_tokens" in item
        )  # Should have indices since they're in the file

    def test_dataset_raises_error_when_indices_required_but_missing(self, tmp_path):
        """Test that dataset raises error when indices required but not present."""
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame(
            {
                "sequence_id": ["p1", "p2"],
                "sequence": ["MVLSPADKTNVKA", "MNIFEMLRIDKGL"],
            }
        )
        df.to_parquet(parquet_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            TokenizedDataset(
                str(parquet_path), max_length=32, require_structure_tokens=True
            )

    def test_dataset_seq_content(self, tmp_path):
        """Test that sequence content is correctly loaded."""
        parquet_path = tmp_path / "test.parquet"
        sequences = ["MVLSPADKTNVKA", "MNIFEMLRIDKGL", "ACDEFGHIK"]
        df = pd.DataFrame(
            {
                "sequence_id": ["p1", "p2", "p3"],
                "sequence": sequences,
            }
        )
        df.to_parquet(parquet_path, index=False)

        ds = TokenizedDataset(
            str(parquet_path), max_length=32, require_structure_tokens=False
        )

        for i, seq in enumerate(sequences):
            item = ds[i]
            assert item["sequence"] == seq

    def test_parquet_dataset_without_indices(self, tmp_path):
        """Test that Parquet dataset works without indices column."""
        pytest.importorskip("pyarrow")

        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame(
            {
                "sequence_id": ["p1", "p2"],
                "sequence": ["MVLSPADKTNVKA", "MNIFEMLRIDKGL"],
            }
        )
        df.to_parquet(parquet_path)

        ds = TokenizedDataset(
            str(parquet_path), max_length=32, require_structure_tokens=False
        )

        assert len(ds) == 2
        item = ds[0]
        assert "sequence" in item
        assert "structure_tokens" not in item


class TestDummyMLMDataset:
    """Tests for DummyMLMDataset."""

    def test_dummy_mlm_dataset_length(self):
        """Test that DummyMLMDataset has correct length."""
        num_samples = 100
        ds = DummyMLMDataset(num_samples=num_samples, seq_len=30)

        assert len(ds) == num_samples

    def test_dummy_mlm_dataset_item_keys(self):
        """Test that DummyMLMDataset returns items with correct keys."""
        ds = DummyMLMDataset(num_samples=10, seq_len=30)

        item = ds[0]
        assert "sequence_id" in item
        assert "sequence" in item

    def test_dummy_mlm_dataset_seq_length(self):
        """Test that DummyMLMDataset returns sequences of correct length."""
        seq_len = 50
        ds = DummyMLMDataset(num_samples=10, seq_len=seq_len)

        item = ds[0]
        assert len(item["sequence"]) == seq_len

    def test_dummy_mlm_dataset_seq_characters(self):
        """Test that DummyMLMDataset uses valid amino acid characters."""
        valid_aa = set("LAGVSERTIPDKQNFYMHWC")
        ds = DummyMLMDataset(num_samples=100, seq_len=100)

        for i in range(min(10, len(ds))):
            item = ds[i]
            seq = item["sequence"]
            for char in seq:
                assert char in valid_aa, f"Invalid character {char} in sequence"

    def test_dummy_mlm_dataset_unique_pids(self):
        """Test that DummyMLMDataset generates unique PIDs."""
        ds = DummyMLMDataset(num_samples=100, seq_len=30)

        pids = set()
        for i in range(len(ds)):
            pid = ds[i]["sequence_id"]
            assert pid not in pids, f"Duplicate PID: {pid}"
            pids.add(pid)


class TestIterableDatasetMLM:
    """Tests for IterableTokenizedDataset with require_structure_tokens=False."""

    def test_iterable_dataset_without_indices(self, tmp_path):
        """Test that iterable dataset works without indices column."""
        pytest.importorskip("pyarrow")
        from stok.data.dataset import IterableTokenizedDataset

        # Create directory with parquet shards
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()

        for i in range(2):
            df = pd.DataFrame(
                {
                    "sequence_id": [f"p{i}_{j}" for j in range(5)],
                    "sequence": [f"MVLSPADKTNVKA{j}" for j in range(5)],
                }
            )
            df.to_parquet(shard_dir / f"shard_{i}.parquet")

        ds = IterableTokenizedDataset(
            str(shard_dir),
            max_length=32,
            require_structure_tokens=False,
            shuffle_shards=False,
            shuffle_rows=False,
        )

        items = list(ds)
        assert len(items) == 10

        for item in items:
            assert "sequence" in item
            assert "sequence_id" in item
            # indices should not be present since we set require_structure_tokens=False
            # and the parquet files don't have an indices column


def test_optional_missing_label_array_preserves_all_positions(tmp_path):
    from stok.data.dataset import TokenizedDataset

    source = tmp_path / "mixed.parquet"
    pd.DataFrame(
        [
            {
                "sequence_id": "unlabeled",
                "sequence": "LAG",
                "structure_tokens": [None] * 3,
            },
            {
                "sequence_id": "labeled",
                "sequence": "LAG",
                "structure_tokens": [0, 1, 2],
            },
        ]
    ).to_parquet(source)
    dataset = TokenizedDataset(
        str(source), max_length=6, require_structure_tokens=False
    )
    assert dataset.has_labels
    assert dataset[0]["structure_tokens"].tolist() == [-1] * 3
    assert dataset[1]["structure_tokens"][:3].tolist() == [0, 1, 2]
