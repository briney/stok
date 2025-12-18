"""Comprehensive integration tests for P@L (Precision@L) metric with structure data.

These tests use real-world PDB files from the CAMEO benchmark to ensure
the p_at_l metric is correctly computed and logged when training with
structure-based eval datasets.
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
from click.testing import CliRunner
from omegaconf import OmegaConf

from stok.cli.cli import cli
from stok.eval.registry import build_metrics, _get_dataset_has_coords
from stok.eval.metrics.contact import PrecisionAtLMetric, _compute_contact_map
from stok.data.structure_dataset import StructureFolderDataset
from stok.data.collate import mlm_collate
from stok.utils.tokenizer import Tokenizer


# Path to real CAMEO test data
CAMEO_TEST_DATA = Path(__file__).parent.parent / "test_data" / "cameo"


class TestStructureDatasetPipeline:
    """Test the data pipeline from PDB files through collation."""

    def test_structure_folder_dataset_loads_cameo_files(self):
        """Verify StructureFolderDataset correctly loads real CAMEO PDB files."""
        if not CAMEO_TEST_DATA.exists():
            pytest.skip("CAMEO test data not found")

        ds = StructureFolderDataset(
            folder_path=str(CAMEO_TEST_DATA),
            max_length=256,
        )

        assert len(ds) > 0, "Dataset should contain files"

        # Check first sample
        sample = ds[0]
        assert "seq" in sample, "Sample should have 'seq' key"
        assert "coords" in sample, "Sample should have 'coords' key"
        assert isinstance(sample["seq"], str), "Sequence should be a string"
        assert isinstance(sample["coords"], torch.Tensor), "Coords should be a tensor"

        # Verify coords shape: [max_length, 3, 3]
        assert sample["coords"].shape == (256, 3, 3), (
            f"Coords shape should be (256, 3, 3), got {sample['coords'].shape}"
        )

        # Verify sequence length is reasonable for real proteins
        assert len(sample["seq"]) >= 10, (
            f"Real protein sequences should be >= 10 residues, got {len(sample['seq'])}"
        )

    def test_mlm_collate_preserves_coords(self):
        """Verify mlm_collate correctly handles and returns coordinates."""
        if not CAMEO_TEST_DATA.exists():
            pytest.skip("CAMEO test data not found")

        ds = StructureFolderDataset(
            folder_path=str(CAMEO_TEST_DATA),
            max_length=128,
        )
        tokenizer = Tokenizer()

        # Create a small batch
        batch = [ds[i] for i in range(min(3, len(ds)))]

        result = mlm_collate(
            batch,
            tokenizer,
            max_len=128,
            mask_prob=0.15,
        )

        # Should return 3-tuple with coords
        assert len(result) == 3, (
            f"mlm_collate should return 3 elements (tokens, labels, coords), got {len(result)}"
        )

        tokens, labels, coords = result

        assert coords is not None, "Coords should not be None"
        assert coords.shape[0] == len(batch), "Batch size mismatch"
        assert coords.shape[1] == 128, "Coords length should match max_len"
        assert coords.shape[2:] == (3, 3), "Coords should have shape [B, L, 3, 3]"


class TestContactMapComputation:
    """Test the contact map computation logic."""

    def test_contact_map_basic(self):
        """Verify contact map computation works with valid coords."""
        B, L = 2, 20
        coords = torch.randn(B, L, 3, 3) * 5.0  # Random coords

        contact_map = _compute_contact_map(coords, threshold=8.0)

        # Check shape
        assert contact_map.shape == (B, L, L), (
            f"Contact map shape should be ({B}, {L}, {L})"
        )

        # Should be a boolean tensor
        assert contact_map.dtype == torch.bool, "Contact map should be boolean"

    def test_contact_map_with_nan_padding(self):
        """Verify contact map handles NaN padding correctly."""
        B, L = 2, 20
        coords = torch.randn(B, L, 3, 3) * 5.0

        # Add NaN padding for last 5 positions
        coords[:, 15:, :, :] = float("nan")

        contact_map = _compute_contact_map(coords, threshold=8.0)

        # Check shape
        assert contact_map.shape == (B, L, L), (
            f"Contact map shape should be ({B}, {L}, {L})"
        )

        # Valid region (positions 0-14) should not have NaN-related issues
        valid_region = contact_map[:, :15, :15]
        # These should be valid boolean values
        assert valid_region.dtype == torch.bool

        # NaN positions should be False (no contacts)
        # Check that NaN row/col positions are all False
        nan_row_contacts = contact_map[:, 15:, :15]  # NaN rows with valid cols
        nan_col_contacts = contact_map[:, :15, 15:]  # Valid rows with NaN cols
        assert not nan_row_contacts.any(), "NaN rows should have no contacts"
        assert not nan_col_contacts.any(), "NaN cols should have no contacts"


class TestPAtLMetricDirectly:
    """Test PrecisionAtLMetric directly with controlled inputs."""

    def test_p_at_l_metric_instantiation(self):
        """Verify P@L metric can be instantiated."""
        metric = PrecisionAtLMetric(
            contact_threshold=8.0,
            min_seq_sep=6,
            use_attention=False,  # Force fallback path
        )
        assert metric.name == "p_at_l"
        assert metric.requires_coords is True
        assert metric.objectives == {"mlm"}

    def test_p_at_l_metric_update_without_exception(self):
        """Verify P@L metric update doesn't throw with valid inputs."""
        metric = PrecisionAtLMetric(
            contact_threshold=8.0,
            min_seq_sep=3,  # Lower threshold for testing
            use_attention=True,
        )

        # Create mock inputs
        B, L, H = 2, 32, 4
        tokens = torch.randint(4, 24, (B, L))  # Amino acid tokens
        labels = torch.full((B, L), -100)  # All masked

        # Create realistic coords
        coords = torch.randn(B, L, 3, 3) * 10.0  # Scale to protein-like distances

        # Create mock outputs with attention weights (tuple of per-layer tensors)
        attentions = tuple(
            torch.softmax(torch.randn(B, H, L, L), dim=-1)
            for _ in range(2)  # 2 layers
        )

        outputs = {
            "logits": torch.randn(B, L, 32),  # vocab_size=32
            "loss": torch.tensor(2.5),
            "attentions": attentions,
        }

        # Create mock config
        cfg = OmegaConf.create(
            {
                "model": {
                    "encoder": {"pad_id": 1},
                    "classifier": {"ignore_index": -100},
                }
            }
        )

        # This should not raise
        try:
            metric.update(outputs, tokens, labels, coords, cfg)
        except Exception as e:
            pytest.fail(f"metric.update raised an exception: {e}")

        # Compute should return valid result
        result = metric.compute()
        assert "p_at_l" in result, "Result should contain 'p_at_l' key"
        assert 0.0 <= result["p_at_l"] <= 1.0, "P@L should be between 0 and 1"

    def test_p_at_l_metric_with_attention_fallback(self):
        """Test P@L metric falls back to logits when no attention provided."""
        metric = PrecisionAtLMetric(
            contact_threshold=8.0,
            min_seq_sep=3,
            use_attention=True,  # Will try attention first
        )

        B, L = 2, 32
        tokens = torch.randint(4, 24, (B, L))
        labels = torch.full((B, L), -100)
        coords = torch.randn(B, L, 3, 3) * 10.0

        # No attention weights provided - should fall back to logits
        outputs = {
            "logits": torch.randn(B, L, 32),
            "loss": torch.tensor(2.5),
        }

        cfg = OmegaConf.create(
            {
                "model": {
                    "encoder": {"pad_id": 1},
                    "classifier": {"ignore_index": -100},
                }
            }
        )

        # Should not raise even without attention
        metric.update(outputs, tokens, labels, coords, cfg)
        result = metric.compute()
        assert "p_at_l" in result


class TestMetricBuildingWithStructureFolder:
    """Test that metrics are correctly built for structure folder datasets."""

    def test_get_dataset_has_coords_structure_format(self, tmp_path):
        """Verify _get_dataset_has_coords returns True for structure format."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "eval": {
                        "cameo": {
                            "path": str(tmp_path),
                            "format": "structure",
                        }
                    }
                }
            }
        )

        has_coords = _get_dataset_has_coords(cfg, "cameo", default_has_coords=False)
        assert has_coords is True, "Structure format should have coords"

    def test_p_at_l_metric_built_for_mlm_with_coords(self, tmp_path):
        """Verify p_at_l metric is built when conditions are met."""
        cfg = OmegaConf.create(
            {
                "train": {"eval": {"metrics": {}}},  # No explicit disable
                "data": {
                    "eval": {
                        "cameo": {
                            "path": str(tmp_path),
                            "format": "structure",
                        }
                    }
                },
            }
        )

        metrics = build_metrics(
            cfg=cfg,
            objective="mlm",
            decoder=None,
            has_coords=False,  # Default is False, but structure format overrides
            eval_name="cameo",
        )

        metric_names = [m.name for m in metrics]
        assert "p_at_l" in metric_names, (
            f"p_at_l should be built for MLM with structure folder. Got metrics: {metric_names}"
        )

    def test_p_at_l_metric_not_built_for_codebook(self, tmp_path):
        """Verify p_at_l metric is NOT built for codebook objective."""
        cfg = OmegaConf.create(
            {
                "train": {"eval": {"metrics": {}}},
                "data": {
                    "eval": {
                        "cameo": {
                            "path": str(tmp_path),
                            "format": "structure",
                        }
                    }
                },
            }
        )

        metrics = build_metrics(
            cfg=cfg,
            objective="codebook",  # Not MLM
            decoder=None,
            has_coords=True,
            eval_name="cameo",
        )

        metric_names = [m.name for m in metrics]
        assert "p_at_l" not in metric_names, (
            f"p_at_l should NOT be built for codebook objective. Got metrics: {metric_names}"
        )


class TestEvaluatorAttentionPropagation:
    """Test that attention weights flow correctly through the evaluator."""

    def test_evaluator_needs_attentions_detection(self, tmp_path):
        """Test that evaluator detects when attention weights are needed."""
        from stok.eval.evaluator import Evaluator
        from stok.models.stok import STokModel

        cfg = OmegaConf.create(
            {
                "train": {
                    "objective": "mlm",
                    "eval": {"metrics": {"p_at_l": {"enabled": True}}},
                    "decoding": {"eval_enabled": False},
                },
                "data": {
                    "load_coords": False,
                    "max_len": 128,
                    "eval": {
                        "cameo": {
                            "path": str(tmp_path),
                            "format": "structure",
                        }
                    },
                },
                "model": {
                    "encoder": {
                        "vocab_size": 32,
                        "pad_id": 1,
                        "d_model": 64,
                        "n_heads": 4,
                        "n_layers": 2,
                        "ffn_mult": 1.0,
                        "dropout": 0.0,
                        "attn_dropout": 0.0,
                    },
                    "classifier": {"ignore_index": -100},
                    "codebook": {"preset": "lite"},
                },
            }
        )

        # Create model
        model = STokModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=1.0,
            dropout=0.0,
            attn_dropout=0.0,
            head_type="mlm",
        )

        evaluator = Evaluator(
            cfg=cfg,
            model=model,
            accelerator=None,
            decoder=None,
        )

        # Check that metrics were built correctly
        metrics = evaluator._get_metrics("cameo")
        metric_names = [m.name for m in metrics]

        assert "p_at_l" in metric_names, (
            f"p_at_l metric should be built. Got: {metric_names}"
        )

        # Check that needs_attentions is True for cameo dataset
        assert evaluator._needs_attentions("cameo") is True, (
            "Evaluator should detect that cameo dataset needs attention weights"
        )


class TestEndToEndMLMWithCameoEval:
    """End-to-end integration tests using real CAMEO data."""

    def test_mlm_training_with_cameo_structure_eval_p_at_l_computed(self, tmp_path):
        """
        Full integration test: MLM training with CAMEO structure folder eval.
        Verifies that p_at_l metric is actually computed and logged.
        """
        if not CAMEO_TEST_DATA.exists():
            pytest.skip("CAMEO test data not found")

        # Create minimal MLM training data
        train_csv = tmp_path / "train.csv"
        df = pd.DataFrame(
            {
                "pid": [f"train_{i}" for i in range(10)],
                "protein_sequence": [
                    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"[:30] for _ in range(10)
                ],
            }
        )
        df.to_csv(train_csv, index=False)

        runner = CliRunner()
        overrides = [
            "train.objective=mlm",
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            "data.max_len=256",  # Enough for real proteins
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Use CAMEO structure folder for eval
            f"+data.eval.cameo.path={CAMEO_TEST_DATA.as_posix()}",
            "+data.eval.cameo.format=structure",
            # Explicitly enable p_at_l
            "train.eval.metrics.p_at_l.enabled=true",
            "train.eval.metrics.p_at_l.min_seq_sep=6",
            # Short run with eval
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])

        # Debug: print full output if assertion fails
        if result.exit_code != 0:
            print("STDERR:", result.output)

        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "Training complete." in result.output

        # Verify eval was performed
        assert "eval/cameo" in result.output, "Eval should have run for cameo dataset"

        # Check for P@L metric in output (may show as "P@L" in formatted output)
        p_at_l_logged = "P@L" in result.output or "p_at_l" in result.output

        if not p_at_l_logged:
            # Print debug info
            print("=== DEBUG: Full output ===")
            print(result.output)
            pytest.fail("P@L metric was not logged. See output above.")

    def test_mlm_training_with_cameo_autodetect(self, tmp_path):
        """
        Test auto-detection of structure folder format.
        Pass the directory directly without explicit format=structure.
        """
        if not CAMEO_TEST_DATA.exists():
            pytest.skip("CAMEO test data not found")

        train_csv = tmp_path / "train.csv"
        df = pd.DataFrame(
            {
                "pid": [f"train_{i}" for i in range(10)],
                "protein_sequence": [
                    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"[:30] for _ in range(10)
                ],
            }
        )
        df.to_csv(train_csv, index=False)

        runner = CliRunner()
        overrides = [
            "train.objective=mlm",
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            "data.max_len=256",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Pass directory directly - should auto-detect as structure folder
            f"+data.eval.cameo={CAMEO_TEST_DATA.as_posix()}",
            "train.eval.metrics.p_at_l.enabled=true",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "eval/cameo" in result.output

    def test_mlm_training_p_at_l_disabled_by_default_no_coords(self, tmp_path):
        """
        Verify p_at_l is NOT computed when coords are not available.
        This ensures the metric correctly respects its requirements.
        """
        # Create CSV dataset without coords
        train_csv = tmp_path / "train.csv"
        eval_csv = tmp_path / "eval.csv"

        df = pd.DataFrame(
            {
                "pid": [f"train_{i}" for i in range(10)],
                "protein_sequence": [
                    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"[:30] for _ in range(10)
                ],
            }
        )
        df.to_csv(train_csv, index=False)

        eval_df = pd.DataFrame(
            {
                "pid": [f"eval_{i}" for i in range(5)],
                "protein_sequence": [
                    "MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQ"[:25] for _ in range(5)
                ],
            }
        )
        eval_df.to_csv(eval_csv, index=False)

        runner = CliRunner()
        overrides = [
            "train.objective=mlm",
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            "data.max_len=64",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            f"+data.eval.validation={eval_csv.as_posix()}",
            # Don't enable p_at_l explicitly - it requires coords
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "eval/validation" in result.output

        # P@L should NOT be logged because there are no coords
        assert "P@L" not in result.output, (
            "P@L should not be logged when coords are not available"
        )

