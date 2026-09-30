"""Unit tests for structure metrics."""

import torch
from omegaconf import OmegaConf

from stok.eval.metrics.structure import (
    FAPEMetric,
    LDDTMetric,
    PredNaNFracMetric,
    RMSDMetric,
    TMScoreMetric,
)


def _make_cfg():
    """Create a minimal config for testing."""
    return OmegaConf.create(
        {"model": {"classifier": {"ignore_index": -100}, "encoder": {"pad_id": 1}}}
    )


def _stable_ncac_coords(batch: int, length: int) -> torch.Tensor:
    """Generate geometrically stable N-CA-C coordinates [B, L, 3, 3]."""
    g = torch.Generator().manual_seed(1234)

    ca = torch.randn((batch, length, 3), generator=g)
    x_dir = torch.randn((batch, length, 3), generator=g)
    y_dir = torch.randn((batch, length, 3), generator=g)
    x_dir = x_dir / (torch.linalg.norm(x_dir, dim=-1, keepdim=True).clamp_min(1e-3))
    y_dir = y_dir / (torch.linalg.norm(y_dir, dim=-1, keepdim=True).clamp_min(1e-3))
    n = ca - 1.45 * x_dir
    c = ca + 1.52 * y_dir
    coords = torch.stack([n, ca, c], dim=-2)
    return coords


class TestLDDTMetric:
    """Tests for LDDTMetric."""

    def test_lddt_metric_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = LDDTMetric()
        assert metric.name == "lddt"
        assert metric.objectives == {"codebook"}
        assert metric.requires_decoder is True
        assert metric.requires_coords is True

    def test_lddt_metric_identical_structures(self):
        """Test lDDT is 1.0 for identical structures."""
        metric = LDDTMetric()
        cfg = _make_cfg()

        B, L = 2, 8
        coords = _stable_ncac_coords(B, L)
        tokens = torch.full((B, L), 4, dtype=torch.long)

        outputs = {"pred_coords": coords.clone()}
        metric.update(outputs, tokens, torch.zeros_like(tokens), coords, cfg)
        result = metric.compute()

        assert abs(result["lddt"] - 1.0) < 0.01

    def test_lddt_metric_skips_missing_coords(self):
        """Test metric skips when coords are missing."""
        metric = LDDTMetric()
        cfg = _make_cfg()

        outputs = {"pred_coords": None}
        tokens = torch.full((1, 8), 4, dtype=torch.long)

        metric.update(outputs, tokens, torch.zeros_like(tokens), None, cfg)
        result = metric.compute()

        assert "lddt" not in result
        assert result["lddt/num_skipped"] == 1

    def test_lddt_metric_accumulates(self):
        """Test lDDT accumulates across batches."""
        metric = LDDTMetric()
        cfg = _make_cfg()

        B, L = 1, 8
        coords = _stable_ncac_coords(B, L)
        tokens = torch.full((B, L), 4, dtype=torch.long)

        # Two batches with identical structures
        metric.update(
            {"pred_coords": coords.clone()},
            tokens,
            torch.zeros_like(tokens),
            coords,
            cfg,
        )
        metric.update(
            {"pred_coords": coords.clone()},
            tokens,
            torch.zeros_like(tokens),
            coords,
            cfg,
        )
        result = metric.compute()

        assert abs(result["lddt"] - 1.0) < 0.01


class TestTMScoreMetric:
    """Tests for TMScoreMetric."""

    def test_tm_metric_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = TMScoreMetric()
        assert metric.name == "tm"
        assert metric.objectives == {"codebook"}
        assert metric.requires_decoder is True
        assert metric.requires_coords is True

    def test_tm_metric_identical_structures(self):
        """Test TM-score is 1.0 for identical structures."""
        metric = TMScoreMetric()
        cfg = _make_cfg()

        B, L = 2, 10
        coords = _stable_ncac_coords(B, L)
        tokens = torch.full((B, L), 4, dtype=torch.long)

        outputs = {"pred_coords": coords.clone()}
        metric.update(outputs, tokens, torch.zeros_like(tokens), coords, cfg)
        result = metric.compute()

        assert abs(result["tm"] - 1.0) < 0.01


class TestRMSDMetric:
    """Tests for RMSDMetric."""

    def test_rmsd_metric_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = RMSDMetric()
        assert metric.name == "rmsd"
        assert metric.objectives == {"codebook"}
        assert metric.requires_decoder is True
        assert metric.requires_coords is True

    def test_rmsd_metric_identical_structures(self):
        """Test RMSD is 0 for identical structures."""
        metric = RMSDMetric()
        cfg = _make_cfg()

        B, L = 2, 10
        coords = _stable_ncac_coords(B, L)
        tokens = torch.full((B, L), 4, dtype=torch.long)

        outputs = {"pred_coords": coords.clone()}
        metric.update(outputs, tokens, torch.zeros_like(tokens), coords, cfg)
        result = metric.compute()

        assert result["rmsd"] < 0.01

    def test_rmsd_metric_config_options(self):
        """Test RMSD metric accepts config options."""
        metric = RMSDMetric(align=False, atom_set="backbone")
        assert metric.align is False
        assert metric.atom_set == "backbone"


class TestFAPEMetric:
    """Tests for FAPEMetric."""

    def test_fape_metric_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = FAPEMetric()
        assert metric.name == "fape_loss"
        assert metric.objectives == {"codebook"}
        assert metric.requires_decoder is True
        assert metric.requires_coords is True

    def test_fape_metric_identical_structures(self):
        """Test FAPE is 0 for identical structures."""
        metric = FAPEMetric()
        cfg = _make_cfg()

        B, L = 2, 10
        coords = _stable_ncac_coords(B, L)
        tokens = torch.full((B, L), 4, dtype=torch.long)

        outputs = {"pred_coords": coords.clone()}
        metric.update(outputs, tokens, torch.zeros_like(tokens), coords, cfg)
        result = metric.compute()

        assert result["fape_loss"] < 0.01

    def test_fape_metric_config_options(self):
        """Test FAPE metric accepts config options."""
        metric = FAPEMetric(clamp=5.0, length_scale=5.0)
        assert metric.clamp == 5.0
        assert metric.length_scale == 5.0


class TestPredNaNFracMetric:
    """Tests for PredNaNFracMetric."""

    def test_pred_nan_frac_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = PredNaNFracMetric()
        assert metric.name == "pred_nan_frac"
        assert metric.objectives == {"codebook"}
        assert metric.requires_decoder is True
        assert metric.requires_coords is False  # Only needs pred_coords

    def test_pred_nan_frac_no_nans(self):
        """Test NaN fraction is 0 when no NaNs present."""
        metric = PredNaNFracMetric()
        cfg = _make_cfg()

        B, L = 2, 10
        pred_coords = torch.randn(B, L, 3, 3)
        tokens = torch.full((B, L), 4, dtype=torch.long)

        outputs = {"pred_coords": pred_coords}
        metric.update(outputs, tokens, torch.zeros_like(tokens), None, cfg)
        result = metric.compute()

        assert result["pred_nan_frac"] == 0.0

    def test_pred_nan_frac_all_nans(self):
        """Test NaN fraction is 1 when all NaNs."""
        metric = PredNaNFracMetric()
        cfg = _make_cfg()

        B, L = 2, 10
        pred_coords = torch.full((B, L, 3, 3), float("nan"))
        tokens = torch.full((B, L), 4, dtype=torch.long)

        outputs = {"pred_coords": pred_coords}
        metric.update(outputs, tokens, torch.zeros_like(tokens), None, cfg)
        result = metric.compute()

        assert result["pred_nan_frac"] == 1.0

    def test_pred_nan_frac_partial_nans(self):
        """Test NaN fraction with partial NaNs."""
        metric = PredNaNFracMetric()
        cfg = _make_cfg()

        B, L = 1, 10
        pred_coords = torch.randn(B, L, 3, 3)
        # Set half to NaN
        pred_coords[:, : L // 2, :, :] = float("nan")
        tokens = torch.full((B, L), 4, dtype=torch.long)

        outputs = {"pred_coords": pred_coords}
        metric.update(outputs, tokens, torch.zeros_like(tokens), None, cfg)
        result = metric.compute()

        assert abs(result["pred_nan_frac"] - 0.5) < 0.01


def test_structure_scores_are_protein_weighted_and_report_missing():
    import pytest

    true = _stable_ncac_coords(5, 6)
    pred = true + torch.arange(5)[:, None, None, None] * torch.randn_like(true) * 0.1
    true[-1] = float("nan")
    tokens = torch.full((5, 6), 4)
    for metric_type in (LDDTMetric, TMScoreMetric, RMSDMetric, FAPEMetric):
        results = []
        for size in (1, 2, 3):
            metric = metric_type()
            for start in range(0, 5, size):
                metric.update(
                    {"pred_coords": pred[start : start + size]},
                    tokens[start : start + size],
                    None,
                    true[start : start + size],
                    _make_cfg(),
                )
            result = metric.compute()
            assert result[f"{metric.name}/num_valid"] == 4
            assert result[f"{metric.name}/num_skipped"] == 1
            results.append(result[metric.name])
        assert results == pytest.approx([results[0]] * 3)


def test_bad_structure_predictions_raise_and_count_failure():
    import pytest

    true = _stable_ncac_coords(1, 6)
    pred = torch.full_like(true, float("nan"))
    tokens = torch.full((1, 6), 4)
    for metric_type in (LDDTMetric, TMScoreMetric, RMSDMetric, FAPEMetric):
        metric = metric_type()
        with pytest.raises(ValueError, match="Nonfinite predictions"):
            metric.update({"pred_coords": pred}, tokens, None, true, _make_cfg())
        assert metric.num_failed == 1
        assert metric.name not in metric.compute()


def test_partial_structure_failure_cannot_report_favorable_subset():
    import pytest

    true = _stable_ncac_coords(1, 6)
    tokens = torch.full((1, 6), 4)
    metric = RMSDMetric()
    metric.update({"pred_coords": true}, tokens, None, true, _make_cfg())
    with pytest.raises(ValueError):
        metric.update(
            {"pred_coords": torch.full_like(true, float("nan"))},
            tokens,
            None,
            true,
            _make_cfg(),
        )
    assert "rmsd" not in metric.compute()
