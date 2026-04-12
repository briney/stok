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
        tokens = torch.zeros(B, L, dtype=torch.long)

        outputs = {"pred_coords": coords.clone()}
        metric.update(outputs, tokens, torch.zeros_like(tokens), coords, cfg)
        result = metric.compute()

        assert abs(result["lddt"] - 1.0) < 0.01

    def test_lddt_metric_skips_missing_coords(self):
        """Test metric skips when coords are missing."""
        metric = LDDTMetric()
        cfg = _make_cfg()

        outputs = {"pred_coords": None}
        tokens = torch.zeros(1, 8, dtype=torch.long)

        metric.update(outputs, tokens, torch.zeros_like(tokens), None, cfg)
        result = metric.compute()

        assert result["lddt"] == 0.0  # No valid updates

    def test_lddt_metric_accumulates(self):
        """Test lDDT accumulates across batches."""
        metric = LDDTMetric()
        cfg = _make_cfg()

        B, L = 1, 8
        coords = _stable_ncac_coords(B, L)
        tokens = torch.zeros(B, L, dtype=torch.long)

        # Two batches with identical structures
        metric.update({"pred_coords": coords.clone()}, tokens, torch.zeros_like(tokens), coords, cfg)
        metric.update({"pred_coords": coords.clone()}, tokens, torch.zeros_like(tokens), coords, cfg)
        result = metric.compute()

        assert abs(result["lddt"] - 1.0) < 0.01


class TestTMScoreMetric:
    """Tests for TMScoreMetric."""

    def test_tm_metric_initialization(self):
        """Test metric initializes with correct defaults."""
        metric = TMScoreMetric()
        assert metric.name == "tm_score"
        assert metric.objectives == {"codebook"}
        assert metric.requires_decoder is True
        assert metric.requires_coords is True

    def test_tm_metric_identical_structures(self):
        """Test TM-score is 1.0 for identical structures."""
        metric = TMScoreMetric()
        cfg = _make_cfg()

        B, L = 2, 10
        coords = _stable_ncac_coords(B, L)
        tokens = torch.zeros(B, L, dtype=torch.long)

        outputs = {"pred_coords": coords.clone()}
        metric.update(outputs, tokens, torch.zeros_like(tokens), coords, cfg)
        result = metric.compute()

        assert abs(result["tm_score"] - 1.0) < 0.01


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
        tokens = torch.zeros(B, L, dtype=torch.long)

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
        assert metric.name == "fape"
        assert metric.objectives == {"codebook"}
        assert metric.requires_decoder is True
        assert metric.requires_coords is True

    def test_fape_metric_identical_structures(self):
        """Test FAPE is 0 for identical structures."""
        metric = FAPEMetric()
        cfg = _make_cfg()

        B, L = 2, 10
        coords = _stable_ncac_coords(B, L)
        tokens = torch.zeros(B, L, dtype=torch.long)

        outputs = {"pred_coords": coords.clone()}
        metric.update(outputs, tokens, torch.zeros_like(tokens), coords, cfg)
        result = metric.compute()

        assert result["fape"] < 0.01

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
        tokens = torch.zeros(B, L, dtype=torch.long)

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
        tokens = torch.zeros(B, L, dtype=torch.long)

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
        tokens = torch.zeros(B, L, dtype=torch.long)

        outputs = {"pred_coords": pred_coords}
        metric.update(outputs, tokens, torch.zeros_like(tokens), None, cfg)
        result = metric.compute()

        assert abs(result["pred_nan_frac"] - 0.5) < 0.01

