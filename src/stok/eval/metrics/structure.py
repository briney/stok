"""Structure-based evaluation metrics."""

from __future__ import annotations

from typing import ClassVar

import torch
from omegaconf import DictConfig

from stok.eval.base import MetricBase
from stok.eval.registry import register_metric
from stok.utils.losses import fape_loss
from stok.utils.metrics import lddt_ca, rmsd, tm_score


@register_metric("lddt")
class LDDTMetric(MetricBase):
    """Local Distance Difference Test (lDDT) metric.

    Computes the lDDT-Ca score between predicted and ground truth structures.
    This is a superposition-free metric that measures local structural similarity.
    """

    name: ClassVar[str] = "lddt"
    objectives: ClassVar[set[str] | None] = {"codebook"}
    requires_decoder: ClassVar[bool] = True
    requires_coords: ClassVar[bool] = True

    def __init__(self, **kwargs):
        """Initialize lDDT metric.

        Args:
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self._lddt_sum: float = 0.0
        self._count: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate lDDT from a batch."""
        pred_coords = outputs.get("pred_coords")
        if pred_coords is None or coords is None:
            return

        with torch.no_grad():
            try:
                pad_id = cfg.model.encoder.pad_id
                res_mask = tokens != pad_id
                lddt_b, _ = lddt_ca(pred_coords, coords, residue_mask=res_mask)
                self._lddt_sum += float(lddt_b.mean().item())
                self._count += 1.0
            except Exception:
                pass  # Skip batch on error

    def compute(self) -> dict[str, float]:
        """Compute average lDDT."""
        return {self.name: self._lddt_sum / max(1.0, self._count)}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._lddt_sum = 0.0
        self._count = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._lddt_sum, self._count])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._lddt_sum = float(t[0].item())
            self._count = float(t[1].item())


@register_metric("tm_score")
class TMScoreMetric(MetricBase):
    """Template Modeling (TM) score metric.

    Computes the TM-score between predicted and ground truth structures.
    This is a length-normalized structural similarity metric.
    """

    name: ClassVar[str] = "tm_score"
    objectives: ClassVar[set[str] | None] = {"codebook"}
    requires_decoder: ClassVar[bool] = True
    requires_coords: ClassVar[bool] = True

    def __init__(self, **kwargs):
        """Initialize TM-score metric.

        Args:
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self._tm_sum: float = 0.0
        self._count: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate TM-score from a batch."""
        pred_coords = outputs.get("pred_coords")
        if pred_coords is None or coords is None:
            return

        with torch.no_grad():
            try:
                pad_id = cfg.model.encoder.pad_id
                res_mask = tokens != pad_id
                tm_b, _ = tm_score(pred_coords, coords, residue_mask=res_mask)
                self._tm_sum += float(tm_b.mean().item())
                self._count += 1.0
            except Exception:
                pass  # Skip batch on error

    def compute(self) -> dict[str, float]:
        """Compute average TM-score."""
        return {self.name: self._tm_sum / max(1.0, self._count)}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._tm_sum = 0.0
        self._count = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._tm_sum, self._count])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._tm_sum = float(t[0].item())
            self._count = float(t[1].item())


@register_metric("rmsd")
class RMSDMetric(MetricBase):
    """Root Mean Square Deviation (RMSD) metric.

    Computes the RMSD in Angstroms between predicted and ground truth structures
    after optimal superposition (Kabsch alignment).
    """

    name: ClassVar[str] = "rmsd"
    objectives: ClassVar[set[str] | None] = {"codebook"}
    requires_decoder: ClassVar[bool] = True
    requires_coords: ClassVar[bool] = True

    def __init__(self, align: bool = True, atom_set: str = "CA", **kwargs):
        """Initialize RMSD metric.

        Args:
            align: Whether to align structures before computing RMSD.
            atom_set: Atoms to use ("CA" or "backbone").
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self.align = align
        self.atom_set = atom_set
        self._rmsd_sum: float = 0.0
        self._count: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate RMSD from a batch."""
        pred_coords = outputs.get("pred_coords")
        if pred_coords is None or coords is None:
            return

        with torch.no_grad():
            try:
                pad_id = cfg.model.encoder.pad_id
                res_mask = tokens != pad_id
                rmsd_b = rmsd(
                    pred_coords,
                    coords,
                    residue_mask=res_mask,
                    align=self.align,
                    atom_set=self.atom_set,
                )
                self._rmsd_sum += float(rmsd_b.mean().item())
                self._count += 1.0
            except Exception:
                pass  # Skip batch on error

    def compute(self) -> dict[str, float]:
        """Compute average RMSD."""
        return {self.name: self._rmsd_sum / max(1.0, self._count)}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._rmsd_sum = 0.0
        self._count = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._rmsd_sum, self._count])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._rmsd_sum = float(t[0].item())
            self._count = float(t[1].item())


@register_metric("fape")
class FAPEMetric(MetricBase):
    """Frame-Aligned Point Error (FAPE) metric.

    Computes the FAPE loss between predicted and ground truth structures.
    This is used in AlphaFold2 and measures structure quality in local frames.
    """

    name: ClassVar[str] = "fape"
    objectives: ClassVar[set[str] | None] = {"codebook"}
    requires_decoder: ClassVar[bool] = True
    requires_coords: ClassVar[bool] = True

    def __init__(self, clamp: float = 10.0, length_scale: float = 10.0, **kwargs):
        """Initialize FAPE metric.

        Args:
            clamp: Maximum error value to clamp to.
            length_scale: Length scale for normalization.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self.clamp = clamp
        self.length_scale = length_scale
        self._fape_sum: float = 0.0
        self._count: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate FAPE from a batch."""
        pred_coords = outputs.get("pred_coords")
        if pred_coords is None or coords is None:
            return

        with torch.no_grad():
            try:
                pad_id = cfg.model.encoder.pad_id
                res_mask = tokens != pad_id
                fape = fape_loss(
                    pred_coords,
                    coords,
                    residue_mask=res_mask,
                    clamp=self.clamp,
                    length_scale=self.length_scale,
                )
                self._fape_sum += float(fape.item())
                self._count += 1.0
            except Exception:
                pass  # Skip batch on error

    def compute(self) -> dict[str, float]:
        """Compute average FAPE."""
        return {self.name: self._fape_sum / max(1.0, self._count)}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._fape_sum = 0.0
        self._count = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._fape_sum, self._count])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._fape_sum = float(t[0].item())
            self._count = float(t[1].item())


@register_metric("pred_nan_frac")
class PredNaNFracMetric(MetricBase):
    """Predicted NaN fraction diagnostic metric.

    Tracks the fraction of predicted coordinates that are NaN, which can
    indicate training instability or issues with the decoder.
    """

    name: ClassVar[str] = "pred_nan_frac"
    objectives: ClassVar[set[str] | None] = {"codebook"}
    requires_decoder: ClassVar[bool] = True
    requires_coords: ClassVar[bool] = False  # Only needs predicted coords

    def __init__(self, **kwargs):
        """Initialize predicted NaN fraction metric.

        Args:
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self._nan_frac_sum: float = 0.0
        self._count: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate NaN fraction from a batch."""
        pred_coords = outputs.get("pred_coords")
        if pred_coords is None:
            return

        with torch.no_grad():
            nan_frac = torch.isnan(pred_coords).float().mean().item()
            self._nan_frac_sum += float(nan_frac)
            self._count += 1.0

    def compute(self) -> dict[str, float]:
        """Compute average NaN fraction."""
        return {self.name: self._nan_frac_sum / max(1.0, self._count)}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._nan_frac_sum = 0.0
        self._count = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._nan_frac_sum, self._count])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._nan_frac_sum = float(t[0].item())
            self._count = float(t[1].item())
