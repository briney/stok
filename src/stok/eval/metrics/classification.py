"""Classification-based evaluation metrics."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from omegaconf import DictConfig

from stok.eval.base import MetricBase
from stok.eval.registry import register_metric


@register_metric("accuracy")
class AccuracyMetric(MetricBase):
    """Token-level classification accuracy metric.

    Computes the fraction of correctly predicted tokens, ignoring positions
    marked with the ignore_index.
    """

    name: ClassVar[str] = "acc"
    objectives: ClassVar[set[str] | None] = {"codebook"}
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = False

    def __init__(self, ignore_index: int = -100, **kwargs):
        """Initialize accuracy metric.

        Args:
            ignore_index: Label index to ignore in accuracy computation.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self._correct: float = 0.0
        self._total: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate accuracy from a batch."""
        logits = outputs["logits"]
        ignore_idx = cfg.model.classifier.get("ignore_index", self.ignore_index)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != ignore_idx
            if mask.sum().item() > 0:
                self._correct += (preds[mask] == labels[mask]).sum().item()
                self._total += mask.sum().item()

    def compute(self) -> dict[str, float]:
        """Compute accuracy from accumulated values."""
        acc = self._correct / max(1.0, self._total)
        return {self.name: acc}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._correct = 0.0
        self._total = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._correct, self._total])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._correct = float(t[0].item())
            self._total = float(t[1].item())


@register_metric("masked_accuracy")
class MaskedAccuracyMetric(MetricBase):
    """Masked token accuracy for MLM pre-training.

    Identical computation to AccuracyMetric but restricted to MLM objective
    and uses a different metric name for clarity.
    """

    name: ClassVar[str] = "mask_acc"
    objectives: ClassVar[set[str] | None] = {"mlm"}
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = False

    def __init__(self, ignore_index: int = -100, **kwargs):
        """Initialize masked accuracy metric.

        Args:
            ignore_index: Label index to ignore in accuracy computation.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self._correct: float = 0.0
        self._total: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate masked accuracy from a batch."""
        logits = outputs["logits"]
        ignore_idx = cfg.model.classifier.get("ignore_index", self.ignore_index)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != ignore_idx
            if mask.sum().item() > 0:
                self._correct += (preds[mask] == labels[mask]).sum().item()
                self._total += mask.sum().item()

    def compute(self) -> dict[str, float]:
        """Compute masked accuracy from accumulated values."""
        acc = self._correct / max(1.0, self._total)
        return {self.name: acc}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._correct = 0.0
        self._total = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._correct, self._total])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._correct = float(t[0].item())
            self._total = float(t[1].item())


@register_metric("perplexity")
class PerplexityMetric(MetricBase):
    """Perplexity metric computed as exp(cross-entropy loss).

    Applies to both MLM and codebook objectives.
    """

    name: ClassVar[str] = "ppl"
    objectives: ClassVar[set[str] | None] = None  # Works for all objectives
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = False

    def __init__(self, **kwargs):
        """Initialize perplexity metric.

        Args:
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self._loss_sum: float = 0.0
        self._batch_count: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate loss for perplexity computation."""
        # Use classification_loss if available (more specific), else total loss
        loss_tensor = outputs.get("classification_loss", outputs.get("loss"))
        if loss_tensor is not None:
            with torch.no_grad():
                self._loss_sum += float(loss_tensor.item())
                self._batch_count += 1.0

    def compute(self) -> dict[str, float]:
        """Compute perplexity from accumulated loss."""
        if self._batch_count > 0:
            avg_loss = self._loss_sum / self._batch_count
            ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
        else:
            ppl = float("inf")
        return {self.name: ppl, "cls_loss": self._loss_sum / max(1.0, self._batch_count)}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._loss_sum = 0.0
        self._batch_count = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._loss_sum, self._batch_count])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._loss_sum = float(t[0].item())
            self._batch_count = float(t[1].item())

