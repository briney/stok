"""MDLM-specific evaluation metrics for sequence and structure tracks."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from stok.eval.base import MetricBase
from stok.eval.registry import register_metric


@register_metric("mdlm_seq_accuracy")
class MDLMSeqAccuracy(MetricBase):
    """Accuracy of sequence predictions at masked positions.

    Computes the fraction of correctly predicted tokens at positions
    where the sequence track was masked during diffusion.
    """

    name: ClassVar[str] = "mdlm_seq_acc"
    objectives: ClassVar[set[str] | None] = {"mdlm"}
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = False

    def __init__(self, ignore_index: int = -100, **kwargs):
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
        """Accumulate accuracy from masked sequence positions."""
        seq_logits = outputs.get("seq_logits")
        if seq_logits is None:
            return
        seq_targets = outputs.get("seq_targets", labels)
        with torch.no_grad():
            preds = seq_logits.argmax(dim=-1)  # [B, L]
            valid = seq_targets != self.ignore_index
            if valid.sum().item() > 0:
                self._correct += (preds[valid] == seq_targets[valid]).sum().item()
                self._total += valid.sum().item()

    def compute(self) -> dict[str, float]:
        if self._total == 0:
            return {self.name: float("nan")}
        return {self.name: self._correct / self._total}

    def reset(self) -> None:
        self._correct = 0.0
        self._total = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        return [torch.tensor([self._correct, self._total])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        if tensors:
            t = tensors[0]
            self._correct = float(t[0].item())
            self._total = float(t[1].item())


@register_metric("mdlm_struct_accuracy")
class MDLMStructAccuracy(MetricBase):
    """Accuracy of structure predictions at masked positions.

    Computes the fraction of correctly predicted structure tokens at
    positions where the structure track was masked during diffusion.
    """

    name: ClassVar[str] = "mdlm_struct_acc"
    objectives: ClassVar[set[str] | None] = {"mdlm"}
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = False

    def __init__(self, ignore_index: int = -100, **kwargs):
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
        """Accumulate accuracy from masked structure positions."""
        struct_logits = outputs.get("struct_logits")
        struct_targets = outputs.get("struct_targets")
        if struct_logits is None or struct_targets is None:
            return
        with torch.no_grad():
            preds = struct_logits.argmax(dim=-1)  # [B, L]
            valid = struct_targets != self.ignore_index
            if valid.sum().item() > 0:
                self._correct += (preds[valid] == struct_targets[valid]).sum().item()
                self._total += valid.sum().item()

    def compute(self) -> dict[str, float]:
        if self._total == 0:
            return {self.name: float("nan")}
        return {self.name: self._correct / self._total}

    def reset(self) -> None:
        self._correct = 0.0
        self._total = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        return [torch.tensor([self._correct, self._total])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        if tensors:
            t = tensors[0]
            self._correct = float(t[0].item())
            self._total = float(t[1].item())


@register_metric("mdlm_seq_perplexity")
class MDLMSeqPerplexity(MetricBase):
    """Perplexity of sequence predictions at masked positions.

    Computes exp(CE loss) at masked sequence positions, averaged across
    all evaluation batches.
    """

    name: ClassVar[str] = "mdlm_seq_ppl"
    objectives: ClassVar[set[str] | None] = {"mdlm"}
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = False

    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self._loss_sum: float = 0.0
        self._count: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate CE loss at masked sequence positions."""
        seq_logits = outputs.get("seq_logits")
        seq_targets = outputs.get("seq_targets", labels)
        if seq_logits is None:
            return
        with torch.no_grad():
            valid = seq_targets != self.ignore_index
            if valid.sum().item() == 0:
                return
            V = seq_logits.size(-1)
            loss = F.cross_entropy(
                seq_logits.reshape(-1, V),
                seq_targets.reshape(-1),
                ignore_index=self.ignore_index,
                reduction="mean",
            )
            self._loss_sum += float(loss.item())
            self._count += 1.0

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {self.name: float("nan")}
        avg_loss = self._loss_sum / self._count
        ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
        return {self.name: ppl}

    def reset(self) -> None:
        self._loss_sum = 0.0
        self._count = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        return [torch.tensor([self._loss_sum, self._count])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        if tensors:
            t = tensors[0]
            self._loss_sum = float(t[0].item())
            self._count = float(t[1].item())


@register_metric("mdlm_struct_perplexity")
class MDLMStructPerplexity(MetricBase):
    """Perplexity of structure predictions at masked positions.

    Computes exp(CE loss) at masked structure positions, averaged across
    all evaluation batches.
    """

    name: ClassVar[str] = "mdlm_struct_ppl"
    objectives: ClassVar[set[str] | None] = {"mdlm"}
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = False

    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self._loss_sum: float = 0.0
        self._count: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate CE loss at masked structure positions."""
        struct_logits = outputs.get("struct_logits")
        struct_targets = outputs.get("struct_targets")
        if struct_logits is None or struct_targets is None:
            return
        with torch.no_grad():
            valid = struct_targets != self.ignore_index
            if valid.sum().item() == 0:
                return
            C = struct_logits.size(-1)
            loss = F.cross_entropy(
                struct_logits.reshape(-1, C),
                struct_targets.reshape(-1),
                ignore_index=self.ignore_index,
                reduction="mean",
            )
            self._loss_sum += float(loss.item())
            self._count += 1.0

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {self.name: float("nan")}
        avg_loss = self._loss_sum / self._count
        ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
        return {self.name: ppl}

    def reset(self) -> None:
        self._loss_sum = 0.0
        self._count = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        return [torch.tensor([self._loss_sum, self._count])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        if tensors:
            t = tensors[0]
            self._loss_sum = float(t[0].item())
            self._count = float(t[1].item())
