"""Base protocol and abstract class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

import torch
from omegaconf import DictConfig


@runtime_checkable
class Metric(Protocol):
    """Protocol defining the interface for evaluation metrics.

    All metrics must implement this protocol to be used with the evaluation system.

    Attributes:
        name: Unique identifier for the metric (used in logging).
        objectives: Set of training objectives this metric applies to.
            None means the metric applies to all objectives.
        requires_decoder: Whether this metric requires a decoder model.
        requires_coords: Whether this metric requires coordinate data.
    """

    name: ClassVar[str]
    objectives: ClassVar[set[str] | None]
    requires_decoder: ClassVar[bool]
    requires_coords: ClassVar[bool]

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate metric values from a single batch.

        Args:
            outputs: Model outputs dictionary containing at minimum 'logits' and 'loss'.
                May also contain 'pred_coords' if decoder was used.
            tokens: Input token IDs [B, L].
            labels: Target labels [B, L].
            coords: Ground truth coordinates [B, L, 3, 3] or None.
            cfg: Full configuration object.
        """
        ...

    def compute(self) -> dict[str, float]:
        """Compute final metric value(s) from accumulated state.

        Returns:
            Dictionary mapping metric names to float values.
        """
        ...

    def reset(self) -> None:
        """Reset accumulated state for a new evaluation run."""
        ...

    def state_tensors(self) -> list[torch.Tensor]:
        """Return internal state as tensors for distributed aggregation.

        Returns:
            List of tensors representing the metric's accumulated state.
        """
        ...

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Restore state from gathered tensors (for distributed training).

        Args:
            tensors: List of tensors as returned by state_tensors(),
                potentially aggregated across processes.
        """
        ...


class MetricBase(ABC):
    """Abstract base class for metrics with default implementations.

    Provides default implementations for state_tensors() and load_state_tensors()
    that work for simple scalar accumulators. Subclasses should override these
    if they have more complex state.

    Class Attributes:
        name: Unique identifier for the metric.
        objectives: Set of training objectives this metric applies to.
            None means the metric applies to all objectives.
        requires_decoder: Whether this metric requires a decoder model.
        requires_coords: Whether this metric requires coordinate data.
    """

    name: ClassVar[str] = ""
    objectives: ClassVar[set[str] | None] = None
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = False

    def __init__(self, **kwargs):
        """Initialize metric with optional configuration.

        Args:
            **kwargs: Metric-specific configuration parameters.
        """
        pass

    @abstractmethod
    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate metric values from a single batch."""
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute final metric value(s) from accumulated state."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state for a new evaluation run."""
        ...

    def state_tensors(self) -> list[torch.Tensor]:
        """Return internal state as tensors for distributed aggregation.

        Default implementation returns an empty list. Override this if the
        metric needs to aggregate state across distributed processes.

        Returns:
            List of tensors representing the metric's accumulated state.
        """
        return []

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Restore state from gathered tensors.

        Default implementation does nothing. Override this if the metric
        needs to aggregate state across distributed processes.

        Args:
            tensors: List of tensors as returned by state_tensors().
        """
        pass
