"""Evaluator class for orchestrating metric computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from stok.eval.base import Metric
from stok.eval.registry import build_metrics

if TYPE_CHECKING:
    pass


def _get_model_device(model: nn.Module, accelerator) -> torch.device:
    """Resolve the device to place tensors on."""
    if accelerator is not None:
        return accelerator.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _unwrap_model(model: nn.Module, accelerator) -> nn.Module:
    """Unwrap model from Accelerate/DDP if needed."""
    return accelerator.unwrap_model(model) if accelerator is not None else model


class Evaluator:
    """Orchestrates evaluation metric computation.

    The Evaluator handles:
    - Building metrics based on configuration and available resources
    - Running evaluation forward passes
    - Decoding predictions for structure metrics
    - Distributed aggregation of metric state
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        accelerator,
        decoder: nn.Module | None = None,
    ):
        """Initialize the evaluator.

        Args:
            cfg: Full configuration object.
            model: The model to evaluate.
            accelerator: Accelerate accelerator instance (or None).
            decoder: Optional decoder model for structure metrics.
        """
        self.cfg = cfg
        self.model = model
        self.accelerator = accelerator
        self.decoder = decoder

        # Determine objective and resource availability
        self.objective = str(cfg.train.get("objective", "codebook")).lower()
        self.has_coords = bool(cfg.data.get("load_coords", False))

        # Decoding configuration
        decoding_cfg = cfg.train.get("decoding", {})
        self.eval_decode_enabled = bool(decoding_cfg.get("eval_enabled", False))
        self.decode_method = str(decoding_cfg.get("eval_method", "argmax"))
        self.decode_temperature = float(decoding_cfg.get("temperature", 1.0))
        self.decode_top_p = float(decoding_cfg.get("top_p", 0.9))

        # Cache for metrics per eval dataset
        self._metrics_cache: dict[str, list[Metric]] = {}

    def _get_metrics(self, eval_name: str | None = None) -> list[Metric]:
        """Get or build metrics for an eval dataset.

        Args:
            eval_name: Name of the eval dataset (for per-dataset overrides).

        Returns:
            List of metric instances.
        """
        cache_key = eval_name or "__default__"
        if cache_key not in self._metrics_cache:
            self._metrics_cache[cache_key] = build_metrics(
                cfg=self.cfg,
                objective=self.objective,
                decoder=self.decoder,
                has_coords=self.has_coords,
                eval_name=eval_name,
            )
        return self._metrics_cache[cache_key]

    def _decode_predictions(
        self,
        outputs: dict,
        tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        """Decode model outputs to predicted coordinates.

        Args:
            outputs: Model outputs containing logits.
            tokens: Input token IDs [B, L].

        Returns:
            Predicted coordinates [B, L, 3, 3] or None.
        """
        if self.decoder is None:
            return None

        if not self.eval_decode_enabled:
            return None

        # Import decoding utilities
        from stok.utils.decoding import decode_coords, indices_to_codes, sample_indices_top_p

        logits = outputs["logits"]
        pad_id = int(self.cfg.model.encoder.pad_id)
        res_mask = tokens != pad_id

        # Get codebook from model
        unwrapped = _unwrap_model(self.model, self.accelerator)
        codebook = unwrapped.classifier.E

        with torch.no_grad():
            if self.decode_method == "top_p":
                temperature = max(1e-8, self.decode_temperature)
                probs = torch.softmax(logits / temperature, dim=-1)
                idx = sample_indices_top_p(probs, top_p=self.decode_top_p, temperature=1.0)
            else:  # argmax
                idx = logits.argmax(dim=-1)

            codes = indices_to_codes(codebook, idx)
            pred_coords = decode_coords(self.decoder, codes, res_mask)

        return pred_coords

    def _gather_metric_states(self, metrics: list[Metric]) -> None:
        """Aggregate metric states across distributed processes.

        Args:
            metrics: List of metrics to aggregate.
        """
        if self.accelerator is None:
            return

        device = self.accelerator.device

        for metric in metrics:
            state_tensors = metric.state_tensors()
            if not state_tensors:
                continue

            # Gather and sum each state tensor separately
            gathered_tensors = []
            for t in state_tensors:
                t_device = t.to(device)
                gathered = self.accelerator.gather_for_metrics(t_device)

                # Sum across processes
                # gathered may be same shape as t (single process) or have extra leading dim
                if gathered.shape == t_device.shape:
                    # Single process, no aggregation needed
                    summed = gathered
                else:
                    # Multi-process: sum along the process dimension (dim=0)
                    summed = gathered.sum(dim=0)

                gathered_tensors.append(summed)

            metric.load_state_tensors(gathered_tensors)

    def evaluate(
        self,
        eval_loader: DataLoader,
        eval_name: str,
    ) -> dict[str, float]:
        """Run evaluation on a dataset.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            eval_name: Name of the evaluation dataset.

        Returns:
            Dictionary mapping metric names to values.
        """
        metrics = self._get_metrics(eval_name)

        # Reset all metrics
        for metric in metrics:
            metric.reset()

        # Check if any metrics require decoding
        needs_decoding = any(
            getattr(m, "requires_decoder", False) or getattr(m, "requires_coords", False)
            for m in metrics
        )

        self.model.eval()
        ignore_index = int(self.cfg.model.classifier.get("ignore_index", -100))

        with torch.no_grad():
            for batch in eval_loader:
                # Unpack batch
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    tokens, labels, coords = batch
                else:
                    tokens, labels = batch
                    coords = None

                # Move to device if not using accelerator
                if self.accelerator is None:
                    device = _get_model_device(self.model, self.accelerator)
                    tokens = tokens.to(device)
                    labels = labels.to(device)
                    if coords is not None:
                        coords = coords.to(device)

                # Forward pass
                outputs = self.model(
                    tokens=tokens,
                    labels=labels,
                    ignore_index=ignore_index,
                )

                # Decode predictions for structure metrics
                if needs_decoding and self.decoder is not None and "pred_coords" not in outputs:
                    pred_coords = self._decode_predictions(outputs, tokens)
                    if pred_coords is not None:
                        outputs["pred_coords"] = pred_coords

                # Update all metrics
                for metric in metrics:
                    try:
                        metric.update(outputs, tokens, labels, coords, self.cfg)
                    except Exception:
                        # Continue on metric failure (graceful degradation)
                        pass

        # Aggregate across distributed processes
        self._gather_metric_states(metrics)

        # Compute final metric values
        results: dict[str, float] = {}
        for metric in metrics:
            try:
                computed = metric.compute()
                results.update(computed)
            except Exception:
                pass

        self.model.train()
        return results

    def evaluate_all(
        self,
        eval_loaders: dict[str, DataLoader],
    ) -> dict[str, dict[str, float]]:
        """Run evaluation on all datasets.

        Args:
            eval_loaders: Dictionary mapping dataset names to DataLoaders.

        Returns:
            Dictionary mapping dataset names to their metric results.
        """
        all_results: dict[str, dict[str, float]] = {}
        for eval_name, eval_loader in eval_loaders.items():
            all_results[eval_name] = self.evaluate(eval_loader, eval_name)
        return all_results

    def clear_cache(self) -> None:
        """Clear the metrics cache (e.g., after config changes)."""
        self._metrics_cache.clear()

