"""Evaluator class for orchestrating metric computation."""

from __future__ import annotations

from typing import TYPE_CHECKING
import random

import numpy as np

import torch
import torch.nn as nn
from accelerate.utils import gather_object
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from stok.utils.masking import residue_mask_from_tokens
from stok.eval.base import Metric
from stok.eval.registry import build_metrics, resolve_eval_metrics

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
        self._capabilities: dict[str, tuple[bool, bool, dict]] = {}

        # Cache for whether attention weights are needed per eval dataset
        self._needs_attentions_cache: dict[str, bool] = {}

    def _get_metrics(self, eval_name: str | None = None) -> list[Metric]:
        """Get or build metrics for an eval dataset.

        Args:
            eval_name: Name of the eval dataset (for per-dataset overrides).

        Returns:
            List of metric instances.
        """
        cache_key = eval_name or "__default__"
        if cache_key not in self._metrics_cache:
            has_coords, has_labels, resolved = self._capabilities.get(cache_key,
                (self.has_coords, True, resolve_eval_metrics(self.cfg, eval_name, objective=self.objective)))
            metrics = build_metrics(
                cfg=self.cfg,
                objective=self.objective,
                decoder=self.decoder,
                has_coords=has_coords,
                has_labels=has_labels,
                resolved=resolved,
                eval_name=eval_name,
            )
            self._metrics_cache[cache_key] = metrics

            # Check if any metric needs attention weights (e.g., p_at_l)
            self._needs_attentions_cache[cache_key] = any(
                getattr(m, "name", "") == "p_at_l" and
                (m.use_attention or m.use_logistic_regression) for m in metrics
            )

        return self._metrics_cache[cache_key]

    def _needs_attentions(self, eval_name: str | None = None) -> bool:
        """Check if attention weights are needed for an eval dataset.

        Args:
            eval_name: Name of the eval dataset.

        Returns:
            True if any metric for this dataset needs attention weights.
        """
        cache_key = eval_name or "__default__"
        # Ensure metrics are built (which populates the cache)
        if cache_key not in self._needs_attentions_cache:
            self._get_metrics(eval_name)
        return self._needs_attentions_cache.get(cache_key, False)

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

        # Import decoding utilities
        from stok.utils.decoding import (
            decode_token_aligned_coords,
            indices_to_codes,
            sample_indices_top_p,
        )

        logits = outputs["logits"]
        pad_id = int(self.cfg.model.encoder.pad_id)
        res_mask = residue_mask_from_tokens(tokens, pad_id=pad_id,
            bos_id=int(self.cfg.model.encoder.get("bos_id", 0)),
            eos_id=int(self.cfg.model.encoder.get("eos_id", 2)))

        # Get codebook from model
        unwrapped = _unwrap_model(self.model, self.accelerator)
        codebook = unwrapped.classifier.E

        with torch.no_grad():
            if self.decode_method == "top_p":
                temperature = max(1e-8, self.decode_temperature)
                probs = torch.softmax(logits / temperature, dim=-1)
                idx = sample_indices_top_p(
                    probs, top_p=self.decode_top_p, temperature=1.0
                )
            else:  # argmax
                idx = logits.argmax(dim=-1)

            codes = indices_to_codes(codebook, idx)
            pred_coords = decode_token_aligned_coords(self.decoder, codes, res_mask)

        return pred_coords

    def _gather_metric_states(self, metrics: list[Metric]) -> None:
        if self.accelerator is None:
            return
        states = []
        for metric in metrics:
            objects = metric.state_objects()
            if objects is not None:
                size = sum(s["features"].numel() * s["features"].element_size() +
                           s["labels"].numel() * s["labels"].element_size() for s in objects)
                total = self.accelerator.gather(torch.tensor([size], device=self.accelerator.device)).sum().item()
                if total > metric.logreg_max_feature_bytes:
                    raise ValueError(f"P@L total feature bytes {total} exceed logreg_max_feature_bytes")
                metric.load_state_objects(gather_object(objects))
            tensors = metric.state_tensors()
            if tensors:
                states.append((metric, tensors))
        if not states:
            return
        flat = torch.cat([t.flatten().double() for _, tensors in states for t in tensors]).to(self.accelerator.device)
        gathered = self.accelerator.gather(flat)
        summed = gathered.reshape(-1, flat.numel()).sum(0)
        offset = 0
        for metric, tensors in states:
            restored = []
            for tensor in tensors:
                restored.append(summed[offset:offset+tensor.numel()].reshape(tensor.shape))
                offset += tensor.numel()
            metric.load_state_tensors(restored)

    def _raise_eval_errors(self, error, eval_name):
        errors = gather_object([error]) if self.accelerator else [error]
        if any(errors):
            raise RuntimeError(f"Evaluation dataset {eval_name}: {errors}")

    def evaluate(self, eval_loader: DataLoader, eval_name: str) -> dict[str, float]:
        """Isolate all evaluation randomness, including DataLoader iteration.

        Stochastic decoding is repeatable at a fixed configuration; changing
        batching can change top-p draws. MLM masking is per-sample invariant.
        """
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        seed = int(self.cfg.train.get("eval", {}).get("seed", self.cfg.train.get("seed", 1337)))
        try:
            with torch.random.fork_rng(devices=devices):
                torch.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed_all(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                return self._evaluate(eval_loader, eval_name)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _evaluate(self, eval_loader: DataLoader, eval_name: str) -> dict[str, float]:
        dataset = eval_loader.dataset
        has_labels = self.objective == "mlm" or getattr(dataset, "has_labels", True)
        self._capabilities[eval_name] = (getattr(dataset, "has_coords", self.has_coords), has_labels,
            getattr(eval_loader, "metric_configs", resolve_eval_metrics(self.cfg, eval_name, objective=self.objective)))
        self._metrics_cache.pop(eval_name, None)
        metrics = self._get_metrics(eval_name)
        for metric in metrics:
            metric.reset()
        needs_decoding = any(metric.requires_decoder for metric in metrics)
        needs_attentions = self._needs_attentions(eval_name)
        n_layers = int(self.cfg.model.encoder.get("n_layers", 12))
        attention_indices = tuple(sorted({i for m in metrics if m.name == "p_at_l"
            for i in m.required_attention_layers(n_layers)}))
        incoming_training = self.model.training
        self.model.eval()
        eval_model = _unwrap_model(self.model, self.accelerator)
        ignore_index = int(self.cfg.model.classifier.get("ignore_index", -100))
        device = _get_model_device(self.model, self.accelerator)
        context = "loader/forward"
        error = None
        try:
            try:
                with torch.no_grad():
                    for batch in eval_loader:
                        context = "forward"
                        tokens, labels = (t.to(device) for t in batch[:2])
                        coords = batch[2].to(device) if len(batch) == 3 else None
                        outputs = eval_model(tokens=tokens, labels=labels if has_labels else None,
                            ignore_index=ignore_index, output_attentions=needs_attentions,
                            **({"attention_layer_indices": attention_indices} if needs_attentions else {}))
                        outputs["residue_mask"] = residue_mask_from_tokens(tokens,
                            pad_id=int(self.cfg.model.encoder.pad_id),
                            bos_id=int(self.cfg.model.encoder.get("bos_id", 0)),
                            eos_id=int(self.cfg.model.encoder.get("eos_id", 2)))
                        if needs_decoding and "pred_coords" not in outputs:
                            outputs["pred_coords"] = self._decode_predictions(outputs, tokens)
                        for metric in metrics:
                            context = metric.name
                            failed_before = metric.num_failed
                            try:
                                metric.update(outputs, tokens, labels if has_labels else None, coords, self.cfg)
                            except Exception:
                                if metric.num_failed == failed_before:
                                    metric.num_failed += 1
                                raise
            except Exception as exc:
                error = f"{context}: {type(exc).__name__}: {exc}"
            self._raise_eval_errors(error, eval_name)
            self._gather_metric_states(metrics)
            results = {}
            error = None
            try:
                for metric in metrics:
                    context = metric.name
                    computed = metric.compute()
                    results.update(computed)
                    if getattr(metric, "explicit", False) and metric.num_valid == 0:
                        raise ValueError(f"{metric.name} unavailable (num_valid=0, "
                            f"num_skipped={metric.num_skipped}, num_failed={metric.num_failed})")
            except Exception as exc:
                error = f"{context}: {type(exc).__name__}: {exc}"
            self._raise_eval_errors(error, eval_name)
            return results
        finally:
            self.model.train(incoming_training)

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
        self._needs_attentions_cache.clear()
