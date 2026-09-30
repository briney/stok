"""Token-weighted classification metrics with explicit observation counts."""
import math
import torch
from stok.eval.base import MetricBase
from stok.eval.registry import register_metric
from stok.utils.losses import token_ce_loss


def _valid_labels(labels, tokens, cfg):
    if labels is None:
        return torch.zeros_like(tokens, dtype=torch.bool)
    return labels != cfg.model.classifier.get("ignore_index", -100)


@register_metric("accuracy")
class AccuracyMetric(MetricBase):
    name = "acc"
    objectives = {"codebook"}

    def __init__(self, ignore_index=-100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.reset()

    def update(self, outputs, tokens, labels, coords, cfg):
        valid = _valid_labels(labels, tokens, cfg)
        count = int(valid.sum())
        self.num_skipped += valid.numel() - count
        if not count:
            return
        logits = outputs["logits"]
        if ((labels[valid] < 0) | (labels[valid] >= logits.size(-1))).any():
            self.num_failed += count
            raise ValueError("Invalid target class ID in accuracy")
        if not torch.isfinite(logits[valid]).all():
            self.num_failed += count
            raise ValueError("Nonfinite classification predictions")
        self._correct += int((logits.argmax(-1)[valid] == labels[valid]).sum())
        self._total += count
        self.num_valid += count

    def compute(self):
        result = self.diagnostics()
        if self._total and not self.num_failed:
            result[self.name] = self._correct / self._total
        return result

    def reset(self):
        self._correct = self._total = 0.
        self.reset_population()

    def state_tensors(self):
        return [torch.tensor([self._correct, self._total, *self.population_values()], dtype=torch.float64)]

    def load_state_tensors(self, tensors):
        if tensors:
            self._correct, self._total = tensors[0][:2].tolist()
            self.load_population(tensors[0], self._total)


@register_metric("masked_accuracy")
class MaskedAccuracyMetric(AccuracyMetric):
    name = "mask_acc"
    objectives = {"mlm"}


@register_metric("perplexity")
class PerplexityMetric(MetricBase):
    name = "ppl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def update(self, outputs, tokens, labels, coords, cfg):
        valid = _valid_labels(labels, tokens, cfg)
        count = int(valid.sum())
        self.num_skipped += valid.numel() - count
        if not count:
            return
        if "logits" in outputs:
            loss_sum = token_ce_loss(outputs["logits"], labels,
                cfg.model.classifier.get("ignore_index", -100), reduction="sum")
        else:
            # Compatibility for callers supplying a mean CE instead of logits.
            loss_sum = outputs.get("classification_loss", outputs.get("loss"))
            if loss_sum is None:
                raise ValueError("Perplexity requires logits or classification loss")
            loss_sum = loss_sum * count
        if not torch.isfinite(loss_sum):
            self.num_failed += count
            raise ValueError("Nonfinite classification loss")
        self._loss_sum += float(loss_sum)
        self._token_count += count
        self.num_valid += count

    def compute(self):
        result = self.diagnostics()
        if self._token_count and not self.num_failed:
            mean = self._loss_sum / self._token_count
            result[self.name] = math.exp(mean) if mean < 709 else float("inf")
        return result

    def reset(self):
        self._loss_sum = 0.
        self._token_count = 0
        self.reset_population()

    def state_tensors(self):
        return [torch.tensor([self._loss_sum, self._token_count, *self.population_values()], dtype=torch.float64)]

    def load_state_tensors(self, tensors):
        if tensors:
            self._loss_sum, self._token_count = tensors[0][:2].tolist()
            self.load_population(tensors[0], self._token_count)
