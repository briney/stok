"""Protein-weighted structure scores; absent observations are not zero scores."""
import torch
from stok.eval.base import MetricBase
from stok.eval.registry import register_metric
from stok.utils.losses import fape_loss
from stok.utils.masking import residue_mask_from_tokens
from stok.utils.metrics import lddt_ca, rmsd, tm_score


class _StructureMetric(MetricBase):
    objectives = {"codebook"}
    requires_decoder = True
    requires_coords = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def update(self, outputs, tokens, labels, coords, cfg):
        pred = outputs.get("pred_coords")
        if pred is None or (self.requires_coords and coords is None):
            self.num_skipped += tokens.size(0)
            return
        mask = outputs.get("residue_mask")
        if mask is None:
            mask = residue_mask_from_tokens(tokens, pad_id=int(cfg.model.encoder.pad_id),
                bos_id=int(cfg.model.encoder.get("bos_id", 0)), eos_id=int(cfg.model.encoder.get("eos_id", 2)))
        with torch.no_grad():
            for i in range(tokens.size(0)):
                try:
                    score = self.score(pred[i:i+1], coords[i:i+1] if coords is not None else None,
                                       mask[i:i+1])
                    if score is None:
                        self.num_skipped += 1
                        continue
                    if not torch.isfinite(score):
                        raise ValueError("Nonfinite structure score on an evaluable protein")
                    self._sum += float(score)
                    self._count += 1
                    self.num_valid += 1
                except Exception:
                    self.num_failed += 1
                    raise

    def compute(self):
        result = self.diagnostics()
        if self._count and not self.num_failed:
            result[self.name] = self._sum / self._count
        return result

    def reset(self):
        self._sum = 0.
        self._count = 0
        self.reset_population()

    def state_tensors(self):
        return [torch.tensor([self._sum, self._count, *self.population_values()], dtype=torch.float64)]

    def load_state_tensors(self, tensors):
        if tensors:
            self._sum, self._count = tensors[0][:2].tolist()
            self.load_population(tensors[0], self._count)


def _target_mask(coords, mask, all_atoms=False):
    required = coords.flatten(-2) if all_atoms else coords[:, :, 1]
    return mask & torch.isfinite(required).all(-1)


def _can_align(coords, mask):
    points = coords[0, mask[0], 1].float()
    return len(points) >= 3 and int(torch.linalg.matrix_rank(points - points.mean(0))) >= 2


@register_metric("lddt")
class LDDTMetric(_StructureMetric):
    name = "lddt"

    def score(self, pred, true, mask):
        valid = _target_mask(true, mask)
        points = true[0, valid[0], 1].float()
        if len(points) < 2:
            return None
        distances = torch.cdist(points, points)
        if not (torch.triu(distances <= 15., diagonal=1)).any():
            return None
        return lddt_ca(pred, true, valid)[0][0]


@register_metric("tm_score")
class TMScoreMetric(_StructureMetric):
    """Kabsch-aligned C-alpha TM score, not a TM-align optimization."""
    name = "tm"

    def score(self, pred, true, mask):
        valid = _target_mask(true, mask)
        if not _can_align(true, valid):
            return None
        return tm_score(pred, true, valid)[0][0]


@register_metric("rmsd")
class RMSDMetric(_StructureMetric):
    name = "rmsd"

    def __init__(self, align=True, atom_set="CA", **kwargs):
        super().__init__(**kwargs)
        self.align, self.atom_set = align, atom_set

    def score(self, pred, true, mask):
        valid = _target_mask(true, mask, self.atom_set.lower() == "backbone")
        if not valid.any() or (self.align and not _can_align(true, valid)):
            return None
        return rmsd(pred, true, valid, align=self.align, atom_set=self.atom_set)[0]


@register_metric("fape")
class FAPEMetric(_StructureMetric):
    name = "fape_loss"

    def __init__(self, clamp=10., length_scale=10., **kwargs):
        super().__init__(**kwargs)
        self.clamp, self.length_scale = clamp, length_scale

    def score(self, pred, true, mask):
        valid = _target_mask(true, mask, True)
        if not valid.any():
            return None
        return fape_loss(pred, true, valid, clamp=self.clamp, length_scale=self.length_scale)


@register_metric("pred_nan_frac")
class PredNaNFracMetric(_StructureMetric):
    name = "pred_nan_frac"
    requires_coords = False

    def score(self, pred, true, mask):
        if not mask.any():
            return None
        return torch.isnan(pred[mask]).float().mean()
