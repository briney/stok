"""Per-protein structure scores; None means the protein is not evaluable."""

import torch
from stok.utils.metrics import lddt_ca, rmsd, tm_score


def _target_mask(coords, mask):
    return mask & torch.isfinite(coords[:, :, 1]).all(-1)


def _can_align(coords, mask):
    points = coords[0, mask[0], 1].float()
    return (
        len(points) >= 3 and int(torch.linalg.matrix_rank(points - points.mean(0))) >= 2
    )


def lddt_score(pred, true, mask):
    valid = _target_mask(true, mask)
    points = true[0, valid[0], 1].float()
    if len(points) < 2:
        return None
    distances = torch.cdist(points, points)
    if not (torch.triu(distances <= 15.0, diagonal=1)).any():
        return None
    return lddt_ca(pred, true, valid)[0][0]


def rmsd_score(pred, true, mask):
    valid = _target_mask(true, mask)
    if not valid.any() or not _can_align(true, valid):
        return None
    return rmsd(pred, true, valid, align=True, atom_set="CA")[0]


def tm_score_value(pred, true, mask):
    """Kabsch-aligned C-alpha TM score, not a TM-align optimization."""
    valid = _target_mask(true, mask)
    if not _can_align(true, valid):
        return None
    return tm_score(pred, true, valid)[0][0]


# Order and names match the evaluator's geometry state rows and metric keys.
STRUCTURE_SCORES = {"lddt": lddt_score, "rmsd": rmsd_score, "tm": tm_score_value}
