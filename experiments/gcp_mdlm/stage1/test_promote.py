import pandas as pd
import pytest

from experiments.gcp_mdlm.stage1.promote import assert_promotion, build_report


def _table(nlls):
    n = len(nlls)
    return pd.DataFrame(
        {
            "sequence_id": [f"p{i}" for i in range(n)],
            "n_res": [10] * n,
            "mean_nll": nlls,
            "top1": [0.5] * n,
            "top5": [0.9] * n,
        }
    )


def test_grounding_wins_when_prototype_nll_clearly_lower():
    tables = {
        "frequency": _table([2.0, 2.1, 2.05, 2.0]),
        "independent": _table([1.0, 1.1, 1.05, 1.0]),
        "prototype": _table([0.5, 0.55, 0.52, 0.5]),
    }
    report = build_report(tables, n_boot=1000, seed=0)
    assert report["arms"]["prototype"]["mean_nll"] < report["arms"]["independent"]["mean_nll"]
    assert_promotion(report)
    assert report["verdict"] == "grounding_wins"


def test_assert_fails_when_head_does_not_beat_floor():
    tables = {
        "frequency": _table([1.0, 1.0, 1.0, 1.0]),
        "independent": _table([2.0, 2.0, 2.0, 2.0]),  # worse than floor
        "prototype": _table([0.5, 0.5, 0.5, 0.5]),
    }
    report = build_report(tables, n_boot=1000, seed=0)
    with pytest.raises(AssertionError, match="independent.*floor"):
        assert_promotion(report)
