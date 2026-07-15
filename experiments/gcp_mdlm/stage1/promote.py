"""Aggregate Stage 1 arms and emit a machine-checked promotion verdict."""

from __future__ import annotations

import pandas as pd

from .metrics import paired_bootstrap_ci


def _arm_summary(table: pd.DataFrame) -> dict:
    return {
        "mean_nll": float(table["mean_nll"].mean()),
        "top1": float(table["top1"].mean()),
        "top5": float(table["top5"].mean()),
    }


def build_report(tables: dict[str, pd.DataFrame], *, n_boot: int = 10000, seed: int = 0) -> dict:
    """Summarize each arm and compute the prototype-vs-independent NLL bootstrap CI."""
    arms = {name: _arm_summary(table) for name, table in tables.items()}
    proto = tables["prototype"].sort_values("sequence_id")
    indep = tables["independent"].sort_values("sequence_id")
    mean_diff, lo, hi = paired_bootstrap_ci(
        proto["mean_nll"].to_numpy(), indep["mean_nll"].to_numpy(), n_boot=n_boot, seed=seed
    )
    return {
        "arms": arms,
        "prototype_minus_independent_nll": {"mean": mean_diff, "lo": lo, "hi": hi},
    }


def assert_promotion(report: dict) -> None:
    """Assert both learned heads beat the floor; set the grounding verdict.

    Verdict (lower NLL is better, so prototype-minus-independent < 0 favors prototype):
      - grounding_wins:  CI upper bound < 0 (prototype clearly lower NLL)
      - grounding_loses: CI lower bound > 0 (prototype clearly higher NLL)
      - grounding_ties:  CI spans 0
    """
    floor = report["arms"]["frequency"]["mean_nll"]
    for name in ("independent", "prototype"):
        assert report["arms"][name]["mean_nll"] < floor, (
            f"{name} mean NLL {report['arms'][name]['mean_nll']:.4f} "
            f"does not beat frequency floor {floor:.4f}"
        )
    ci = report["prototype_minus_independent_nll"]
    if ci["hi"] < 0:
        report["verdict"] = "grounding_wins"
    elif ci["lo"] > 0:
        report["verdict"] = "grounding_loses"
    else:
        report["verdict"] = "grounding_ties"
