from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .types import RunStatus, TensorComparison, classify_run


@dataclass(frozen=True)
class RunSummary:
    status: RunStatus
    complete: bool
    weights_pass: bool
    core_pass: bool
    public_pass: bool
    input_count: int
    completed_count: int
    core_failures: tuple[str, ...]
    public_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


def summarize_run(
    *,
    weight_pass: bool,
    core_comparisons: Sequence[TensorComparison],
    public_comparisons: Sequence[TensorComparison],
    input_count: int,
    completed_count: int,
) -> RunSummary:
    if input_count < 0 or completed_count < 0 or completed_count > input_count:
        raise ValueError(
            "Run counts must satisfy 0 <= completed_count <= input_count, "
            f"got {completed_count}/{input_count}"
        )
    complete = input_count == completed_count
    core_failures = tuple(item.name for item in core_comparisons if not item.passed)
    public_failures = tuple(item.name for item in public_comparisons if not item.passed)
    core_pass = complete and bool(core_comparisons) and not core_failures
    public_pass = complete and bool(public_comparisons) and not public_failures
    status = classify_run(
        weights_pass=weight_pass and complete,
        core_pass=core_pass,
        public_pass=public_pass,
    )
    return RunSummary(
        status=status,
        complete=complete,
        weights_pass=weight_pass,
        core_pass=core_pass,
        public_pass=public_pass,
        input_count=input_count,
        completed_count=completed_count,
        core_failures=core_failures,
        public_failures=public_failures,
    )


def render_markdown(summary: RunSummary) -> str:
    return "\n".join(
        [
            "# GCP-VQVAE Encoder Parity Qualification",
            "",
            f"- Status: `{summary.status.value}`",
            f"- Inputs completed: `{summary.completed_count}/{summary.input_count}`",
            f"- Weight parity: `{'PASS' if summary.weights_pass else 'FAIL'}`",
            f"- Shared-input core parity: `{'PASS' if summary.core_pass else 'FAIL'}`",
            f"- Public pipeline parity: `{'PASS' if summary.public_pass else 'FAIL'}`",
            f"- Core failures: `{list(summary.core_failures)}`",
            f"- Public failures: `{list(summary.public_failures)}`",
            "",
        ]
    )
