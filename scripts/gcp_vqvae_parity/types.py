from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

# Pandas/PyArrow persist qualification counts and shape dimensions as signed int64.
MAX_STORAGE_INTEGER = 2**63 - 1


class RunStatus(StrEnum):
    FULLY_QUALIFIED = "fully_qualified"
    CORE_QUALIFIED = "core_qualified"
    NOT_QUALIFIED = "not_qualified"


@dataclass(frozen=True)
class TensorComparison:
    name: str
    shape: tuple[int, ...]
    compared: int
    mismatched: int
    exact: bool
    passed: bool
    mask_equal: bool = True
    finite_pattern_equal: bool = True
    within_tolerance: bool = False
    max_abs: float = 0.0
    max_rel: float = 0.0
    p50_abs: float = 0.0
    p95_abs: float = 0.0
    p99_abs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_run(*, weights_pass: bool, core_pass: bool, public_pass: bool) -> RunStatus:
    if not weights_pass or not core_pass:
        return RunStatus.NOT_QUALIFIED
    if public_pass:
        return RunStatus.FULLY_QUALIFIED
    return RunStatus.CORE_QUALIFIED
