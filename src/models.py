"""Data model dataclasses shared across all modules."""
from __future__ import annotations

import dataclasses
from typing import List


@dataclasses.dataclass
class CheckResult:
    check_id: str
    check_name: str
    category: str
    severity: str       # "BLOCKER", "WARNING", "INFO"
    passed: bool
    message: str
    details: dict
    recommendation: str

    def __post_init__(self) -> None:
        # Ensure passed is always a plain Python bool (not np.bool_)
        self.passed = bool(self.passed)


@dataclasses.dataclass
class AuditReport:
    dataset_name: str
    timestamp: str
    total_checks: int
    passed: int
    failed: int
    blockers: int
    warnings: int
    infos: int
    checks: List[CheckResult]
    summary: str
