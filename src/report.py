"""Report generation: JSON export and human-readable console output."""
from __future__ import annotations

import json
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AuditReport

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_GRAY = "\033[90m"
_WHITE = "\033[97m"


def _severity_color(severity: str, passed: bool) -> str:
    if passed:
        return _GREEN
    if severity == "BLOCKER":
        return _RED
    if severity == "WARNING":
        return _YELLOW
    return _CYAN


def print_human_report(report: "AuditReport") -> None:
    """Print a human-readable, color-coded audit report to stdout."""
    sep = "-" * 60

    print(f"\n{_BOLD}{_WHITE}{'=' * 60}{_RESET}")
    print(f"{_BOLD}{_WHITE}  DATASET AUDIT REPORT{_RESET}")
    print(f"{_BOLD}{_WHITE}{'=' * 60}{_RESET}")
    print(f"  Dataset  : {_BOLD}{report.dataset_name}{_RESET}")
    print(f"  Timestamp: {report.timestamp}")
    print(f"  Summary  : {report.summary}")
    print()

    # Scorecard
    blocker_color = _RED if report.blockers > 0 else _GREEN
    warn_color = _YELLOW if report.warnings > 0 else _GREEN
    print(f"  {_BOLD}Checks:{_RESET}  {report.total_checks} total  |  "
          f"{_GREEN}{report.passed} passed{_RESET}  |  "
          f"{blocker_color}{report.blockers} BLOCKERS{_RESET}  |  "
          f"{warn_color}{report.warnings} WARNINGS{_RESET}  |  "
          f"{_CYAN}{report.infos} INFO{_RESET}")
    print(f"{_GRAY}{sep}{_RESET}")

    # Group by category
    categories = ["temporal", "schema", "integrity", "statistical", "consistency"]
    category_labels = {
        "temporal": "Temporalidad",
        "schema": "Schema / Metadata",
        "integrity": "Integridad",
        "statistical": "Estadística / Anomalías",
        "consistency": "Consistencia inter-ejecución",
    }

    for cat in categories:
        cat_checks = [c for c in report.checks if c.category == cat]
        if not cat_checks:
            continue
        print(f"\n  {_BOLD}{category_labels.get(cat, cat).upper()}{_RESET}")
        for chk in cat_checks:
            color = _severity_color(chk.severity, chk.passed)
            status = "+" if chk.passed else "x"
            sev_tag = f"[{chk.severity}]" if not chk.passed else "[OK]    "
            print(f"    {color}{status} {chk.check_id:<4} {sev_tag:<10}{_RESET} {chk.message}")
            if not chk.passed and chk.recommendation:
                print(f"         {_GRAY}>> {chk.recommendation}{_RESET}")
            if not chk.passed and chk.details:
                # Print at most a few details inline
                detail_items = list(chk.details.items())[:3]
                for k, v in detail_items:
                    if isinstance(v, list) and len(v) > 5:
                        v = v[:5] + [f"... ({len(chk.details[k]) - 5} more)"]
                    print(f"         {_GRAY}{k}: {v}{_RESET}")

    print(f"\n{_GRAY}{sep}{_RESET}\n")


def export_report_json(report: "AuditReport", path: str | Path) -> None:
    """Export the AuditReport to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
