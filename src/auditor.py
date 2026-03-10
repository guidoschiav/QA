"""Main audit orchestrator."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pandas as pd

from .models import CheckResult, AuditReport
from .utils import normalize_to_long, parse_dates
from .checks import temporal, schema, integrity, statistical, consistency


def _apply_severity_overrides(results: List[CheckResult], overrides: dict) -> List[CheckResult]:
    """Apply user-defined severity overrides keyed by check_id."""
    if not overrides:
        return results
    for r in results:
        key = f"{r.check_id}_{r.check_name.lower().replace(' ', '_')}"
        if r.check_id in overrides:
            r.severity = overrides[r.check_id]
        elif key in overrides:
            r.severity = overrides[key]
    return results


def audit_dataset(
    df: pd.DataFrame,
    config: dict,
    prev_snapshot: pd.DataFrame | None = None,
) -> AuditReport:
    """
    Run the full audit pipeline on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to audit (long or wide format).
    config : dict
        Loaded configuration dict (from config_loader).
    prev_snapshot : pd.DataFrame | None
        Previous snapshot for consistency checks.

    Returns
    -------
    AuditReport
    """
    # Normalize to long format
    long_df, cfg = normalize_to_long(df, config)

    # Parse dates in-place (vectorized)
    date_col = cfg.get("date_column", "Date")
    if date_col in long_df.columns:
        long_df = long_df.copy()
        long_df[date_col] = parse_dates(long_df[date_col])

    all_results: List[CheckResult] = []

    # --- Run each check category ---
    all_results.extend(schema.run_checks(long_df, cfg))
    all_results.extend(temporal.run_checks(long_df, cfg))
    all_results.extend(integrity.run_checks(long_df, cfg))
    all_results.extend(statistical.run_checks(long_df, cfg))

    if prev_snapshot is not None:
        # Normalize snapshot too
        prev_long, _ = normalize_to_long(prev_snapshot, config)
        if date_col in prev_long.columns:
            prev_long = prev_long.copy()
            prev_long[date_col] = parse_dates(prev_long[date_col])
        all_results.extend(consistency.run_checks(long_df, cfg, prev_long))
    else:
        # Add skipped consistency checks
        all_results.extend(consistency.run_checks_no_snapshot(cfg))

    # Apply severity overrides
    overrides = cfg.get("severity_overrides", {})
    all_results = _apply_severity_overrides(all_results, overrides)

    # Compute summary stats
    passed = sum(1 for r in all_results if r.passed)
    failed = len(all_results) - passed
    blockers = sum(1 for r in all_results if not r.passed and r.severity == "BLOCKER")
    warnings = sum(1 for r in all_results if not r.passed and r.severity == "WARNING")
    infos = sum(1 for r in all_results if not r.passed and r.severity == "INFO")

    if blockers > 0:
        summary = f"FAILED — {blockers} blocker(s) detected. Dataset should not be published."
    elif warnings > 0:
        summary = f"WARNINGS — {warnings} issue(s) require review."
    elif failed > 0:
        summary = f"INFO — {failed} informational finding(s)."
    else:
        summary = "PASSED — All checks passed."

    return AuditReport(
        dataset_name=config.get("dataset_name", "unknown"),
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        total_checks=len(all_results),
        passed=passed,
        failed=failed,
        blockers=blockers,
        warnings=warnings,
        infos=infos,
        checks=all_results,
        summary=summary,
    )
