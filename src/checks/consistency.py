"""Consistency inter-run checks: C1–C3."""
from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np

from ..models import CheckResult
from ..utils import get_value_columns, get_dimension_columns


def run_checks(
    df: pd.DataFrame,
    config: dict,
    prev_snapshot: pd.DataFrame,
) -> List[CheckResult]:
    return [
        _c1_historical_rewrites(df, config, prev_snapshot),
        _c2_pct_cells_modified(df, config, prev_snapshot),
        _c3_revision_whitelist(df, config, prev_snapshot),
    ]


def run_checks_no_snapshot(config: dict) -> List[CheckResult]:
    """Return skipped results when no snapshot is available."""
    results = []
    for check_id, name in [
        ("C1", "historical_rewrites"),
        ("C2", "pct_cells_modified"),
        ("C3", "revision_whitelist"),
    ]:
        results.append(CheckResult(
            check_id=check_id,
            check_name=name,
            category="consistency",
            severity="INFO",
            passed=True,
            message="Skipped — no previous snapshot available",
            details={"skipped": True},
            recommendation="Run again after first snapshot is saved.",
        ))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_key_index(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return df indexed by (date_col + dim_cols), value columns only."""
    date_col = config.get("date_column", "Date")
    dim_cols = get_dimension_columns(df, config)
    value_cols = get_value_columns(df, config)

    key_cols = [date_col] + dim_cols
    keep_cols = [c for c in key_cols + value_cols if c in df.columns]
    subset = df[keep_cols].copy()

    # Normalize date to string for alignment
    if date_col in subset.columns:
        subset[date_col] = pd.to_datetime(subset[date_col]).dt.strftime("%Y-%m-%d")

    # Drop duplicates on key (keep first) to allow clean alignment
    subset = subset.drop_duplicates(subset=key_cols)
    subset = subset.set_index(key_cols)
    return subset


# ---------------------------------------------------------------------------
# C1 — Historical rewrites detection
# ---------------------------------------------------------------------------

def _c1_historical_rewrites(
    df: pd.DataFrame, config: dict, prev: pd.DataFrame
) -> CheckResult:
    try:
        curr_idx = _build_key_index(df, config)
        prev_idx = _build_key_index(prev, config)
    except Exception as e:
        return CheckResult(
            check_id="C1",
            check_name="historical_rewrites",
            category="consistency",
            severity="WARNING",
            passed=True,
            message=f"Could not align snapshots for comparison: {e}",
            details={},
            recommendation="",
        )

    # Align on common keys
    common_keys = curr_idx.index.intersection(prev_idx.index)
    if len(common_keys) == 0:
        return CheckResult(
            check_id="C1",
            check_name="historical_rewrites",
            category="consistency",
            severity="INFO",
            passed=True,
            message="No overlapping keys between current and previous snapshot",
            details={"common_keys": 0},
            recommendation="",
        )

    curr_common = curr_idx.loc[common_keys]
    prev_common = prev_idx.loc[common_keys]

    # Align columns
    common_cols = curr_common.columns.intersection(prev_common.columns)
    if len(common_cols) == 0:
        return CheckResult(
            check_id="C1",
            check_name="historical_rewrites",
            category="consistency",
            severity="INFO",
            passed=True,
            message="No common value columns to compare",
            details={},
            recommendation="",
        )

    curr_vals = curr_common[common_cols]
    prev_vals = prev_common[common_cols]

    # Vectorized comparison — handle NaN equality
    changed = ~(
        (curr_vals == prev_vals) |
        (curr_vals.isnull() & prev_vals.isnull())
    )
    n_changed_rows = int(changed.any(axis=1).sum())
    n_total_rows = len(common_keys)

    passed = n_changed_rows == 0
    return CheckResult(
        check_id="C1",
        check_name="historical_rewrites",
        category="consistency",
        severity="WARNING",
        passed=passed,
        message=(
            "No historical rewrites detected"
            if passed
            else f"{n_changed_rows}/{n_total_rows} historical records changed vs previous snapshot"
        ),
        details={
            "changed_rows": n_changed_rows,
            "total_common_rows": n_total_rows,
            "pct_changed": round(n_changed_rows / n_total_rows, 4) if n_total_rows > 0 else 0,
        },
        recommendation=(
            ""
            if passed
            else "Check if revisions are expected. Use revision_whitelist to suppress known revisions."
        ),
    )


# ---------------------------------------------------------------------------
# C2 — Percentage of modified cells
# ---------------------------------------------------------------------------

def _c2_pct_cells_modified(
    df: pd.DataFrame, config: dict, prev: pd.DataFrame
) -> CheckResult:
    try:
        curr_idx = _build_key_index(df, config)
        prev_idx = _build_key_index(prev, config)
    except Exception:
        return CheckResult(
            check_id="C2",
            check_name="pct_cells_modified",
            category="consistency",
            severity="INFO",
            passed=True,
            message="Could not align snapshots",
            details={},
            recommendation="",
        )

    common_keys = curr_idx.index.intersection(prev_idx.index)
    if len(common_keys) == 0:
        return CheckResult(
            check_id="C2",
            check_name="pct_cells_modified",
            category="consistency",
            severity="INFO",
            passed=True,
            message="No overlapping rows to compute cell modification rate",
            details={},
            recommendation="",
        )

    curr_common = curr_idx.loc[common_keys]
    prev_common = prev_idx.loc[common_keys]
    common_cols = curr_common.columns.intersection(prev_common.columns)

    if len(common_cols) == 0:
        return CheckResult(
            check_id="C2",
            check_name="pct_cells_modified",
            category="consistency",
            severity="INFO",
            passed=True,
            message="No common value columns",
            details={},
            recommendation="",
        )

    curr_vals = curr_common[common_cols]
    prev_vals = prev_common[common_cols]

    changed_cells = ~(
        (curr_vals == prev_vals) |
        (curr_vals.isnull() & prev_vals.isnull())
    )
    n_changed_cells = int(changed_cells.values.sum())
    total_cells = curr_vals.size
    pct = n_changed_cells / total_cells if total_cells > 0 else 0.0

    max_pct = config.get("revision_whitelist", {}).get("max_revision_pct", 0.10)
    passed = pct <= max_pct

    return CheckResult(
        check_id="C2",
        check_name="pct_cells_modified",
        category="consistency",
        severity="INFO",
        passed=passed,
        message=(
            f"{pct:.1%} of cells modified (within {max_pct:.0%} threshold)"
            if passed
            else f"{pct:.1%} of cells modified (exceeds {max_pct:.0%} threshold)"
        ),
        details={
            "changed_cells": n_changed_cells,
            "total_cells": total_cells,
            "pct_changed": round(pct, 4),
            "threshold": max_pct,
        },
        recommendation=(
            ""
            if passed
            else "Large number of cell revisions — verify this is expected restatement."
        ),
    )


# ---------------------------------------------------------------------------
# C3 — Revision whitelist validation
# ---------------------------------------------------------------------------

def _c3_revision_whitelist(
    df: pd.DataFrame, config: dict, prev: pd.DataFrame
) -> CheckResult:
    wl_config = config.get("revision_whitelist", {}) or {}
    enabled = wl_config.get("enabled", False)

    if not enabled:
        return CheckResult(
            check_id="C3",
            check_name="revision_whitelist",
            category="consistency",
            severity="INFO",
            passed=True,
            message="Revision whitelist check disabled",
            details={"enabled": False},
            recommendation="",
        )

    max_age_days = wl_config.get("max_revision_age_days", 90)
    max_pct = wl_config.get("max_revision_pct", 0.10)
    date_col = config.get("date_column", "Date")

    try:
        curr_idx = _build_key_index(df, config)
        prev_idx = _build_key_index(prev, config)
    except Exception as e:
        return CheckResult(
            check_id="C3",
            check_name="revision_whitelist",
            category="consistency",
            severity="INFO",
            passed=True,
            message=f"Could not align snapshots: {e}",
            details={},
            recommendation="",
        )

    common_keys = curr_idx.index.intersection(prev_idx.index)
    if len(common_keys) == 0:
        return CheckResult(
            check_id="C3",
            check_name="revision_whitelist",
            category="consistency",
            severity="INFO",
            passed=True,
            message="No common rows — whitelist check N/A",
            details={},
            recommendation="",
        )

    curr_common = curr_idx.loc[common_keys]
    prev_common = prev_idx.loc[common_keys]
    common_cols = curr_common.columns.intersection(prev_common.columns)

    changed_mask = ~(
        (curr_common[common_cols] == prev_common[common_cols]) |
        (curr_common[common_cols].isnull() & prev_common[common_cols].isnull())
    )
    changed_rows_mask = changed_mask.any(axis=1)

    if not changed_rows_mask.any():
        return CheckResult(
            check_id="C3",
            check_name="revision_whitelist",
            category="consistency",
            severity="INFO",
            passed=True,
            message="No revisions detected — whitelist check N/A",
            details={},
            recommendation="",
        )

    # Extract date level from index (first level if multi-index)
    changed_idx = changed_rows_mask[changed_rows_mask].index
    if isinstance(changed_idx, pd.MultiIndex):
        changed_dates = pd.to_datetime([idx[0] for idx in changed_idx], errors="coerce")
    else:
        changed_dates = pd.to_datetime(changed_idx, errors="coerce")

    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=max_age_days)

    # Revisions beyond max_age_days are unexpected
    old_revisions = changed_dates[changed_dates < cutoff]
    n_old = len(old_revisions)
    n_recent = len(changed_dates) - n_old
    total_rows = len(common_keys)
    pct_revised = len(changed_dates) / total_rows if total_rows > 0 else 0

    issues = []
    if n_old > 0:
        issues.append(f"{n_old} revision(s) older than {max_age_days} days")
    if pct_revised > max_pct:
        issues.append(f"{pct_revised:.1%} revised exceeds max {max_pct:.0%}")

    passed = len(issues) == 0
    return CheckResult(
        check_id="C3",
        check_name="revision_whitelist",
        category="consistency",
        severity="WARNING",
        passed=passed,
        message=(
            f"Revisions within whitelist ({n_recent} recent, 0 old)"
            if passed
            else "Revision whitelist violations: " + "; ".join(issues)
        ),
        details={
            "total_revised_rows": len(changed_dates),
            "recent_revisions": n_recent,
            "old_revisions_beyond_age": n_old,
            "pct_revised": round(pct_revised, 4),
            "max_age_days": max_age_days,
            "max_pct": max_pct,
        },
        recommendation=(
            ""
            if passed
            else "Investigate old revisions — may indicate a bug in the data pipeline."
        ),
    )
