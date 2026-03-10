"""Data integrity checks: I1–I4."""
from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np

from ..models import CheckResult
from ..utils import get_value_columns


def run_checks(df: pd.DataFrame, config: dict) -> List[CheckResult]:
    return [
        _i1_null_percentage(df, config),
        _i2_suspicious_zeros(df, config),
        _i3_exact_duplicates(df, config),
        _i4_value_range(df, config),
    ]


# ---------------------------------------------------------------------------
# I1 — Null percentage per column
# ---------------------------------------------------------------------------

def _i1_null_percentage(df: pd.DataFrame, config: dict) -> CheckResult:
    max_null_pct = config.get("max_null_pct", 0.05)
    n = len(df)
    if n == 0:
        return CheckResult(
            check_id="I1",
            check_name="null_percentage",
            category="integrity",
            severity="WARNING",
            passed=False,
            message="DataFrame is empty",
            details={},
            recommendation="Check data source.",
        )

    null_pcts = (df.isnull().sum() / n)
    violating = null_pcts[null_pcts > max_null_pct]
    passed = len(violating) == 0

    details = {col: round(float(pct), 4) for col, pct in violating.items()}
    return CheckResult(
        check_id="I1",
        check_name="null_percentage",
        category="integrity",
        severity="WARNING",
        passed=passed,
        message=(
            f"Null % within threshold ({max_null_pct:.0%}) for all columns"
            if passed
            else f"{len(violating)} column(s) exceed {max_null_pct:.0%} null threshold"
        ),
        details={"violating_columns": details, "threshold": max_null_pct},
        recommendation="" if passed else "Investigate missing data in source and fill or flag.",
    )


# ---------------------------------------------------------------------------
# I2 — Suspicious zeros
# ---------------------------------------------------------------------------

def _i2_suspicious_zeros(df: pd.DataFrame, config: dict) -> CheckResult:
    allow_zeros = config.get("allow_zero_values", True)
    value_cols = get_value_columns(df, config)

    if allow_zeros or not value_cols:
        return CheckResult(
            check_id="I2",
            check_name="suspicious_zeros",
            category="integrity",
            severity="WARNING",
            passed=True,
            message="Zero-value check skipped (allow_zero_values=true or no value columns)",
            details={},
            recommendation="",
        )

    zero_counts = {}
    for vc in value_cols:
        if vc in df.columns:
            n_zeros = int((df[vc] == 0).sum())
            if n_zeros > 0:
                zero_counts[vc] = n_zeros

    passed = len(zero_counts) == 0
    return CheckResult(
        check_id="I2",
        check_name="suspicious_zeros",
        category="integrity",
        severity="WARNING",
        passed=passed,
        message=(
            "No suspicious zeros found in value column(s)"
            if passed
            else f"Zeros found (allow_zero_values=false): {zero_counts}"
        ),
        details={"zero_counts": zero_counts},
        recommendation=(
            ""
            if passed
            else "Verify zeros are valid or set allow_zero_values: true if expected."
        ),
    )


# ---------------------------------------------------------------------------
# I3 — Exact duplicate rows
# ---------------------------------------------------------------------------

def _i3_exact_duplicates(df: pd.DataFrame, config: dict) -> CheckResult:
    n_dups = int(df.duplicated().sum())
    passed = n_dups == 0
    return CheckResult(
        check_id="I3",
        check_name="exact_duplicate_rows",
        category="integrity",
        severity="BLOCKER",
        passed=passed,
        message=(
            "No exact duplicate rows"
            if passed
            else f"{n_dups} exact duplicate row(s) found"
        ),
        details={"exact_duplicate_count": n_dups},
        recommendation="" if passed else "Run df.drop_duplicates() before publishing.",
    )


# ---------------------------------------------------------------------------
# I4 — Value range
# ---------------------------------------------------------------------------

def _i4_value_range(df: pd.DataFrame, config: dict) -> CheckResult:
    value_range = config.get("value_range", {}) or {}
    vmin = value_range.get("min")
    vmax = value_range.get("max")
    value_cols = get_value_columns(df, config)

    if (vmin is None and vmax is None) or not value_cols:
        return CheckResult(
            check_id="I4",
            check_name="value_range",
            category="integrity",
            severity="WARNING",
            passed=True,
            message="Value range check skipped (no limits configured or no value columns)",
            details={},
            recommendation="",
        )

    issues = []
    details: dict = {}

    for vc in value_cols:
        if vc not in df.columns:
            continue
        col = df[vc].dropna()
        if len(col) == 0:
            continue

        col_min = float(col.min())
        col_max = float(col.max())
        details[f"{vc}_actual_min"] = col_min
        details[f"{vc}_actual_max"] = col_max

        if vmin is not None and col_min < vmin:
            n_below = int((col < vmin).sum())
            issues.append(f"'{vc}' has {n_below} value(s) below min={vmin}")
            details[f"{vc}_below_min"] = n_below

        if vmax is not None and col_max > vmax:
            n_above = int((col > vmax).sum())
            issues.append(f"'{vc}' has {n_above} value(s) above max={vmax}")
            details[f"{vc}_above_max"] = n_above

    passed = len(issues) == 0
    return CheckResult(
        check_id="I4",
        check_name="value_range",
        category="integrity",
        severity="WARNING",
        passed=passed,
        message=(
            f"All values within configured range [{vmin}, {vmax}]"
            if passed
            else "Value range violations: " + "; ".join(issues)
        ),
        details=details,
        recommendation=(
            ""
            if passed
            else "Investigate out-of-range records in source data."
        ),
    )
