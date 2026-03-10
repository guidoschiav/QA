"""Schema / Metadata checks: S1–S3."""
from __future__ import annotations

from typing import List

import pandas as pd

from ..models import CheckResult


def run_checks(df: pd.DataFrame, config: dict) -> List[CheckResult]:
    return [
        _s1_required_columns(df, config),
        _s2_data_types(df, config),
        _s3_dimension_values(df, config),
    ]


# ---------------------------------------------------------------------------
# S1 — Required columns present
# ---------------------------------------------------------------------------

def _s1_required_columns(df: pd.DataFrame, config: dict) -> CheckResult:
    date_col = config.get("date_column", "Date")
    dim_cols = config.get("dimension_columns", [])
    fmt = config.get("format", "long")

    required = [date_col] + list(dim_cols)
    if fmt == "long":
        vc = config.get("value_column", "Value")
        required.append(vc)
    else:
        required.extend(config.get("value_columns", []))

    missing = [c for c in required if c not in df.columns]
    passed = len(missing) == 0

    return CheckResult(
        check_id="S1",
        check_name="required_columns_present",
        category="schema",
        severity="BLOCKER",
        passed=passed,
        message=(
            "All required columns present"
            if passed
            else f"Missing columns: {missing}"
        ),
        details={"missing": missing, "present": list(df.columns)},
        recommendation="" if passed else f"Add missing columns: {missing}",
    )


# ---------------------------------------------------------------------------
# S2 — Data types
# ---------------------------------------------------------------------------

def _s2_data_types(df: pd.DataFrame, config: dict) -> CheckResult:
    date_col = config.get("date_column", "Date")
    fmt = config.get("format", "long")
    issues = []
    details: dict = {}

    # Date column: must be datetime or parseable string
    if date_col in df.columns:
        col = df[date_col]
        if not pd.api.types.is_datetime64_any_dtype(col):
            # Try parsing sample
            sample = col.dropna().head(5)
            try:
                pd.to_datetime(sample, format="%Y-%m-%d")
            except Exception:
                issues.append(f"'{date_col}' is not parseable as YYYY-MM-DD")
                details[date_col] = str(col.dtype)

    # Value columns: must be numeric
    if fmt == "long":
        vc = config.get("value_column", "Value")
        if vc in df.columns and not pd.api.types.is_numeric_dtype(df[vc]):
            issues.append(f"'{vc}' is not numeric (dtype: {df[vc].dtype})")
            details[vc] = str(df[vc].dtype)
    else:
        for vc in config.get("value_columns", []):
            if vc in df.columns and not pd.api.types.is_numeric_dtype(df[vc]):
                issues.append(f"'{vc}' is not numeric (dtype: {df[vc].dtype})")
                details[vc] = str(df[vc].dtype)

    passed = len(issues) == 0
    return CheckResult(
        check_id="S2",
        check_name="data_types",
        category="schema",
        severity="WARNING",
        passed=passed,
        message=(
            "All column dtypes are correct"
            if passed
            else "Type issues: " + "; ".join(issues)
        ),
        details=details,
        recommendation="" if passed else "Cast columns to correct types before auditing.",
    )


# ---------------------------------------------------------------------------
# S3 — Dimension values vs expected list
# ---------------------------------------------------------------------------

def _s3_dimension_values(df: pd.DataFrame, config: dict) -> CheckResult:
    expected_dims: dict = config.get("expected_dimensions", {}) or {}
    issues = []
    details: dict = {}

    for col, expected_values in expected_dims.items():
        if expected_values is None:
            continue
        if col not in df.columns:
            continue

        actual_values = set(df[col].dropna().unique())
        expected_set = set(expected_values)

        new_vals = sorted(actual_values - expected_set)
        missing_vals = sorted(expected_set - actual_values)

        if new_vals:
            issues.append(f"'{col}' has {len(new_vals)} new unexpected value(s)")
            details[f"{col}_new"] = new_vals[:10]
        if missing_vals:
            issues.append(f"'{col}' is missing {len(missing_vals)} expected value(s)")
            details[f"{col}_missing"] = missing_vals[:10]

    passed = len(issues) == 0
    return CheckResult(
        check_id="S3",
        check_name="dimension_values",
        category="schema",
        severity="WARNING",
        passed=passed,
        message=(
            "All dimension values match expected lists"
            if passed
            else "Dimension issues: " + "; ".join(issues)
        ),
        details=details,
        recommendation=(
            ""
            if passed
            else "Update expected_dimensions config or fix entity names in source."
        ),
    )
