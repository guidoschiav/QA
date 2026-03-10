"""Statistical / anomaly checks: A1–A3."""
from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np
from scipy import stats

from ..models import CheckResult
from ..utils import get_value_columns, get_dimension_columns


def run_checks(df: pd.DataFrame, config: dict) -> List[CheckResult]:
    return [
        _a1_zscore_outliers(df, config),
        _a2_abrupt_pct_change(df, config),
        _a3_distribution_summary(df, config),
    ]


# ---------------------------------------------------------------------------
# A1 — Z-score outliers per series/entity
# ---------------------------------------------------------------------------

def _a1_zscore_outliers(df: pd.DataFrame, config: dict) -> CheckResult:
    threshold = config.get("outlier_zscore_threshold", 3.5)
    value_cols = get_value_columns(df, config)
    dim_cols = get_dimension_columns(df, config)

    if not value_cols:
        return CheckResult(
            check_id="A1",
            check_name="zscore_outliers",
            category="statistical",
            severity="WARNING",
            passed=True,
            message="Z-score check skipped: no value columns",
            details={},
            recommendation="",
        )

    total_outliers = 0
    details: dict = {}

    for vc in value_cols:
        if vc not in df.columns:
            continue
        col = df[vc].dropna()
        if len(col) < 10:
            continue

        if dim_cols:
            # Compute z-scores within each group (vectorized via groupby transform)
            group_mean = df.groupby(dim_cols)[vc].transform("mean")
            group_std = df.groupby(dim_cols)[vc].transform("std")
            valid_std = group_std > 0
            z_scores = pd.Series(np.nan, index=df.index)
            z_scores[valid_std] = ((df[vc] - group_mean) / group_std).abs()[valid_std]
        else:
            z_scores = pd.Series(np.abs(stats.zscore(col, nan_policy="omit")), index=col.index)
            z_scores = z_scores.reindex(df.index)

        outlier_mask = z_scores > threshold
        n_outliers = int(outlier_mask.sum())
        total_outliers += n_outliers

        if n_outliers > 0:
            pct = n_outliers / len(df)
            details[vc] = {"count": n_outliers, "pct": round(pct, 4)}
            # Sample worst offenders
            worst_idx = z_scores.nlargest(3).index
            details[f"{vc}_worst_z"] = [round(float(z_scores[i]), 2) for i in worst_idx if pd.notna(z_scores[i])]

    passed = total_outliers == 0
    return CheckResult(
        check_id="A1",
        check_name="zscore_outliers",
        category="statistical",
        severity="WARNING",
        passed=passed,
        message=(
            f"No z-score outliers found (threshold={threshold})"
            if passed
            else f"{total_outliers} outlier(s) found across value columns (|z| > {threshold})"
        ),
        details=details,
        recommendation=(
            ""
            if passed
            else "Investigate outlier records — may be data errors or genuine extremes."
        ),
    )


# ---------------------------------------------------------------------------
# A2 — Abrupt percentage change vs rolling window
# ---------------------------------------------------------------------------

def _a2_abrupt_pct_change(df: pd.DataFrame, config: dict) -> CheckResult:
    threshold = config.get("pct_change_threshold", 0.5)
    date_col = config.get("date_column", "Date")
    value_cols = get_value_columns(df, config)
    dim_cols = get_dimension_columns(df, config)

    if not value_cols or date_col not in df.columns:
        return CheckResult(
            check_id="A2",
            check_name="abrupt_pct_change",
            category="statistical",
            severity="WARNING",
            passed=True,
            message="Pct-change check skipped: no value columns or date column",
            details={},
            recommendation="",
        )

    total_spikes = 0
    details: dict = {}

    sort_cols = [date_col] + dim_cols
    working = df.sort_values(sort_cols).copy()

    for vc in value_cols:
        if vc not in working.columns:
            continue

        if dim_cols:
            # pct_change within each group
            pct_ch = working.groupby(dim_cols)[vc].pct_change().abs()
        else:
            pct_ch = working[vc].pct_change().abs()

        spike_mask = pct_ch > threshold
        n_spikes = int(spike_mask.sum())
        total_spikes += n_spikes

        if n_spikes > 0:
            details[vc] = {
                "count": n_spikes,
                "max_pct_change": round(float(pct_ch.max()), 4),
            }

    passed = total_spikes == 0
    return CheckResult(
        check_id="A2",
        check_name="abrupt_pct_change",
        category="statistical",
        severity="WARNING",
        passed=passed,
        message=(
            f"No abrupt changes found (threshold={threshold:.0%})"
            if passed
            else f"{total_spikes} abrupt change(s) detected (>{threshold:.0%})"
        ),
        details=details,
        recommendation=(
            ""
            if passed
            else "Review spikes — may indicate data errors, restatements, or real events."
        ),
    )


# ---------------------------------------------------------------------------
# A3 — Distribution summary (vs snapshot if available)
# ---------------------------------------------------------------------------

def _a3_distribution_summary(df: pd.DataFrame, config: dict) -> CheckResult:
    value_cols = get_value_columns(df, config)

    if not value_cols:
        return CheckResult(
            check_id="A3",
            check_name="distribution_summary",
            category="statistical",
            severity="INFO",
            passed=True,
            message="Distribution check skipped: no value columns",
            details={},
            recommendation="",
        )

    details: dict = {}
    for vc in value_cols:
        if vc not in df.columns:
            continue
        col = df[vc].dropna()
        if len(col) == 0:
            continue
        details[vc] = {
            "count": len(col),
            "mean": round(float(col.mean()), 4),
            "std": round(float(col.std()), 4),
            "min": round(float(col.min()), 4),
            "p25": round(float(col.quantile(0.25)), 4),
            "median": round(float(col.median()), 4),
            "p75": round(float(col.quantile(0.75)), 4),
            "max": round(float(col.max()), 4),
            "skew": round(float(col.skew()), 4),
        }

    return CheckResult(
        check_id="A3",
        check_name="distribution_summary",
        category="statistical",
        severity="INFO",
        passed=True,
        message="Distribution statistics computed",
        details=details,
        recommendation="Compare distribution metrics across runs to detect drift.",
    )
