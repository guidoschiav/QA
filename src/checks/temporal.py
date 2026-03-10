"""Temporal checks: T1–T6."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pandas as pd
import numpy as np

from ..models import CheckResult
from ..utils import infer_frequency, generate_expected_dates, get_holidays


def run_checks(df: pd.DataFrame, config: dict) -> List[CheckResult]:
    results = []
    date_col = config.get("date_column", "Date")

    if date_col not in df.columns:
        # Schema check will catch this; skip temporal
        return []

    dates = df[date_col].dropna()
    if len(dates) == 0:
        return []

    unique_dates = dates.sort_values().unique()
    date_index = pd.DatetimeIndex(unique_dates)

    results.append(_t1_frequency(date_index, config))
    results.append(_t2_gaps(date_index, config))
    results.append(_t3_duplicates(df, config))
    results.append(_t4_staleness(date_index, config))
    results.append(_t5_date_range(date_index, config))
    results.append(_t6_day_patterns(date_index, config))

    return results


# ---------------------------------------------------------------------------
# T1 — Frequency inference
# ---------------------------------------------------------------------------

def _t1_frequency(date_index: pd.DatetimeIndex, config: dict) -> CheckResult:
    from ..utils import infer_frequency

    expected = config.get("expected_frequency", "D")
    inferred = infer_frequency(pd.Series(date_index))

    passed = inferred == expected
    return CheckResult(
        check_id="T1",
        check_name="frequency_mismatch",
        category="temporal",
        severity="BLOCKER",
        passed=passed,
        message=(
            f"Frequency OK: inferred '{inferred}' matches expected '{expected}'"
            if passed
            else f"Frequency mismatch: inferred '{inferred}', expected '{expected}'"
        ),
        details={"inferred": inferred, "expected": expected, "n_unique_dates": len(date_index)},
        recommendation=(
            ""
            if passed
            else f"Check scraper schedule. Expected '{expected}' but got '{inferred}'."
        ),
    )


# ---------------------------------------------------------------------------
# T2 — Gap detection
# ---------------------------------------------------------------------------

def _t2_gaps(date_index: pd.DatetimeIndex, config: dict) -> CheckResult:
    if len(date_index) < 2:
        return CheckResult(
            check_id="T2",
            check_name="gap_detection",
            category="temporal",
            severity="BLOCKER",
            passed=False,
            message="Not enough dates to check gaps (< 2 unique dates)",
            details={},
            recommendation="Verify data source has sufficient rows.",
        )

    freq = config.get("expected_frequency", "D")
    holidays = get_holidays(config.get("holidays_country"))
    start, end = date_index.min(), date_index.max()
    expected = generate_expected_dates(start, end, freq, holidays)

    actual_set = set(date_index.normalize())
    expected_set = set(expected.normalize())
    gaps = sorted(expected_set - actual_set)

    passed = len(gaps) == 0
    gap_strs = [str(g.date()) for g in gaps[:20]]
    return CheckResult(
        check_id="T2",
        check_name="gap_detection",
        category="temporal",
        severity="BLOCKER",
        passed=passed,
        message=(
            "No gaps found in date sequence"
            if passed
            else f"{len(gaps)} gap(s) found in expected {freq} sequence"
        ),
        details={"gap_count": len(gaps), "gaps_sample": gap_strs},
        recommendation="" if passed else "Fill missing dates or verify scraper missed these periods.",
    )


# ---------------------------------------------------------------------------
# T3 — Duplicate records by Date + dimensions
# ---------------------------------------------------------------------------

def _t3_duplicates(df: pd.DataFrame, config: dict) -> CheckResult:
    date_col = config.get("date_column", "Date")
    dim_cols = config.get("dimension_columns", [])
    key_cols = [date_col] + [c for c in dim_cols if c in df.columns]

    dup_mask = df.duplicated(subset=key_cols, keep=False)
    n_dups = dup_mask.sum()
    passed = n_dups == 0

    sample = []
    if not passed:
        sample = df[dup_mask][key_cols].head(5).astype(str).values.tolist()

    return CheckResult(
        check_id="T3",
        check_name="duplicate_date_dimension_keys",
        category="temporal",
        severity="BLOCKER",
        passed=passed,
        message=(
            "No duplicate (Date+dimensions) keys found"
            if passed
            else f"{n_dups} duplicate key rows found"
        ),
        details={"duplicate_rows": n_dups, "sample": sample},
        recommendation="" if passed else "Deduplicate by key before publishing.",
    )


# ---------------------------------------------------------------------------
# T4 — Staleness
# ---------------------------------------------------------------------------

def _t4_staleness(date_index: pd.DatetimeIndex, config: dict) -> CheckResult:
    lag = config.get("acceptable_lag_days", 3)
    last_date = date_index.max().date()
    today = date.today()
    expected_latest = today - timedelta(days=lag)
    days_stale = (expected_latest - last_date).days

    passed = days_stale <= 0
    return CheckResult(
        check_id="T4",
        check_name="staleness",
        category="temporal",
        severity="WARNING",
        passed=passed,
        message=(
            f"Data is current (last date: {last_date}, lag: {lag} days)"
            if passed
            else f"Data is stale by {days_stale} day(s) — last date: {last_date}"
        ),
        details={
            "last_date": str(last_date),
            "today": str(today),
            "acceptable_lag_days": lag,
            "days_stale": max(days_stale, 0),
        },
        recommendation=(
            ""
            if passed
            else f"Check scraper. Expected last date >= {expected_latest}."
        ),
    )


# ---------------------------------------------------------------------------
# T5 — Date range validation
# ---------------------------------------------------------------------------

def _t5_date_range(date_index: pd.DatetimeIndex, config: dict) -> CheckResult:
    actual_start = date_index.min().date()
    actual_end = date_index.max().date()
    expected_start = config.get("expected_start")
    expected_end = config.get("expected_end")

    issues = []
    details: dict = {
        "actual_start": str(actual_start),
        "actual_end": str(actual_end),
    }

    if expected_start:
        exp_start = pd.Timestamp(expected_start).date()
        details["expected_start"] = str(exp_start)
        if actual_start > exp_start:
            diff = (actual_start - exp_start).days
            issues.append(f"starts {diff} day(s) late (expected {exp_start})")

    if expected_end:
        exp_end = pd.Timestamp(expected_end).date()
        details["expected_end"] = str(exp_end)
        if actual_end < exp_end:
            diff = (exp_end - actual_end).days
            issues.append(f"ends {diff} day(s) early (expected {exp_end})")

    passed = len(issues) == 0
    return CheckResult(
        check_id="T5",
        check_name="date_range_validation",
        category="temporal",
        severity="WARNING",
        passed=passed,
        message=(
            f"Date range OK: {actual_start} to {actual_end}"
            if passed
            else "Date range issues: " + "; ".join(issues)
        ),
        details=details,
        recommendation="" if passed else "Verify data source coverage.",
    )


# ---------------------------------------------------------------------------
# T6 — Day pattern detection
# ---------------------------------------------------------------------------

def _t6_day_patterns(date_index: pd.DatetimeIndex, config: dict) -> CheckResult:
    freq = config.get("expected_frequency", "D")
    holidays_set = get_holidays(config.get("holidays_country"))

    dow_counts = pd.Series(date_index.dayofweek).value_counts().sort_index()
    n_weekends = int(dow_counts.get(5, 0) + dow_counts.get(6, 0))
    n_holidays = sum(1 for d in date_index if str(d.date()) in holidays_set)

    issues = []
    details: dict = {
        "weekend_dates_count": n_weekends,
        "holiday_dates_count": n_holidays,
        "day_of_week_distribution": dow_counts.to_dict(),
    }

    if freq == "B" and n_weekends > 0:
        issues.append(f"{n_weekends} weekend date(s) found in business-day dataset")
    if freq == "B" and n_holidays > 0 and config.get("holidays_country"):
        issues.append(f"{n_holidays} holiday date(s) found")

    passed = len(issues) == 0
    return CheckResult(
        check_id="T6",
        check_name="day_pattern_detection",
        category="temporal",
        severity="INFO",
        passed=passed,
        message=(
            "Day pattern consistent with expected frequency"
            if passed
            else "Day pattern anomalies: " + "; ".join(issues)
        ),
        details=details,
        recommendation="" if passed else "Review whether weekends/holidays should be excluded.",
    )
