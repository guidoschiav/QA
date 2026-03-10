"""Test fixtures — DataFrames that trigger known audit conditions."""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import date, timedelta


def _bdays(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=periods)


# ---------------------------------------------------------------------------
# 1. df_clean — passes all checks
# ---------------------------------------------------------------------------

def make_df_clean() -> pd.DataFrame:
    dates = _bdays("2024-01-02", 60)
    regions = ["CABA", "GBA", "Interior"]
    records = []
    rng = np.random.default_rng(42)
    for d in dates:
        for r in regions:
            records.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Region": r,
                "Value": float(rng.integers(1000, 50000)),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. df_gaps — temporal gaps and staleness
# ---------------------------------------------------------------------------

def make_df_gaps() -> pd.DataFrame:
    """Missing dates in the middle + stale end date."""
    all_dates = _bdays("2024-01-02", 60)
    # Remove 5 days in the middle to create gaps
    gap_indices = {10, 11, 12, 25, 26}
    dates = [d for i, d in enumerate(all_dates) if i not in gap_indices]
    # Make it stale: shift end date 30 days back
    dates = [d - timedelta(days=30) for d in dates]

    regions = ["CABA", "GBA"]
    records = []
    rng = np.random.default_rng(7)
    for d in dates:
        for r in regions:
            records.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Region": r,
                "Value": float(rng.integers(500, 10000)),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. df_dirty — nulls, duplicates, outliers, zero values
# ---------------------------------------------------------------------------

def make_df_dirty() -> pd.DataFrame:
    dates = _bdays("2024-01-02", 40)
    regions = ["CABA", "GBA", "Interior"]
    records = []
    rng = np.random.default_rng(99)
    for d in dates:
        for r in regions:
            value = float(rng.integers(1000, 30000))
            records.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Region": r,
                "Value": value,
            })
    df = pd.DataFrame(records)

    # Inject nulls (>5% in Value)
    null_indices = rng.choice(len(df), size=20, replace=False)
    df.loc[null_indices, "Value"] = np.nan

    # Inject zero values
    zero_indices = rng.choice(len(df), size=5, replace=False)
    df.loc[zero_indices, "Value"] = 0.0

    # Inject outlier
    df.loc[0, "Value"] = 9_999_999.0  # extreme value

    # Add exact duplicate rows
    dup_rows = df.iloc[:3].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Add duplicate key (same Date+Region)
    df = pd.concat([df, df.iloc[[5]].copy()], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# 4. df_revised — for consistency checks against a "previous" snapshot
# ---------------------------------------------------------------------------

def make_df_snapshot_prev() -> pd.DataFrame:
    """The 'previous' snapshot — 40 business days, clean."""
    dates = _bdays("2024-01-02", 40)
    regions = ["CABA", "GBA", "Interior"]
    records = []
    rng = np.random.default_rng(123)
    for d in dates:
        for r in regions:
            records.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Region": r,
                "Value": float(rng.integers(5000, 20000)),
            })
    return pd.DataFrame(records)


def make_df_revised() -> pd.DataFrame:
    """Current snapshot with historical rewrites: 20% of old rows changed."""
    prev = make_df_snapshot_prev()
    # Add 10 new days
    new_dates = _bdays("2024-03-01", 10)
    regions = ["CABA", "GBA", "Interior"]
    rng = np.random.default_rng(456)
    new_records = []
    for d in new_dates:
        for r in regions:
            new_records.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Region": r,
                "Value": float(rng.integers(5000, 20000)),
            })

    current = pd.concat([prev, pd.DataFrame(new_records)], ignore_index=True)

    # Rewrite ~25% of historical rows (simulate revision)
    hist_mask = current["Date"] < "2024-02-01"
    hist_indices = current[hist_mask].index
    rewrite_indices = rng.choice(hist_indices, size=len(hist_indices) // 4, replace=False)
    current.loc[rewrite_indices, "Value"] *= rng.uniform(0.8, 1.2, size=len(rewrite_indices))

    return current


# ---------------------------------------------------------------------------
# 5. df_wide — wide format dataset
# ---------------------------------------------------------------------------

def make_df_wide() -> pd.DataFrame:
    dates = _bdays("2024-01-02", 30)
    rng = np.random.default_rng(77)
    records = []
    for d in dates:
        records.append({
            "Date": d.strftime("%Y-%m-%d"),
            "Sales": float(rng.integers(10000, 100000)),
            "Units": float(rng.integers(100, 5000)),
        })
    return pd.DataFrame(records)


# Config matching each fixture
CLEAN_CONFIG = {
    "dataset_name": "test_clean",
    "format": "long",
    "date_column": "Date",
    "dimension_columns": ["Region"],
    "value_column": "Value",
    "expected_frequency": "B",
    "expected_start": "2024-01-02",
    "acceptable_lag_days": 365,  # high lag so staleness doesn't fail
    "holidays_country": "AR",
    "expected_dimensions": {"Region": ["CABA", "GBA", "Interior"]},
    "max_null_pct": 0.05,
    "allow_zero_values": False,
    "value_range": {"min": 0, "max": 100000},
    "outlier_zscore_threshold": 5.0,
    "pct_change_threshold": 5.0,
    "revision_whitelist": {"enabled": False},
    "severity_overrides": {},
}

GAPS_CONFIG = {
    "dataset_name": "test_gaps",
    "format": "long",
    "date_column": "Date",
    "dimension_columns": ["Region"],
    "value_column": "Value",
    "expected_frequency": "B",
    "acceptable_lag_days": 1,
    "holidays_country": "AR",
    "expected_dimensions": {},
    "max_null_pct": 0.05,
    "allow_zero_values": True,
    "value_range": {"min": None, "max": None},
    "outlier_zscore_threshold": 3.5,
    "pct_change_threshold": 0.5,
    "revision_whitelist": {"enabled": False},
    "severity_overrides": {},
}

DIRTY_CONFIG = {
    "dataset_name": "test_dirty",
    "format": "long",
    "date_column": "Date",
    "dimension_columns": ["Region"],
    "value_column": "Value",
    "expected_frequency": "B",
    "acceptable_lag_days": 365,
    "holidays_country": None,
    "expected_dimensions": {},
    "max_null_pct": 0.01,  # tight threshold → will fail
    "allow_zero_values": False,
    "value_range": {"min": 0, "max": 100000},
    "outlier_zscore_threshold": 3.0,
    "pct_change_threshold": 0.5,
    "revision_whitelist": {"enabled": False},
    "severity_overrides": {},
}

CONSISTENCY_CONFIG = {
    "dataset_name": "test_consistency",
    "format": "long",
    "date_column": "Date",
    "dimension_columns": ["Region"],
    "value_column": "Value",
    "expected_frequency": "B",
    "acceptable_lag_days": 365,
    "holidays_country": None,
    "expected_dimensions": {},
    "max_null_pct": 0.05,
    "allow_zero_values": True,
    "value_range": {"min": None, "max": None},
    "outlier_zscore_threshold": 3.5,
    "pct_change_threshold": 0.5,
    "revision_whitelist": {
        "enabled": True,
        "max_revision_age_days": 90,
        "max_revision_pct": 0.05,  # tight → will fail
    },
    "severity_overrides": {},
}

WIDE_CONFIG = {
    "dataset_name": "test_wide",
    "format": "wide",
    "date_column": "Date",
    "dimension_columns": [],
    "value_columns": ["Sales", "Units"],
    "expected_frequency": "B",
    "acceptable_lag_days": 365,
    "holidays_country": None,
    "expected_dimensions": {},
    "max_null_pct": 0.05,
    "allow_zero_values": True,
    "value_range": {"min": None, "max": None},
    "outlier_zscore_threshold": 3.5,
    "pct_change_threshold": 0.5,
    "revision_whitelist": {"enabled": False},
    "severity_overrides": {},
}
