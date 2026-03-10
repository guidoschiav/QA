"""Utility helpers: format detection, date parsing, wide→long conversion, holidays."""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# Argentine holidays 2024-2025 (hardcoded — no external library)
# ---------------------------------------------------------------------------
AR_HOLIDAYS: set[str] = {
    # 2024
    "2024-01-01", "2024-02-12", "2024-02-13", "2024-03-24", "2024-03-28",
    "2024-03-29", "2024-04-02", "2024-05-01", "2024-05-25", "2024-06-17",
    "2024-06-20", "2024-07-09", "2024-08-17", "2024-10-12", "2024-11-18",
    "2024-12-08", "2024-12-25",
    # 2025
    "2025-01-01", "2025-03-03", "2025-03-04", "2025-03-24", "2025-04-02",
    "2025-04-17", "2025-04-18", "2025-05-01", "2025-05-25", "2025-06-16",
    "2025-06-20", "2025-07-09", "2025-08-17", "2025-10-12", "2025-11-20",
    "2025-11-21", "2025-12-08", "2025-12-25",
}


def get_holidays(country: str | None) -> set[str]:
    if country and country.upper() == "AR":
        return AR_HOLIDAYS
    return set()


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a series to datetime64[ns], accepting strings or datetimes."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")


# ---------------------------------------------------------------------------
# Format detection & wide→long conversion
# ---------------------------------------------------------------------------

def detect_format(df: pd.DataFrame, config: dict) -> str:
    """Return 'long' or 'wide' based on config, falling back to heuristics."""
    if "format" in config:
        return config["format"]
    # Heuristic: if there's a 'Value' column → long
    if "Value" in df.columns or config.get("value_column") in df.columns:
        return "long"
    return "wide"


def wide_to_long(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Convert a wide-format DataFrame to long format.
    Expects date_column and value_columns in config.
    Returns DataFrame with columns: [date_column, 'variable', 'Value']
    where 'variable' is added to dimension_columns.
    """
    date_col = config.get("date_column", "Date")
    value_cols = config.get("value_columns", [])
    dim_cols = [c for c in df.columns if c != date_col and c not in value_cols]

    id_vars = [date_col] + dim_cols
    long_df = df.melt(id_vars=id_vars, value_vars=value_cols,
                      var_name="variable", value_name="Value")
    return long_df


def normalize_to_long(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Ensure DataFrame is in long format.
    Returns (long_df, updated_config) with dimension_columns adjusted.
    """
    fmt = detect_format(df, config)
    if fmt == "wide":
        long_df = wide_to_long(df, config)
        updated_config = dict(config)
        updated_config["format"] = "long"
        updated_config["value_column"] = "Value"
        dim_cols = list(config.get("dimension_columns", []))
        if "variable" not in dim_cols:
            dim_cols = dim_cols + ["variable"]
        updated_config["dimension_columns"] = dim_cols
        return long_df, updated_config
    return df, config


# ---------------------------------------------------------------------------
# Frequency inference
# ---------------------------------------------------------------------------

FREQ_ALIASES = {
    "D": "daily",
    "B": "business_days",
    "W": "weekly",
    "M": "monthly",
    "Q": "quarterly",
    "A": "annual",
}


def infer_frequency(dates: pd.Series) -> str | None:
    """
    Infer the dominant frequency of a date series.
    Returns pandas frequency string or None.
    """
    sorted_dates = dates.dropna().sort_values().unique()
    if len(sorted_dates) < 2:
        return None

    sorted_dates = pd.DatetimeIndex(sorted_dates)
    diffs = pd.Series(sorted_dates[1:] - sorted_dates[:-1]).dt.days

    # Use median to be robust to gaps
    median_diff = diffs.median()

    if median_diff <= 1.5:
        # Distinguish daily (D) from business days (B):
        # if no weekend dates are present, it's likely business days
        dt_index = pd.DatetimeIndex(sorted_dates)
        has_weekends = (dt_index.dayofweek >= 5).any()
        return "D" if has_weekends else "B"
    elif median_diff <= 5:
        return "B"
    elif median_diff <= 10:
        return "W"
    elif median_diff <= 40:
        return "M"
    elif median_diff <= 100:
        return "Q"
    else:
        return "A"


def generate_expected_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
    holidays: set[str] | None = None,
) -> pd.DatetimeIndex:
    """Generate expected date range given frequency."""
    if frequency == "B":
        expected = pd.bdate_range(start=start, end=end)
        if holidays:
            holiday_ts = pd.DatetimeIndex([pd.Timestamp(h) for h in holidays])
            expected = expected.difference(holiday_ts)
    elif frequency == "D":
        expected = pd.date_range(start=start, end=end, freq="D")
    elif frequency == "W":
        expected = pd.date_range(start=start, end=end, freq="W")
    elif frequency == "M":
        expected = pd.date_range(start=start, end=end, freq="MS")
    elif frequency == "Q":
        expected = pd.date_range(start=start, end=end, freq="QS")
    elif frequency == "A":
        expected = pd.date_range(start=start, end=end, freq="YS")
    else:
        expected = pd.date_range(start=start, end=end, freq="D")
    return expected


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def get_value_columns(df: pd.DataFrame, config: dict) -> List[str]:
    """Return list of numeric value column names based on config format."""
    fmt = config.get("format", "long")
    if fmt == "long":
        vc = config.get("value_column", "Value")
        return [vc] if vc in df.columns else []
    else:
        return config.get("value_columns", [])


def get_dimension_columns(df: pd.DataFrame, config: dict) -> List[str]:
    """Return list of dimension column names that exist in df."""
    dims = config.get("dimension_columns", [])
    return [d for d in dims if d in df.columns]
