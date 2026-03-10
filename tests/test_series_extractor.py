"""Tests for src/series_extractor.py"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.series_extractor import (
    ExtractionResult,
    Series,
    extract_series_long,
    extract_series_wide,
    extract_series_wide_transposed,
    extract_series_auto_platform,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_wide_df():
    dates = pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "Date": dates,
        "GDP": range(100, 112),
        "CPI": [float(i) * 1.5 for i in range(12)],
        "Notes": ["ok"] * 12,
    })


def _make_long_df():
    dates = pd.date_range("2023-01-01", periods=6, freq="MS").strftime("%Y-%m-%d")
    n = len(dates)
    return pd.DataFrame({
        "Date": list(dates) * 2,
        "Country": ["Argentina"] * n + ["Brazil"] * n,
        "Value": list(range(10, 10 + n)) + list(range(20, 20 + n)),
    })


def _make_wide_transposed_df():
    dates = pd.date_range("2023-01-01", periods=6, freq="MS").strftime("%Y-%m-%d")
    data = {"Entity": ["GDP", "CPI"]}
    for d in dates:
        data[d] = [float(i) for i in range(2)]
    return pd.DataFrame(data)


# ── extract_series_wide ────────────────────────────────────────────────────────

class TestExtractSeriesWide:
    def test_returns_extraction_result(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP", "CPI"])
        assert isinstance(result, ExtractionResult)

    def test_dataset_format_wide(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP", "CPI"])
        assert result.dataset_format == "wide"

    def test_series_count_matches_value_cols(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP", "CPI"])
        assert result.total_series == 2
        assert len(result.series) == 2

    def test_series_names_are_column_names(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP", "CPI"])
        names = {s.name for s in result.series}
        assert names == {"GDP", "CPI"}

    def test_series_has_correct_row_count(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP"])
        s = result.series[0]
        assert s.row_count == 12

    def test_dates_are_datetime64(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP"])
        assert pd.api.types.is_datetime64_any_dtype(result.series[0].dates)

    def test_values_are_float(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP"])
        assert result.series[0].values.dtype == np.float64 or \
               pd.api.types.is_float_dtype(result.series[0].values)

    def test_date_range_populated(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP"])
        dr = result.series[0].date_range
        assert dr[0] == "2023-01-01"
        assert dr[1] == "2023-12-01"

    def test_preview_has_entries(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP"])
        assert len(result.series[0].preview) > 0

    def test_ignore_cols_excluded(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP", "CPI"], ignore_cols=["CPI"])
        assert result.total_series == 1
        assert result.series[0].name == "GDP"

    def test_non_numeric_col_skipped_gracefully(self):
        df = _make_wide_df()
        # Notes column has strings — should produce series with 0 valid or be skipped
        result = extract_series_wide(df, "Date", ["GDP", "Notes"])
        # GDP should be extracted; Notes may be skipped or have 0 rows
        names = [s.name for s in result.series]
        assert "GDP" in names

    def test_source_columns_includes_date(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP"])
        assert "Date" in result.series[0].source_columns
        assert "GDP" in result.series[0].source_columns

    def test_extraction_log_not_empty(self):
        df = _make_wide_df()
        result = extract_series_wide(df, "Date", ["GDP"])
        assert len(result.extraction_log) > 0


# ── extract_series_long ────────────────────────────────────────────────────────

class TestExtractSeriesLong:
    def test_returns_extraction_result(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", ["Country"], "Value")
        assert isinstance(result, ExtractionResult)

    def test_dataset_format_long(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", ["Country"], "Value")
        assert result.dataset_format == "long"

    def test_series_count_equals_unique_keys(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", ["Country"], "Value")
        assert result.total_series == 2

    def test_series_names_include_dim_values(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", ["Country"], "Value")
        names = {s.name for s in result.series}
        assert "Argentina" in names
        assert "Brazil" in names

    def test_row_counts_correct(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", ["Country"], "Value")
        for s in result.series:
            assert s.row_count == 6

    def test_no_dimension_cols_single_series(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", [], "Value")
        assert result.total_series == 1
        assert result.series[0].name == "(all)"

    def test_multi_dim_key_joined_by_pipe(self):
        df = pd.DataFrame({
            "Date": ["2023-01-01"] * 4,
            "Country": ["AR", "AR", "BR", "BR"],
            "Sector": ["A", "B", "A", "B"],
            "Value": [1.0, 2.0, 3.0, 4.0],
        })
        result = extract_series_long(df, "Date", ["Country", "Sector"], "Value")
        names = {s.name for s in result.series}
        assert "AR | A" in names
        assert "BR | B" in names

    def test_metadata_has_key_values(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", ["Country"], "Value")
        for s in result.series:
            assert "key_values" in s.metadata
            assert "Country" in s.metadata["key_values"]

    def test_dates_sorted(self):
        df = _make_long_df().sample(frac=1, random_state=42)  # shuffle
        result = extract_series_long(df, "Date", ["Country"], "Value")
        for s in result.series:
            assert s.dates.is_monotonic_increasing

    def test_series_sorted_alphabetically(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", ["Country"], "Value")
        names = [s.name for s in result.series]
        assert names == sorted(names)

    def test_extraction_log_row_info(self):
        df = _make_long_df()
        result = extract_series_long(df, "Date", ["Country"], "Value")
        log_text = " ".join(result.extraction_log)
        assert "12" in log_text  # 12 total rows


# ── extract_series_wide_transposed ─────────────────────────────────────────────

class TestExtractSeriesWideTransposed:
    def test_returns_extraction_result(self):
        df = _make_wide_transposed_df()
        result = extract_series_wide_transposed(df, entity_col="Entity")
        assert isinstance(result, ExtractionResult)

    def test_dataset_format(self):
        df = _make_wide_transposed_df()
        result = extract_series_wide_transposed(df, entity_col="Entity")
        assert result.dataset_format == "wide_transposed"

    def test_series_count_equals_rows(self):
        df = _make_wide_transposed_df()
        result = extract_series_wide_transposed(df, entity_col="Entity")
        assert result.total_series == 2

    def test_series_names_from_entity_col(self):
        df = _make_wide_transposed_df()
        result = extract_series_wide_transposed(df, entity_col="Entity")
        names = {s.name for s in result.series}
        assert "GDP" in names
        assert "CPI" in names

    def test_dates_parsed_from_headers(self):
        df = _make_wide_transposed_df()
        result = extract_series_wide_transposed(df, entity_col="Entity")
        for s in result.series:
            assert pd.api.types.is_datetime64_any_dtype(s.dates)

    def test_row_count_matches_date_cols(self):
        df = _make_wide_transposed_df()
        n_date_cols = len(df.columns) - 1
        result = extract_series_wide_transposed(df, entity_col="Entity")
        for s in result.series:
            assert s.row_count <= n_date_cols

    def test_non_date_headers_skipped(self):
        df = pd.DataFrame({
            "Entity": ["GDP"],
            "not_a_date": [1.0],
            "also_not": [2.0],
            "2023-01-01": [3.0],
        })
        result = extract_series_wide_transposed(df, entity_col="Entity")
        assert result.total_series == 1
        assert result.series[0].row_count == 1

    def test_defaults_entity_col_to_first_column(self):
        df = _make_wide_transposed_df()
        result = extract_series_wide_transposed(df)
        assert result.total_series == 2


# ── extract_series_auto_platform ──────────────────────────────────────────────

class TestExtractSeriesAutoPlatform:
    def test_long_single_numeric(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=6, freq="MS").strftime("%Y-%m-%d"),
            "Country": ["AR"] * 3 + ["BR"] * 3,
            "Value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        result = extract_series_auto_platform(df)
        assert result.dataset_format == "long"
        assert result.total_series == 2

    def test_wide_multi_numeric(self):
        df = _make_wide_df().drop(columns=["Notes"])
        result = extract_series_auto_platform(df)
        assert result.dataset_format == "wide"
        assert result.total_series == 2

    def test_alphacast_date_column_detected(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=4, freq="MS").strftime("%Y-%m-%d"),
            "Value": [1.0, 2.0, 3.0, 4.0],
        })
        result = extract_series_auto_platform(df)
        assert result.date_column == "Date"

    def test_date_column_case_insensitive(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=4, freq="MS").strftime("%Y-%m-%d"),
            "Value": [1.0, 2.0, 3.0, 4.0],
        })
        result = extract_series_auto_platform(df)
        assert result.date_column.lower() == "date"

    def test_extraction_log_populated(self):
        df = _make_long_df()
        result = extract_series_auto_platform(df)
        assert len(result.extraction_log) > 0
