"""Tests for src/canonicalizer.py"""
import pandas as pd
import numpy as np
import pytest

from src.canonicalizer import (
    canonicalize,
    detect_header_row,
    detect_date_format,
    detect_number_format,
    compose_date_column,
    CUSTOM_NULL_VALUES,
)
from src.alphacast_profile import ALPHACAST_PROFILE, apply_alphacast_profile


def make_long_df():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Region": ["CABA", "GBA", "CABA"],
        "Product": ["A", "A", "B"],
        "Value": [100.0, 200.0, 150.0],
    })


def make_wide_df():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "CABA": [100.0, 110.0],
        "GBA": [200.0, 210.0],
        "Interior": [300.0, 310.0],
    })


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Original tests â€” unchanged
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestLongFormat:
    def test_basic_long(self):
        df = make_long_df()
        result = canonicalize(df, date_col="Date", key_cols=["Region", "Product"],
                               value_col="Value", fmt="long")
        assert list(result.df.columns) == ["Date", "Key", "Value"]
        assert len(result.df) == 3
        assert result.format_original == "long"

    def test_date_parsed(self):
        df = make_long_df()
        result = canonicalize(df, date_col="Date", key_cols=["Region"], value_col="Value")
        assert pd.api.types.is_datetime64_any_dtype(result.df["Date"])

    def test_composite_key(self):
        df = make_long_df()
        result = canonicalize(df, date_col="Date", key_cols=["Region", "Product"],
                               value_col="Value")
        assert "caba|a" in result.df["Key"].values

    def test_no_key_cols(self):
        df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Value": [1.0, 2.0]})
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert (result.df["Key"] == "_total_").all()

    def test_nat_rows_dropped(self):
        df = pd.DataFrame({
            "Date": ["2024-01-01", "not-a-date", "2024-01-03"],
            "Value": [1.0, 2.0, 3.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert len(result.df) == 2
        assert any("Dropped" in l for l in result.normalization_log)

    def test_key_normalized_lowercase(self):
        df = make_long_df()
        result = canonicalize(df, date_col="Date", key_cols=["Region"], value_col="Value",
                               normalizations={"lowercase_keys": True})
        assert all(k == k.lower() for k in result.df["Key"])

    def test_duplicate_dedup(self):
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-01"],
            "Region": ["CABA", "CABA"],
            "Value": [100.0, 200.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=["Region"], value_col="Value")
        assert len(result.df) == 1
        assert result.df.iloc[0]["Value"] == 200.0  # kept last


class TestWideFormat:
    def test_wide_to_long(self):
        df = make_wide_df()
        result = canonicalize(df, date_col="Date", key_cols=[], value_col=None,
                               value_cols=["CABA", "GBA", "Interior"], fmt="wide")
        assert list(result.df.columns) == ["Date", "Key", "Value"]
        assert len(result.df) == 6  # 2 dates Ã— 3 cols
        assert result.format_original == "wide"

    def test_wide_key_contains_col_name(self):
        df = make_wide_df()
        result = canonicalize(df, date_col="Date", key_cols=[], value_col=None,
                               value_cols=["CABA", "GBA"], fmt="wide")
        assert "caba" in result.df["Key"].values


class TestValueParsing:
    def test_currency_stripped(self):
        df = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Value": ["$1,234.56"],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert abs(result.df.iloc[0]["Value"] - 1234.56) < 0.01

    def test_european_decimal(self):
        df = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Value": ["1.234,56"],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert abs(result.df.iloc[0]["Value"] - 1234.56) < 0.01

    def test_numeric_column_unchanged(self):
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Value": [100.5, 200.3],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert abs(result.df.iloc[0]["Value"] - 100.5) < 1e-9


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# New tests â€” header detection
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDetectHeaderRow:
    def test_junk_rows_at_top(self):
        """3 junk rows + 1 header row â†’ detect header at index 3."""
        df_raw = pd.DataFrame([
            ["Report Q1-2025", None, None, None],       # row 0: junk (1 string)
            ["Source: Government", None, None, None],   # row 1: junk (1 string)
            [None, None, None, None],                   # row 2: empty junk (0 strings)
            ["Date", "Region", "Product", "Value"],     # row 3: HEADER (4 strings)
            ["2025-01-01", "CABA", "A", 100.0],         # row 4: data
            ["2025-01-02", "GBA", "B", 200.0],          # row 5: data
        ])
        assert detect_header_row(df_raw) == 3

    def test_already_loaded_returns_zero(self):
        """DataFrame with string column names â†’ returns 0 (header already applied)."""
        df = make_long_df()
        assert detect_header_row(df) == 0

    def test_no_junk_rows(self):
        """DataFrame loaded without header (int columns), header is row 0."""
        df_raw = pd.DataFrame([
            ["Date", "Region", "Value"],   # row 0: HEADER
            ["2025-01-01", "CABA", 100.0], # row 1: data
        ])
        # columns are 0, 1, 2 (ints)
        assert detect_header_row(df_raw) == 0

    def test_header_row_applied_via_canonicalize(self):
        """canonicalize with header_row='auto' re-interprets a raw DataFrame."""
        df_raw = pd.DataFrame([
            ["Informe Enero 2025", None, None],  # junk
            ["Date", "Region", "Value"],          # header at row 1
            ["2025-01-01", "CABA", "100.0"],      # data
            ["2025-01-02", "GBA", "200.0"],        # data
        ])
        result = canonicalize(
            df_raw,
            date_col="Date",
            key_cols=["Region"],
            value_col="Value",
            header_row="auto",
        )
        assert len(result.df) == 2
        assert any("Header detected" in l for l in result.normalization_log)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# New tests â€” date format detection & parsing
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDateFormats:
    def test_dmy_slash(self):
        """DD/MM/YYYY â†’ parsed as 2025-03-15."""
        df = pd.DataFrame({
            "Date": ["15/03/2025", "20/03/2025"],
            "Value": [100.0, 200.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert result.df.iloc[0]["Date"] == pd.Timestamp("2025-03-15")
        assert result.df.iloc[1]["Date"] == pd.Timestamp("2025-03-20")

    def test_detect_date_format_dmy(self):
        """detect_date_format identifies DD/MM/YYYY as dayfirst=True."""
        s = pd.Series(["15/03/2025", "20/03/2025", "28/01/2025"])
        info = detect_date_format(s)
        assert info["dayfirst"] is True
        assert info["format"] == "dmy"

    def test_spanish_month_text(self):
        """'Ene 2025' â†’ 2025-01-01."""
        df = pd.DataFrame({
            "Date": ["Ene 2025", "Feb 2025", "Mar 2025"],
            "Value": [100.0, 200.0, 300.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert result.df.iloc[0]["Date"] == pd.Timestamp("2025-01-01")
        assert result.df.iloc[1]["Date"] == pd.Timestamp("2025-02-01")

    def test_spanish_month_full(self):
        """'Enero 2025' â†’ 2025-01-01."""
        df = pd.DataFrame({
            "Date": ["Enero 2025", "Febrero 2025"],
            "Value": [100.0, 200.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert result.df.iloc[0]["Date"] == pd.Timestamp("2025-01-01")
        assert result.df.iloc[1]["Date"] == pd.Timestamp("2025-02-01")

    def test_quarterly_q1(self):
        """'Q1 2025' â†’ 2025-01-01."""
        df = pd.DataFrame({
            "Date": ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"],
            "Value": [100.0, 200.0, 300.0, 400.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        expected = [
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-04-01"),
            pd.Timestamp("2025-07-01"),
            pd.Timestamp("2025-10-01"),
        ]
        for i, exp in enumerate(expected):
            assert result.df.iloc[i]["Date"] == exp

    def test_quarterly_year_q_format(self):
        """'2025-Q1' â†’ 2025-01-01."""
        df = pd.DataFrame({
            "Date": ["2025-Q1", "2025-Q2"],
            "Value": [100.0, 200.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert result.df.iloc[0]["Date"] == pd.Timestamp("2025-01-01")
        assert result.df.iloc[1]["Date"] == pd.Timestamp("2025-04-01")

    def test_detect_date_format_quarterly(self):
        """detect_date_format identifies quarterly format."""
        s = pd.Series(["Q1 2025", "Q2 2025", "Q3 2025"])
        info = detect_date_format(s)
        assert "quarterly" in info["format"]
        assert info["frequency_hint"] == "Q"

    def test_detect_date_format_iso(self):
        """Standard ISO dates get high confidence."""
        s = pd.Series(["2025-01-01", "2025-02-01", "2025-03-01"])
        info = detect_date_format(s)
        assert info["format"] == "iso"
        assert info["confidence"] == "high"

    def test_iso_week(self):
        """'2025-W03' â†’ Monday of week 3 of 2025."""
        df = pd.DataFrame({
            "Date": ["2025-W03", "2025-W04"],
            "Value": [100.0, 200.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        # Week 3 of 2025 starts on Monday 2025-01-13
        assert result.df.iloc[0]["Date"] == pd.Timestamp.fromisocalendar(2025, 3, 1)

    def test_detect_date_format_campaign_year_span(self):
        """detect_date_format identifies campaign year span format."""
        s = pd.Series(["2025/2026", "2024-2025", "2023/24"])
        info = detect_date_format(s)
        assert info["format"] == "campaign_year_span"
        assert info["frequency_hint"] == "A"

    def test_campaign_year_span(self):
        """Campaign strings map to the first year as YYYY-01-01."""
        df = pd.DataFrame({
            "Date": ["2025/2026", "2024-2025", "2023/24"],
            "Value": [100.0, 200.0, 300.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        parsed_dates = list(result.df["Date"])
        assert parsed_dates == [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2025-01-01"),
        ]

    def test_year_only(self):
        """'2025' â†’ 2025-01-01."""
        df = pd.DataFrame({
            "Date": ["2023", "2024", "2025"],
            "Value": [100.0, 200.0, 300.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert result.df.iloc[0]["Date"] == pd.Timestamp("2023-01-01")
        assert result.df.iloc[2]["Date"] == pd.Timestamp("2025-01-01")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# New tests â€” number format detection & parsing
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestNumberFormats:
    def _make_df(self, values):
        dates = [f"2024-01-{i+1:02d}" for i in range(len(values))]
        return pd.DataFrame({"Date": dates, "Value": values})

    def test_european_number_format(self):
        """'1.234,56' â†’ 1234.56 via auto-detection."""
        df = self._make_df(["1.234,56", "2.345,67"])
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert abs(result.df.iloc[0]["Value"] - 1234.56) < 0.01
        assert abs(result.df.iloc[1]["Value"] - 2345.67) < 0.01

    def test_currency_dollar(self):
        """'$1,234.56' â†’ 1234.56."""
        df = self._make_df(["$1,234.56", "$2,345.67"])
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert abs(result.df.iloc[0]["Value"] - 1234.56) < 0.01

    def test_currency_euro(self):
        """'€1.234,56' -> 1234.56 (European format with euro sign)."""
        df = self._make_df(["€1.234,56"])
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert abs(result.df.iloc[0]["Value"] - 1234.56) < 0.01

    def test_percentage_values(self):
        """'15.5%' â†’ 15.5 (strip % sign, do not divide)."""
        df = self._make_df(["15.5%", "3.2%"])
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert abs(result.df.iloc[0]["Value"] - 15.5) < 0.01
        assert abs(result.df.iloc[1]["Value"] - 3.2) < 0.01

    def test_parentheses_negative(self):
        """'(500)' â†’ -500.0."""
        df = self._make_df(["(500)", "(1200)"])
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert abs(result.df.iloc[0]["Value"] - (-500.0)) < 0.01
        assert abs(result.df.iloc[1]["Value"] - (-1200.0)) < 0.01

    def test_detect_number_format_european(self):
        s = pd.Series(["1.234,56", "2.345,67"])
        fmt = detect_number_format(s)
        assert fmt["decimal"] == ","
        assert fmt["thousands"] == "."

    def test_detect_number_format_standard(self):
        s = pd.Series(["1,234.56", "2,345.67"])
        fmt = detect_number_format(s)
        assert fmt["decimal"] == "."
        assert fmt["thousands"] == ","

    def test_detect_number_format_percentage(self):
        s = pd.Series(["15.5%", "3.2%"])
        fmt = detect_number_format(s)
        assert fmt["is_percentage"] is True

    def test_detect_number_format_currency(self):
        s = pd.Series(["$1,234.56", "$2,345.67"])
        fmt = detect_number_format(s)
        assert fmt["currency"] == "$"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# New tests â€” null values
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestNullValues:
    def _make_df(self, values):
        return pd.DataFrame({
            "Date": [f"2024-01-{i+1:02d}" for i in range(len(values))],
            "Value": values,
        })

    def test_custom_null_nd(self):
        """'n/d' in value column â†’ treated as NaN."""
        df = self._make_df(["100.0", "n/d", "200.0"])
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert result.df.iloc[1]["Value"] != result.df.iloc[1]["Value"]  # isnan

    def test_custom_null_dots(self):
        """'...' â†’ treated as NaN."""
        df = self._make_df(["100.0", "...", "300.0"])
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert pd.isna(result.df.iloc[1]["Value"])

    def test_custom_null_dash(self):
        """'-' (single dash) â†’ treated as NaN."""
        df = self._make_df(["100.0", "-", "300.0"])
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value")
        assert pd.isna(result.df.iloc[1]["Value"])

    def test_extra_custom_null(self):
        """User-supplied null value 'missing' â†’ treated as NaN."""
        df = self._make_df(["100.0", "missing", "300.0"])
        result = canonicalize(
            df, date_col="Date", key_cols=[], value_col="Value",
            custom_null_values=["missing"],
        )
        assert pd.isna(result.df.iloc[1]["Value"])

    def test_custom_null_values_constant(self):
        """CUSTOM_NULL_VALUES contains expected strings."""
        assert "n/d" in CUSTOM_NULL_VALUES
        assert "..." in CUSTOM_NULL_VALUES
        assert "N/A" in CUSTOM_NULL_VALUES
        assert "#REF!" in CUSTOM_NULL_VALUES


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# New tests â€” scale
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestScale:
    def test_scale_1000(self):
        """scale=1000 multiplies all values by 1000."""
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Value": [1.5, 2.5],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value", scale=1000.0)
        assert abs(result.df.iloc[0]["Value"] - 1500.0) < 1e-9
        assert abs(result.df.iloc[1]["Value"] - 2500.0) < 1e-9

    def test_scale_1_no_change(self):
        """scale=1.0 (default) leaves values unchanged."""
        df = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Value": [42.0],
        })
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value", scale=1.0)
        assert abs(result.df.iloc[0]["Value"] - 42.0) < 1e-9

    def test_scale_logged(self):
        """scale != 1 is logged in normalization_log."""
        df = pd.DataFrame({"Date": ["2024-01-01"], "Value": [1.0]})
        result = canonicalize(df, date_col="Date", key_cols=[], value_col="Value", scale=0.001)
        assert any("scale" in l.lower() for l in result.normalization_log)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# New tests â€” skip_rows_bottom
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestSkipRowsBottom:
    def test_skip_one_footer(self):
        """skip_rows_bottom=1 removes the last row (e.g. 'Total' row)."""
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02", "Total"],
            "Value": [100.0, 200.0, 300.0],
        })
        result = canonicalize(
            df, date_col="Date", key_cols=[], value_col="Value",
            skip_rows_bottom=1,
        )
        assert len(result.df) == 2

    def test_skip_zero_no_change(self):
        df = make_long_df()
        result_no_skip = canonicalize(df, date_col="Date", key_cols=["Region"], value_col="Value")
        result_skip0 = canonicalize(df, date_col="Date", key_cols=["Region"], value_col="Value",
                                    skip_rows_bottom=0)
        assert len(result_no_skip.df) == len(result_skip0.df)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# New tests â€” Alphacast profile
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestAlphacastProfile:
    def test_profile_has_required_keys(self):
        assert "header_row" in ALPHACAST_PROFILE
        assert "date_column_names" in ALPHACAST_PROFILE
        assert "decimal" in ALPHACAST_PROFILE
        assert "encoding" in ALPHACAST_PROFILE

    def test_apply_profile_fills_platform(self):
        config = {
            "source_name": "Raw",
            "platform_name": "Alphacast",
            "tolerance": 0.01,
            "source": {
                "format": "long",
                "date_column": "Date",
                "key_columns": ["Country"],
                "value_column": "Value",
            },
        }
        result = apply_alphacast_profile(config)
        assert "platform" in result
        plat = result["platform"]
        assert plat["date_column"] == "Date"
        assert plat["header_row"] == 0
        assert plat["normalizations"]["lowercase_keys"] is True

    def test_apply_profile_user_values_take_priority(self):
        """Explicit platform config values override profile defaults."""
        config = {
            "source": {"format": "long", "date_column": "Date",
                       "key_columns": ["R"], "value_column": "V"},
            "platform": {
                "date_column": "report_date",   # user override
                "key_columns": ["region"],
            },
        }
        result = apply_alphacast_profile(config)
        plat = result["platform"]
        assert plat["date_column"] == "report_date"   # user value kept
        assert plat["key_columns"] == ["region"]       # user value kept
        assert plat["header_row"] == 0                 # profile default filled in

    def test_apply_profile_returns_valid_config(self):
        """Output of apply_alphacast_profile can be used in a compare operation."""
        config = {
            "source_name": "Source",
            "platform_name": "Alphacast",
            "tolerance": 0.01,
            "source": {
                "format": "long",
                "date_column": "Date",
                "key_columns": ["Region"],
                "value_column": "Value",
            },
        }
        result = apply_alphacast_profile(config)
        # Should have all required keys for comparator.py
        assert result["source"]["format"] == "long"
        assert result["platform"]["format"] == "long"  # inherited from source
        assert isinstance(result["platform"]["number_format"], dict)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# New tests â€” compose_date_column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestComposeDateColumn:
    def _df(self, years, periods):
        return pd.DataFrame({"AÃ±o": years, "Periodo": periods})

    def test_month_numeric_basic(self):
        """year + numeric month â†’ correct timestamps."""
        df = self._df([2024, 2024, 2024], [1, 6, 12])
        result = compose_date_column(df, "AÃ±o", "Periodo", "month_numeric")
        assert result.iloc[0] == pd.Timestamp("2024-01-01")
        assert result.iloc[1] == pd.Timestamp("2024-06-01")
        assert result.iloc[2] == pd.Timestamp("2024-12-01")

    def test_month_text_es(self):
        """Spanish month names â†’ correct timestamps."""
        df = self._df([2024, 2024, 2024], ["enero", "feb", "Marzo"])
        result = compose_date_column(df, "AÃ±o", "Periodo", "month_text_es")
        assert result.iloc[0] == pd.Timestamp("2024-01-01")
        assert result.iloc[1] == pd.Timestamp("2024-02-01")
        assert result.iloc[2] == pd.Timestamp("2024-03-01")

    def test_month_text_en(self):
        """English month names â†’ correct timestamps."""
        df = self._df([2023, 2023], ["January", "Dec"])
        result = compose_date_column(df, "AÃ±o", "Periodo", "month_text_en")
        assert result.iloc[0] == pd.Timestamp("2023-01-01")
        assert result.iloc[1] == pd.Timestamp("2023-12-01")

    def test_quarter_numeric(self):
        """Numeric quarters 1-4 â†’ first month of each quarter."""
        df = self._df([2025, 2025, 2025, 2025], [1, 2, 3, 4])
        result = compose_date_column(df, "AÃ±o", "Periodo", "quarter_numeric")
        assert result.iloc[0] == pd.Timestamp("2025-01-01")
        assert result.iloc[1] == pd.Timestamp("2025-04-01")
        assert result.iloc[2] == pd.Timestamp("2025-07-01")
        assert result.iloc[3] == pd.Timestamp("2025-10-01")

    def test_quarter_text_q(self):
        """'Q1'/'Q2'/'Q3'/'Q4' â†’ first month of each quarter."""
        df = self._df([2024, 2024, 2024, 2024], ["Q1", "Q2", "Q3", "Q4"])
        result = compose_date_column(df, "AÃ±o", "Periodo", "quarter_text_q")
        assert result.iloc[0] == pd.Timestamp("2024-01-01")
        assert result.iloc[1] == pd.Timestamp("2024-04-01")
        assert result.iloc[2] == pd.Timestamp("2024-07-01")
        assert result.iloc[3] == pd.Timestamp("2024-10-01")

    def test_quarter_text_t(self):
        """'1T'/'2T'/'3T'/'4T' (Spanish format) â†’ first month of each quarter."""
        df = self._df([2023, 2023, 2023], ["1T", "2T", "4T"])
        result = compose_date_column(df, "AÃ±o", "Periodo", "quarter_text_t")
        assert result.iloc[0] == pd.Timestamp("2023-01-01")
        assert result.iloc[1] == pd.Timestamp("2023-04-01")
        assert result.iloc[2] == pd.Timestamp("2023-10-01")

    def test_year_forward_fill(self):
        """NaN years are forward-filled (Excel merged-cell pattern)."""
        df = self._df([2024, None, None, 2025], [1, 2, 3, 1])
        result = compose_date_column(df, "AÃ±o", "Periodo", "month_numeric")
        assert result.iloc[0] == pd.Timestamp("2024-01-01")
        assert result.iloc[1] == pd.Timestamp("2024-02-01")  # ffill from 2024
        assert result.iloc[2] == pd.Timestamp("2024-03-01")  # ffill from 2024
        assert result.iloc[3] == pd.Timestamp("2025-01-01")

    def test_invalid_period_yields_nat(self):
        """Unparseable period values â†’ NaT (no crash)."""
        df = self._df([2024, 2024], ["invalid_month", None])
        result = compose_date_column(df, "AÃ±o", "Periodo", "month_text_es")
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

    def test_unknown_period_type_all_nat(self):
        """Unknown period_type â†’ all NaT."""
        df = self._df([2024, 2024], [1, 2])
        result = compose_date_column(df, "AÃ±o", "Periodo", "unknown_type")
        assert result.isna().all()


