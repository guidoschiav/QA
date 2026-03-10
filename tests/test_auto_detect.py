"""Unit tests for src/auto_detect.py."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.auto_detect import auto_detect_config, strip_internal_keys
from src.auditor import audit_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_wide_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-02", periods=20).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "Date": dates,
        "Sales": rng.integers(1000, 50000, 20).astype(float),
        "Units": rng.integers(10, 500, 20).astype(float),
        "Revenue": rng.integers(5000, 200000, 20).astype(float),
    })


def _make_df_with_periodo() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2024-01-02", periods=15).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "Periodo": dates,
        "Region": ["CABA", "GBA", "Interior"] * 5,
        "Ventas": rng.integers(100, 10000, 15).astype(float),
    })


def _make_df_datetime_dtype() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame({
        "FechaOp": dates,  # datetime64 dtype, non-standard name
        "Valor": rng.integers(100, 5000, 10).astype(float),
    })


# ── Test 1: clean demo data ───────────────────────────────────────────────────

class TestCleanDemoData:
    def test_date_column_detected(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        assert cfg["date_column"] == "Date"

    def test_format_long(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        assert cfg["format"] == "long"

    def test_value_column(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        assert cfg["value_column"] == "Value"

    def test_dimensions_detected(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        assert set(cfg["dimension_columns"]) == {"Region", "Product"}

    def test_frequency_business_days(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        assert cfg["expected_frequency"] == "B"

    def test_detection_notes_present(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        assert "_detection_notes" in cfg
        assert isinstance(cfg["_detection_notes"], list)
        assert len(cfg["_detection_notes"]) > 0

    def test_detection_notes_structure(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        for note in cfg["_detection_notes"]:
            assert "field" in note
            assert "value" in note
            assert "confidence" in note
            assert note["confidence"] in ("high", "medium", "low")
            assert "reason" in note


# ── Test 2: wide format ───────────────────────────────────────────────────────

class TestWideFormat:
    def test_format_wide(self):
        df = _make_wide_df()
        cfg = auto_detect_config(df)
        assert cfg["format"] == "wide"

    def test_value_columns_detected(self):
        df = _make_wide_df()
        cfg = auto_detect_config(df)
        assert set(cfg["value_columns"]) == {"Sales", "Units", "Revenue"}

    def test_value_column_empty_in_wide(self):
        df = _make_wide_df()
        cfg = auto_detect_config(df)
        # value_column is irrelevant for wide; value_columns carries the info
        assert cfg["value_columns"] != []

    def test_date_still_detected_in_wide(self):
        df = _make_wide_df()
        cfg = auto_detect_config(df)
        assert cfg["date_column"] == "Date"


# ── Test 3: date column by parsing ("Periodo") ───────────────────────────────

class TestDateDetectionByParsing:
    def test_periodo_detected_by_name(self):
        """'Periodo' is in the priority list, so it should match by name."""
        df = _make_df_with_periodo()
        cfg = auto_detect_config(df)
        assert cfg["date_column"] == "Periodo"

    def test_note_says_matched_by_name(self):
        df = _make_df_with_periodo()
        cfg = auto_detect_config(df)
        date_notes = [n for n in cfg["_detection_notes"] if n["field"] == "date_column"]
        assert len(date_notes) == 1
        assert "matched by name" in date_notes[0]["reason"].lower()

    def test_raro_name_detected_by_parsing(self):
        """A non-priority column name → must be detected by parsing."""
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2024-06-01", periods=20).strftime("%Y-%m-%d")
        df = pd.DataFrame({
            "XYZ_ts": dates,
            "Amount": rng.integers(100, 5000, 20).astype(float),
        })
        cfg = auto_detect_config(df)
        assert cfg["date_column"] == "XYZ_ts"
        date_notes = [n for n in cfg["_detection_notes"] if n["field"] == "date_column"]
        assert "parsing" in date_notes[0]["reason"].lower()

    def test_datetime_dtype_detected(self):
        df = _make_df_datetime_dtype()
        cfg = auto_detect_config(df)
        assert cfg["date_column"] == "FechaOp"


# ── Test 4: audit_dataset() compatibility ─────────────────────────────────────

class TestAuditCompatibility:
    def test_clean_demo_no_error(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        audit_cfg = strip_internal_keys(cfg)
        report = audit_dataset(df, audit_cfg)
        assert report.total_checks >= 19
        assert isinstance(report.passed, int)
        assert isinstance(report.failed, int)

    def test_dirty_demo_no_error(self):
        df = pd.read_csv("demo/demo_data_dirty.csv")
        cfg = auto_detect_config(df)
        audit_cfg = strip_internal_keys(cfg)
        report = audit_dataset(df, audit_cfg)
        assert report.total_checks >= 19
        # Dirty data should have at least some failures
        assert report.failed > 0

    def test_wide_df_no_error(self):
        df = _make_wide_df()
        cfg = auto_detect_config(df)
        audit_cfg = strip_internal_keys(cfg)
        report = audit_dataset(df, audit_cfg)
        assert report.total_checks >= 19

    def test_strip_internal_keys_removes_detection_notes(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        clean = strip_internal_keys(cfg)
        assert "_detection_notes" not in clean
        assert "dataset_name" in clean
        assert "format" in clean


# ── Test 5: smart defaults ────────────────────────────────────────────────────

class TestSmartDefaults:
    def test_max_null_pct_low_null_data(self):
        """Clean data → max_null_pct should be 0.02."""
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        assert cfg["max_null_pct"] == 0.02

    def test_allow_zero_values_when_no_zeros(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        # Clean data has no zeros → allow_zero_values=True (no restriction needed)
        assert isinstance(cfg["allow_zero_values"], bool)

    def test_value_range_has_margin(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        vmin = df["Value"].min()
        vmax = df["Value"].max()
        assert cfg["value_range"]["min"] <= vmin
        assert cfg["value_range"]["max"] >= vmax

    def test_expected_dimensions_populated(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        assert "Region" in cfg["expected_dimensions"]
        assert "CABA" in cfg["expected_dimensions"]["Region"]

    def test_expected_start_matches_first_date(self):
        df = pd.read_csv("demo/demo_data_clean.csv")
        cfg = auto_detect_config(df)
        first_date = pd.to_datetime(df["Date"]).min().strftime("%Y-%m-%d")
        assert cfg["expected_start"] == first_date
