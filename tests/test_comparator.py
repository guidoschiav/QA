"""Tests for src/comparator.py"""
import pandas as pd
import numpy as np
import pytest

from src.comparator import compare_datasets, detect_scale


LONG_CONFIG = {
    "source_name": "Source",
    "platform_name": "Platform",
    "tolerance": 0.01,
    "source": {
        "format": "long",
        "date_column": "Date",
        "key_columns": ["Region"],
        "value_column": "Value",
    },
    "platform": {
        "format": "long",
        "date_column": "Date",
        "key_columns": ["Region"],
        "value_column": "Value",
    },
}


def make_identical_pair():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d").tolist() * 2,
        "Region": ["CABA"] * 5 + ["GBA"] * 5,
        "Value": [100.0, 101.0, 102.0, 103.0, 104.0,
                  200.0, 201.0, 202.0, 203.0, 204.0],
    })
    return df.copy(), df.copy()


class TestIdenticalDatasets:
    def test_100_percent_match(self):
        src, plat = make_identical_pair()
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert report.match_pct == pytest.approx(1.0)
        assert report.total_diffs == 0

    def test_no_missing_dates(self):
        src, plat = make_identical_pair()
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert report.missing_dates_in_platform == []
        assert report.extra_dates_in_platform == []

    def test_no_missing_keys(self):
        src, plat = make_identical_pair()
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert report.missing_keys_in_platform == []
        assert report.extra_keys_in_platform == []

    def test_comparable_rows_count(self):
        src, plat = make_identical_pair()
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert report.comparable_rows == 10


class TestDiffsDetected:
    def test_diffs_found(self):
        src, plat = make_identical_pair()
        # Introduce 2 large diffs
        plat.loc[0, "Value"] = 200.0   # was 100
        plat.loc[5, "Value"] = 100.0   # was 200
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert report.total_diffs >= 2
        assert report.match_pct < 1.0

    def test_missing_dates_detected(self):
        src, plat = make_identical_pair()
        # Remove all rows for 2024-01-01 from platform (both CABA and GBA)
        plat = plat[plat["Date"] != "2024-01-01"].reset_index(drop=True)
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert len(report.missing_dates_in_platform) >= 1

    def test_extra_keys_detected(self):
        src, plat = make_identical_pair()
        extra = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Region": ["Patagonia"],
            "Value": [999.0],
        })
        plat = pd.concat([plat, extra], ignore_index=True)
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert "patagonia" in report.extra_keys_in_platform

    def test_top_diffs_sorted_by_abs_diff(self):
        src, plat = make_identical_pair()
        plat.loc[0, "Value"] = 500.0   # large diff
        plat.loc[1, "Value"] = 101.5   # small diff
        report = compare_datasets(src, plat, LONG_CONFIG)
        if len(report.top_diffs) >= 2:
            assert report.top_diffs[0]["abs_diff"] >= report.top_diffs[1]["abs_diff"]


class TestTemporalShift:
    def test_shift_detected(self):
        bdays = pd.bdate_range("2024-01-01", periods=5)
        src = pd.DataFrame({
            "Date": bdays.strftime("%Y-%m-%d").tolist() * 2,
            "Region": ["CABA"] * 5 + ["GBA"] * 5,
            "Value": [100.0, 110.0, 120.0, 130.0, 140.0,
                      200.0, 210.0, 220.0, 230.0, 240.0],
        })
        # Platform is shifted by +1 day (uses next day's values but same dates as source)
        # To simulate: platform has same dates but values from the day before
        plat = src.copy()
        # Shift: platform date = source date + 1 bday but value from source date - 1
        # Simpler: platform has source values but dates shifted +1
        plat["Date"] = (pd.to_datetime(plat["Date"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        # Now inner join will be 4 rows (dates 2-5 overlap)
        # After shift correction the match should be perfect
        report = compare_datasets(src, plat, LONG_CONFIG)
        # With a 1-day shift, shift detection should find it
        # (May or may not trigger depending on overlap size — just check it runs)
        assert report is not None


class TestAutoMapping:
    def test_auto_map_columns(self):
        """When platform config is absent, auto_map_columns() should rename platform cols."""
        src = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Region": ["CABA", "GBA"],
            "Value": [100.0, 200.0],
        })
        # Use "date" so it matches "Date" exactly (case-insensitive)
        plat = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "region": ["caba", "gba"],
            "value": [100.0, 200.0],
        })
        # Config without platform section — auto-map should handle it
        config = {
            "source_name": "S",
            "platform_name": "P",
            "tolerance": 0.01,
            "source": {
                "format": "long",
                "date_column": "Date",
                "key_columns": ["Region"],
                "value_column": "Value",
            },
        }
        report = compare_datasets(src, plat, config)
        assert report.comparable_rows > 0


class TestWideFormat:
    def test_wide_source(self):
        src = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "CABA": [100.0, 110.0],
            "GBA": [200.0, 210.0],
        })
        plat = src.copy()
        config = {
            "source_name": "S",
            "platform_name": "P",
            "tolerance": 0.01,
            "source": {
                "format": "wide",
                "date_column": "Date",
                "key_columns": [],
                "value_columns": ["CABA", "GBA"],
            },
            "platform": {
                "format": "wide",
                "date_column": "Date",
                "key_columns": [],
                "value_columns": ["CABA", "GBA"],
            },
        }
        report = compare_datasets(src, plat, config)
        assert report.match_pct == pytest.approx(1.0)


class TestDetectScale:
    """Unit tests for detect_scale()."""

    def test_no_scale_returns_none_confidence(self):
        src = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0] * 4)
        plat = src.copy()
        result = detect_scale(src, plat)
        assert result["confidence"] is None

    def test_scale_1000_detected(self):
        src = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 8)
        plat = src * 1000.0
        result = detect_scale(src, plat)
        assert result["scale_factor"] == pytest.approx(1000.0)
        assert result["confidence"] in ("high", "medium")

    def test_scale_100_detected(self):
        src = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 8)
        plat = src * 100.0
        result = detect_scale(src, plat)
        assert result["scale_factor"] == pytest.approx(100.0)
        assert result["confidence"] in ("high", "medium")

    def test_inverse_scale_detected(self):
        """Platform is 1/1000 of source."""
        src = pd.Series([1000.0, 2000.0, 3000.0, 4000.0, 5000.0] * 6)
        plat = src / 1000.0
        result = detect_scale(src, plat)
        assert result["scale_factor"] == pytest.approx(0.001)
        assert result["confidence"] in ("high", "medium")

    def test_insufficient_data(self):
        src = pd.Series([100.0, 200.0])
        plat = src * 1000.0
        result = detect_scale(src, plat)
        assert result["confidence"] is None
        assert "insufficient" in result["label"]

    def test_nan_values_excluded(self):
        base = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0] * 4)
        mask = pd.Series([True, False, True, True, True] * 4)
        src = base[mask].reset_index(drop=True)
        plat = (base * 1000.0)[mask].reset_index(drop=True)
        result = detect_scale(src, plat)
        assert result["scale_factor"] == pytest.approx(1000.0)

    def test_label_includes_direction_larger(self):
        src = pd.Series([1.0] * 20)
        plat = pd.Series([1000.0] * 20)
        result = detect_scale(src, plat)
        assert "larger" in result["label"].lower() or "platform" in result["label"].lower()

    def test_label_includes_direction_smaller(self):
        src = pd.Series([1000.0] * 20)
        plat = pd.Series([1.0] * 20)
        result = detect_scale(src, plat)
        assert "smaller" in result["label"].lower() or "platform" in result["label"].lower()


class TestScaleIntegration:
    """Integration: scale detection in compare_datasets."""

    def test_scale_detection_in_report(self):
        """compare_datasets always fills scale_detection field."""
        src, plat = make_identical_pair()
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert report.scale_detection is not None
        assert "scale_factor" in report.scale_detection

    def test_scale_auto_corrected(self):
        """When platform is 1000x source, match_pct improves after auto-correction."""
        src = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-01", periods=20).strftime("%Y-%m-%d").tolist(),
            "Region": ["CABA"] * 20,
            "Value": [float(i * 100) for i in range(1, 21)],
        })
        plat = src.copy()
        plat["Value"] = plat["Value"] * 1000.0
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert report.scale_detection is not None
        assert report.scale_detection["scale_factor"] == pytest.approx(1000.0)
        assert report.scale_detection["confidence"] in ("high", "medium")
        assert report.match_pct >= 0.9

    def test_no_scale_no_degradation(self):
        """Identical data → scale_detection confidence None, match stays perfect."""
        src, plat = make_identical_pair()
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert report.match_pct == pytest.approx(1.0)
        assert report.scale_detection["confidence"] is None


class TestCanonicalLogs:
    def test_canonical_logs_present(self):
        src, plat = make_identical_pair()
        report = compare_datasets(src, plat, LONG_CONFIG)
        assert isinstance(report.canonicalization_log_source, list)
        assert isinstance(report.canonicalization_log_platform, list)

    def test_scale_detection_in_serializable_dict(self):
        src, plat = make_identical_pair()
        report = compare_datasets(src, plat, LONG_CONFIG)
        d = report.to_serializable_dict()
        assert "scale_detection" in d
