"""Unit tests for individual check categories."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.checks import temporal, schema, integrity, statistical, consistency
from src.models import CheckResult
from src.config_loader import config_from_dict
from tests.fixtures import (
    make_df_clean, make_df_gaps, make_df_dirty,
    make_df_revised, make_df_snapshot_prev, make_df_wide,
    CLEAN_CONFIG, GAPS_CONFIG, DIRTY_CONFIG, CONSISTENCY_CONFIG, WIDE_CONFIG,
)


# ---------------------------------------------------------------------------
# Temporal checks
# ---------------------------------------------------------------------------

class TestTemporalChecks:
    def test_t1_frequency_correct(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        results = temporal.run_checks(df, config)
        t1 = next(r for r in results if r.check_id == "T1")
        assert t1.passed, f"T1 should pass on clean data: {t1.message}"

    def test_t2_gaps_detected(self):
        df = make_df_gaps()
        config = config_from_dict(GAPS_CONFIG)
        results = temporal.run_checks(df, config)
        t2 = next(r for r in results if r.check_id == "T2")
        assert not t2.passed, "T2 should detect gaps in df_gaps"
        assert t2.details["gap_count"] > 0

    def test_t3_duplicates_detected(self):
        df = make_df_dirty()
        config = config_from_dict(DIRTY_CONFIG)
        results = temporal.run_checks(df, config)
        t3 = next(r for r in results if r.check_id == "T3")
        assert not t3.passed, "T3 should detect duplicate keys in df_dirty"

    def test_t4_staleness_detected(self):
        df = make_df_gaps()
        config = config_from_dict(GAPS_CONFIG)
        results = temporal.run_checks(df, config)
        t4 = next(r for r in results if r.check_id == "T4")
        assert not t4.passed, "T4 should detect stale data in df_gaps"
        assert t4.details["days_stale"] > 0

    def test_t5_date_range_ok(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        results = temporal.run_checks(df, config)
        t5 = next(r for r in results if r.check_id == "T5")
        # expected_start = 2024-01-02 and actual starts at 2024-01-02
        assert t5.passed, f"T5 should pass: {t5.message}"

    def test_t6_day_patterns(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        results = temporal.run_checks(df, config)
        t6 = next(r for r in results if r.check_id == "T6")
        assert t6.check_id == "T6"
        assert "day_of_week_distribution" in t6.details


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------

class TestSchemaChecks:
    def test_s1_missing_column(self):
        df = make_df_clean().drop(columns=["Value"])
        config = config_from_dict(CLEAN_CONFIG)
        results = schema.run_checks(df, config)
        s1 = next(r for r in results if r.check_id == "S1")
        assert not s1.passed
        assert "Value" in s1.details["missing"]

    def test_s1_all_present(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        results = schema.run_checks(df, config)
        s1 = next(r for r in results if r.check_id == "S1")
        assert s1.passed

    def test_s2_wrong_type(self):
        df = make_df_clean()
        df["Value"] = df["Value"].astype(str)  # coerce to string
        config = config_from_dict(CLEAN_CONFIG)
        results = schema.run_checks(df, config)
        s2 = next(r for r in results if r.check_id == "S2")
        assert not s2.passed

    def test_s3_new_dimension_value(self):
        df = make_df_clean()
        df.loc[0, "Region"] = "Patagonia"  # unexpected
        config = config_from_dict(CLEAN_CONFIG)
        results = schema.run_checks(df, config)
        s3 = next(r for r in results if r.check_id == "S3")
        assert not s3.passed
        assert "Region_new" in s3.details

    def test_s3_missing_dimension_value(self):
        df = make_df_clean()
        df = df[df["Region"] != "Interior"]
        config = config_from_dict(CLEAN_CONFIG)
        results = schema.run_checks(df, config)
        s3 = next(r for r in results if r.check_id == "S3")
        assert not s3.passed
        assert "Region_missing" in s3.details


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

class TestIntegrityChecks:
    def test_i1_nulls_exceed_threshold(self):
        df = make_df_dirty()
        config = config_from_dict(DIRTY_CONFIG)
        results = integrity.run_checks(df, config)
        i1 = next(r for r in results if r.check_id == "I1")
        assert not i1.passed, "I1 should fail with high null %"

    def test_i1_within_threshold(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        results = integrity.run_checks(df, config)
        i1 = next(r for r in results if r.check_id == "I1")
        assert i1.passed

    def test_i2_zeros_detected(self):
        df = make_df_dirty()
        config = config_from_dict(DIRTY_CONFIG)
        results = integrity.run_checks(df, config)
        i2 = next(r for r in results if r.check_id == "I2")
        assert not i2.passed, "I2 should detect zeros when allow_zero_values=False"

    def test_i3_exact_duplicates(self):
        df = make_df_dirty()
        config = config_from_dict(DIRTY_CONFIG)
        results = integrity.run_checks(df, config)
        i3 = next(r for r in results if r.check_id == "I3")
        assert not i3.passed
        assert i3.details["exact_duplicate_count"] > 0

    def test_i4_value_out_of_range(self):
        df = make_df_dirty()
        config = config_from_dict(DIRTY_CONFIG)
        results = integrity.run_checks(df, config)
        i4 = next(r for r in results if r.check_id == "I4")
        assert not i4.passed  # outlier 9_999_999 > max 100_000


# ---------------------------------------------------------------------------
# Statistical checks
# ---------------------------------------------------------------------------

class TestStatisticalChecks:
    def test_a1_outliers_detected(self):
        df = make_df_dirty()
        config = config_from_dict(DIRTY_CONFIG)
        results = statistical.run_checks(df, config)
        a1 = next(r for r in results if r.check_id == "A1")
        assert not a1.passed, "A1 should detect extreme outlier in df_dirty"

    def test_a1_no_outliers_clean(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        results = statistical.run_checks(df, config)
        a1 = next(r for r in results if r.check_id == "A1")
        assert a1.passed, f"A1 should pass on clean data: {a1.message}"

    def test_a2_abrupt_change(self):
        # Create data with a huge spike
        df = make_df_clean()
        df.loc[df.index[5], "Value"] = 999_999.0
        config = config_from_dict(CLEAN_CONFIG)
        # Lower threshold
        cfg = config_from_dict({**CLEAN_CONFIG, "pct_change_threshold": 0.5})
        results = statistical.run_checks(df, cfg)
        a2 = next(r for r in results if r.check_id == "A2")
        assert not a2.passed

    def test_a3_distribution_summary(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        results = statistical.run_checks(df, config)
        a3 = next(r for r in results if r.check_id == "A3")
        assert a3.check_id == "A3"
        assert "Value" in a3.details
        assert "mean" in a3.details["Value"]


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

class TestConsistencyChecks:
    def test_c1_rewrites_detected(self):
        prev = make_df_snapshot_prev()
        current = make_df_revised()
        config = config_from_dict(CONSISTENCY_CONFIG)
        results = consistency.run_checks(current, config, prev)
        c1 = next(r for r in results if r.check_id == "C1")
        assert not c1.passed, "C1 should detect rewrites in df_revised"
        assert c1.details["changed_rows"] > 0

    def test_c1_no_rewrites(self):
        prev = make_df_clean()
        current = make_df_clean()  # identical
        config = config_from_dict(CLEAN_CONFIG)
        results = consistency.run_checks(current, config, prev)
        c1 = next(r for r in results if r.check_id == "C1")
        assert c1.passed, "C1 should pass when data is unchanged"

    def test_c2_pct_modified(self):
        prev = make_df_snapshot_prev()
        current = make_df_revised()
        config = config_from_dict(CONSISTENCY_CONFIG)
        results = consistency.run_checks(current, config, prev)
        c2 = next(r for r in results if r.check_id == "C2")
        assert c2.details["pct_changed"] >= 0

    def test_c3_whitelist(self):
        prev = make_df_snapshot_prev()
        current = make_df_revised()
        config = config_from_dict(CONSISTENCY_CONFIG)
        results = consistency.run_checks(current, config, prev)
        c3 = next(r for r in results if r.check_id == "C3")
        assert c3.check_id == "C3"

    def test_no_snapshot_returns_skipped(self):
        config = config_from_dict(CLEAN_CONFIG)
        results = consistency.run_checks_no_snapshot(config)
        assert len(results) == 3
        assert all(r.details.get("skipped") for r in results)


# ---------------------------------------------------------------------------
# Wide format
# ---------------------------------------------------------------------------

class TestWideFormat:
    def test_wide_to_long_schema_check(self):
        """Wide dataset should pass S1 after normalization."""
        from src.utils import normalize_to_long
        df = make_df_wide()
        config = config_from_dict(WIDE_CONFIG)
        long_df, updated_cfg = normalize_to_long(df, config)
        assert "Value" in long_df.columns
        assert "variable" in long_df.columns
        results = schema.run_checks(long_df, updated_cfg)
        s1 = next(r for r in results if r.check_id == "S1")
        assert s1.passed, f"S1 should pass for normalized wide data: {s1.message}"
