"""Integration tests for audit_dataset()."""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.auditor import audit_dataset
from src.models import AuditReport, CheckResult
from src.config_loader import config_from_dict
from tests.fixtures import (
    make_df_clean, make_df_dirty, make_df_gaps,
    make_df_revised, make_df_snapshot_prev, make_df_wide,
    CLEAN_CONFIG, DIRTY_CONFIG, GAPS_CONFIG, CONSISTENCY_CONFIG, WIDE_CONFIG,
)


class TestAuditDataset:
    def test_returns_audit_report(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        report = audit_dataset(df, config)
        assert isinstance(report, AuditReport)

    def test_report_structure(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        report = audit_dataset(df, config)
        # Check all required fields
        assert report.dataset_name == "test_clean"
        assert report.timestamp.endswith("Z")
        assert report.total_checks > 0
        assert report.passed + report.failed == report.total_checks
        assert report.blockers + report.warnings + report.infos == report.failed
        assert isinstance(report.checks, list)
        assert len(report.checks) == report.total_checks
        assert isinstance(report.summary, str)

    def test_check_result_structure(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        report = audit_dataset(df, config)
        for chk in report.checks:
            assert isinstance(chk, CheckResult)
            assert chk.check_id
            assert chk.category in ("temporal", "schema", "integrity", "statistical", "consistency")
            assert chk.severity in ("BLOCKER", "WARNING", "INFO")
            assert isinstance(chk.passed, bool)
            assert isinstance(chk.details, dict)

    def test_clean_dataset_has_no_blockers(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        report = audit_dataset(df, config)
        assert report.blockers == 0, (
            f"Clean dataset should have 0 blockers, got {report.blockers}. "
            f"Failing checks: {[c for c in report.checks if not c.passed and c.severity == 'BLOCKER']}"
        )

    def test_dirty_dataset_has_blockers(self):
        df = make_df_dirty()
        config = config_from_dict(DIRTY_CONFIG)
        report = audit_dataset(df, config)
        assert report.blockers > 0, "Dirty dataset should have at least one blocker"
        assert report.failed > 0

    def test_all_categories_represented(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        report = audit_dataset(df, config)
        categories = {c.category for c in report.checks}
        expected = {"temporal", "schema", "integrity", "statistical", "consistency"}
        assert categories == expected

    def test_minimum_19_checks(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        report = audit_dataset(df, config)
        assert report.total_checks >= 19, (
            f"Expected at least 19 checks, got {report.total_checks}"
        )

    def test_with_snapshot_runs_consistency(self):
        prev = make_df_snapshot_prev()
        current = make_df_revised()
        config = config_from_dict(CONSISTENCY_CONFIG)
        report = audit_dataset(current, config, prev_snapshot=prev)
        consistency_checks = [c for c in report.checks if c.category == "consistency"]
        assert len(consistency_checks) == 3
        # At least one should not be skipped
        assert any(not c.details.get("skipped") for c in consistency_checks)

    def test_without_snapshot_consistency_skipped(self):
        df = make_df_clean()
        config = config_from_dict(CLEAN_CONFIG)
        report = audit_dataset(df, config, prev_snapshot=None)
        consistency_checks = [c for c in report.checks if c.category == "consistency"]
        assert len(consistency_checks) == 3
        assert all(c.details.get("skipped") for c in consistency_checks)

    def test_wide_format_integration(self):
        df = make_df_wide()
        config = config_from_dict(WIDE_CONFIG)
        report = audit_dataset(df, config)
        assert isinstance(report, AuditReport)
        assert report.total_checks >= 19

    def test_gaps_detected_in_report(self):
        df = make_df_gaps()
        config = config_from_dict(GAPS_CONFIG)
        report = audit_dataset(df, config)
        t2 = next((c for c in report.checks if c.check_id == "T2"), None)
        assert t2 is not None
        assert not t2.passed, "T2 should detect gaps"

    def test_summary_reflects_blockers(self):
        df = make_df_dirty()
        config = config_from_dict(DIRTY_CONFIG)
        report = audit_dataset(df, config)
        if report.blockers > 0:
            assert "FAILED" in report.summary or "blocker" in report.summary.lower()

    def test_severity_overrides_applied(self):
        df = make_df_gaps()
        config = config_from_dict({
            **GAPS_CONFIG,
            "severity_overrides": {"T4": "INFO"},
        })
        report = audit_dataset(df, config)
        t4 = next((c for c in report.checks if c.check_id == "T4"), None)
        if t4 and not t4.passed:
            assert t4.severity == "INFO"
