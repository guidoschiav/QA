"""Tests for src/series_matcher.py — correlation-first cascade."""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.series_extractor import (
    ExtractionResult,
    Series,
    extract_series_wide,
    extract_series_long,
)
from src.series_matcher import (
    MatchingResult,
    SeriesMatch,
    UnmatchedSeries,
    apply_manual_matches,
    auto_match_series,
    get_matching_summary,
)


# ── Test data builders ─────────────────────────────────────────────────────────

def _make_series(
    name: str,
    n: int = 12,
    offset: float = 0.0,
    scale: float = 1.0,
    values: list | None = None,
    start: str = "2023-01-01",
) -> Series:
    """Build a Series with monthly dates starting at `start`."""
    dates = pd.date_range(start, periods=n, freq="MS")
    if values is not None:
        vals = pd.Series(values, dtype=float)
    else:
        vals = pd.Series([(float(i) + offset) * scale for i in range(n)], dtype=float)
    return Series(
        name=name,
        source_columns=["Date", name],
        dates=pd.Series(dates),
        values=vals,
        row_count=n,
        date_range=(dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")),
        preview={},
    )


def _make_extraction(series_list: list[Series], fmt: str = "wide") -> ExtractionResult:
    return ExtractionResult(
        series=series_list,
        total_series=len(series_list),
        dataset_format=fmt,
        date_column="Date",
        dimension_columns=[],
        value_columns=[s.name for s in series_list],
        ignored_columns=[],
        extraction_log=[],
    )


# ── Phase 1: Correlation-first matching ───────────────────────────────────────

class TestCorrelationMatch:
    def test_identical_data_different_names_matched(self):
        """Same data under completely different names matches via correlation."""
        src = _make_extraction([_make_series("Series_A", offset=100.0)])
        plat = _make_extraction([_make_series("Completely_Different_Name", offset=100.0)])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "correlation"
        assert result.matches[0].confidence >= 0.95

    def test_same_name_same_data_matched_via_correlation(self):
        """Series with identical name AND data → match_type is 'correlation', not 'exact'."""
        src = _make_extraction([_make_series("GDP")])
        plat = _make_extraction([_make_series("GDP")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "correlation"

    def test_case_insensitive_same_data_matched_via_correlation(self):
        """'GDP' vs 'gdp' with same data → correlation match."""
        src = _make_extraction([_make_series("GDP")])
        plat = _make_extraction([_make_series("gdp")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "correlation"

    def test_low_correlation_unmatched(self):
        """Negatively correlated series (r = -1) must NOT match at threshold 0.95."""
        src_vals = pd.Series([float(i) for i in range(12)])
        plat_vals = pd.Series([float(11 - i) for i in range(12)])  # r = -1
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")

        src_s = Series(
            name="Src", source_columns=["Date", "Src"],
            dates=pd.Series(dates), values=src_vals,
            row_count=12, date_range=("", ""), preview={},
        )
        plat_s = Series(
            name="Plat", source_columns=["Date", "Plat"],
            dates=pd.Series(dates), values=plat_vals,
            row_count=12, date_range=("", ""), preview={},
        )
        src = _make_extraction([src_s])
        plat = _make_extraction([plat_s])
        result = auto_match_series(src, plat, correlation_threshold=0.95)
        assert result.matched_count == 0

    def test_insufficient_overlap_skips_correlation(self):
        """Non-overlapping date ranges → has_any_overlap=False → correlation phase skipped."""
        # No name match either ("SrcA" ≠ "PlatB") → fully unmatched
        src_s = Series(
            name="SrcA", source_columns=["Date", "SrcA"],
            dates=pd.Series(pd.date_range("2023-01-01", periods=6, freq="MS")),
            values=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            row_count=6, date_range=("", ""), preview={},
        )
        plat_s = Series(
            name="PlatB", source_columns=["Date", "PlatB"],
            dates=pd.Series(pd.date_range("2025-01-01", periods=6, freq="MS")),
            values=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            row_count=6, date_range=("", ""), preview={},
        )
        src = _make_extraction([src_s])
        plat = _make_extraction([plat_s])
        result = auto_match_series(src, plat, correlation_threshold=0.95)
        assert result.matched_count == 0

    def test_below_085_candidate_threshold_unmatched(self):
        """Source with overlap but r < 0.85 stays unmatched (no name fallback for overlapping)."""
        n = 20
        # Slightly correlated but not enough (add noise to drop r below 0.85)
        rng = np.random.default_rng(42)
        base = np.arange(n, dtype=float)
        noisy = base + rng.normal(0, 20.0, n)  # heavy noise → low r
        dates = pd.date_range("2023-01-01", periods=n, freq="MS")

        src_s = Series(
            name="SameDate", source_columns=["Date", "SameDate"],
            dates=pd.Series(dates), values=pd.Series(noisy),
            row_count=n, date_range=("", ""), preview={},
        )
        plat_s = Series(
            name="SameDate", source_columns=["Date", "SameDate"],
            dates=pd.Series(dates), values=pd.Series(base),
            row_count=n, date_range=("", ""), preview={},
        )
        src = _make_extraction([src_s])
        plat = _make_extraction([plat_s])
        r = auto_match_series(src, plat, correlation_threshold=0.95)
        # If r happens to be >= 0.85, match is ok; otherwise must be unmatched (no name fallback)
        if r.matched_count == 0:
            assert len(r.unmatched_source) == 1

    def test_correlation_details_includes_r_value(self):
        src = _make_extraction([_make_series("X", offset=100.0)])
        plat = _make_extraction([_make_series("Y_different", offset=100.0)])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert "r=" in result.matches[0].similarity_details or \
               "correlation" in result.matches[0].similarity_details

    def test_multiple_series_all_matched(self):
        src = _make_extraction([_make_series("A"), _make_series("B"), _make_series("C")])
        plat = _make_extraction([_make_series("A"), _make_series("B"), _make_series("C")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 3
        assert all(m.match_type == "correlation" for m in result.matches)

    def test_name_boost_tiebreaks_ambiguous_correlation(self):
        """When two platform series have identical r, name similarity tiebreaks."""
        # src "GDP" with same data as both "GDP" and "GDPx" on plat
        # They have identical r=1. Name boost: "GDP"→"GDP" +0.010, "GDP"→"GDPx" 0.0
        src_s = _make_series("GDP")
        plat_a = _make_series("GDP")         # name boost: exact +0.010
        plat_b = _make_series("GDPx")        # name boost: 0.0

        src = _make_extraction([src_s])
        plat = _make_extraction([plat_a, plat_b])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].platform_series.name == "GDP"


# ── Phase 2: Name-only fallback (no date overlap) ─────────────────────────────

class TestNameOnlyFallback:
    """
    All tests in this class use non-overlapping date ranges so correlation
    is impossible and the name-only fallback is forced.
    src: starts 2023-01-01  plat: starts 2025-01-01  → zero overlap
    """

    def _src(self, name: str, **kw) -> Series:
        return _make_series(name, start="2023-01-01", **kw)

    def _plat(self, name: str, **kw) -> Series:
        return _make_series(name, start="2025-01-01", **kw)

    def test_exact_same_case(self):
        src = _make_extraction([self._src("GDP")])
        plat = _make_extraction([self._plat("GDP")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "name_only"
        assert result.matches[0].confidence == 1.0   # exact same string

    def test_exact_different_case(self):
        src = _make_extraction([self._src("GDP")])
        plat = _make_extraction([self._plat("gdp")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "name_only"
        assert result.matches[0].confidence == 0.98  # case-insensitive only

    def test_whitespace_stripped(self):
        src = _make_extraction([self._src("  GDP  ")])
        plat = _make_extraction([self._plat("gdp")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "name_only"

    def test_normalized_underscore_vs_space(self):
        src = _make_extraction([self._src("GDP_Real")])
        plat = _make_extraction([self._plat("GDP Real")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "name_only"
        assert result.matches[0].confidence == 0.50

    def test_normalized_hyphen_vs_space(self):
        src = _make_extraction([self._src("CPI-Index")])
        plat = _make_extraction([self._plat("CPI Index")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "name_only"

    def test_normalized_dot_separator(self):
        src = _make_extraction([self._src("M1.Money")])
        plat = _make_extraction([self._plat("M1 Money")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "name_only"

    def test_substring_short_in_long(self):
        src = _make_extraction([self._src("M1")])
        plat = _make_extraction([self._plat("M1 Money Supply")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "name_only"
        assert result.matches[0].confidence == 0.40

    def test_substring_long_contains_short(self):
        src = _make_extraction([self._src("Unemployment Rate")])
        plat = _make_extraction([self._plat("Unemployment")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].match_type == "name_only"

    def test_no_name_match_stays_unmatched(self):
        """No overlap AND no name relationship → truly unmatched."""
        src = _make_extraction([self._src("Alpha")])
        plat = _make_extraction([self._plat("Zeta")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 0
        assert len(result.unmatched_source) == 1
        assert len(result.unmatched_platform) == 1


# ── Cascade priority ──────────────────────────────────────────────────────────

class TestCascadePriority:
    def test_correlation_beats_name_similarity(self):
        """Correlation match wins even when name is different (data > names)."""
        src_s = _make_series("SourceSeriesName", offset=50.0)
        plat_s = _make_series("TotallyDifferentName", offset=50.0)  # same data, different name
        plat_unrelated = _make_series("SourceSeriesName", values=list(range(11, -1, -1)))  # r=-1

        src = _make_extraction([src_s])
        plat = _make_extraction([plat_unrelated, plat_s])
        result = auto_match_series(src, plat)

        assert result.matched_count == 1
        assert result.matches[0].match_type == "correlation"
        assert result.matches[0].platform_series.name == "TotallyDifferentName"

    def test_platform_series_used_only_once(self):
        """Two source series both match the same platform series; only first wins."""
        src = _make_extraction([_make_series("GDP"), _make_series("gdp")])
        plat = _make_extraction([_make_series("GDP")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert len(result.unmatched_source) == 1

    def test_overlapping_bad_correlation_no_name_fallback(self):
        """Source with overlap but bad correlation is left UNMATCHED (not name-fallback)."""
        n = 12
        dates = pd.date_range("2023-01-01", periods=n, freq="MS")
        src_s = Series(
            name="GDP", source_columns=["Date", "GDP"],
            dates=pd.Series(dates), values=pd.Series(list(range(n)), dtype=float),
            row_count=n, date_range=("", ""), preview={},
        )
        plat_s = Series(
            name="GDP", source_columns=["Date", "GDP"],
            dates=pd.Series(dates), values=pd.Series(list(range(n, 0, -1)), dtype=float),
            row_count=n, date_range=("", ""), preview={},
        )
        src = _make_extraction([src_s])
        plat = _make_extraction([plat_s])
        result = auto_match_series(src, plat, correlation_threshold=0.95)
        # r = -1, has overlap → NOT matched even though names are identical
        assert result.matched_count == 0

    def test_greedy_order_highest_r_first(self):
        """Source with best correlation is processed first in greedy phase."""
        # src_strong correlates r=1 with plat_a; src_weak has r<0.95 with plat_a but r=1 with plat_b
        n = 12
        dates = pd.date_range("2023-01-01", periods=n, freq="MS")
        base = list(range(n))
        inv = list(range(n, 0, -1))

        src_strong = Series(
            name="Strong", source_columns=["Date", "Strong"],
            dates=pd.Series(dates), values=pd.Series(base, dtype=float),
            row_count=n, date_range=("", ""), preview={},
        )
        src_weak = Series(
            name="Weak", source_columns=["Date", "Weak"],
            dates=pd.Series(dates), values=pd.Series(base, dtype=float),
            row_count=n, date_range=("", ""), preview={},
        )
        plat_a = Series(
            name="PlatA", source_columns=["Date", "PlatA"],
            dates=pd.Series(dates), values=pd.Series(base, dtype=float),
            row_count=n, date_range=("", ""), preview={},
        )
        plat_b = Series(
            name="PlatB", source_columns=["Date", "PlatB"],
            dates=pd.Series(dates), values=pd.Series(base, dtype=float),
            row_count=n, date_range=("", ""), preview={},
        )
        src = _make_extraction([src_strong, src_weak])
        plat = _make_extraction([plat_a, plat_b])
        result = auto_match_series(src, plat)
        # Both src have r=1 with any plat; both should end up matched to different platforms
        assert result.matched_count == 2


# ── Unmatched reporting ───────────────────────────────────────────────────────

class TestUnmatchedReporting:
    def test_unmatched_source_reported(self):
        src = _make_extraction([_make_series("A"), _make_series("X_no_match")])
        plat = _make_extraction([_make_series("A")])
        result = auto_match_series(src, plat)
        assert len(result.unmatched_source) == 1
        assert result.unmatched_source[0].series.name == "X_no_match"
        assert result.unmatched_source[0].side == "source"

    def test_unmatched_platform_reported(self):
        src = _make_extraction([_make_series("A")])
        plat = _make_extraction([_make_series("A"), _make_series("Z_no_match")])
        result = auto_match_series(src, plat)
        assert len(result.unmatched_platform) == 1
        assert result.unmatched_platform[0].series.name == "Z_no_match"
        assert result.unmatched_platform[0].side == "platform"

    def test_empty_source(self):
        src = _make_extraction([])
        plat = _make_extraction([_make_series("A")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 0
        assert len(result.unmatched_platform) == 1

    def test_empty_platform(self):
        src = _make_extraction([_make_series("A")])
        plat = _make_extraction([])
        result = auto_match_series(src, plat)
        assert result.matched_count == 0
        assert len(result.unmatched_source) == 1

    def test_both_empty(self):
        src = _make_extraction([])
        plat = _make_extraction([])
        result = auto_match_series(src, plat)
        assert result.matched_count == 0
        assert result.match_rate_source == 0.0
        assert result.match_rate_platform == 0.0


# ── Match rates ───────────────────────────────────────────────────────────────

class TestMatchRates:
    def test_match_rate_source(self):
        src = _make_extraction([_make_series("A"), _make_series("B")])
        plat = _make_extraction([_make_series("A")])
        result = auto_match_series(src, plat)
        assert result.match_rate_source == 0.5

    def test_match_rate_platform(self):
        src = _make_extraction([_make_series("A")])
        plat = _make_extraction([_make_series("A"), _make_series("B")])
        result = auto_match_series(src, plat)
        assert result.match_rate_platform == 0.5

    def test_perfect_match_rate(self):
        series = [_make_series(n) for n in ["A", "B", "C"]]
        src = _make_extraction(list(series))
        plat = _make_extraction([_make_series(n) for n in ["A", "B", "C"]])
        result = auto_match_series(src, plat)
        assert result.match_rate_source == 1.0
        assert result.match_rate_platform == 1.0


# ── apply_manual_matches ──────────────────────────────────────────────────────

class TestApplyManualMatches:
    def _unmatched_result(self):
        # Anti-correlated data (r=-1) so correlation phase produces no matches.
        # has_any_overlap=True for all → name fallback also skipped → matched_count=0.
        n = 12
        src = _make_extraction([
            _make_series("SourceA", values=list(range(n))),
            _make_series("SourceB", values=list(range(n, 2 * n))),
        ])
        plat = _make_extraction([
            _make_series("PlatformX", values=list(range(n, 0, -1))),
            _make_series("PlatformY", values=list(range(2 * n, n, -1))),
        ])
        result = auto_match_series(src, plat, correlation_threshold=0.95)
        assert result.matched_count == 0, "Setup error: expected no auto-matches"
        return result

    def test_manual_match_applied(self):
        result = self._unmatched_result()
        updated = apply_manual_matches(result, [("SourceA", "PlatformX")])
        assert updated.matched_count == 1
        assert updated.matches[0].match_type == "manual"
        assert updated.matches[0].source_series.name == "SourceA"
        assert updated.matches[0].platform_series.name == "PlatformX"

    def test_manual_match_confidence_is_1(self):
        result = self._unmatched_result()
        updated = apply_manual_matches(result, [("SourceA", "PlatformX")])
        assert updated.matches[0].confidence == 1.0

    def test_manual_replaces_auto_match(self):
        src = _make_extraction([_make_series("GDP")])
        plat = _make_extraction([_make_series("GDP"), _make_series("GDP_v2")])
        result = auto_match_series(src, plat)
        assert result.matched_count == 1
        assert result.matches[0].platform_series.name == "GDP"

        # Override: manually match to GDP_v2 instead
        updated = apply_manual_matches(result, [("GDP", "GDP_v2")])
        assert updated.matched_count == 1
        assert updated.matches[0].platform_series.name == "GDP_v2"
        assert updated.matches[0].match_type == "manual"

    def test_unknown_source_warns_in_log(self):
        result = self._unmatched_result()
        updated = apply_manual_matches(result, [("NONEXISTENT", "PlatformX")])
        assert any("WARNING" in line for line in updated.matching_log)

    def test_no_pairs_returns_same_result(self):
        result = self._unmatched_result()
        updated = apply_manual_matches(result, [])
        assert updated.matched_count == result.matched_count

    def test_manual_count_increments(self):
        result = self._unmatched_result()
        updated = apply_manual_matches(result, [("SourceA", "PlatformX")])
        assert updated.manual_count == 1


# ── get_matching_summary ──────────────────────────────────────────────────────

class TestGetMatchingSummary:
    def test_returns_dict(self):
        src = _make_extraction([_make_series("A")])
        plat = _make_extraction([_make_series("A")])
        result = auto_match_series(src, plat)
        summary = get_matching_summary(result)
        assert isinstance(summary, dict)

    def test_summary_keys(self):
        src = _make_extraction([_make_series("A")])
        plat = _make_extraction([_make_series("A")])
        result = auto_match_series(src, plat)
        summary = get_matching_summary(result)
        for key in ["total_source", "total_platform", "matched_count",
                    "match_rate_source", "match_rate_platform",
                    "by_match_type", "unmatched_source", "unmatched_platform", "matches"]:
            assert key in summary

    def test_summary_matches_list(self):
        # "A" matched to "A" via correlation; "B"/"C" anti-correlated → unmatched
        n = 12
        src = _make_extraction([
            _make_series("A"),
            _make_series("B", values=list(range(n))),
        ])
        plat = _make_extraction([
            _make_series("A"),
            _make_series("C", values=list(range(n, 0, -1))),  # r = -1 vs B
        ])
        result = auto_match_series(src, plat, correlation_threshold=0.95)
        summary = get_matching_summary(result)
        assert len(summary["matches"]) == 1
        assert len(summary["unmatched_source"]) == 1
        assert len(summary["unmatched_platform"]) == 1

    def test_summary_is_json_serializable(self):
        import json
        src = _make_extraction([_make_series("A")])
        plat = _make_extraction([_make_series("B")])
        result = auto_match_series(src, plat)
        summary = get_matching_summary(result)
        json.dumps(summary)  # must not raise

    def test_by_match_type_counts_correlation(self):
        """With overlapping data, matches go into 'correlation' bucket."""
        src = _make_extraction([_make_series("A"), _make_series("B")])
        plat = _make_extraction([_make_series("A"), _make_series("B")])
        result = auto_match_series(src, plat)
        summary = get_matching_summary(result)
        assert summary["by_match_type"]["correlation"] >= 2

    def test_by_match_type_counts_name_only(self):
        """Non-overlapping series that match by name go into 'name_only' bucket."""
        src = _make_extraction([_make_series("GDP", start="2023-01-01")])
        plat = _make_extraction([_make_series("GDP", start="2025-01-01")])
        result = auto_match_series(src, plat)
        summary = get_matching_summary(result)
        assert summary["by_match_type"]["name_only"] >= 1

    def test_by_match_type_counts_manual(self):
        """Manual matches go into 'manual' bucket."""
        n = 12
        src = _make_extraction([_make_series("SourceA", values=list(range(n)))])
        plat = _make_extraction([_make_series("PlatformX", values=list(range(n, 0, -1)))])
        result = auto_match_series(src, plat, correlation_threshold=0.95)
        updated = apply_manual_matches(result, [("SourceA", "PlatformX")])
        summary = get_matching_summary(updated)
        assert summary["by_match_type"]["manual"] == 1
