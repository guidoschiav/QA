"""Comparator: compare two datasets and produce a DiffReport."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.canonicalizer import CanonicalDF, canonicalize
from src.auto_mapper import auto_map_columns, mapping_to_rename_dict

if TYPE_CHECKING:
    from src.series_matcher import MatchingResult

# Standard scale factors to test against (powers of 10)
_STANDARD_FACTORS = [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e6, 1e9]


@dataclass
class DiffReport:
    # ── Metadata ──────────────────────────────────────────────────────────────
    timestamp: str
    source_name: str
    platform_name: str

    # ── Coverage ──────────────────────────────────────────────────────────────
    total_source_rows: int
    total_platform_rows: int
    comparable_rows: int           # rows in inner join
    date_range_source: tuple[str, str]
    date_range_platform: tuple[str, str]
    missing_dates_in_platform: list[str]   # dates in source, not in platform
    extra_dates_in_platform: list[str]     # dates in platform, not in source
    missing_keys_in_platform: list[str]    # keys in source, not in platform
    extra_keys_in_platform: list[str]      # keys in platform, not in source
    duplicates_in_source: int
    duplicates_in_platform: int

    # ── Value comparison ──────────────────────────────────────────────────────
    match_pct: float               # % of comparable rows within tolerance
    total_diffs: int               # rows outside tolerance
    mean_abs_diff: float
    mean_rel_diff_pct: float
    max_abs_diff: float
    max_rel_diff_pct: float
    top_diffs: list[dict]          # top 20 rows by abs_diff

    # ── Diagnostics ──────────────────────────────────────────────────────────
    possible_temporal_shift: str | None   # e.g. "+1 day" if shift improves match

    # ── Raw data ──────────────────────────────────────────────────────────────
    all_diffs: pd.DataFrame | None = field(default=None, repr=False)

    # ── Config & logs ─────────────────────────────────────────────────────────
    config_used: dict = field(default_factory=dict)
    canonicalization_log_source: list[str] = field(default_factory=list)
    canonicalization_log_platform: list[str] = field(default_factory=list)
    column_mappings: list[dict] = field(default_factory=list)
    scale_detection: dict | None = field(default=None)

    # ── Per-series results (populated by compare_matched_series) ──────────────
    per_series_results: list[dict] | None = field(default=None)
    series_summary: dict | None = field(default=None)

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (excludes all_diffs DataFrame)."""
        return {
            "timestamp": self.timestamp,
            "source_name": self.source_name,
            "platform_name": self.platform_name,
            "total_source_rows": self.total_source_rows,
            "total_platform_rows": self.total_platform_rows,
            "comparable_rows": self.comparable_rows,
            "date_range_source": list(self.date_range_source),
            "date_range_platform": list(self.date_range_platform),
            "missing_dates_in_platform": self.missing_dates_in_platform,
            "extra_dates_in_platform": self.extra_dates_in_platform,
            "missing_keys_in_platform": self.missing_keys_in_platform,
            "extra_keys_in_platform": self.extra_keys_in_platform,
            "duplicates_in_source": self.duplicates_in_source,
            "duplicates_in_platform": self.duplicates_in_platform,
            "match_pct": round(self.match_pct, 4),
            "total_diffs": self.total_diffs,
            "mean_abs_diff": round(self.mean_abs_diff, 6),
            "mean_rel_diff_pct": round(self.mean_rel_diff_pct, 4),
            "max_abs_diff": round(self.max_abs_diff, 6),
            "max_rel_diff_pct": round(self.max_rel_diff_pct, 4),
            "top_diffs": self.top_diffs,
            "possible_temporal_shift": self.possible_temporal_shift,
            "scale_detection": self.scale_detection,
            "config_used": self.config_used,
            "canonicalization_log_source": self.canonicalization_log_source,
            "canonicalization_log_platform": self.canonicalization_log_platform,
            "column_mappings": self.column_mappings,
            "per_series_results": self.per_series_results,
            "series_summary": self.series_summary,
        }


def detect_scale(
    source_values: pd.Series,
    platform_values: pd.Series,
    sample_size: int = 200,
) -> dict:
    """
    Detect if platform values are scaled relative to source values.

    Computes platform/source ratios, finds the median, then matches it to the
    nearest standard scale factor (powers of 10). Confidence is based on ratio
    consistency (low std/median) and log-distance to the nearest standard factor.

    Returns:
        dict with keys:
          ratio_median  – median of platform/source ratios
          scale_factor  – nearest standard factor (e.g. 1000.0)
          confidence    – "high" | "medium" | "low" | None
          suggested_correction – same as scale_factor (divide platform by this)
          label         – human-readable description
    """
    src = source_values.reset_index(drop=True)
    plat = platform_values.reset_index(drop=True)

    mask = (
        src.notna() & plat.notna()
        & (src.abs() > 1e-10) & (plat.abs() > 1e-10)
    )
    src_clean = src[mask]
    plat_clean = plat[mask]

    if len(src_clean) < 3:
        return {
            "ratio_median": None,
            "scale_factor": 1.0,
            "confidence": None,
            "suggested_correction": 1.0,
            "label": "insufficient data",
        }

    if len(src_clean) > sample_size:
        idx = src_clean.sample(sample_size, random_state=42).index
        src_clean = src_clean.loc[idx]
        plat_clean = plat_clean.loc[idx]

    ratios = plat_clean.values / src_clean.values
    median_ratio = float(np.median(ratios))

    if median_ratio <= 0:
        return {
            "ratio_median": round(median_ratio, 8),
            "scale_factor": 1.0,
            "confidence": None,
            "suggested_correction": 1.0,
            "label": "non-positive ratio — mixed signs",
        }

    # Filter extreme outliers (within 5x of median) before computing std
    ratios_filtered = ratios[(ratios / median_ratio >= 0.2) & (ratios / median_ratio <= 5.0)]
    if len(ratios_filtered) < 3:
        ratios_filtered = ratios
    median_ratio = float(np.median(ratios_filtered))

    # Log-distance to standard factors
    log_median = np.log10(abs(median_ratio))
    log_factors = [np.log10(abs(f)) for f in _STANDARD_FACTORS]
    distances = [abs(log_median - lf) for lf in log_factors]
    best_idx = int(np.argmin(distances))
    scale_factor = _STANDARD_FACTORS[best_idx]
    min_distance = distances[best_idx]

    # Confidence: based on consistency (rel_std) and closeness to standard factor
    rel_std = (
        float(np.std(ratios_filtered) / abs(median_ratio))
        if abs(median_ratio) > 0 else 1.0
    )

    if scale_factor == 1.0 and min_distance < 0.05:
        confidence = None   # essentially no scale difference
    elif min_distance < 0.1 and rel_std < 0.1:
        confidence = "high"
    elif min_distance < 0.3 and rel_std < 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    # Human-readable label
    if confidence is None:
        label = "no scale difference detected"
    elif scale_factor > 1.0:
        label = f"platform × {scale_factor:g} vs source (platform is larger)"
    elif scale_factor < 1.0:
        inv = 1.0 / scale_factor
        label = f"platform ÷ {inv:g} vs source (platform is smaller)"
    else:
        label = "no scale difference detected"

    return {
        "ratio_median": round(median_ratio, 8),
        "scale_factor": scale_factor,
        "confidence": confidence,
        "suggested_correction": scale_factor,
        "label": label,
    }


def _canonicalize_from_config(df: pd.DataFrame, cfg_side: dict, side_name: str) -> CanonicalDF:
    """
    Canonicalize one side (source or platform) using config spec.

    Expected cfg_side keys:
        date_column, key_columns (list), value_column (long) or value_columns (wide),
        format ("long"/"wide"), normalizations (optional dict),
        header_row, date_format, dayfirst, number_format, scale, skip_rows_bottom
    """
    fmt = cfg_side.get("format", "long")
    date_col = cfg_side["date_column"]
    key_cols = cfg_side.get("key_columns", [])
    value_col = cfg_side.get("value_column") if fmt == "long" else None
    value_cols = cfg_side.get("value_columns") if fmt == "wide" else None
    normalizations = cfg_side.get("normalizations", {})

    return canonicalize(
        df=df,
        date_col=date_col,
        key_cols=key_cols,
        value_col=value_col,
        value_cols=value_cols,
        fmt=fmt,
        normalizations=normalizations,
        header_row=cfg_side.get("header_row", "auto"),
        skip_rows_bottom=cfg_side.get("skip_rows_bottom", 0),
        date_format=cfg_side.get("date_format"),
        dayfirst=cfg_side.get("dayfirst"),
        number_format=cfg_side.get("number_format"),
        scale=cfg_side.get("scale", 1.0),
        custom_null_values=cfg_side.get("custom_null_values"),
    )


def _detect_temporal_shift(
    src: pd.DataFrame, plat: pd.DataFrame, tolerance: float
) -> str | None:
    """
    Test if shifting platform by ±1 day improves match rate vs base.
    Returns "+1 day", "-1 day", or None.
    """
    base = _match_pct_inner(src, plat, tolerance)
    best_shift = None
    best_pct = base

    for days in [1, -1]:
        shifted = plat.copy()
        shifted["Date"] = shifted["Date"] + pd.Timedelta(days=days)
        pct = _match_pct_inner(src, shifted, tolerance)
        if pct > best_pct + 0.05:  # at least 5pp improvement to declare shift
            best_pct = pct
            best_shift = f"+{days} day" if days > 0 else f"{days} day"

    return best_shift


def _match_pct_inner(src: pd.DataFrame, plat: pd.DataFrame, tolerance: float) -> float:
    """Compute % of inner-join rows where abs_diff <= tolerance * |src_value|."""
    merged = src.merge(plat, on=["Date", "Key"], suffixes=("_src", "_plat"))
    if len(merged) == 0:
        return 0.0
    abs_diff = (merged["Value_src"] - merged["Value_plat"]).abs()
    rel_tol = tolerance * merged["Value_src"].abs().clip(lower=1e-10)
    match = (abs_diff <= rel_tol) | (abs_diff <= tolerance)
    return match.mean()


def compare_datasets(
    df_source: pd.DataFrame,
    df_platform: pd.DataFrame,
    config: dict,
) -> DiffReport:
    """
    Compare two datasets and produce a DiffReport.

    Config structure:
    {
        "source": { date_column, key_columns, value_column/value_columns, format,
                    normalizations, header_row, date_format, dayfirst,
                    number_format, scale, skip_rows_bottom },
        "platform": { ... same keys ... },
        "tolerance": 0.01,   # relative tolerance (1%)
        "source_name": "Source",
        "platform_name": "Platform",
    }

    If "platform" config is omitted, auto_map_columns() is used to map source → platform columns.
    """
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_name = config.get("source_name", "Source")
    platform_name = config.get("platform_name", "Platform")
    tolerance = float(config.get("tolerance", 0.01))

    src_cfg = config.get("source", {})
    plat_cfg = config.get("platform", {})

    # ── Auto-map platform columns if no explicit platform config ──────────────
    column_mappings_out: list[dict] = []
    if not plat_cfg or not plat_cfg.get("date_column"):
        mappings = auto_map_columns(list(df_source.columns), list(df_platform.columns))
        rename_map = mapping_to_rename_dict(mappings)
        df_platform = df_platform.rename(columns=rename_map)
        # Use same structure as source after renaming
        plat_cfg = dict(src_cfg)
        column_mappings_out = [
            {"source": m.source_col, "platform": m.platform_col, "confidence": m.confidence}
            for m in mappings
        ]

    # ── Canonicalize both sides ───────────────────────────────────────────────
    can_src = _canonicalize_from_config(df_source, src_cfg, "source")
    can_plat = _canonicalize_from_config(df_platform, plat_cfg, "platform")

    src_df = can_src.df
    plat_df = can_plat.df

    # ── Duplicates ────────────────────────────────────────────────────────────
    dups_src = int(df_source.duplicated().sum())
    dups_plat = int(df_platform.duplicated().sum())

    # ── Date coverage ─────────────────────────────────────────────────────────
    src_dates = set(src_df["Date"].dt.normalize().unique())
    plat_dates = set(plat_df["Date"].dt.normalize().unique())

    missing_dates = sorted(src_dates - plat_dates)
    extra_dates = sorted(plat_dates - src_dates)

    def _fmt_dates(dates) -> list[str]:
        return [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in dates]

    # ── Key coverage ──────────────────────────────────────────────────────────
    src_keys = set(src_df["Key"].unique())
    plat_keys = set(plat_df["Key"].unique())
    missing_keys = sorted(src_keys - plat_keys)
    extra_keys = sorted(plat_keys - src_keys)

    # ── Date ranges ───────────────────────────────────────────────────────────
    def _date_range(df: pd.DataFrame) -> tuple[str, str]:
        if len(df) == 0:
            return ("N/A", "N/A")
        mn = df["Date"].min().strftime("%Y-%m-%d")
        mx = df["Date"].max().strftime("%Y-%m-%d")
        return (mn, mx)

    # ── Inner join for value comparison ───────────────────────────────────────
    merged = src_df.merge(plat_df, on=["Date", "Key"], suffixes=("_src", "_plat"))
    comparable_rows = len(merged)

    if comparable_rows == 0:
        # No overlap at all
        return DiffReport(
            timestamp=ts,
            source_name=source_name,
            platform_name=platform_name,
            total_source_rows=len(src_df),
            total_platform_rows=len(plat_df),
            comparable_rows=0,
            date_range_source=_date_range(src_df),
            date_range_platform=_date_range(plat_df),
            missing_dates_in_platform=_fmt_dates(missing_dates),
            extra_dates_in_platform=_fmt_dates(extra_dates),
            missing_keys_in_platform=missing_keys,
            extra_keys_in_platform=extra_keys,
            duplicates_in_source=dups_src,
            duplicates_in_platform=dups_plat,
            match_pct=0.0,
            total_diffs=0,
            mean_abs_diff=0.0,
            mean_rel_diff_pct=0.0,
            max_abs_diff=0.0,
            max_rel_diff_pct=0.0,
            top_diffs=[],
            possible_temporal_shift=None,
            all_diffs=None,
            config_used=config,
            canonicalization_log_source=can_src.normalization_log,
            canonicalization_log_platform=can_plat.normalization_log,
            column_mappings=column_mappings_out,
            scale_detection=None,
        )

    # ── Scale detection ───────────────────────────────────────────────────────
    scale_info = detect_scale(merged["Value_src"], merged["Value_plat"])

    # Auto-correct if confidence is high or medium and scale is not 1.0.
    # Platform (Alphacast) is the ground truth — scale Source to match Platform.
    if scale_info["confidence"] in ("high", "medium") and scale_info["scale_factor"] != 1.0:
        merged = merged.copy()
        merged["Value_src"] = merged["Value_src"] * scale_info["scale_factor"]
        scale_info["_auto_corrected"] = True
    else:
        scale_info["_auto_corrected"] = False

    # ── Value diff computation ────────────────────────────────────────────────
    merged["abs_diff"] = (merged["Value_src"] - merged["Value_plat"]).abs()
    merged["rel_diff_pct"] = (
        merged["abs_diff"] / merged["Value_src"].abs().clip(lower=1e-10) * 100
    )

    # Tolerance: abs <= tolerance OR rel <= tolerance*100%
    abs_tol = tolerance
    rel_tol_pct = tolerance * 100
    matched = (merged["abs_diff"] <= abs_tol) | (merged["rel_diff_pct"] <= rel_tol_pct)

    match_pct = float(matched.mean())
    diffs_df = merged[~matched].copy()
    total_diffs = int((~matched).sum())

    mean_abs = float(merged["abs_diff"].mean()) if comparable_rows > 0 else 0.0
    mean_rel = float(merged["rel_diff_pct"].mean()) if comparable_rows > 0 else 0.0
    max_abs = float(merged["abs_diff"].max()) if comparable_rows > 0 else 0.0
    max_rel = float(merged["rel_diff_pct"].max()) if comparable_rows > 0 else 0.0

    top_diffs_df = merged.nlargest(20, "abs_diff")[
        ["Date", "Key", "Value_src", "Value_plat", "abs_diff", "rel_diff_pct"]
    ].copy()
    top_diffs_df["Date"] = top_diffs_df["Date"].dt.strftime("%Y-%m-%d")
    top_diffs_df["abs_diff"] = top_diffs_df["abs_diff"].round(4)
    top_diffs_df["rel_diff_pct"] = top_diffs_df["rel_diff_pct"].round(2)
    top_diffs = top_diffs_df.to_dict(orient="records")

    # ── Temporal shift detection ──────────────────────────────────────────────
    shift = _detect_temporal_shift(src_df, plat_df, tolerance)

    # ── Build all_diffs with extra info ──────────────────────────────────────
    merged["Date_str"] = merged["Date"].dt.strftime("%Y-%m-%d")
    merged["matched"] = matched

    return DiffReport(
        timestamp=ts,
        source_name=source_name,
        platform_name=platform_name,
        total_source_rows=len(src_df),
        total_platform_rows=len(plat_df),
        comparable_rows=comparable_rows,
        date_range_source=_date_range(src_df),
        date_range_platform=_date_range(plat_df),
        missing_dates_in_platform=_fmt_dates(missing_dates),
        extra_dates_in_platform=_fmt_dates(extra_dates),
        missing_keys_in_platform=missing_keys,
        extra_keys_in_platform=extra_keys,
        duplicates_in_source=dups_src,
        duplicates_in_platform=dups_plat,
        match_pct=match_pct,
        total_diffs=total_diffs,
        mean_abs_diff=mean_abs,
        mean_rel_diff_pct=mean_rel,
        max_abs_diff=max_abs,
        max_rel_diff_pct=max_rel,
        top_diffs=top_diffs,
        possible_temporal_shift=shift,
        all_diffs=merged,
        config_used=config,
        canonicalization_log_source=can_src.normalization_log,
        canonicalization_log_platform=can_plat.normalization_log,
        column_mappings=column_mappings_out,
        scale_detection=scale_info,
    )


def compare_matched_series(
    matching_result: "MatchingResult",
    config: dict,
) -> DiffReport:
    """
    Compare series-level data using a MatchingResult from series_matcher.

    For each matched pair, aligns the overlapping date range and computes diffs.
    Aggregates the results into a single DiffReport, with per_series_results and
    series_summary populated.

    Config keys (all optional):
        tolerance       – relative tolerance (default 0.01)
        source_name     – label for source side
        platform_name   – label for platform side
    """
    from src.series_matcher import MatchingResult as _MR, get_matching_summary

    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tolerance = float(config.get("tolerance", 0.01))
    source_name = config.get("source_name", "Source")
    platform_name = config.get("platform_name", "Platform")

    per_series: list[dict] = []
    all_diffs_parts: list[pd.DataFrame] = []

    total_rows = 0
    total_comparable = 0
    total_matched_rows = 0

    for match in matching_result.matches:
        src_s = match.source_series
        plat_s = match.platform_series

        src_df = pd.DataFrame({"Date": src_s.dates.values, "Value_src": src_s.values.values})
        plat_df = pd.DataFrame({"Date": plat_s.dates.values, "Value_plat": plat_s.values.values})

        src_df = src_df.dropna()
        plat_df = plat_df.dropna()

        # Align on date
        merged = src_df.merge(plat_df, on="Date", how="inner")
        n_comparable = len(merged)
        total_rows += len(src_df)
        total_comparable += n_comparable

        if n_comparable == 0:
            per_series.append({
                "source_name": src_s.name,
                "platform_name": plat_s.name,
                "match_type": match.match_type,
                "match_confidence": round(match.confidence, 4),
                "comparable_rows": 0,
                "match_pct": None,
                "mean_abs_diff": None,
                "max_abs_diff": None,
                "scale_detection": None,
            })
            continue

        # Scale detection per series
        scale_info = detect_scale(merged["Value_src"], merged["Value_plat"])
        if scale_info["confidence"] in ("high", "medium") and scale_info["scale_factor"] != 1.0:
            merged = merged.copy()
            merged["Value_src"] = merged["Value_src"] * scale_info["scale_factor"]
            scale_info["_auto_corrected"] = True
        else:
            scale_info["_auto_corrected"] = False

        merged["abs_diff"] = (merged["Value_src"] - merged["Value_plat"]).abs()
        merged["rel_diff_pct"] = (
            merged["abs_diff"] / merged["Value_src"].abs().clip(lower=1e-10) * 100
        )
        matched = (merged["abs_diff"] <= tolerance) | (merged["rel_diff_pct"] <= tolerance * 100)
        n_matched = int(matched.sum())
        total_matched_rows += n_matched

        merged["Key"] = src_s.name
        all_diffs_parts.append(merged[~matched].copy())

        per_series.append({
            "source_name": src_s.name,
            "platform_name": plat_s.name,
            "match_type": match.match_type,
            "match_confidence": round(match.confidence, 4),
            "comparable_rows": n_comparable,
            "match_pct": round(float(matched.mean()), 4),
            "mean_abs_diff": round(float(merged["abs_diff"].mean()), 6),
            "max_abs_diff": round(float(merged["abs_diff"].max()), 6),
            "scale_detection": {k: v for k, v in scale_info.items() if k != "_auto_corrected"},
        })

    # Aggregate summary
    match_summary = get_matching_summary(matching_result)
    n_series_perfect = sum(
        1 for r in per_series
        if r["match_pct"] is not None and r["match_pct"] >= 1.0
    )
    n_series_diffs = sum(
        1 for r in per_series
        if r["match_pct"] is not None and r["match_pct"] < 1.0
    )
    overall_match_pct = (
        total_matched_rows / total_comparable if total_comparable > 0 else 0.0
    )

    series_summary = {
        "total_series_matched": matching_result.matched_count,
        "unmatched_source_series": [u.series.name for u in matching_result.unmatched_source],
        "unmatched_platform_series": [u.series.name for u in matching_result.unmatched_platform],
        "series_perfect_match": n_series_perfect,
        "series_with_diffs": n_series_diffs,
        "series_no_overlap": sum(1 for r in per_series if r["comparable_rows"] == 0),
        "overall_match_pct": round(overall_match_pct, 4),
        "matching_summary": match_summary,
    }

    # Build aggregate all_diffs
    if all_diffs_parts:
        all_diffs_df = pd.concat(all_diffs_parts, ignore_index=True)
        all_diffs_df["Date_str"] = all_diffs_df["Date"].dt.strftime("%Y-%m-%d")
        all_diffs_df["matched"] = False
    else:
        all_diffs_df = None

    # Top diffs across all series
    top_diffs: list[dict] = []
    if all_diffs_df is not None and len(all_diffs_df) > 0:
        top_diffs_df = all_diffs_df.nlargest(20, "abs_diff")[
            ["Date", "Key", "Value_src", "Value_plat", "abs_diff", "rel_diff_pct"]
        ].copy()
        top_diffs_df["Date"] = top_diffs_df["Date"].dt.strftime("%Y-%m-%d")
        top_diffs_df["abs_diff"] = top_diffs_df["abs_diff"].round(4)
        top_diffs_df["rel_diff_pct"] = top_diffs_df["rel_diff_pct"].round(2)
        top_diffs = top_diffs_df.to_dict(orient="records")

    total_diffs = len(all_diffs_df) if all_diffs_df is not None else 0

    return DiffReport(
        timestamp=ts,
        source_name=source_name,
        platform_name=platform_name,
        total_source_rows=total_rows,
        total_platform_rows=sum(
            len(m.platform_series.dates) for m in matching_result.matches
        ),
        comparable_rows=total_comparable,
        date_range_source=("N/A", "N/A"),
        date_range_platform=("N/A", "N/A"),
        missing_dates_in_platform=[],
        extra_dates_in_platform=[],
        missing_keys_in_platform=[u.series.name for u in matching_result.unmatched_source],
        extra_keys_in_platform=[u.series.name for u in matching_result.unmatched_platform],
        duplicates_in_source=0,
        duplicates_in_platform=0,
        match_pct=overall_match_pct,
        total_diffs=total_diffs,
        mean_abs_diff=0.0,
        mean_rel_diff_pct=0.0,
        max_abs_diff=0.0,
        max_rel_diff_pct=0.0,
        top_diffs=top_diffs,
        possible_temporal_shift=None,
        all_diffs=all_diffs_df,
        config_used=config,
        canonicalization_log_source=[],
        canonicalization_log_platform=[],
        column_mappings=[],
        scale_detection=None,
        per_series_results=per_series,
        series_summary=series_summary,
    )
