"""Match source series to platform series — correlation-first cascade."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.series_extractor import ExtractionResult, Series


# ── Name normalization ─────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Lowercase, strip, collapse separators to space."""
    s = name.strip().lower()
    s = re.sub(r"[_\-\.\|,;:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_boost(src_name: str, plat_name: str) -> float:
    """
    Small confidence boost for tiebreaking when correlation values are nearly equal.
    Exact name match → +0.010 · Normalized match → +0.005 · Substring match → +0.002
    """
    if src_name.strip().lower() == plat_name.strip().lower():
        return 0.010
    sn = _norm(src_name)
    pn = _norm(plat_name)
    if sn and pn and sn == pn:
        return 0.005
    if sn and pn and (sn in pn or pn in sn):
        return 0.002
    return 0.0


# ── Public dataclasses ─────────────────────────────────────────────────────────

@dataclass
class SeriesMatch:
    """A matched pair of source and platform series."""
    source_series: Series
    platform_series: Series
    match_type: str          # "correlation", "name_only", "manual"
    confidence: float
    similarity_details: str  # Human-readable explanation


@dataclass
class UnmatchedSeries:
    """A series that could not be matched."""
    series: Series
    side: str                # "source" or "platform"
    reason: str


@dataclass
class MatchingResult:
    """Full result of matching source series against platform series."""
    matches: list[SeriesMatch]
    unmatched_source: list[UnmatchedSeries]
    unmatched_platform: list[UnmatchedSeries]
    total_source: int
    total_platform: int
    matched_count: int
    match_rate_source: float
    match_rate_platform: float
    matching_log: list[str]

    # Summary counts by match_type
    exact_count: int = 0
    normalized_count: int = 0
    substring_count: int = 0
    correlation_count: int = 0
    name_only_count: int = 0
    manual_count: int = 0


# ── Correlation helper ─────────────────────────────────────────────────────────

def _pearson_on_overlap(
    src: Series, plat: Series, min_overlap: int = 5
) -> float | None:
    """
    Compute Pearson r on the overlapping date range between two series.
    Returns None if overlap < min_overlap or std == 0 or all-NaN.
    """
    src_df = pd.DataFrame({"date": src.dates.values, "src": src.values.values})
    plat_df = pd.DataFrame({"date": plat.dates.values, "plat": plat.values.values})

    if src_df.empty or plat_df.empty:
        return None

    merged = src_df.merge(plat_df, on="date", how="inner").dropna()
    if len(merged) < min_overlap:
        return None

    sv = merged["src"].values.astype(float)
    pv = merged["plat"].values.astype(float)

    if np.std(sv) == 0 or np.std(pv) == 0:
        return None

    r = float(np.corrcoef(sv, pv)[0, 1])
    return r if np.isfinite(r) else None


# ── Core cascade ───────────────────────────────────────────────────────────────

def auto_match_series(
    source_extraction: ExtractionResult,
    platform_extraction: ExtractionResult,
    correlation_threshold: float = 0.95,
    min_overlap: int = 5,
) -> MatchingResult:
    """
    Match platform series to source series using a correlation-first cascade.
    Direction: for each PLATFORM series, find the best matching SOURCE.

    Phase 1 — Data correlation (PRIMARY)
        For every (source, platform) pair that shares >= min_overlap dates,
        compute Pearson r.
        Greedy: pick the platform with the best available source r first:
        - r >= correlation_threshold with no close competitor (within 0.02):
            direct match, match_type="correlation"
        - r >= 0.85 but ambiguous (tie within 0.02):
            tiebreak by name similarity boost, match_type="correlation"
        - best r < 0.85 for all sources: no correlation match → see Phase 2

    Phase 2 â€” Name-only fallback (for any still-unmatched platform series)
        If a platform series remains unmatched after the correlation phase,
        try name matching: exact â†’ normalized â†’ substring.
        match_type = "name_only"

    This keeps correlation as the primary signal, but still allows intuitive
    name-based pairing when correlations are weak or unavailable.
    Source series not used by any platform remain in unmatched_source (pool leftover).
    """
    log: list[str] = []
    matches: list[SeriesMatch] = []

    src_series = list(source_extraction.series)
    plat_series = list(platform_extraction.series)

    used_src: set[int] = set()
    used_plat: set[int] = set()

    # ── Phase 1a: Compute full correlation matrix ──────────────────────────────
    # corr_matrix[(si, pi)] = Pearson r or None
    corr_matrix: dict[tuple[int, int], float | None] = {}
    for si in range(len(src_series)):
        for pi in range(len(plat_series)):
            corr_matrix[(si, pi)] = _pearson_on_overlap(
                src_series[si], plat_series[pi], min_overlap=min_overlap
            )

    # has_any_overlap_plat[pi]: True if plat_series[pi] has computable r with
    # at least one source series (i.e., enough common dates exist).
    has_any_overlap_plat: dict[int, bool] = {
        pi: any(corr_matrix[(si, pi)] is not None for si in range(len(src_series)))
        for pi in range(len(plat_series))
    }

    # ── Phase 1b: Greedy correlation-based matching (platform-first) ──────────
    # Each iteration picks the platform with the highest available source r.
    # Repeat until no more matches above 0.85 can be made.
    _keep_going = True
    while _keep_going:
        _keep_going = False

        # Find the platform (not yet matched) with the best available source r
        best_pi: int | None = None
        best_top_r: float = 0.85 - 1e-9  # must beat 0.85 to qualify

        for pi in range(len(plat_series)):
            if pi in used_plat:
                continue
            for si in range(len(src_series)):
                if si in used_src:
                    continue
                r = corr_matrix.get((si, pi))
                if r is not None and r > best_top_r:
                    best_top_r = r
                    best_pi = pi

        if best_pi is None:
            break  # no more candidates above 0.85

        pi = best_pi
        plat = plat_series[pi]

        # Collect all source candidates >= 0.85 for this platform
        candidates: list[tuple[float, int]] = []
        for si in range(len(src_series)):
            if si in used_src:
                continue
            r = corr_matrix.get((si, pi))
            if r is not None and r >= 0.85:
                candidates.append((r, si))
        candidates.sort(reverse=True)

        if not candidates:
            continue

        top_r, top_si = candidates[0]
        # Close competitors: within 0.02 of top_r
        tie_threshold = top_r - 0.02
        close = [(r2, si2) for r2, si2 in candidates if r2 >= tie_threshold]

        if top_r >= correlation_threshold and len(close) == 1:
            # Unambiguous strong match
            src = src_series[top_si]
            detail = f"r={top_r:.4f}"
            matches.append(SeriesMatch(src, plat, "correlation", top_r, detail))
            log.append(f"[corr-strong] '{plat.name}' ← '{src.name}' (r={top_r:.4f})")
            used_src.add(top_si)
            used_plat.add(pi)
        else:
            # Tiebreak by name boost
            scored: list[tuple[float, float, int, float]] = []
            for r2, si2 in close:
                boost = _name_boost(src_series[si2].name, plat.name)
                scored.append((r2 + boost, r2, si2, boost))
            scored.sort(reverse=True)
            _, best_r, best_si2, best_boost = scored[0]
            src = src_series[best_si2]
            detail = f"r={best_r:.4f}" + (" (desempate por nombre)" if best_boost > 0 else "")
            matches.append(SeriesMatch(src, plat, "correlation", best_r, detail))
            log.append(f"[corr-tiebrk] '{plat.name}' ← '{src.name}' (r={best_r:.4f})")
            used_src.add(best_si2)
            used_plat.add(pi)

        _keep_going = True

    log.append(f"Correlation phase matched: {len(matches)}")

    # â”€â”€ Phase 2: Name-only fallback for remaining unmatched platform series â”€â”€
    src_lower_list = [s.name.strip().lower() for s in src_series]
    src_norm_list = [_norm(s.name) for s in src_series]

    for pi, plat in enumerate(plat_series):
        if pi in used_plat:
            continue

        plat_lower = plat.name.strip().lower()
        plat_n = _norm(plat.name)
        matched = False
        overlap_note = (
            " (con overlap pero correlación insuficiente)"
            if has_any_overlap_plat[pi]
            else " (sin datos comparables)"
        )

        # a. Exact
        for si in range(len(src_series)):
            if si in used_src:
                continue
            if plat_lower == src_lower_list[si]:
                src = src_series[si]
                conf = 1.0 if plat.name == src.name else 0.98
                detail = f"name_only: exact match{overlap_note}"
                matches.append(SeriesMatch(src, plat, "name_only", conf, detail))
                log.append(f"[name-exact] '{plat.name}' ← '{src.name}'")
                used_src.add(si)
                used_plat.add(pi)
                matched = True
                break
        if matched:
            continue

        # b. Normalized
        for si in range(len(src_series)):
            if si in used_src:
                continue
            if plat_n and plat_n == src_norm_list[si]:
                src = src_series[si]
                detail = f"name_only: normalized match{overlap_note}"
                matches.append(SeriesMatch(src, plat, "name_only", 0.50, detail))
                log.append(f"[name-norm] '{plat.name}' ← '{src.name}'")
                used_src.add(si)
                used_plat.add(pi)
                matched = True
                break
        if matched:
            continue

        # c. Substring
        best_si_fb: int | None = None
        best_len: int = -1
        for si in range(len(src_series)):
            if si in used_src:
                continue
            sn = src_norm_list[si]
            if plat_n and sn and (plat_n in sn or sn in plat_n):
                overlap_len = len(plat_n) if plat_n in sn else len(sn)
                if overlap_len > best_len:
                    best_len = overlap_len
                    best_si_fb = si
        if best_si_fb is not None:
            src = src_series[best_si_fb]
            detail = f"name_only: substring match{overlap_note}"
            matches.append(SeriesMatch(src, plat, "name_only", 0.40, detail))
            log.append(f"[name-substr] '{plat.name}' ← '{src.name}'")
            used_src.add(best_si_fb)
            used_plat.add(pi)

    log.append(f"Name-only fallback matched: {len([m for m in matches if m.match_type == 'name_only'])}")

    # ── Build unmatched lists ──────────────────────────────────────────────────
    unmatched_src = [
        UnmatchedSeries(src_series[si], "source", "No matching platform series found")
        for si in range(len(src_series))
        if si not in used_src
    ]
    unmatched_plat = [
        UnmatchedSeries(plat_series[pi], "platform", "No matching source series found")
        for pi in range(len(plat_series))
        if pi not in used_plat
    ]

    # ── Summary counts ─────────────────────────────────────────────────────────
    type_counts: dict[str, int] = {
        "exact": 0, "normalized": 0, "substring": 0,
        "correlation": 0, "name_only": 0, "manual": 0,
    }
    for m in matches:
        type_counts[m.match_type] = type_counts.get(m.match_type, 0) + 1

    n_src = len(src_series)
    n_plat = len(plat_series)
    n_matched = len(matches)
    log.append(
        f"Total: {n_matched}/{n_src} source matched, "
        f"{n_matched}/{n_plat} platform matched"
    )

    return MatchingResult(
        matches=matches,
        unmatched_source=unmatched_src,
        unmatched_platform=unmatched_plat,
        total_source=n_src,
        total_platform=n_plat,
        matched_count=n_matched,
        match_rate_source=n_matched / n_src if n_src > 0 else 0.0,
        match_rate_platform=n_matched / n_plat if n_plat > 0 else 0.0,
        matching_log=log,
        exact_count=type_counts["exact"],
        normalized_count=type_counts["normalized"],
        substring_count=type_counts["substring"],
        correlation_count=type_counts["correlation"],
        name_only_count=type_counts["name_only"],
        manual_count=type_counts["manual"],
    )


# ── Manual override ────────────────────────────────────────────────────────────

def apply_manual_matches(
    result: MatchingResult,
    manual_pairs: list[tuple[str, str]],
) -> MatchingResult:
    """
    Apply manual (user-specified) source→platform name pairs to a MatchingResult.
    Any existing auto-match involving either name is removed first.
    Returns a new MatchingResult.
    """
    if not manual_pairs:
        return result

    log = list(result.matching_log)

    src_by_name = {s.series.name: s.series for s in result.unmatched_source}
    plat_by_name = {s.series.name: s.series for s in result.unmatched_platform}
    for m in result.matches:
        src_by_name[m.source_series.name] = m.source_series
        plat_by_name[m.platform_series.name] = m.platform_series

    new_matches = list(result.matches)
    new_unmatched_src = list(result.unmatched_source)
    new_unmatched_plat = list(result.unmatched_platform)

    for src_name, plat_name in manual_pairs:
        if src_name not in src_by_name:
            log.append(f"[manual] WARNING: source '{src_name}' not found — skipped")
            continue
        if plat_name not in plat_by_name:
            log.append(f"[manual] WARNING: platform '{plat_name}' not found — skipped")
            continue

        new_matches = [
            m for m in new_matches
            if m.source_series.name != src_name and m.platform_series.name != plat_name
        ]
        new_unmatched_src = [u for u in new_unmatched_src if u.series.name != src_name]
        new_unmatched_plat = [u for u in new_unmatched_plat if u.series.name != plat_name]

        new_matches.append(SeriesMatch(
            source_series=src_by_name[src_name],
            platform_series=plat_by_name[plat_name],
            match_type="manual",
            confidence=1.0,
            similarity_details=f"Manual match: '{src_name}' → '{plat_name}'",
        ))
        log.append(f"[manual]    '{src_name}' → '{plat_name}'")

    type_counts: dict[str, int] = {
        "exact": 0, "normalized": 0, "substring": 0,
        "correlation": 0, "name_only": 0, "manual": 0,
    }
    for m in new_matches:
        type_counts[m.match_type] = type_counts.get(m.match_type, 0) + 1

    n_src = result.total_source
    n_plat = result.total_platform
    n_matched = len(new_matches)

    return MatchingResult(
        matches=new_matches,
        unmatched_source=new_unmatched_src,
        unmatched_platform=new_unmatched_plat,
        total_source=n_src,
        total_platform=n_plat,
        matched_count=n_matched,
        match_rate_source=n_matched / n_src if n_src > 0 else 0.0,
        match_rate_platform=n_matched / n_plat if n_plat > 0 else 0.0,
        matching_log=log,
        exact_count=type_counts["exact"],
        normalized_count=type_counts["normalized"],
        substring_count=type_counts["substring"],
        correlation_count=type_counts["correlation"],
        name_only_count=type_counts["name_only"],
        manual_count=type_counts["manual"],
    )


# ── Summary ────────────────────────────────────────────────────────────────────

def get_matching_summary(result: MatchingResult) -> dict:
    """Return a JSON-serializable summary of the matching result."""
    return {
        "total_source": result.total_source,
        "total_platform": result.total_platform,
        "matched_count": result.matched_count,
        "match_rate_source": round(result.match_rate_source, 4),
        "match_rate_platform": round(result.match_rate_platform, 4),
        "by_match_type": {
            "exact": result.exact_count,
            "normalized": result.normalized_count,
            "substring": result.substring_count,
            "correlation": result.correlation_count,
            "name_only": result.name_only_count,
            "manual": result.manual_count,
        },
        "unmatched_source": [u.series.name for u in result.unmatched_source],
        "unmatched_platform": [u.series.name for u in result.unmatched_platform],
        "matches": [
            {
                "source": m.source_series.name,
                "platform": m.platform_series.name,
                "match_type": m.match_type,
                "confidence": round(m.confidence, 4),
                "detail": m.similarity_details,
            }
            for m in result.matches
        ],
    }
