"""Auto-map columns between source and platform datasets."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ColumnMapping:
    source_col: str
    platform_col: str | None
    confidence: str          # "exact", "similar", "manual", "unmatched"
    similarity_score: float  # 1.0 = exact, 0.0 = unmatched


def _normalize(col: str) -> str:
    """Lowercase, strip, replace underscores/hyphens/dots with space, collapse spaces."""
    s = col.strip().lower()
    s = re.sub(r"[_\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def auto_map_columns(
    source_cols: list[str],
    platform_cols: list[str],
) -> list[ColumnMapping]:
    """
    Auto-map source columns to platform columns.

    Strategy:
    1. Exact match (case-insensitive)
    2. Normalized string match (strip/lower/replace separators)
    3. Unmatched if neither matches
    """
    platform_lower: dict[str, str] = {c.lower(): c for c in platform_cols}
    platform_norm: dict[str, str] = {_normalize(c): c for c in platform_cols}

    used_platform: set[str] = set()
    mappings: list[ColumnMapping] = []

    for src in source_cols:
        src_lower = src.lower()
        src_norm = _normalize(src)

        # 1. Exact case-insensitive match
        if src_lower in platform_lower:
            plat = platform_lower[src_lower]
            used_platform.add(plat)
            score = 1.0 if src == plat else 0.95
            confidence = "exact"
            mappings.append(ColumnMapping(src, plat, confidence, score))
            continue

        # 2. Normalized match
        if src_norm in platform_norm:
            plat = platform_norm[src_norm]
            if plat not in used_platform:
                used_platform.add(plat)
                mappings.append(ColumnMapping(src, plat, "similar", 0.80))
                continue

        # 3. Unmatched
        mappings.append(ColumnMapping(src, None, "unmatched", 0.0))

    return mappings


def mapping_to_rename_dict(mappings: list[ColumnMapping]) -> dict[str, str]:
    """Return {platform_col: source_col} rename dict for matched columns."""
    return {
        m.platform_col: m.source_col
        for m in mappings
        if m.platform_col is not None
    }
