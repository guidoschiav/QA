"""Canonicalizer: normalize any dataset to (Date, Key, Value) form."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_NULL_VALUES: list[str] = [
    "", "n/a", "N/A", "NA", "na", "#N/A", "#NA", "null", "NULL", "None",
    "...", "-", "--", "---", ".", "..",
    "nd", "ND", "n.d.", "n/d", "s/d", "S/D", "s.d.",
    "#REF!", "#VALUE!", "#DIV/0!", "#ERROR!",
    "nan", "NaN", "NAN",
]

MESES_ES: dict[str, int] = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

MESES_EN: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

_QUARTER_TO_MONTH: dict[int, int] = {1: 1, 2: 4, 3: 7, 4: 10}


# ─────────────────────────────────────────────────────────────────────────────
# CanonicalDF dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CanonicalDF:
    df: pd.DataFrame          # columns: Date (datetime64), Key (str), Value (float64)
    date_column_original: str
    key_columns_original: list[str]
    value_column_original: str
    format_original: str      # "long" or "wide"
    normalization_log: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Header detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_numeric_string(v) -> bool:
    """Return True if v can be converted to float."""
    try:
        float(str(v).replace(",", "."))
        return True
    except (ValueError, TypeError):
        return False


def detect_header_row(df_raw: pd.DataFrame, max_scan: int = 20) -> int:
    """
    Scan the first max_scan rows of a raw DataFrame and detect which row
    contains the actual column headers.

    If the DataFrame already has string column names (not integer indices),
    the header is assumed to be already applied and returns 0.

    Heuristic: the header row has the highest count of non-numeric, non-null
    string cells. Ties are broken by preferring the later row (since headers
    usually appear just before data).

    Args:
        df_raw: DataFrame, typically loaded with header=None so columns are ints.
        max_scan: Maximum number of rows to inspect.

    Returns:
        0-based row index of the detected header row.
    """
    # If columns are already named strings → header already applied
    if not all(isinstance(c, (int, np.integer)) for c in df_raw.columns):
        return 0

    n = min(max_scan, len(df_raw))
    best_row, best_score = 0, -1

    for i in range(n):
        row = df_raw.iloc[i]
        score = sum(
            1
            for v in row
            if v is not None
            and not (isinstance(v, float) and np.isnan(v))
            and isinstance(v, str)
            and str(v).strip()
            and not _is_numeric_string(v)
        )
        if score >= best_score:   # >= to prefer later rows on ties
            best_score = score
            best_row = i

    return best_row


# ─────────────────────────────────────────────────────────────────────────────
# Date format detection & parsing
# ─────────────────────────────────────────────────────────────────────────────

# Ordered patterns (most specific first)
_DATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("timestamp",           re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")),
    ("iso",                 re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}$")),
    ("iso_week",            re.compile(r"^\d{4}-W\d{2}$")),
    ("campaign_year_span",  re.compile(r"^\d{4}(?:[/-])(?:\d{4}|\d{2})$")),
    ("year_only",           re.compile(r"^\d{4}$")),
    ("quarterly_q_year",    re.compile(r"^Q[1-4]\s?\d{4}$", re.I)),
    ("quarterly_year_q",    re.compile(r"^\d{4}[-_]Q[1-4]$", re.I)),
    ("quarterly_t_year",    re.compile(r"^[1-4]T\s?\d{4}$", re.I)),
    ("quarterly_year_t",    re.compile(r"^\d{4}[-_]T[1-4]$", re.I)),
    # Spanish BEFORE English: "Feb", "Mar" etc. appear in both but ES has unique
    # entries (Ene, Abr, Ago, Dic) that identify the language unambiguously.
    # Ambiguous abbreviations (Feb/Mar/Jun/Jul/Sep/Oct/Nov/May) are in both
    # dicts with the same numeric value, so parsing is correct either way.
    ("month_text_es_short", re.compile(
        r"^(Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic)-\d{2}$", re.I)),
    ("month_text_es",       re.compile(
        r"^(Ene(?:ro)?|Feb(?:rero)?|Mar(?:zo)?|Abr(?:il)?|May(?:o)?|Jun(?:io)?|"
        r"Jul(?:io)?|Ago(?:sto)?|Sep(?:tiembre)?|Oct(?:ubre)?|Nov(?:iembre)?|"
        r"Dic(?:iembre)?)\s+\d{2,4}$", re.I)),
    ("month_text_en_short", re.compile(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$", re.I)),
    ("month_text_en",       re.compile(
        r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{2,4}$", re.I)),
    ("dmy",                 re.compile(r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}$")),
]

_QUARTERLY_FORMATS = frozenset([
    "quarterly_q_year", "quarterly_year_q", "quarterly_t_year", "quarterly_year_t",
])
_MONTH_TEXT_ES = frozenset(["month_text_es", "month_text_es_short"])
_MONTH_TEXT_EN = frozenset(["month_text_en", "month_text_en_short"])


def detect_date_format(series: pd.Series, sample_size: int = 100) -> dict:
    """
    Analyze a sample of a date column and detect its format.

    Formats detected:
    - ISO: "2025-03-05", "2025/03/05"
    - Timestamp: "2025-03-05T14:30:00Z"
    - EU/Latam: "05/03/2025", "05-03-2025"
    - US: "03/05/2025"
    - English text: "Jan 2025", "January 2025", "Jan-25"
    - Spanish text: "Ene 2025", "Enero 2025", "Ene-25"
    - Quarterly: "Q1 2025", "2025-Q1", "1T2025"
    - ISO week: "2025-W03"
    - Year only: "2025"

    DD/MM vs MM/DD disambiguation:
    - If any first-component > 12 → definitely DD/MM (dayfirst=True, high confidence)
    - If all components <= 12 → assume DD/MM as Latam convention (medium confidence)

    Returns:
        dict with keys: format, dayfirst, confidence, frequency_hint
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return {"format": "iso", "dayfirst": False, "confidence": "high", "frequency_hint": "D"}

    sample = series.dropna().astype(str).str.strip().head(sample_size)
    if len(sample) == 0:
        return {"format": "iso", "dayfirst": False, "confidence": "low", "frequency_hint": "D"}

    # Vote: each value votes for the first pattern it matches
    votes: dict[str, int] = {}
    for val in sample:
        for fmt_name, pat in _DATE_PATTERNS:
            if pat.match(val):
                votes[fmt_name] = votes.get(fmt_name, 0) + 1
                break

    if not votes:
        return {"format": "iso", "dayfirst": False, "confidence": "low", "frequency_hint": "D"}

    winner = max(votes, key=votes.__getitem__)
    vote_pct = votes[winner] / len(sample)
    confidence = "high" if vote_pct >= 0.9 else ("medium" if vote_pct >= 0.6 else "low")

    # Determine frequency hint
    frequency_hint = "D"
    if winner in _QUARTERLY_FORMATS:
        frequency_hint = "Q"
    elif winner in (_MONTH_TEXT_ES | _MONTH_TEXT_EN | {"month_text_en_short", "month_text_es_short"}):
        frequency_hint = "M"
    elif winner in ("campaign_year_span", "year_only"):
        frequency_hint = "A"
    elif winner == "iso_week":
        frequency_hint = "W"

    # DMY vs MDY disambiguation
    dayfirst = False
    if winner == "dmy":
        _re_dmy = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{4}$")
        first_parts = []
        for val in sample:
            m = _re_dmy.match(val)
            if m:
                first_parts.append(int(m.group(1)))
        if first_parts:
            if max(first_parts) > 12:
                dayfirst = True
                confidence = "high"
            else:
                # Ambiguous — assume Latam (DD/MM) convention
                dayfirst = True
                confidence = "medium"

    return {
        "format": winner,
        "dayfirst": dayfirst,
        "confidence": confidence,
        "frequency_hint": frequency_hint,
    }


def _parse_quarterly(series: pd.Series) -> pd.Series:
    """Parse quarterly strings like 'Q1 2025', '2025-Q1', '1T2025' to Timestamp."""
    _patterns = [
        (re.compile(r"^Q([1-4])\s?(\d{4})$", re.I),  "q_y"),   # Q1 2025
        (re.compile(r"^(\d{4})[-_]Q([1-4])$", re.I), "y_q"),   # 2025-Q1
        (re.compile(r"^([1-4])T\s?(\d{4})$", re.I),  "q_y"),   # 1T2025 (Spanish)
        (re.compile(r"^(\d{4})[-_]T([1-4])$", re.I), "y_q"),   # 2025-T1
    ]

    def _parse_one(v):
        s = str(v).strip()
        for pat, order in _patterns:
            m = pat.match(s)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                q, y = (a, b) if order == "q_y" else (b, a)
                try:
                    return pd.Timestamp(year=y, month=_QUARTER_TO_MONTH[q], day=1)
                except Exception:
                    return pd.NaT
        return pd.NaT

    return series.map(_parse_one)


def _parse_month_text(series: pd.Series, lang: str = "es") -> pd.Series:
    """Parse 'Ene 2025', 'Jan-25', 'Enero 2025' etc. to first-of-month Timestamp."""
    month_map = MESES_ES if lang == "es" else MESES_EN
    _re = re.compile(r"^(\w+)[\s\-](\d{2,4})$")

    def _parse_one(v):
        s = str(v).strip()
        m = _re.match(s)
        if not m:
            return pd.NaT
        month_str = m.group(1).lower()
        year_str = m.group(2)
        month = month_map.get(month_str)
        if month is None:
            return pd.NaT
        year = int(year_str)
        if year < 100:
            year += 2000   # "25" → 2025
        try:
            return pd.Timestamp(year=year, month=month, day=1)
        except Exception:
            return pd.NaT

    return series.map(_parse_one)


def _parse_campaign_year_span(series: pd.Series) -> pd.Series:
    """Parse campaign strings like 2025/2026, 2025-2026, 2025/26 to YYYY-01-01."""
    _re = re.compile(r"^(\d{4})(?:[/-])(\d{2}|\d{4})$")

    def _parse_one(v):
        s = str(v).strip()
        m = _re.match(s)
        if not m:
            return pd.NaT
        start_year = int(m.group(1))
        end_part = m.group(2)
        if len(end_part) == 2:
            end_year = (start_year // 100) * 100 + int(end_part)
        else:
            end_year = int(end_part)
        if end_year != start_year + 1:
            return pd.NaT
        try:
            return pd.Timestamp(year=start_year, month=1, day=1)
        except Exception:
            return pd.NaT

    return series.map(_parse_one)


def _parse_iso_week(series: pd.Series) -> pd.Series:
    """Parse '2025-W03' to the Monday of that ISO week."""
    _re = re.compile(r"^(\d{4})-W(\d{2})$")

    def _parse_one(v):
        s = str(v).strip()
        m = _re.match(s)
        if not m:
            return pd.NaT
        year, week = int(m.group(1)), int(m.group(2))
        try:
            return pd.Timestamp.fromisocalendar(year, week, 1)
        except Exception:
            return pd.NaT

    return series.map(_parse_one)


def _apply_date_parsing(
    series: pd.Series,
    fmt_info: dict,
    explicit_fmt: str | None,
    explicit_dayfirst: bool | None,
) -> pd.Series:
    """
    Parse a date series using explicit or auto-detected format info.

    Args:
        series: Raw date column.
        fmt_info: Output of detect_date_format().
        explicit_fmt: strptime format string or special name
                      ('quarterly', 'month_text_es', 'month_text_en',
                       'year_only', 'iso_week'). None = use fmt_info.
        explicit_dayfirst: Override for dayfirst. None = use fmt_info.
    """
    # Resolve which format to use
    fmt_type = explicit_fmt if explicit_fmt is not None else fmt_info.get("format", "iso")
    dayfirst = explicit_dayfirst if explicit_dayfirst is not None else fmt_info.get("dayfirst", False)

    # Quarterly formats
    if fmt_type in (_QUARTERLY_FORMATS | {"quarterly"}):
        return _parse_quarterly(series)

    # Spanish month text
    if fmt_type in (_MONTH_TEXT_ES | {"month_text_es"}):
        return _parse_month_text(series, "es")

    # English month text
    if fmt_type in (_MONTH_TEXT_EN | {"month_text_en"}):
        return _parse_month_text(series, "en")

    # Campaign year span
    if fmt_type == "campaign_year_span":
        return _parse_campaign_year_span(series)

    # Year only
    if fmt_type == "year_only":
        return pd.to_datetime(
            series.astype(str).str.strip() + "-01-01", errors="coerce"
        )

    # ISO week
    if fmt_type == "iso_week":
        return _parse_iso_week(series)

    # Timestamp / ISO: standard pandas parsing
    if fmt_type in ("timestamp", "iso"):
        return pd.to_datetime(series, errors="coerce")

    # Explicit strptime format string (not a special name)
    if explicit_fmt is not None and explicit_fmt not in (
        "quarterly", "month_text_es", "month_text_en", "campaign_year_span", "year_only", "iso_week",
    ):
        return pd.to_datetime(series, format=explicit_fmt, errors="coerce")

    # DMY or unknown: use dayfirst hint
    return pd.to_datetime(series, dayfirst=dayfirst, errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# Number format detection & parsing
# ─────────────────────────────────────────────────────────────────────────────

def detect_number_format(series: pd.Series, sample_size: int = 100) -> dict:
    """
    Detect the numeric format of a value column.

    Detects:
    - Decimal separator: '.' (standard) or ',' (European)
    - Thousands separator: ',' or '.' or ' ' or ''
    - Currency symbols: $, €, £, ¥, AR$, USD, EUR → removed
    - Percentages: "15%" → 15
    - Parentheses for negatives: "(500)" → -500

    Heuristic for decimal:
    - If '1.234,56' pattern (dot thousands, comma decimal) → European
    - If '1,234.56' pattern (comma thousands, dot decimal) → Standard
    - If value ends with exactly 2 digits after last comma → comma is decimal
    - Otherwise → dot is decimal

    Returns:
        dict: {"decimal": str, "thousands": str, "currency": str|None, "is_percentage": bool}
    """
    if pd.api.types.is_numeric_dtype(series):
        return {"decimal": ".", "thousands": "", "currency": None, "is_percentage": False}

    sample = series.dropna().astype(str).str.strip().head(sample_size)
    if len(sample) == 0:
        return {"decimal": ".", "thousands": "", "currency": None, "is_percentage": False}

    # Detect currency (order matters: multi-char tokens before single-char)
    currency: str | None = None
    _currency_patterns = [r"AR\$", r"USD", r"EUR", r"\$", r"€", r"£", r"¥"]
    for pat in _currency_patterns:
        hits = sample.str.contains(pat, regex=True)
        if hits.any():
            m = re.search(pat, sample[hits].iloc[0])
            if m:
                currency = m.group(0)
            break

    # Detect percentage
    is_percentage = bool(sample.str.endswith("%").any())

    # Clean sample for separator analysis
    clean = sample.copy()
    if currency:
        clean = clean.str.replace(re.escape(currency), "", regex=False)
    clean = (
        clean.str.replace(r"[$€£¥]", "", regex=True)
             .str.replace(r"\s", "", regex=True)
             .str.strip("%")
             .str.lstrip("(")
             .str.rstrip(")")
    )

    # European pattern: 1.234,56 or 1.234 (dot thousands, comma decimal)
    has_european = clean.str.match(r"^-?\d{1,3}(\.\d{3})+(,\d*)?$").any()
    # Standard pattern: 1,234.56 or 1,234 (comma thousands, dot decimal)
    has_standard = clean.str.match(r"^-?\d{1,3}(,\d{3})+(\.\d*)?$").any()

    if has_european and not has_standard:
        decimal, thousands = ",", "."
    elif has_standard and not has_european:
        decimal, thousands = ".", ","
    elif has_european and has_standard:
        # Mixed — prefer European (Latam convention)
        decimal, thousands = ",", "."
    else:
        # No thousands separator pattern — detect by last separator
        has_comma = bool(clean.str.contains(",").any())
        has_dot = bool(clean.str.contains(r"\.").any())
        if has_comma and not has_dot:
            decimal, thousands = ",", ""
        else:
            decimal, thousands = ".", ""

    return {
        "decimal": decimal,
        "thousands": thousands,
        "currency": currency,
        "is_percentage": is_percentage,
    }


def _parse_value_with_format(series: pd.Series, fmt: dict) -> pd.Series:
    """
    Parse a value series to float using a pre-determined number format dict
    (output of detect_number_format).

    Handles: currency stripping, parentheses negatives, thousands/decimal
    separators, percentage division.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    s = series.astype(str).str.strip()

    # Parentheses for negatives: "(500)" → "-500"
    _re_parens = re.compile(r"^\(([0-9,.\s]+)\)$")
    has_parens = s.str.match(r"^\([\d,.\s]+\)$").any()
    if has_parens:
        s = s.str.replace(r"^\((.+)\)$", r"-\1", regex=True)

    # Remove currency
    currency = fmt.get("currency")
    if currency:
        s = s.str.replace(re.escape(currency), "", regex=False)
    s = s.str.replace(r"AR\$|USD|EUR|[$€£¥]", "", regex=True)

    # Strip whitespace
    s = s.str.replace(r"\s", "", regex=True)

    # Handle percentage: strip '%' (do NOT divide — value is already in its unit)
    if fmt.get("is_percentage"):
        s = s.str.rstrip("%")

    # Remove thousands separator and normalize decimal
    decimal = fmt.get("decimal", ".")
    thousands = fmt.get("thousands", "")

    if decimal == ",":
        if thousands == ".":
            s = s.str.replace(".", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
    else:
        if thousands == ",":
            s = s.str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def _parse_value(series: pd.Series) -> pd.Series:
    """
    Coerce values to float with auto-detected format.
    Kept for backward compatibility — internally uses detect_number_format.
    """
    fmt = detect_number_format(series)
    return _parse_value_with_format(series, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Key normalization (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_string(s: str) -> str:
    """Lowercase, strip, collapse spaces."""
    return re.sub(r"\s+", " ", s.strip().lower())


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def canonicalize(
    df: pd.DataFrame,
    date_col: str,
    key_cols: list[str],
    value_col: str | None = None,
    value_cols: list[str] | None = None,
    fmt: str = "long",
    normalizations: dict | None = None,
    # New parameters:
    header_row: int | str = "auto",
    skip_rows_bottom: int = 0,
    date_format: str | None = None,    # strptime format string or special name; None = auto
    dayfirst: bool | None = None,      # None = auto-detect from date column
    number_format: dict | None = None, # None = auto-detect from value column
    scale: float = 1.0,                # multiply all values by this factor
    custom_null_values: list[str] | None = None,
) -> CanonicalDF:
    """
    Normalize a dataset to canonical (Date, Key, Value) form.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.
        key_cols: Dimension/key columns.
        value_col: Value column name (long format).
        value_cols: List of value column names (wide format).
        fmt: "long" or "wide".
        normalizations: Optional dict with keys:
            - lowercase_keys: bool (default True)
            - strip_keys: bool (default True)
        header_row: Row index of the header row, or "auto" to detect.
            Use when df was loaded with header=None (junk rows at top).
        skip_rows_bottom: Number of rows to drop from the bottom (footers, totals).
        date_format: Explicit date format string (strptime) or special name:
            'quarterly', 'month_text_es', 'month_text_en', 'campaign_year_span', 'year_only', 'iso_week'.
            None = auto-detect.
        dayfirst: Override for DD/MM ambiguity. None = auto-detect.
        number_format: dict from detect_number_format(). None = auto-detect.
        scale: Multiply all values by this factor after parsing (e.g. 1000 to convert
            thousands to units). Applied AFTER null/NaN detection.
        custom_null_values: Additional strings to treat as NaN (merged with
            CUSTOM_NULL_VALUES).

    Returns:
        CanonicalDF with df having columns: Date (datetime64), Key (str), Value (float64).

    """
    log: list[str] = []
    norms = normalizations or {}
    lowercase_keys = norms.get("lowercase_keys", True)
    strip_keys = norms.get("strip_keys", True)

    work = df.copy()

    # ── PRE-1: header_row detection & re-interpretation ──────────────────────
    if isinstance(header_row, str) and header_row == "auto":
        detected_hr = detect_header_row(work)
    else:
        detected_hr = int(header_row)

    # Re-interpret if columns are integer indices OR if a non-zero header was specified
    _has_int_cols = all(isinstance(c, (int, np.integer)) for c in work.columns)
    if detected_hr > 0 or _has_int_cols:
        new_cols = work.iloc[detected_hr].astype(str).tolist()
        work = work.iloc[detected_hr + 1:].copy()
        work.columns = new_cols
        work = work.reset_index(drop=True)
        if detected_hr > 0:
            log.append(f"Header detected at row {detected_hr}; skipped {detected_hr} junk row(s).")
        else:
            log.append("Re-interpreted raw DataFrame (integer column indices → header from row 0).")

    # Safety-net: ensure all column names are strings (handles numeric headers like years)
    work.columns = [str(c) for c in work.columns]

    # ── PRE-2: skip rows at bottom ────────────────────────────────────────────
    if skip_rows_bottom > 0 and len(work) > skip_rows_bottom:
        work = work.iloc[: len(work) - skip_rows_bottom].copy()
        log.append(f"Dropped {skip_rows_bottom} row(s) from bottom.")

    # ── PRE-3: replace custom null values in string columns ───────────────────
    null_vals = list(CUSTOM_NULL_VALUES)
    if custom_null_values:
        null_vals = null_vals + [v for v in custom_null_values if v not in null_vals]

    str_cols = [c for c in work.columns
                if not pd.api.types.is_numeric_dtype(work[c])
                and not pd.api.types.is_datetime64_any_dtype(work[c])]
    if str_cols:
        # Replace exact matches only (not substrings)
        for col in str_cols:
            work[col] = work[col].astype(str).replace(null_vals, pd.NA)
        log.append(f"Applied null-value substitution on {len(str_cols)} string column(s).")

    # ── PRE-4: pre-detect number format (on original column, before melt) ────
    _num_fmt = number_format
    if _num_fmt is None:
        if fmt == "long" and value_col and value_col in work.columns:
            _num_fmt = detect_number_format(work[value_col])
        elif fmt == "wide" and value_cols:
            available_vcs = [c for c in value_cols if c in work.columns]
            if available_vcs:
                combined = pd.concat([work[c].astype(str) for c in available_vcs], ignore_index=True)
                _num_fmt = detect_number_format(combined)
        if _num_fmt is not None:
            log.append(
                f"Number format auto-detected: decimal='{_num_fmt['decimal']}', "
                f"thousands='{_num_fmt['thousands']}'"
                + (f", currency='{_num_fmt['currency']}'" if _num_fmt.get("currency") else "")
                + ("." if not _num_fmt.get("is_percentage") else ", percentage=True.")
            )
    if _num_fmt is None:
        _num_fmt = {"decimal": ".", "thousands": "", "currency": None, "is_percentage": False}

    # ── Step 1: Parse dates ──────────────────────────────────────────────────
    if date_col not in work.columns:
        raise KeyError(f"Date column '{date_col}' not found. Available: {list(work.columns)}")

    if not pd.api.types.is_datetime64_any_dtype(work[date_col]):
        # Auto-detect format if not given
        _fmt_info = detect_date_format(work[date_col])
        work[date_col] = _apply_date_parsing(work[date_col], _fmt_info, date_format, dayfirst)
        log.append(
            f"Parsed '{date_col}' to datetime "
            f"(format={_fmt_info['format']}, confidence={_fmt_info['confidence']})."
        )

    nat_count = work[date_col].isna().sum()
    if nat_count > 0:
        work = work.dropna(subset=[date_col])
        log.append(f"Dropped {nat_count} rows with NaT in '{date_col}'.")

    # ── Step 2: Wide → Long ──────────────────────────────────────────────────
    format_original = fmt
    value_column_original = value_col or "Value"

    if fmt == "wide":
        if not value_cols:
            raise ValueError("value_cols must be provided for wide format.")
        id_vars = [date_col] + (key_cols or [])
        work = work.melt(
            id_vars=id_vars, value_vars=value_cols,
            var_name="_key_wide", value_name="Value",
        )
        key_cols = (key_cols or []) + ["_key_wide"]
        value_col = "Value"
        value_column_original = ", ".join(value_cols)
        log.append(f"Melted wide→long: {len(value_cols)} value columns.")

    if value_col is None:
        raise ValueError("value_col must be provided for long format.")

    # ── Step 3: Build composite Key ──────────────────────────────────────────
    if key_cols:
        key_series_list = []
        for kc in key_cols:
            s = work[kc].astype(str)
            if strip_keys:
                s = s.str.strip()
            if lowercase_keys:
                s = s.str.lower()
            key_series_list.append(s)
        if len(key_series_list) == 1:
            work["Key"] = key_series_list[0]
        else:
            work["Key"] = key_series_list[0].str.cat(key_series_list[1:], sep="|")
        log.append(f"Key built from {key_cols} with sep '|'.")
    else:
        work["Key"] = "_total_"
        log.append("No key columns — Key set to '_total_'.")

    # ── Step 4: Parse Value ──────────────────────────────────────────────────
    work["Value"] = _parse_value_with_format(work[value_col], _num_fmt)
    null_vals_count = work["Value"].isna().sum()
    if null_vals_count > 0:
        log.append(f"Warning: {null_vals_count} null values in Value column after parsing.")

    # ── Step 5: Apply scale ──────────────────────────────────────────────────
    if scale != 1.0:
        work["Value"] = work["Value"] * scale
        log.append(f"Applied scale factor {scale}.")

    # ── Step 6: Select & rename columns ─────────────────────────────────────
    work = work.rename(columns={date_col: "Date"})
    result = work[["Date", "Key", "Value"]].copy()

    # ── Step 7: Sort and deduplicate ─────────────────────────────────────────
    result = result.sort_values(["Date", "Key"]).reset_index(drop=True)
    dups = result.duplicated(subset=["Date", "Key"]).sum()
    if dups > 0:
        result = result.drop_duplicates(subset=["Date", "Key"], keep="last")
        log.append(f"Deduplicated {dups} duplicate (Date, Key) pairs — kept last.")

    return CanonicalDF(
        df=result,
        date_column_original=date_col,
        key_columns_original=key_cols or [],
        value_column_original=value_column_original,
        format_original=format_original,
        normalization_log=log,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Date composition from separate Año + Mes/Trimestre columns
# ─────────────────────────────────────────────────────────────────────────────

def compose_date_column(
    df: pd.DataFrame,
    year_col: str,
    period_col: str,
    period_type: str,
) -> pd.Series:
    """
    Build a datetime64 Series from separate year and month/quarter columns.

    period_type:
        "month_numeric"   – period_col has int 1-12
        "month_text_es"   – period_col has Spanish month names (Enero/Ene/…)
        "month_text_en"   – period_col has English month names (January/Jan/…)
        "quarter_numeric" – period_col has int 1-4
        "quarter_text_q"  – period_col has Q1/Q2/Q3/Q4
        "quarter_text_t"  – period_col has 1T/2T/3T/4T

    Excel merged-cell handling: year_col is forward-filled before parsing so
    that NaN rows inherit the value of the last non-null cell above them.

    Returns pd.Series of dtype datetime64[ns]; rows that cannot be parsed
    yield NaT.
    """
    years = df[year_col].copy()

    # Forward fill for Excel merged cells
    years = years.ffill()

    # Parse year as int (handles 2024, 2024.0, "2024", "2024.0")
    years_int = pd.to_numeric(years, errors="coerce").fillna(0).astype(int)

    # Parse period column → month number (1-12)
    periods = df[period_col]

    if period_type == "month_numeric":
        month_nums = pd.to_numeric(periods, errors="coerce")

    elif period_type in ("month_text_es", "month_text_en"):
        month_map = MESES_ES if period_type == "month_text_es" else MESES_EN

        def _parse_month_txt(v) -> float:
            if pd.isna(v):
                return float("nan")
            s = str(v).strip().lower()
            if s in month_map:
                return float(month_map[s])
            if len(s) >= 3 and s[:3] in month_map:
                return float(month_map[s[:3]])
            return float("nan")

        month_nums = periods.map(_parse_month_txt)

    elif period_type == "quarter_numeric":
        qtrs = pd.to_numeric(periods, errors="coerce")
        month_nums = qtrs.map(
            lambda q: float(_QUARTER_TO_MONTH[int(q)]) if pd.notna(q) and int(q) in _QUARTER_TO_MONTH else float("nan")
        )

    elif period_type == "quarter_text_q":
        def _parse_q(v) -> float:
            if pd.isna(v):
                return float("nan")
            m = re.search(r"[qQ](\d)", str(v))
            if m:
                q = int(m.group(1))
                return float(_QUARTER_TO_MONTH.get(q, float("nan")))
            return float("nan")
        month_nums = periods.map(_parse_q)

    elif period_type == "quarter_text_t":
        def _parse_t(v) -> float:
            if pd.isna(v):
                return float("nan")
            m = re.search(r"(\d)[tT]", str(v))
            if m:
                q = int(m.group(1))
                return float(_QUARTER_TO_MONTH.get(q, float("nan")))
            return float("nan")
        month_nums = periods.map(_parse_t)

    else:
        month_nums = pd.Series([float("nan")] * len(df), index=df.index)

    # Build datetime Series
    result = []
    for yr, mo in zip(years_int, month_nums):
        if pd.isna(mo) or yr == 0:
            result.append(pd.NaT)
        else:
            try:
                result.append(pd.Timestamp(year=int(yr), month=int(mo), day=1))
            except Exception:
                result.append(pd.NaT)

    return pd.Series(result, dtype="datetime64[ns]", index=df.index)
