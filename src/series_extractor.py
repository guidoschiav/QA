"""Series extractor: extract individual time series from DataFrames."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.canonicalizer import (
    CUSTOM_NULL_VALUES,
    MESES_ES,
    detect_date_format,
    detect_header_row,
    detect_number_format,
    _parse_value_with_format,
)
from src.alphacast_profile import ALPHACAST_PROFILE


# ── Quarterly parser ──────────────────────────────────────────────────────────

_Q_TO_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}

_QUARTERLY_PATS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^Q([1-4])\s*(\d{4})$", re.I), "q_yr"),    # Q1 2025
    (re.compile(r"^(\d{4})[-_]Q([1-4])$", re.I), "yr_q"),   # 2025-Q1
    (re.compile(r"^([1-4])T\s*(\d{4})$", re.I), "q_yr"),    # 1T2025
    (re.compile(r"^(\d{4})[-_]T([1-4])$", re.I), "yr_q"),   # 2025-T1
]

_RE_MONTH_ES = re.compile(
    r"^(Ene(?:ro)?|Feb(?:rero)?|Mar(?:zo)?|Abr(?:il)?|May(?:o)?|Jun(?:io)?|"
    r"Jul(?:io)?|Ago(?:sto)?|Sep(?:tiembre)?|Oct(?:ubre)?|Nov(?:iembre)?|"
    r"Dic(?:iembre)?)[- ](\d{2,4})$",
    re.I,
)


def _replace_nulls(series: pd.Series) -> pd.Series:
    """Replace known null strings with NaN."""
    return series.replace(CUSTOM_NULL_VALUES, np.nan)


def _parse_quarterly(series: pd.Series) -> pd.Series:
    """Parse quarterly date strings (Q1 2025, 1T2025, 2025-Q1) → datetime64."""
    result = []
    for val in series:
        ts = pd.NaT
        s = str(val).strip() if pd.notna(val) and val is not None else ""
        for pat, order in _QUARTERLY_PATS:
            m = pat.match(s)
            if m:
                g = m.groups()
                q, yr = (int(g[0]), int(g[1])) if order == "q_yr" else (int(g[1]), int(g[0]))
                try:
                    ts = pd.Timestamp(year=yr, month=_Q_TO_MONTH[q], day=1)
                except Exception:
                    pass
                break
        result.append(ts)
    return pd.Series(result, dtype="datetime64[ns]", index=series.index)


def _parse_month_es(series: pd.Series) -> pd.Series:
    """Parse Spanish month-year strings (Ene 2025, Enero-25) → datetime64."""
    result = []
    for val in series:
        ts = pd.NaT
        s = str(val).strip() if pd.notna(val) and val is not None else ""
        m = _RE_MONTH_ES.match(s)
        if m:
            mes_str = m.group(1).lower()
            yr_str = m.group(2)
            yr = int(yr_str) if len(yr_str) == 4 else 2000 + int(yr_str)
            month = MESES_ES.get(mes_str[:3]) or MESES_ES.get(mes_str)
            if month:
                try:
                    ts = pd.Timestamp(year=yr, month=month, day=1)
                except Exception:
                    pass
        result.append(ts)
    return pd.Series(result, dtype="datetime64[ns]", index=series.index)


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse a series of values to datetime64 using detect_date_format hints."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    s = _replace_nulls(series)
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return pd.Series([pd.NaT] * len(series), dtype="datetime64[ns]", index=series.index)

    s_str = s_clean.astype(str).str.strip()
    fmt_info = detect_date_format(s_str)
    fmt_name = fmt_info.get("format", "iso")
    dayfirst = fmt_info.get("dayfirst", False)

    if fmt_name in ("iso", "timestamp"):
        return pd.to_datetime(s, errors="coerce")
    elif fmt_name == "dmy":
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    elif fmt_name == "year_only":
        return pd.to_datetime(
            s.astype(str).str.strip().replace("nan", np.nan).fillna("") + "-01-01",
            errors="coerce",
        )
    elif fmt_name in ("quarterly_q_year", "quarterly_year_q",
                      "quarterly_t_year", "quarterly_year_t"):
        return _parse_quarterly(s.astype(str).str.strip())
    elif fmt_name in ("month_text_es", "month_text_es_short"):
        return _parse_month_es(s.astype(str).str.strip())
    elif fmt_name in ("month_text_en", "month_text_en_short"):
        return pd.to_datetime(s, errors="coerce")
    else:
        return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def _parse_values(series: pd.Series) -> pd.Series:
    """Parse a series of values to float64, auto-detecting number format."""
    s = _replace_nulls(series)
    fmt = detect_number_format(s.dropna().head(100))
    return _parse_value_with_format(s, fmt)


def _make_date_range(dates: pd.Series) -> tuple[str, str]:
    """Return (min_date_str, max_date_str) for a datetime Series."""
    valid = dates.dropna()
    if len(valid) == 0:
        return ("N/A", "N/A")
    return (valid.min().strftime("%Y-%m-%d"), valid.max().strftime("%Y-%m-%d"))


def _make_preview(dates: pd.Series, values: pd.Series, n: int = 3) -> dict:
    """Return {date_str: value} for the first n non-null (date, value) pairs."""
    result: dict = {}
    paired = pd.DataFrame({"d": dates.values, "v": values.values}).dropna().head(n)
    for _, row in paired.iterrows():
        try:
            d_str = pd.Timestamp(row["d"]).strftime("%Y-%m-%d")
        except Exception:
            d_str = str(row["d"])
        result[d_str] = float(row["v"])
    return result


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class Series:
    """Una serie temporal individual: secuencia de (fecha, valor)."""
    name: str                        # Nombre identificador de la serie
    source_columns: list[str]        # Columnas originales que generaron esta serie
    dates: pd.Series                 # Series de fechas (datetime64)
    values: pd.Series                # Series de valores (float64)
    row_count: int                   # Cantidad de puntos de datos
    date_range: tuple[str, str]      # (primera fecha, última fecha)
    preview: dict                    # {fecha: valor} de las primeras 3 filas
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Resultado de extraer series de un dataset."""
    series: list[Series]
    total_series: int
    dataset_format: str              # "long", "wide", "wide_transposed"
    date_column: str
    dimension_columns: list[str]     # Solo para long
    value_columns: list[str]         # Solo para wide
    ignored_columns: list[str]       # Columnas ignoradas
    extraction_log: list[str]        # Log de decisiones


# ── Extraction functions ──────────────────────────────────────────────────────

def extract_series_wide(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    ignore_cols: list[str] | None = None,
) -> ExtractionResult:
    """
    Extract series from a wide-format DataFrame.
    Each column in value_cols becomes one Series.
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    log: list[str] = []
    ignore = set(ignore_cols or [])
    effective_cols = [c for c in value_cols if c not in ignore and c in df.columns]
    ignored = [c for c in df.columns if c not in effective_cols and c != date_col]

    dates_parsed = _parse_dates(df[date_col])
    valid_dates = dates_parsed.notna().sum()
    log.append(f"Parsed {valid_dates}/{len(dates_parsed)} dates from '{date_col}'")

    series_list: list[Series] = []
    for col in effective_cols:
        vals = _parse_values(df[col])
        mask = dates_parsed.notna() & vals.notna()
        d = dates_parsed[mask].reset_index(drop=True)
        v = vals[mask].reset_index(drop=True)
        if len(d) == 0:
            log.append(f"Skipped '{col}' — no valid data points")
            ignored.append(col)
            continue
        series_list.append(Series(
            name=col,
            source_columns=[date_col, col],
            dates=d,
            values=v,
            row_count=len(d),
            date_range=_make_date_range(d),
            preview=_make_preview(d, v),
            metadata={"original_column": col},
        ))

    log.append(f"Extracted {len(series_list)} series from {len(effective_cols)} columns")
    if ignored:
        log.append(f"Ignored: {ignored}")

    return ExtractionResult(
        series=series_list,
        total_series=len(series_list),
        dataset_format="wide",
        date_column=date_col,
        dimension_columns=[],
        value_columns=effective_cols,
        ignored_columns=ignored,
        extraction_log=log,
    )


def extract_series_long(
    df: pd.DataFrame,
    date_col: str,
    dimension_cols: list[str],
    value_col: str,
) -> ExtractionResult:
    """
    Extract series from a long-format DataFrame.
    Each unique combination of dimension_cols values becomes one Series.
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    log: list[str] = []

    dates_parsed = _parse_dates(df[date_col])
    vals_parsed = _parse_values(df[value_col])
    log.append(f"Parsed {dates_parsed.notna().sum()}/{len(dates_parsed)} dates")

    work = df.copy()
    work["_date"] = dates_parsed
    work["_value"] = vals_parsed
    work = work.dropna(subset=["_date", "_value"])
    log.append(f"{len(work)} rows with valid date+value (from {len(df)})")

    if not dimension_cols:
        # No dimensions → single series containing all rows
        d = work["_date"].reset_index(drop=True)
        v = work["_value"].reset_index(drop=True)
        s = Series(
            name="(all)",
            source_columns=[date_col, value_col],
            dates=d, values=v, row_count=len(d),
            date_range=_make_date_range(d),
            preview=_make_preview(d, v),
            metadata={"dimension_columns": []},
        )
        log.append("No dimension columns — treating dataset as single series")
        return ExtractionResult(
            series=[s], total_series=1, dataset_format="long",
            date_column=date_col, dimension_columns=[],
            value_columns=[value_col], ignored_columns=[], extraction_log=log,
        )

    # Build series key from dimension values
    work["_key"] = work[dimension_cols].apply(
        lambda row: " | ".join(str(val).strip() for val in row), axis=1
    )

    series_list: list[Series] = []
    for key, group in sorted(work.groupby("_key", sort=True)):
        g = group.sort_values("_date")
        d = g["_date"].reset_index(drop=True)
        v = g["_value"].reset_index(drop=True)
        series_list.append(Series(
            name=key,
            source_columns=[date_col] + dimension_cols + [value_col],
            dates=d, values=v, row_count=len(d),
            date_range=_make_date_range(d),
            preview=_make_preview(d, v),
            metadata={
                "dimension_columns": dimension_cols,
                "key_values": {
                    col: str(group[col].iloc[0]) if len(group) > 0 else None
                    for col in dimension_cols
                },
            },
        ))

    counts = [s.row_count for s in series_list]
    if counts:
        log.append(
            f"Extracted {len(series_list)} series "
            f"(rows/series: min={min(counts)}, max={max(counts)}, "
            f"avg={sum(counts)/len(counts):.1f})"
        )

    return ExtractionResult(
        series=series_list,
        total_series=len(series_list),
        dataset_format="long",
        date_column=date_col,
        dimension_columns=dimension_cols,
        value_columns=[value_col],
        ignored_columns=[],
        extraction_log=log,
    )


def extract_series_wide_transposed(
    df: pd.DataFrame,
    entity_col: str | None = None,
    header_row: int = 0,
) -> ExtractionResult:
    """
    Extract series from a wide-transposed DataFrame.
    Dates are in column headers; each row is a named entity/series.
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    log: list[str] = []

    if entity_col is None:
        entity_col = str(df.columns[0])
    log.append(f"Entity column: '{entity_col}'")

    date_cols = [c for c in df.columns if c != entity_col]
    headers_series = pd.Series([str(c) for c in date_cols])
    dates_from_headers = _parse_dates(headers_series)
    n_valid = int(dates_from_headers.notna().sum())
    log.append(f"Parsed {n_valid}/{len(date_cols)} column headers as dates")

    if n_valid == 0:
        log.append("ERROR: could not parse any column header as a date")
        return ExtractionResult(
            series=[], total_series=0, dataset_format="wide_transposed",
            date_column=entity_col, dimension_columns=[], value_columns=[],
            ignored_columns=[str(c) for c in date_cols], extraction_log=log,
        )

    series_list: list[Series] = []
    for _, row in df.iterrows():
        entity_name = str(row[entity_col]).strip()
        if not entity_name or entity_name.lower() in ("nan", "none", ""):
            continue

        row_vals = pd.Series([row[c] for c in date_cols], dtype=object)
        vals = _parse_values(row_vals)

        mask = dates_from_headers.notna() & vals.notna()
        d = dates_from_headers[mask].reset_index(drop=True)
        v = vals[mask].reset_index(drop=True)

        if len(d) == 0:
            log.append(f"Skipped '{entity_name}' — no valid data points")
            continue

        valid_src_cols = [str(date_cols[i]) for i, mv in enumerate(mask) if mv]
        series_list.append(Series(
            name=entity_name,
            source_columns=valid_src_cols,
            dates=d, values=v, row_count=len(d),
            date_range=_make_date_range(d),
            preview=_make_preview(d, v),
            metadata={"entity_column": entity_col},
        ))

    log.append(f"Extracted {len(series_list)} series from {len(df)} rows")

    return ExtractionResult(
        series=series_list,
        total_series=len(series_list),
        dataset_format="wide_transposed",
        date_column=entity_col,
        dimension_columns=[entity_col],
        value_columns=[str(c) for c in date_cols],
        ignored_columns=[],
        extraction_log=log,
    )


def extract_series_auto_platform(df: pd.DataFrame) -> ExtractionResult:
    """
    Extract series from a Platform (Alphacast) dataset automatically.
    Uses ALPHACAST_PROFILE to guide date column detection.
    """
    log: list[str] = []

    # 1. Find date column using Alphacast known names
    alphacast_date_names = {c.lower() for c in ALPHACAST_PROFILE["date_column_names"]}
    date_col: str | None = None
    for col in df.columns:
        if col.lower() in alphacast_date_names:
            date_col = col
            log.append(f"Date column found by Alphacast profile: '{date_col}'")
            break

    if date_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                log.append(f"Date column detected by dtype: '{date_col}'")
                break
            sample = df[col].dropna().head(20)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample.astype(str), errors="coerce", format="%Y-%m-%d")
                if float(parsed.notna().mean()) >= 0.8:
                    date_col = col
                    log.append(f"Date column detected by parsing: '{date_col}'")
                    break

    if date_col is None:
        date_col = str(df.columns[0])
        log.append(f"Date column defaulting to first column: '{date_col}'")

    # 2. Classify remaining columns
    other_cols = [c for c in df.columns if c != date_col]
    numeric_cols = [c for c in other_cols if pd.api.types.is_numeric_dtype(df[c])]
    string_cols = [c for c in other_cols if not pd.api.types.is_numeric_dtype(df[c])]

    # Dimension columns: string columns with few unique values (≤50)
    dim_cols = [
        c for c in string_cols
        if df[c].nunique() <= 50
    ]
    for c in dim_cols:
        log.append(f"Dimension column: '{c}' ({df[c].nunique()} unique values)")

    # 3. Decide format
    if len(numeric_cols) == 0:
        # Try first non-date string col as value
        value_col = string_cols[0] if string_cols else df.columns[-1]
        log.append(f"Format: long (no numeric cols, using '{value_col}')")
        return extract_series_long(df, date_col, dim_cols, value_col)
    elif len(numeric_cols) == 1:
        value_col = numeric_cols[0]
        log.append(f"Format: long (single numeric column '{value_col}')")
        return extract_series_long(df, date_col, dim_cols, value_col)
    elif dim_cols:
        # Multiple numeric + dimension cols → long using first numeric
        value_col = numeric_cols[0]
        log.append(
            f"Format: long (dimension columns present, "
            f"{len(numeric_cols)} numeric cols, using '{value_col}')"
        )
        return extract_series_long(df, date_col, dim_cols, value_col)
    else:
        # Many numeric, few/no string → wide
        log.append(f"Format: wide ({len(numeric_cols)} numeric columns)")
        return extract_series_wide(df, date_col, numeric_cols)
