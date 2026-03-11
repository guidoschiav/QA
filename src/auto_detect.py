"""Auto-detect dataset structure and generate a config dict compatible with audit_dataset()."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import infer_frequency

# Priority name list for date column matching (lowercase)
_DATE_NAME_PRIORITY = [
    "date", "fecha", "datetime", "timestamp", "period", "periodo",
    "trade_date", "report_date", "observation_date",
]

# Substrings that suggest a numeric column is an ID, not a value
_ID_INDICATORS = ("id", "code", "num", "key", "idx", "index")


def auto_detect_config(df: pd.DataFrame) -> dict:
    """
    Analyze a DataFrame and auto-generate a config dict compatible with audit_dataset().

    Returns a dict with the same structure as config_from_dict() output, plus an
    extra '_detection_notes' field with detection metadata.

    Each note is a dict:
        {"field": str, "value": any, "confidence": "high"|"medium"|"low", "reason": str}
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    notes: list[dict] = []

    date_col, n = _detect_date_column(df)
    notes.extend(n)

    fmt, value_col, value_cols, n = _detect_format(df, date_col)
    notes.extend(n)

    if fmt == "long":
        dim_cols, n = _detect_dimensions(df, date_col, value_col)
        notes.extend(n)
    else:
        dim_cols = []
        notes.append({
            "field": "dimension_columns",
            "value": [],
            "confidence": "high",
            "reason": "Wide format â€” dimension detection skipped",
        })

    freq, n = _detect_frequency(df, date_col)
    notes.extend(n)

    defaults, n = _compute_smart_defaults(df, date_col, value_col, value_cols, fmt, dim_cols)
    notes.extend(n)

    config: dict[str, Any] = {
        "dataset_name": "auto_detected",
        "description": "Auto-detected configuration",
        "format": fmt,
        "date_column": date_col if date_col else (df.columns[0] if len(df.columns) > 0 else "Date"),
        "dimension_columns": dim_cols,
        "value_column": value_col or "Value",
        "value_columns": value_cols,
        "expected_frequency": freq,
        "expected_start": defaults.get("expected_start"),
        "expected_end": None,
        "acceptable_lag_days": defaults.get("acceptable_lag_days", 3),
        "holidays_country": None,
        "timezone": None,
        "expected_dimensions": defaults.get("expected_dimensions", {}),
        "max_null_pct": defaults.get("max_null_pct", 0.05),
        "allow_zero_values": defaults.get("allow_zero_values", True),
        "value_range": defaults.get("value_range", {"min": None, "max": None}),
        "outlier_zscore_threshold": 3.5,
        "pct_change_threshold": 0.5,
        "revision_whitelist": {
            "enabled": False,
            "max_revision_age_days": 90,
            "max_revision_pct": 0.10,
        },
        "severity_overrides": {},
        "_detection_notes": notes,
    }
    return config


def strip_internal_keys(config: dict) -> dict:
    """Return a copy of config without internal '_*' keys (safe to pass to audit_dataset)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


# â”€â”€ Detection helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_last_date(series: pd.Series) -> pd.Timestamp | None:
    """
    Try multiple strategies to parse a date series and return the maximum date.
    Returns None if no valid date after 2000-01-01 can be found.

    Strategies tried (in order):
    1. Direct pd.to_datetime() â€” handles most string formats (YYYY-MM-DD, etc.)
    2. Format '%Y%m%d' as string â€” handles YYYYMMDD integers/strings (e.g. 20260115)
    3. pd.to_datetime() with dayfirst=True â€” handles DD/MM/YYYY
    """
    _MIN_VALID = pd.Timestamp("2000-01-01")

    def _get_max(ts: pd.Series) -> pd.Timestamp | None:
        valid = ts.dropna()
        if len(valid) == 0:
            return None
        m = valid.max()
        return m if m >= _MIN_VALID else None

    # Strategy 1: direct parse (works for YYYY-MM-DD, ISO strings, datetime dtype)
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        result = _get_max(parsed)
        if result is not None:
            return result
    except Exception:
        pass

    # Strategy 2: convert to string then parse as YYYYMMDD (handles int columns)
    try:
        parsed = pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
        result = _get_max(parsed)
        if result is not None:
            return result
    except Exception:
        pass

    # Strategy 3: dayfirst (DD/MM/YYYY)
    try:
        parsed = pd.to_datetime(series.astype(str), dayfirst=True, errors="coerce")
        result = _get_max(parsed)
        if result is not None:
            return result
    except Exception:
        pass

    return None


def _detect_date_column(df: pd.DataFrame) -> tuple[str | None, list[dict]]:
    notes: list[dict] = []
    cols_lower = {c.lower(): c for c in df.columns}

    # 1. Priority match by name
    for name in _DATE_NAME_PRIORITY:
        if name in cols_lower:
            col = cols_lower[name]
            notes.append({
                "field": "date_column",
                "value": col,
                "confidence": "high",
                "reason": f"matched by name ('{name}')",
            })
            return col, notes

    # 2. Already datetime dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            notes.append({
                "field": "date_column",
                "value": col,
                "confidence": "high",
                "reason": "already datetime dtype",
            })
            return col, notes

    # 3. Try parsing object/string columns
    best_col: str | None = None
    best_pct = 0.0

    for col in df.columns:
        if df[col].dtype not in (object, "string"):
            continue
        sample = df[col].dropna().head(50)
        if len(sample) == 0:
            continue
        # Try strict YYYY-MM-DD first
        parsed = pd.to_datetime(sample, errors="coerce", format="%Y-%m-%d")
        pct = float(parsed.notna().mean())
        if pct < 0.7:
            # Try flexible parsing
            parsed = pd.to_datetime(sample, errors="coerce")
            pct = float(parsed.notna().mean())
        if pct > best_pct:
            best_pct = pct
            best_col = col

    if best_col and best_pct >= 0.7:
        confidence = "high" if best_pct >= 0.95 else "medium"
        notes.append({
            "field": "date_column",
            "value": best_col,
            "confidence": confidence,
            "reason": f"detected by parsing ({best_pct:.0%} of sample parseable as date)",
        })
        return best_col, notes

    notes.append({
        "field": "date_column",
        "value": None,
        "confidence": "low",
        "reason": "no date column found â€” check column names",
    })
    return None, notes


def _is_id_column(df: pd.DataFrame, col: str) -> bool:
    """Heuristic: column is likely an ID (unique integers, name suggests ID)."""
    name_lower = col.lower()
    if not any(ind in name_lower for ind in _ID_INDICATORS):
        return False
    if not pd.api.types.is_integer_dtype(df[col]):
        return False
    n_non_null = df[col].dropna()
    return len(n_non_null) == n_non_null.nunique()


def _detect_format(
    df: pd.DataFrame, date_col: str | None
) -> tuple[str, str | None, list[str], list[dict]]:
    notes: list[dict] = []
    exclude: set[str] = {date_col} if date_col else set()

    numeric_cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not _is_id_column(df, c)
    ]

    if len(numeric_cols) == 0:
        notes.append({
            "field": "format",
            "value": "long",
            "confidence": "low",
            "reason": "no numeric columns found â€” defaulting to long",
        })
        return "long", None, [], notes

    if len(numeric_cols) == 1:
        vc = numeric_cols[0]
        notes.append({
            "field": "format",
            "value": "long",
            "confidence": "high",
            "reason": f"single numeric column ('{vc}')",
        })
        notes.append({
            "field": "value_column",
            "value": vc,
            "confidence": "high",
            "reason": "only numeric column in dataset",
        })
        return "long", vc, [], notes

    # 2+ numeric columns â†’ wide
    notes.append({
        "field": "format",
        "value": "wide",
        "confidence": "high",
        "reason": f"{len(numeric_cols)} numeric columns: {numeric_cols}",
    })
    notes.append({
        "field": "value_columns",
        "value": numeric_cols,
        "confidence": "high",
        "reason": "all non-date, non-ID numeric columns",
    })
    return "wide", None, numeric_cols, notes


def _detect_dimensions(
    df: pd.DataFrame, date_col: str | None, value_col: str | None
) -> tuple[list[str], list[dict]]:
    notes: list[dict] = []
    exclude: set[str] = {c for c in [date_col, value_col] if c}

    candidates = [
        c for c in df.columns
        if c not in exclude
        and not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    dim_cols: list[str] = []
    n_rows = len(df)

    for col in candidates:
        n_unique = df[col].nunique()
        # Exclude likely free-text fields (high cardinality in small datasets)
        if n_rows < 10_000 and n_unique > 1_000:
            notes.append({
                "field": "dimension_columns",
                "value": col,
                "confidence": "low",
                "reason": f"'{col}' excluded â€” {n_unique} unique values (likely free text)",
            })
        else:
            dim_cols.append(col)

    notes.append({
        "field": "dimension_columns",
        "value": dim_cols,
        "confidence": "high" if dim_cols else "medium",
        "reason": (
            f"string/category columns: {dim_cols}"
            if dim_cols
            else "no string/category columns found"
        ),
    })
    return dim_cols, notes


def _detect_frequency(
    df: pd.DataFrame, date_col: str | None
) -> tuple[str, list[dict]]:
    notes: list[dict] = []

    if not date_col or date_col not in df.columns:
        notes.append({
            "field": "expected_frequency",
            "value": "D",
            "confidence": "low",
            "reason": "no date column â€” defaulting to daily",
        })
        return "D", notes

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values().unique()

    if len(dates) < 3:
        notes.append({
            "field": "expected_frequency",
            "value": "D",
            "confidence": "low",
            "reason": "too few unique dates to infer frequency",
        })
        return "D", notes

    freq = infer_frequency(pd.Series(dates))

    if freq is None:
        notes.append({
            "field": "expected_frequency",
            "value": "D",
            "confidence": "low",
            "reason": "irregular intervals â€” defaulting to daily",
        })
        return "D", notes

    _freq_labels = {
        "B": "Business days", "D": "Daily", "W": "Weekly",
        "M": "Monthly", "Q": "Quarterly", "A": "Annual",
    }
    confidence = "high" if freq in ("B", "D", "W", "M") else "medium"
    notes.append({
        "field": "expected_frequency",
        "value": freq,
        "confidence": confidence,
        "reason": f"inferred as {_freq_labels.get(freq, freq)} from date gaps",
    })
    return freq, notes


def _compute_smart_defaults(
    df: pd.DataFrame,
    date_col: str | None,
    value_col: str | None,
    value_cols: list[str],
    fmt: str,
    dim_cols: list[str],
) -> tuple[dict, list[dict]]:
    notes: list[dict] = []
    result: dict[str, Any] = {}

    vc_list = [value_col] if (fmt == "long" and value_col) else value_cols

    # â”€â”€ max_null_pct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    null_rate = float(df.isnull().values.mean())
    if null_rate < 0.01:
        result["max_null_pct"] = 0.02
        null_reason = f"actual null rate < 1% â€” threshold set to 2%"
    elif null_rate <= 0.05:
        result["max_null_pct"] = round(null_rate + 0.02, 3)
        null_reason = f"actual null rate {null_rate:.1%} â€” threshold set to {result['max_null_pct']:.1%}"
    else:
        result["max_null_pct"] = 0.10
        null_reason = f"high null rate {null_rate:.1%} â€” threshold set to 10% (review source data)"

    notes.append({
        "field": "max_null_pct",
        "value": result["max_null_pct"],
        "confidence": "medium" if null_rate <= 0.05 else "low",
        "reason": null_reason,
    })

    # â”€â”€ allow_zero_values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    zero_pct = 0.0
    for vc in vc_list:
        if vc in df.columns:
            col = df[vc].dropna()
            if len(col) > 0:
                z = float((col == 0).sum()) / len(col)
                zero_pct = max(zero_pct, z)

    if zero_pct >= 0.05:
        result["allow_zero_values"] = True
        zero_reason = f"zeros are {zero_pct:.1%} of values â€” appear intentional"
    elif zero_pct > 0:
        result["allow_zero_values"] = False
        zero_reason = f"zeros are {zero_pct:.1%} of values â€” flagged as suspicious"
    else:
        result["allow_zero_values"] = True
        zero_reason = "no zeros found in value column(s)"

    notes.append({
        "field": "allow_zero_values",
        "value": result["allow_zero_values"],
        "confidence": "medium",
        "reason": zero_reason,
    })

    # â”€â”€ acceptable_lag_days â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if date_col and date_col in df.columns:
        last_date = _parse_last_date(df[date_col])
        today = pd.Timestamp.today().normalize()
        if last_date is not None:
            actual_lag = int((today - last_date.normalize()).days)
        else:
            actual_lag = None

        if actual_lag is not None and 0 <= actual_lag <= 3650:
            result["acceptable_lag_days"] = 2 if actual_lag <= 3 else actual_lag + 1
            notes.append({
                "field": "acceptable_lag_days",
                "value": result["acceptable_lag_days"],
                "confidence": "medium",
                "reason": f"ultima fecha hace {actual_lag} dia(s) ({last_date.date() if last_date else '?'})",
            })
        else:
            result["acceptable_lag_days"] = 3
            notes.append({
                "field": "acceptable_lag_days",
                "value": 3,
                "confidence": "low",
                "reason": (
                    f"no se pudo parsear la fecha correctamente (resultado: {last_date}) â€” default 3 dias"
                    if last_date is None or actual_lag is None
                    else f"lag irrazonable ({actual_lag} dias) â€” verifique formato de fechas â€” default 3 dias"
                ),
            })
    else:
        result["acceptable_lag_days"] = 3

    # â”€â”€ value_range â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vmin_all: float | None = None
    vmax_all: float | None = None
    for vc in vc_list:
        if vc in df.columns:
            col = df[vc].dropna()
            if len(col) > 0:
                vmin = float(col.min())
                vmax = float(col.max())
                vmin_all = vmin if vmin_all is None else min(vmin_all, vmin)
                vmax_all = vmax if vmax_all is None else max(vmax_all, vmax)

    if vmin_all is not None and vmax_all is not None:
        span = vmax_all - vmin_all
        margin = span * 0.10 if span > 0 else abs(vmax_all) * 0.10
        range_min = (max(0.0, vmin_all - margin) if vmin_all >= 0 else vmin_all - margin)
        range_max = vmax_all + margin
        result["value_range"] = {"min": round(range_min, 2), "max": round(range_max, 2)}
        notes.append({
            "field": "value_range",
            "value": result["value_range"],
            "confidence": "medium",
            "reason": f"actual range [{vmin_all:.2f}, {vmax_all:.2f}] + 10% margin",
        })
    else:
        result["value_range"] = {"min": None, "max": None}

    # â”€â”€ expected_dimensions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    exp_dims: dict[str, list | None] = {}
    for col in dim_cols:
        if col in df.columns:
            unique_vals = sorted(str(v) for v in df[col].dropna().unique())
            exp_dims[col] = unique_vals if len(unique_vals) <= 50 else None
    result["expected_dimensions"] = exp_dims

    # â”€â”€ expected_start â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if date_col and date_col in df.columns:
        first = pd.to_datetime(df[date_col], errors="coerce").dropna().min()
        result["expected_start"] = first.strftime("%Y-%m-%d")
    else:
        result["expected_start"] = None

    return result, notes


