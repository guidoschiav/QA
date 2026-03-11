"""Dataset Auditor — Streamlit UI."""
from __future__ import annotations

import csv
import dataclasses
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.auditor import audit_dataset
from src.auto_detect import auto_detect_config, strip_internal_keys
from src.auto_mapper import auto_map_columns
from src.alphacast_profile import ALPHACAST_PROFILE, apply_alphacast_profile
from src.canonicalizer import (
    canonicalize, detect_header_row, detect_date_format, detect_number_format,
    compose_date_column, MESES_ES, MESES_EN,
)
from src.alphacast_client import (
    load_api_key_with_source, save_api_key, delete_api_key,
    validate_api_key, fetch_dataset,
    API_KEY_ENV_VAR, API_KEY_FILE,
)
from src.comparator import compare_datasets, compare_matched_series
from src.config_loader import config_from_dict
from src.report import export_report_json
from src.snapshot_manager import load_snapshot, save_snapshot
from src.series_extractor import (
    extract_series_wide, extract_series_long,
    extract_series_wide_transposed, extract_series_auto_platform,
    ExtractionResult, Series,
)
from src.series_matcher import (
    auto_match_series, apply_manual_matches, get_matching_summary,
    MatchingResult, SeriesMatch, UnmatchedSeries,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dataset Auditor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.badge-blocker  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;
                  border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-warning  { background:#fef3c7; color:#92400e; border:1px solid #fcd34d;
                  border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-info     { background:#dbeafe; color:#1e40af; border:1px solid #93c5fd;
                  border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-ok       { background:#dcfce7; color:#166534; border:1px solid #86efac;
                  border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-diff     { background:#fce7f3; color:#9d174d; border:1px solid #f9a8d4;
                  border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.check-card {
    border-left: 4px solid #e5e7eb; padding: 8px 12px;
    margin-bottom: 6px; border-radius: 0 4px 4px 0; background: #f9fafb;
}
.check-card.blocker { border-left-color: #ef4444; background: #fff5f5; }
.check-card.warning { border-left-color: #f59e0b; background: #fffbeb; }
.check-card.info    { border-left-color: #3b82f6; background: #eff6ff; }
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-box {
    flex: 1; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 12px 16px; text-align: center; background: #fff;
}
.metric-box .metric-value { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.metric-box .metric-label { font-size: 0.8rem; color: #6b7280; margin-top: 2px; }
.footer { font-size: 0.75rem; color: #9ca3af; text-align: center; margin-top: 32px; }
</style>
""", unsafe_allow_html=True)


# ── Bootstrap: load persisted API key once per session ────────────────────────
if "alphacast_api_key" not in st.session_state:
    _secret_key = None
    try:
        _secret_key = st.secrets.get("ALPHACAST_API_KEY") or st.secrets.get("alphacast_api_key")
    except Exception:
        _secret_key = None

    if not _secret_key:
        try:
            _alphacast_section = st.secrets.get("alphacast")
            if hasattr(_alphacast_section, "get"):
                _secret_key = _alphacast_section.get("api_key")
        except Exception:
            _secret_key = None

    if _secret_key:
        st.session_state["alphacast_api_key"] = str(_secret_key).strip()
        st.session_state["alphacast_api_key_source"] = "streamlit_secrets"
    else:
        _persisted_key, _persisted_source = load_api_key_with_source()
        if _persisted_key:
            st.session_state["alphacast_api_key"] = _persisted_key
            st.session_state["alphacast_api_key_source"] = _persisted_source


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 6:
        return "configured"
    return api_key[:6] + "........"


def reveal_directory(path: Path) -> None:
    path.mkdir(exist_ok=True)
    if hasattr(os, "startfile"):
        try:
            os.startfile(str(path))
            return
        except OSError:
            pass
    st.info(
        f"En Streamlit Community Cloud no se pueden abrir carpetas del servidor. "
        f"Usa descargas o revisa la ruta localmente: `{path.name}/`"
    )

def detect_separator(content: bytes) -> str:
    try:
        sample = content[:4096].decode("utf-8", errors="replace")
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def load_csv(content: bytes, separator: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), sep=separator)
    df.columns = [str(c) for c in df.columns]
    return df


def _excel_engine(ext: str) -> str:
    return {"xlsx": "openpyxl", "xls": "xlrd", "xlsb": "pyxlsb"}.get(ext.lstrip("."), "openpyxl")


def load_excel(content: bytes, sheet_name: str, ext: str = ".xlsx") -> pd.DataFrame:
    engine = _excel_engine(ext)
    df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, engine=engine)
    df.columns = [str(c) for c in df.columns]
    return df


def get_excel_sheets(content: bytes, ext: str = ".xlsx") -> list[str]:
    engine = _excel_engine(ext)
    return pd.ExcelFile(io.BytesIO(content), engine=engine).sheet_names


def load_file(uploaded, key_prefix: str) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    content = uploaded.read()
    ext = Path(uploaded.name).suffix.lower()
    if ext == ".csv":
        sep_options = {"Auto": None, "Comma (,)": ",", "Semicolon (;)": ";",
                       "Tab (\\t)": "\t", "Pipe (|)": "|"}
        sep_choice = st.selectbox("Separador CSV", list(sep_options.keys()),
                                  index=0, key=f"{key_prefix}_sep")
        sep = sep_options[sep_choice] or detect_separator(content)
        try:
            return load_csv(content, sep)
        except Exception as e:
            st.error(f"Error al leer CSV: {e}")
    elif ext in (".xlsx", ".xls", ".xlsb"):
        try:
            sheets = get_excel_sheets(content, ext)
            sheet = st.selectbox("Hoja", sheets, key=f"{key_prefix}_sheet")
            return load_excel(content, sheet, ext)
        except Exception as e:
            st.error(f"Error al leer Excel ({ext}): {e}")
    return None


def severity_color(severity: str) -> str:
    return {"BLOCKER": "#ef4444", "WARNING": "#f59e0b", "INFO": "#3b82f6"}.get(severity, "#6b7280")


def severity_badge_class(severity: str) -> str:
    return {"BLOCKER": "badge-blocker", "WARNING": "badge-warning", "INFO": "badge-info"}.get(
        severity, "badge-ok"
    )


def card_class(severity: str) -> str:
    return {"BLOCKER": "blocker", "WARNING": "warning", "INFO": "info"}.get(severity, "")


def save_report(report) -> Path:
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = reports_dir / f"{report.dataset_name}_{ts}.json"
    export_report_json(report, path)
    return path


def load_config_from_bytes(content: bytes) -> dict:
    raw = yaml.safe_load(content.decode("utf-8"))
    return config_from_dict(raw)


def save_auto_config_yaml(config: dict) -> Path:
    configs_dir = Path(__file__).parent / "configs"
    configs_dir.mkdir(exist_ok=True)
    name = config.get("dataset_name", "auto_detected")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = configs_dir / f"{name}_{ts}.yaml"
    clean = strip_internal_keys(config)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(clean, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return path


def build_config_from_controls(
    dataset_name, date_col, fmt, value_col, value_cols, dim_cols, freq,
    max_null_pct, allow_zeros, outlier_threshold, lag_days, detected,
) -> dict:
    return {
        "dataset_name": dataset_name,
        "description": "Auto-detected configuration",
        "format": fmt,
        "date_column": date_col,
        "dimension_columns": dim_cols,
        "value_column": value_col,
        "value_columns": value_cols,
        "expected_frequency": freq,
        "expected_start": detected.get("expected_start"),
        "expected_end": None,
        "acceptable_lag_days": lag_days,
        "holidays_country": None,
        "timezone": None,
        "expected_dimensions": detected.get("expected_dimensions", {}),
        "max_null_pct": max_null_pct,
        "allow_zero_values": allow_zeros,
        "value_range": detected.get("value_range", {"min": None, "max": None}),
        "outlier_zscore_threshold": outlier_threshold,
        "pct_change_threshold": 0.5,
        "revision_whitelist": {"enabled": False, "max_revision_age_days": 90, "max_revision_pct": 0.10},
        "severity_overrides": {},
    }


def _col_idx(col, lst: list) -> int:
    return lst.index(col) if col and col in lst else 0


def _build_compare_html(diff) -> str:
    pct = diff.match_pct * 100
    color = "#22c55e" if pct >= 95 else ("#f59e0b" if pct >= 80 else "#ef4444")
    rows_html = "".join(
        f"<tr><td>{r.get('Date', r.get('date', ''))}</td><td>{r.get('Key', r.get('key', ''))}</td>"
        f"<td>{r.get('Value_src', 0):.4f}</td><td>{r.get('Value_plat', 0):.4f}</td>"
        f"<td>{r.get('rel_diff_pct', 0):.2f}%</td></tr>"
        for r in (diff.top_diffs or [])[:20]
    )
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Compare Report — {diff.source_name} vs {diff.platform_name}</title>
<style>body{{font-family:sans-serif;padding:24px;max-width:900px;margin:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #e5e7eb;padding:6px 10px;text-align:left}}
th{{background:#f9fafb}}.match{{font-size:2rem;font-weight:700;color:{color}}}</style>
</head><body>
<h1>Compare Report</h1>
<p>{diff.source_name} vs {diff.platform_name} &nbsp;·&nbsp; {diff.timestamp}</p>
<p class="match">{pct:.1f}% match</p>
<table><tr><th></th><th>Source</th><th>Platform</th></tr>
<tr><td>Comparable rows</td><td colspan="2">{diff.comparable_rows:,}</td></tr>
<tr><td>Total diffs</td><td colspan="2">{diff.total_diffs}</td></tr>
</table>
{"" if not diff.top_diffs else f"<h2>Top diferencias</h2><table><tr><th>Fecha</th><th>Key</th><th>Source</th><th>Platform</th><th>Dif Rel %</th></tr>{rows_html}</table>"}
</body></html>"""


# ── New helpers for series compare flow ──────────────────────────────────────

def _detect_column_type(df: pd.DataFrame, col: str) -> str:
    """Return a human-readable description of column type."""
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return "📅 Fecha"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "📊 Numérico"
    n_unique = df[col].nunique()
    # Quick date parse check
    sample = df[col].dropna().head(20)
    if len(sample) > 0:
        try:
            parsed = pd.to_datetime(sample.astype(str), errors="coerce")
            if float(parsed.notna().mean()) >= 0.8:
                return "📅 Fecha (parseable)"
        except Exception:
            pass
    if n_unique <= 50:
        return f"🏷️ Categoría ({n_unique} únicos)"
    return f"📝 Texto ({n_unique} únicos)"


def _auto_classify_columns(df: pd.DataFrame, src_format: str) -> dict[str, str]:
    """Auto-classify columns for series extraction."""
    detected_date_col: str | None = None

    # Find date column: look for name hints or datetime dtype
    for col in df.columns:
        col_lower = col.lower()
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            detected_date_col = col
            break
        if any(k in col_lower for k in ("date", "fecha", "time", "periodo", "period")):
            detected_date_col = col
            break

    # Fallback: try parsing first string column
    if detected_date_col is None:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                sample = df[col].dropna().head(20).astype(str)
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if float(parsed.notna().mean()) >= 0.8:
                        detected_date_col = col
                        break
                except Exception:
                    pass

    result: dict[str, str] = {}
    for col in df.columns:
        if col == detected_date_col:
            result[col] = "Fecha"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            result[col] = "Fecha"
        elif pd.api.types.is_numeric_dtype(df[col]):
            result[col] = "Serie" if src_format == "wide" else "Valor"
        elif df[col].nunique() <= 50:
            result[col] = "Dimensión" if src_format == "long" else "Ignorar"
        else:
            result[col] = "Ignorar"
    return result


def _sicon(done: bool, active: bool) -> str:
    if done:
        return "✅"
    if active:
        return "🔄"
    return "⬜"


def _discard_reason(s) -> str | None:
    """Return auto-discard reason for a Series, or None if ok."""
    vals = s.values.dropna()
    if s.row_count == 0 or len(vals) == 0:
        return "sin datos"
    if s.row_count == 1:
        return "1 solo dato"
    if vals.nunique() == 1:
        return "valor constante"
    if len(vals) / s.row_count < 0.1:
        return "mayormente nula"
    return None


def _merge_matching_results(base, additional):
    """Merge additional-source matches into the base MatchingResult."""
    from src.series_matcher import MatchingResult as _MR
    combined_matches = base.matches + additional.matches
    combined_unmatched_src = base.unmatched_source + additional.unmatched_source
    type_counts: dict[str, int] = {
        "exact": 0, "normalized": 0, "substring": 0, "correlation": 0, "manual": 0,
    }
    for _m in combined_matches:
        type_counts[_m.match_type] = type_counts.get(_m.match_type, 0) + 1
    n_matched = len(combined_matches)
    n_src = base.total_source + additional.total_source
    n_plat = base.total_platform
    return _MR(
        matches=combined_matches,
        unmatched_source=combined_unmatched_src,
        unmatched_platform=additional.unmatched_platform,
        total_source=n_src,
        total_platform=n_plat,
        matched_count=n_matched,
        match_rate_source=n_matched / n_src if n_src > 0 else 0.0,
        match_rate_platform=n_matched / n_plat if n_plat > 0 else 0.0,
        matching_log=base.matching_log + ["--- Source adicional ---"] + additional.matching_log,
        exact_count=type_counts["exact"],
        normalized_count=type_counts["normalized"],
        substring_count=type_counts["substring"],
        correlation_count=type_counts["correlation"],
        manual_count=type_counts["manual"],
    )


def _detect_period_type(series: pd.Series) -> str:
    """Auto-detect period type from a Series of month/quarter values."""
    sample = series.dropna().head(30)
    if sample.empty:
        return "month_text_es"
    nums = pd.to_numeric(sample, errors="coerce")
    if float(nums.notna().mean()) > 0.8:
        mx = float(nums.dropna().max())
        return "quarter_numeric" if mx <= 4 else "month_numeric"
    strs = sample.astype(str).str.strip().str.lower()
    if strs.str.match(r"^q\d").any():
        return "quarter_text_q"
    if strs.str.match(r"^\d[t]").any():
        return "quarter_text_t"
    if any(s in MESES_ES or s[:3] in MESES_ES for s in strs if len(s) >= 3):
        return "month_text_es"
    if any(s in MESES_EN or s[:3] in MESES_EN for s in strs if len(s) >= 3):
        return "month_text_en"
    return "month_text_es"


def _show_date_compose_panel(
    df: pd.DataFrame,
    year_col: str,
    period_col: str,
    period_kind: str,
    key_prefix: str,
) -> tuple[str, bool]:
    """Render the date composition UI panel. Returns (period_type, is_valid)."""
    st.markdown(f"📅 **Composición de fecha: Año + {period_kind}**")
    _n_nan_yr = int(df[year_col].isna().sum())
    _dc1, _dc2 = st.columns(2)
    with _dc1:
        st.caption(f"Columna Año: **{year_col}**")
        if _n_nan_yr > 0:
            st.info(f"ℹ️ Se aplicará forward fill ({_n_nan_yr} celdas vacías)")
    with _dc2:
        st.caption(f"Columna {period_kind}: **{period_col}**")

    _auto_pt = _detect_period_type(df[period_col])
    if period_kind == "Mes":
        _pt_map = {"month_numeric": "Numérico 1-12", "month_text_es": "Texto español", "month_text_en": "Texto inglés"}
    else:
        _pt_map = {"quarter_text_q": "Q1/Q2/Q3/Q4", "quarter_text_t": "1T/2T/3T/4T", "quarter_numeric": "Numérico 1-4"}
    _pt_keys = list(_pt_map.keys())
    _pt_labels = list(_pt_map.values())
    _default_idx = _pt_keys.index(_auto_pt) if _auto_pt in _pt_keys else 0
    _sel_lbl = st.radio("Formato:", _pt_labels, index=_default_idx,
                        horizontal=True, key=f"{key_prefix}_pt_radio")
    _period_type = _pt_keys[_pt_labels.index(_sel_lbl)]

    try:
        _prev_df = df[[year_col, period_col]].head(5).copy()
        _prev_dates = compose_date_column(_prev_df, year_col, period_col, _period_type)
        _yr_filled = _prev_df[year_col].ffill()
        _prev_rows = []
        for _i in range(len(_prev_df)):
            _yr_o = _prev_df.iloc[_i][year_col]
            _yr_d = f"NaN→{_yr_filled.iloc[_i]}" if pd.isna(_yr_o) else str(_yr_o)
            _dt = _prev_dates.iloc[_i]
            _prev_rows.append({
                "Año": _yr_d,
                period_kind: str(_prev_df.iloc[_i][period_col]),
                "Fecha compuesta": _dt.strftime("%Y-%m-%d") if pd.notna(_dt) else "❌ Error",
                "": "✅" if pd.notna(_dt) else "❌",
            })
        st.dataframe(pd.DataFrame(_prev_rows), hide_index=True, use_container_width=True)
        return _period_type, True
    except Exception as _e:
        st.error(f"Error en composición de fecha: {_e}")
        return _period_type, False


def _apply_date_composition(
    df: pd.DataFrame,
    cls_dict: dict,
    period_type: str | None,
) -> tuple[pd.DataFrame, str | None]:
    """Apply date composition if cls_dict has Año+Mes/Trimestre columns.

    Returns (df_out, date_col_name). If composed, df_out has '_composed_date' column.
    If not, returns (df, existing Fecha col).
    """
    year_col = next((c for c, t in cls_dict.items() if t == "Año"), None)
    period_col = next((c for c, t in cls_dict.items() if t in ("Mes", "Trimestre")), None)
    if year_col and period_col and period_type:
        df_out = df.copy()
        df_out["_composed_date"] = compose_date_column(df_out, year_col, period_col, period_type)
        return df_out, "_composed_date"
    return df, next((c for c, t in cls_dict.items() if t == "Fecha"), None)


def _auto_classify_platform_columns(df: pd.DataFrame, plat_format: str) -> dict[str, str]:
    """Auto-classify columns for Platform (Alphacast) extraction."""
    detected_date_col: str | None = None
    for col in df.columns:
        col_lower = col.lower()
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            detected_date_col = col
            break
        if any(k in col_lower for k in ("date", "fecha", "time", "periodo", "period")):
            detected_date_col = col
            break
    if detected_date_col is None:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                sample = df[col].dropna().head(20).astype(str)
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if float(parsed.notna().mean()) >= 0.8:
                        detected_date_col = col
                        break
                except Exception:
                    pass
    result: dict[str, str] = {}
    for col in df.columns:
        if col == detected_date_col or pd.api.types.is_datetime64_any_dtype(df[col]):
            result[col] = "Fecha"
        elif pd.api.types.is_numeric_dtype(df[col]):
            result[col] = "Valor"
        else:
            n_unique = df[col].nunique()
            if n_unique <= 2:
                result[col] = "Ignorar"
            elif plat_format == "long":
                result[col] = "Dimensión"
            else:
                result[col] = "Ignorar"
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Dataset Auditor")
    st.caption("v2.0 — Series Compare")
    st.markdown("---")

    # Reports folder
    if st.button("📂 Abrir carpeta reportes", use_container_width=True, key="sidebar_open_reports"):
        _rdir = Path(__file__).parent / "reports"
        _rdir.mkdir(exist_ok=True)
        reveal_directory(_rdir)

    # Demo files
    st.markdown("---")
    st.markdown("**Demo files:**")
    st.caption("demo/demo_series_source_wide.csv")
    st.caption("demo/demo_series_platform_long.csv")

    # YAML config loader
    st.markdown("---")
    with st.expander("⚙️ Cargar config de matching YAML"):
        _yaml_uploader = st.file_uploader(
            "Config YAML", type=["yaml", "yml", "json"], key="sidebar_yaml_uploader"
        )
        if _yaml_uploader is not None:
            try:
                _yaml_cfg = yaml.safe_load(_yaml_uploader.read().decode("utf-8"))
                st.session_state["loaded_matching_yaml"] = _yaml_cfg
                st.success("Config cargada.")
                st.json(_yaml_cfg, expanded=False)
            except Exception as _e:
                st.error(f"Error: {_e}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # AUDIT section (logic unchanged)
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("🔍 Auditoría")

    _audit_tab_mode = st.session_state.get("audit_tab_mode", "Source")

    df_audit_loaded: pd.DataFrame | None = None
    if _audit_tab_mode == "Source":
        df_audit_loaded = st.session_state.get("source_df")
        if df_audit_loaded is None:
            st.caption("Cargá el dataset fuente en Compare (Paso 1).")
    elif _audit_tab_mode == "Platform":
        df_audit_loaded = st.session_state.get("platform_df")
        if df_audit_loaded is None:
            st.caption("Cargá el dataset plataforma en Compare (Paso 1).")
    else:
        audit_file = st.file_uploader(
            "Subí dataset para auditar", type=["csv", "xlsx", "xls", "xlsb"], key="audit_sep_uploader"
        )
        if audit_file is not None:
            df_audit_loaded = load_file(audit_file, "audit_sep")
            if df_audit_loaded is not None:
                st.success(f"{len(df_audit_loaded):,} × {len(df_audit_loaded.columns)}")

    st.markdown("**Config**")
    config_mode = st.radio("Modo config", ["🔍 Auto", "📄 Archivo YAML"],
                           horizontal=True, key="config_mode")
    config_loaded: dict | None = None

    if config_mode == "📄 Archivo YAML":
        uploaded_config = st.file_uploader(
            "Config YAML", type=["yaml", "yml", "json"], key="config_uploader"
        )
        if uploaded_config is not None:
            try:
                config_loaded = load_config_from_bytes(uploaded_config.read())
                st.success(f"Config: **{config_loaded.get('dataset_name', '?')}**")
            except Exception as e:
                st.error(f"Error en config: {e}")
    else:
        if df_audit_loaded is None:
            st.caption("Cargá un dataset primero.")
        else:
            _detect_key = f"{id(df_audit_loaded)}_{df_audit_loaded.shape[0]}_{df_audit_loaded.shape[1]}"
            if st.session_state.get("_detected_key") != _detect_key:
                detected = auto_detect_config(df_audit_loaded)
                st.session_state["_detected_config"] = detected
                st.session_state["_detected_key"] = _detect_key
                for _k in ["ui_date_col", "ui_format", "ui_value_col", "ui_value_cols",
                            "ui_dims", "ui_freq", "ui_max_null", "ui_allow_zeros",
                            "ui_outlier", "ui_lag", "ui_dataset_name"]:
                    st.session_state.pop(_k, None)

            detected = st.session_state["_detected_config"]
            notes = detected.get("_detection_notes", [])

            with st.expander("Configuracion detectada", expanded=False):
                shown: dict[str, dict] = {}
                for note in notes:
                    shown[note["field"]] = note
                for note in shown.values():
                    icon = {"high": "✅", "medium": "⚠️", "low": "❌"}.get(note["confidence"], "❓")
                    val = note["value"]
                    val_str = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)
                    st.markdown(
                        f"{icon} **{note['field']}**: `{val_str}`  \n"
                        f"<span style='font-size:0.78rem;color:#6b7280'>{note['reason']}</span>",
                        unsafe_allow_html=True,
                    )

            all_cols = list(df_audit_loaded.columns)

            def _col_idx_audit(col, lst):
                return lst.index(col) if col and col in lst else 0

            ui_dataset_name = st.text_input(
                "Nombre dataset",
                value=detected.get("dataset_name", "auto_detected"), key="ui_dataset_name",
            )
            ui_date_col = st.selectbox(
                "Columna fecha", all_cols,
                index=_col_idx_audit(detected["date_column"], all_cols), key="ui_date_col",
            )
            fmt_options = ["long", "wide"]
            ui_format = st.selectbox(
                "Formato", fmt_options,
                index=_col_idx_audit(detected["format"], fmt_options), key="ui_format",
            )
            remaining = [c for c in all_cols if c != ui_date_col]
            if ui_format == "long":
                det_vc = detected.get("value_column", "")
                ui_value_col = st.selectbox(
                    "Columna valor", remaining,
                    index=_col_idx_audit(det_vc, remaining), key="ui_value_col",
                )
                non_val = [c for c in remaining if c != ui_value_col]
                det_dims = [d for d in detected.get("dimension_columns", []) if d in non_val]
                ui_dims = st.multiselect("Dimensiones", non_val, default=det_dims, key="ui_dims")
                ui_value_cols: list = []
            else:
                det_vcs = [c for c in detected.get("value_columns", []) if c in remaining]
                if not det_vcs:
                    det_vcs = [c for c in remaining if pd.api.types.is_numeric_dtype(df_audit_loaded[c])]
                ui_value_cols = st.multiselect(
                    "Columnas valor", remaining, default=det_vcs, key="ui_value_cols"
                )
                ui_dims = []
                ui_value_col = "Value"

            freq_map = {"B": "Business days", "D": "Daily", "W": "Weekly", "M": "Monthly"}
            freq_labels = list(freq_map.keys())
            freq_display = [freq_map[k] for k in freq_labels]
            det_freq = detected.get("expected_frequency", "D")
            freq_idx = freq_labels.index(det_freq) if det_freq in freq_labels else 0
            ui_freq_label = st.selectbox("Frecuencia", freq_display, index=freq_idx, key="ui_freq")
            ui_freq = freq_labels[freq_display.index(ui_freq_label)]
            ui_max_null = st.slider(
                "Max nulos (%)", 0, 100,
                int(detected.get("max_null_pct", 0.05) * 100), key="ui_max_null",
            )
            ui_allow_zeros = st.checkbox(
                "Permitir ceros",
                value=bool(detected.get("allow_zero_values", True)), key="ui_allow_zeros",
            )
            ui_outlier = st.slider(
                "Umbral outliers (z)", 1.0, 5.0,
                float(detected.get("outlier_zscore_threshold", 3.5)), step=0.5, key="ui_outlier",
            )
            ui_lag = st.number_input(
                "Lag aceptable (días)", min_value=0,
                value=int(detected.get("acceptable_lag_days", 3)), key="ui_lag",
            )

            config_loaded = build_config_from_controls(
                dataset_name=ui_dataset_name, date_col=ui_date_col, fmt=ui_format,
                value_col=ui_value_col, value_cols=ui_value_cols, dim_cols=ui_dims,
                freq=ui_freq, max_null_pct=ui_max_null / 100, allow_zeros=ui_allow_zeros,
                outlier_threshold=float(ui_outlier), lag_days=int(ui_lag), detected=detected,
            )

            if st.button("Guardar como YAML", use_container_width=True, key="save_yaml_btn"):
                try:
                    saved = save_auto_config_yaml(config_loaded)
                    st.success(f"Guardado en `configs/{saved.name}`")
                except Exception as _e:
                    st.error(f"Error: {_e}")

    can_audit = df_audit_loaded is not None and config_loaded is not None
    run_audit_btn = st.button(
        "▶ Run Audit", disabled=not can_audit,
        use_container_width=True, key="run_audit_btn",
    )

    export_clicked = st.button(
        "Exportar reporte audit",
        disabled="audit_result" not in st.session_state,
        use_container_width=True,
        key="export_audit_btn",
    )

    if st.button("Abrir carpeta reportes ", use_container_width=True, key="open_reports_audit"):
        reports_dir = Path(__file__).parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        reveal_directory(reports_dir)


# ── Run Audit ─────────────────────────────────────────────────────────────────
if run_audit_btn and can_audit:
    with st.spinner("Ejecutando auditoría..."):
        try:
            prev_snapshot = load_snapshot(config_loaded["dataset_name"])
            report = audit_dataset(df_audit_loaded, config_loaded, prev_snapshot=prev_snapshot)
            st.session_state["audit_result"] = report
            st.session_state["audit_df"] = df_audit_loaded.copy()
            st.session_state["audit_config"] = config_loaded.copy()
            st.session_state["audit_ts"] = datetime.now(timezone.utc).isoformat()
            save_snapshot(df_audit_loaded, config_loaded["dataset_name"])
            st.toast("Snapshot guardado.")
        except Exception as exc:
            st.error(f"Error durante la auditoría: {exc}")
            if os.getenv("DEBUG"):
                st.exception(exc)

# ── Export audit report ───────────────────────────────────────────────────────
if export_clicked and "audit_result" in st.session_state:
    try:
        path = save_report(st.session_state["audit_result"])
        st.sidebar.success(f"Reporte guardado:\n`{path.name}`")
    except Exception as exc:
        st.sidebar.error(f"Error al exportar: {exc}")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_compare, tab_audit, tab_settings = st.tabs(["🔄 Compare", "🔍 Audit", "⚙️ Settings"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMPARE (6-step series flow)
# ═════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## 🔄 Compare por Series")

    # ── Compute step completion state ─────────────────────────────────────────
    _s1_done = (
        st.session_state.get("source_df") is not None
        and st.session_state.get("platform_df") is not None
    )
    _s2_done = (
        st.session_state.get("classification_confirmed", False)
        and st.session_state.get("platform_classification_confirmed", False)
    )
    _s3_done = (
        "source_extraction" in st.session_state
        and "platform_extraction" in st.session_state
    )
    _s3_selection_done = st.session_state.get("series_selection_confirmed", False)
    _s4_done = st.session_state.get("matching_confirmed", False)
    _s5_done = "compare_result" in st.session_state

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 1: Cargar Datasets
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander(
        f"{_sicon(_s1_done, True)} PASO 1: Cargar Datasets",
        expanded=not _s1_done,
    ):
        _col_src, _col_plat = st.columns(2)

        with _col_src:
            st.markdown("**📁 Source (Fuente cruda)**")

            # Format selector
            _fmt_options = ["Long", "Wide", "Wide Transpuesto"]
            _fmt_radio = st.radio(
                "Formato:", _fmt_options, horizontal=True, key="source_format_radio"
            )
            _fmt_code_map = {"Long": "long", "Wide": "wide", "Wide Transpuesto": "wide_transposed"}
            _new_fmt_code = _fmt_code_map[_fmt_radio]

            # Detect format change → clear downstream
            if st.session_state.get("_last_source_format") != _new_fmt_code:
                if "_last_source_format" in st.session_state:
                    for _k in ["classification_confirmed", "source_column_classification",
                                "source_extraction", "platform_extraction",
                                "_auto_cls", "_auto_cls_key",
                                "matching_result", "matching_confirmed",
                                "_auto_matching_result", "compare_result"]:
                        st.session_state.pop(_k, None)
                st.session_state["_last_source_format"] = _new_fmt_code
            st.session_state["source_format"] = _new_fmt_code

            src_file = st.file_uploader(
                "CSV o XLSX / XLS / XLSB", type=["csv", "xlsx", "xls", "xlsb"],
                key="cmp_src_uploader", label_visibility="collapsed",
            )
            if src_file is not None:
                _df_new = load_file(src_file, "step1_src")
                if _df_new is not None:
                    _sk = f"{src_file.name}_{_df_new.shape[0]}_{_df_new.shape[1]}"
                    if st.session_state.get("_src_file_key") != _sk:
                        for _k in ["classification_confirmed", "source_column_classification",
                                    "source_extraction", "platform_extraction",
                                    "_auto_cls", "_auto_cls_key",
                                    "matching_result", "matching_confirmed",
                                    "_auto_matching_result", "compare_result"]:
                            st.session_state.pop(_k, None)
                        st.session_state["source_df"] = _df_new
                        st.session_state["_src_file_key"] = _sk
                        st.session_state["_src_filename"] = src_file.name

            _src_df_cur = st.session_state.get("source_df")
            if _src_df_cur is not None:
                st.caption(f"**{len(_src_df_cur):,} filas × {len(_src_df_cur.columns)} cols**")
                st.dataframe(_src_df_cur.head(3), use_container_width=True, hide_index=True)

        with _col_plat:
            st.markdown("**📁 Platform (Alphacast)**")

            # Format selector
            _plat_fmt_options = ["Long", "Wide"]
            _plat_fmt_radio = st.radio(
                "Formato:", _plat_fmt_options, horizontal=True, key="platform_format_radio"
            )
            _plat_fmt_code_map = {"Long": "long", "Wide": "wide"}
            _new_plat_fmt_code = _plat_fmt_code_map[_plat_fmt_radio]

            # Detect format change → clear downstream
            if st.session_state.get("_last_platform_format") != _new_plat_fmt_code:
                if "_last_platform_format" in st.session_state:
                    for _k in ["platform_classification_confirmed", "platform_column_classification",
                                "_auto_cls_plat", "_auto_cls_plat_key",
                                "platform_extraction", "matching_result", "matching_confirmed",
                                "_auto_matching_result", "compare_result",
                                "series_selection_confirmed", "source_selected_series",
                                "platform_selected_series"]:
                        st.session_state.pop(_k, None)
                st.session_state["_last_platform_format"] = _new_plat_fmt_code
            st.session_state["platform_format"] = _new_plat_fmt_code

            # ── Origen: Subir archivo vs Alphacast API ────────────────────────
            _plat_origen = st.radio(
                "Origen:", ["Subir archivo", "Descargar de Alphacast"],
                horizontal=True, key="platform_source_radio",
            )

            # Clear platform_df when switching origin
            if st.session_state.get("_last_plat_origen") != _plat_origen:
                if "_last_plat_origen" in st.session_state:
                    for _k in ["platform_df", "_plat_file_key",
                                "platform_classification_confirmed", "platform_column_classification",
                                "_auto_cls_plat", "_auto_cls_plat_key",
                                "platform_extraction", "matching_result", "matching_confirmed",
                                "_auto_matching_result", "compare_result",
                                "series_selection_confirmed", "source_selected_series",
                                "platform_selected_series"]:
                        st.session_state.pop(_k, None)
                st.session_state["_last_plat_origen"] = _plat_origen
            st.session_state["platform_source_type"] = (
                "file" if _plat_origen == "Subir archivo" else "alphacast"
            )

            if _plat_origen == "Subir archivo":
                plat_file = st.file_uploader(
                    "CSV o XLSX / XLS / XLSB", type=["csv", "xlsx", "xls", "xlsb"],
                    key="cmp_plat_uploader", label_visibility="collapsed",
                )
                if plat_file is not None:
                    _df_plat_new = load_file(plat_file, "step1_plat")
                    if _df_plat_new is not None:
                        _pk = f"{plat_file.name}_{_df_plat_new.shape[0]}_{_df_plat_new.shape[1]}"
                        if st.session_state.get("_plat_file_key") != _pk:
                            for _k in ["platform_classification_confirmed", "platform_column_classification",
                                        "_auto_cls_plat", "_auto_cls_plat_key",
                                        "platform_extraction", "matching_result",
                                        "matching_confirmed", "_auto_matching_result", "compare_result",
                                        "series_selection_confirmed", "source_selected_series",
                                        "platform_selected_series"]:
                                st.session_state.pop(_k, None)
                            st.session_state["platform_df"] = _df_plat_new
                            st.session_state["_plat_file_key"] = _pk
                            st.session_state.pop("alphacast_dataset_id", None)
                            st.session_state.pop("alphacast_dataset_name", None)

            else:
                # ── Alphacast API panel ───────────────────────────────────────
                _ac_key = st.session_state.get("alphacast_api_key")
                _ac_source = st.session_state.get("alphacast_api_key_source")
                _ac_show_input = st.session_state.get("_ac_show_key_input", False)

                if _ac_key and not _ac_show_input:
                    if _ac_source == "streamlit_secrets":
                        st.success(f"API Key configurada desde Streamlit secrets ({mask_api_key(_ac_key)})")
                        st.caption("En Streamlit Community Cloud, cargala en App settings > Secrets con `ALPHACAST_API_KEY`.")
                    elif _ac_source == "env":
                        st.success(f"API Key configurada desde variable de entorno ({mask_api_key(_ac_key)})")
                        st.caption(f"Variable usada: `{API_KEY_ENV_VAR}`")
                    else:
                        st.success(f"API Key configurada ({mask_api_key(_ac_key)})")
                        _kc1, _kc2 = st.columns(2)
                        with _kc1:
                            if st.button("Cambiar key", key="plat_ac_change_key"):
                                st.session_state["_ac_show_key_input"] = True
                                st.rerun()
                        with _kc2:
                            if st.button("Borrar key", key="plat_ac_delete_key"):
                                delete_api_key()
                                st.session_state.pop("alphacast_api_key", None)
                                st.session_state.pop("alphacast_api_key_source", None)
                                st.session_state["_ac_show_key_input"] = False
                                st.rerun()
                else:
                    _new_ac_key = st.text_input(
                        "API Key:", type="password", placeholder="ak_xxxxxxxxxxxxxxxx",
                        key="plat_ac_key_input",
                        help="Encontrala en alphacast.io -> Settings",
                    )
                    st.caption("En Streamlit Community Cloud conviene cargarla en Secrets como `ALPHACAST_API_KEY`.")
                    if st.button("Guardar key", key="plat_ac_save_key"):
                        if _new_ac_key.strip():
                            with st.spinner("Validando..."):
                                if validate_api_key(_new_ac_key.strip()):
                                    save_api_key(_new_ac_key.strip())
                                    st.session_state["alphacast_api_key"] = _new_ac_key.strip()
                                    st.session_state["alphacast_api_key_source"] = "file"
                                    st.session_state["_ac_show_key_input"] = False
                                    st.success("Key valida y guardada.")
                                    st.rerun()
                                else:
                                    st.error("Key invalida. Verifica que sea correcta.")
                        else:
                            st.warning("Ingresa una API key.")

                # Dataset ID input + download (only when key is set)
                _ac_key_ready = bool(st.session_state.get("alphacast_api_key"))
                if _ac_key_ready:
                    _ac_ds_id = st.number_input(
                        "Dataset ID:", min_value=1, step=1,
                        value=st.session_state.get("alphacast_dataset_id") or 1,
                        key="plat_ac_dataset_id",
                        help="Lo encontrás en la URL: alphacast.io/datasets/{ID}",
                    )
                    if st.button("⬇️ Descargar dataset", key="plat_ac_download", type="primary"):
                        with st.spinner("Descargando desde Alphacast..."):
                            _ac_result = fetch_dataset(
                                st.session_state["alphacast_api_key"], int(_ac_ds_id)
                            )
                        if _ac_result is not None:
                            for _k in ["platform_classification_confirmed", "platform_column_classification",
                                        "_auto_cls_plat", "_auto_cls_plat_key",
                                        "platform_extraction", "matching_result",
                                        "matching_confirmed", "_auto_matching_result", "compare_result",
                                        "series_selection_confirmed", "source_selected_series",
                                        "platform_selected_series"]:
                                st.session_state.pop(_k, None)
                            st.session_state["platform_df"] = _ac_result.df
                            st.session_state["_plat_file_key"] = f"alphacast_{_ac_result.id}"
                            st.session_state["alphacast_dataset_id"] = _ac_result.id
                            st.session_state["alphacast_dataset_name"] = _ac_result.name
                            st.rerun()
                        else:
                            st.error(
                                "❌ No se pudo descargar. Verificá el ID y tu conexión."
                            )

                    # Show current Alphacast dataset info if loaded
                    if st.session_state.get("alphacast_dataset_name"):
                        st.caption(
                            f"✅ **{st.session_state['alphacast_dataset_name']}** "
                            f"(ID #{st.session_state['alphacast_dataset_id']})"
                        )

            _plat_df_cur = st.session_state.get("platform_df")
            if _plat_df_cur is not None:
                st.caption(f"**{len(_plat_df_cur):,} filas × {len(_plat_df_cur.columns)} cols**")
                st.dataframe(_plat_df_cur.head(3), use_container_width=True, hide_index=True)

        if _s1_done:
            st.success("Ambos datasets cargados. Continuá con el Paso 2.")
        elif st.session_state.get("source_df") is not None:
            st.info("Cargá el dataset Platform para continuar.")
        elif st.session_state.get("platform_df") is not None:
            st.info("Cargá el dataset Source para continuar.")

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 2: Clasificar Columnas del Source
    # ─────────────────────────────────────────────────────────────────────────
    if _s1_done:
        with st.expander(
            f"{_sicon(_s2_done, not _s2_done)} PASO 2: Clasificar Columnas",
            expanded=not _s2_done,
        ):
            _src_df = st.session_state["source_df"]
            _src_fmt = st.session_state.get("source_format", "long")

            # Cache auto-classification
            _cls_cache_key = f"cls_{id(_src_df)}_{_src_df.shape}_{_src_fmt}"
            if st.session_state.get("_auto_cls_key") != _cls_cache_key:
                st.session_state["_auto_cls"] = _auto_classify_columns(_src_df, _src_fmt)
                st.session_state["_auto_cls_key"] = _cls_cache_key
            _auto_cls = st.session_state["_auto_cls"]

            # Use previously confirmed classification as starting point if same format
            _existing_cls = st.session_state.get("source_column_classification", _auto_cls)

            if _src_fmt == "wide_transposed":
                st.markdown("**Formato Wide Transpuesto** — cada fila es una serie, las columnas son fechas.")
                _all_cols = list(_src_df.columns)
                _entity_col = st.selectbox(
                    "Columna de entidades (nombre de la serie):",
                    _all_cols, key="entity_col_select",
                )
                _header_cols = [c for c in _all_cols if c != _entity_col]
                _headers_sample = pd.Series([str(c) for c in _header_cols[:12]])
                try:
                    _dates_prev = pd.to_datetime(_headers_sample, errors="coerce")
                    _valid_hdrs = _dates_prev.dropna()
                    if len(_valid_hdrs) > 0:
                        _date_strs = [d.strftime("%Y-%m-%d") for d in _valid_hdrs[:6]]
                        st.caption(f"Fechas detectadas en headers: {', '.join(_date_strs)}{'...' if len(_valid_hdrs) > 6 else ''}")
                    else:
                        st.warning("No se detectaron fechas en los headers de columna.")
                except Exception:
                    st.warning("No se pudo parsear los headers como fechas.")

                if st.button("✅ Confirmar (Wide Transpuesto)", key="confirm_cls_transposed"):
                    st.session_state["source_column_classification"] = {"entity_col": _entity_col}
                    st.session_state["classification_confirmed"] = True
                    for _k in ["source_extraction", "platform_extraction", "matching_result",
                                "matching_confirmed", "_auto_matching_result", "compare_result"]:
                        st.session_state.pop(_k, None)
                    st.rerun()

            else:
                # Long or Wide: show data_editor table
                if _src_fmt == "long":
                    _cls_options = ["Fecha", "Año", "Mes", "Trimestre", "Dimensión", "Valor", "Ignorar"]
                    st.caption("**Fecha** = col. de fechas · **Año+Mes** o **Año+Trimestre** = fecha compuesta · **Dimensión** = categoría · **Valor** = valor numérico · **Ignorar** = excluir")
                else:
                    _cls_options = ["Fecha", "Año", "Mes", "Trimestre", "Serie", "Ignorar"]
                    st.caption("**Fecha** = col. de fechas · **Año+Mes** o **Año+Trimestre** = fecha compuesta · **Serie** = columna numérica · **Ignorar** = excluir")

                _cls_rows = []
                for _col in _src_df.columns:
                    _det_type = _detect_column_type(_src_df, _col)
                    _cur_cls = _existing_cls.get(_col, _auto_cls.get(_col, _cls_options[-1]))
                    if _cur_cls not in _cls_options:
                        _cur_cls = _auto_cls.get(_col, _cls_options[-1])
                    _cls_rows.append({
                        "Columna": _col,
                        "Tipo detectado": _det_type,
                        "Tu clasificación": _cur_cls,
                    })

                _cls_df = pd.DataFrame(_cls_rows)
                _edited_cls = st.data_editor(
                    _cls_df,
                    column_config={
                        "Columna": st.column_config.TextColumn(disabled=True),
                        "Tipo detectado": st.column_config.TextColumn(disabled=True),
                        "Tu clasificación": st.column_config.SelectboxColumn(
                            options=_cls_options, required=True,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="classification_editor",
                )

                # Date composition panel (shown when Año + Mes/Trimestre selected)
                _src_cls_dict = dict(zip(_edited_cls["Columna"], _edited_cls["Tu clasificación"]))
                _src_year_col = next((c for c, t in _src_cls_dict.items() if t == "Año"), None)
                _src_period_col = next((c for c, t in _src_cls_dict.items() if t in ("Mes", "Trimestre")), None)
                _src_period_kind = next((t for t in _src_cls_dict.values() if t in ("Mes", "Trimestre")), None)
                _src_compose_valid = True
                _src_period_type = None
                if _src_year_col and _src_period_col:
                    _src_period_type, _src_compose_valid = _show_date_compose_panel(
                        _src_df, _src_year_col, _src_period_col, _src_period_kind, "src_compose"
                    )

                # Validate
                _n_fecha = (_edited_cls["Tu clasificación"] == "Fecha").sum()
                _n_year = (_edited_cls["Tu clasificación"] == "Año").sum()
                _n_period = (_edited_cls["Tu clasificación"].isin(["Mes", "Trimestre"])).sum()
                _has_date = _n_fecha >= 1 or (_n_year >= 1 and _n_period >= 1)
                _can_confirm = False
                if not _has_date:
                    st.error("Necesitás al menos 1 columna Fecha, o columnas Año + Mes/Trimestre.")
                else:
                    if _n_fecha > 1:
                        st.warning("Más de 1 columna Fecha — se usará la primera.")
                    if not _src_compose_valid and _src_year_col:
                        st.error("Error en la composición de fecha. Revisá el formato.")
                    else:
                        _can_confirm = True

                if _src_fmt == "long":
                    _n_val = (_edited_cls["Tu clasificación"] == "Valor").sum()
                    if _n_val == 0:
                        st.error("Debe haber al menos 1 columna de tipo Valor.")
                        _can_confirm = False
                    elif _n_val > 1:
                        st.warning("Más de 1 columna Valor — se usará la primera.")
                elif _src_fmt == "wide":
                    _n_serie = (_edited_cls["Tu clasificación"] == "Serie").sum()
                    if _n_serie == 0:
                        st.error("Debe haber al menos 1 columna de tipo Serie.")
                        _can_confirm = False

                if st.button(
                    "✅ Confirmar clasificación Source", key="confirm_cls",
                    type="primary", disabled=not _can_confirm,
                ):
                    _cls_result = dict(zip(_edited_cls["Columna"], _edited_cls["Tu clasificación"]))
                    st.session_state["source_column_classification"] = _cls_result
                    st.session_state["classification_confirmed"] = True
                    st.session_state["source_period_type"] = _src_period_type
                    for _k in ["source_extraction", "platform_extraction", "matching_result",
                                "matching_confirmed", "_auto_matching_result", "compare_result",
                                "series_selection_confirmed", "source_selected_series",
                                "platform_selected_series"]:
                        st.session_state.pop(_k, None)
                    st.rerun()

            # ── Platform classification ───────────────────────────────────────
            if st.session_state.get("classification_confirmed", False):
                st.markdown("---")
                st.markdown("**📁 Clasificación Platform (Alphacast)**")
                _plat_df_cls = st.session_state["platform_df"]
                _plat_fmt_cls = st.session_state.get("platform_format", "long")

                if st.session_state.get("platform_classification_confirmed", False):
                    _plat_cls_done = st.session_state["platform_column_classification"]
                    st.success(f"Clasificación Platform confirmada: {len(_plat_cls_done)} columnas.")
                    if st.button("✏️ Editar clasificación Platform", key="edit_plat_cls"):
                        st.session_state.pop("platform_classification_confirmed", None)
                        for _k in ["platform_extraction", "matching_result", "matching_confirmed",
                                    "_auto_matching_result", "compare_result",
                                    "series_selection_confirmed", "source_selected_series",
                                    "platform_selected_series"]:
                            st.session_state.pop(_k, None)
                        st.rerun()
                else:
                    _plat_cls_cache_key = f"plat_cls_{id(_plat_df_cls)}_{_plat_df_cls.shape}_{_plat_fmt_cls}"
                    if st.session_state.get("_auto_cls_plat_key") != _plat_cls_cache_key:
                        st.session_state["_auto_cls_plat"] = _auto_classify_platform_columns(
                            _plat_df_cls, _plat_fmt_cls
                        )
                        st.session_state["_auto_cls_plat_key"] = _plat_cls_cache_key
                    _auto_cls_plat = st.session_state["_auto_cls_plat"]
                    _existing_plat_cls = st.session_state.get("platform_column_classification", _auto_cls_plat)

                    if _plat_fmt_cls == "long":
                        _plat_cls_options = ["Fecha", "Año", "Mes", "Trimestre", "Dimensión", "Valor", "Ignorar"]
                        st.caption("**Fecha** = col. de fechas · **Año+Mes** o **Año+Trimestre** = fecha compuesta · **Dimensión** = categoría · **Valor** = valor numérico · **Ignorar** = excluir")
                    else:
                        _plat_cls_options = ["Fecha", "Año", "Mes", "Trimestre", "Valor", "Ignorar"]
                        st.caption("**Fecha** = col. de fechas · **Año+Mes** o **Año+Trimestre** = fecha compuesta · **Valor** = columna numérica · **Ignorar** = excluir")

                    _plat_cls_rows = []
                    for _col in _plat_df_cls.columns:
                        _det_type = _detect_column_type(_plat_df_cls, _col)
                        _cur_plat_cls = _existing_plat_cls.get(_col, _auto_cls_plat.get(_col, _plat_cls_options[-1]))
                        if _cur_plat_cls not in _plat_cls_options:
                            _cur_plat_cls = _auto_cls_plat.get(_col, _plat_cls_options[-1])
                        _n_uniq = _plat_df_cls[_col].nunique() if not pd.api.types.is_numeric_dtype(_plat_df_cls[_col]) else None
                        _type_note = _det_type
                        if _n_uniq is not None and _n_uniq <= 2:
                            _type_note += f" ⚠️ {_n_uniq} valor único"
                        _plat_cls_rows.append({
                            "Columna": _col,
                            "Tipo detectado": _type_note,
                            "Tu clasificación": _cur_plat_cls,
                        })

                    _plat_cls_df = pd.DataFrame(_plat_cls_rows)
                    _edited_plat_cls = st.data_editor(
                        _plat_cls_df,
                        column_config={
                            "Columna": st.column_config.TextColumn(disabled=True),
                            "Tipo detectado": st.column_config.TextColumn(disabled=True),
                            "Tu clasificación": st.column_config.SelectboxColumn(
                                options=_plat_cls_options, required=True,
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                        key="platform_classification_editor",
                    )

                    # Date composition panel for Platform
                    _plat_cls_dict_edit = dict(zip(_edited_plat_cls["Columna"], _edited_plat_cls["Tu clasificación"]))
                    _plat_year_col = next((c for c, t in _plat_cls_dict_edit.items() if t == "Año"), None)
                    _plat_period_col = next((c for c, t in _plat_cls_dict_edit.items() if t in ("Mes", "Trimestre")), None)
                    _plat_period_kind = next((t for t in _plat_cls_dict_edit.values() if t in ("Mes", "Trimestre")), None)
                    _plat_compose_valid = True
                    _plat_period_type_edit = None
                    if _plat_year_col and _plat_period_col:
                        _plat_period_type_edit, _plat_compose_valid = _show_date_compose_panel(
                            _plat_df_cls, _plat_year_col, _plat_period_col, _plat_period_kind, "plat_compose"
                        )

                    _plat_n_fecha = (_edited_plat_cls["Tu clasificación"] == "Fecha").sum()
                    _plat_n_year = (_edited_plat_cls["Tu clasificación"] == "Año").sum()
                    _plat_n_period = (_edited_plat_cls["Tu clasificación"].isin(["Mes", "Trimestre"])).sum()
                    _plat_has_date = _plat_n_fecha >= 1 or (_plat_n_year >= 1 and _plat_n_period >= 1)
                    _plat_can_confirm = False
                    if not _plat_has_date:
                        st.error("Necesitás al menos 1 columna Fecha, o columnas Año + Mes/Trimestre.")
                    else:
                        if not _plat_compose_valid and _plat_year_col:
                            st.error("Error en la composición de fecha. Revisá el formato.")
                        else:
                            _plat_can_confirm = True

                    _plat_n_val = (_edited_plat_cls["Tu clasificación"] == "Valor").sum()
                    if _plat_n_val == 0:
                        st.error("Debe haber al menos 1 columna de tipo Valor.")
                        _plat_can_confirm = False

                    if st.button(
                        "✅ Confirmar clasificación Platform", key="confirm_plat_cls",
                        type="primary", disabled=not _plat_can_confirm,
                    ):
                        _plat_cls_result = dict(zip(
                            _edited_plat_cls["Columna"], _edited_plat_cls["Tu clasificación"]
                        ))
                        st.session_state["platform_column_classification"] = _plat_cls_result
                        st.session_state["platform_classification_confirmed"] = True
                        st.session_state["platform_period_type"] = _plat_period_type_edit
                        for _k in ["platform_extraction", "matching_result", "matching_confirmed",
                                    "_auto_matching_result", "compare_result",
                                    "series_selection_confirmed", "source_selected_series",
                                    "platform_selected_series"]:
                            st.session_state.pop(_k, None)
                        st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 3: Extraer Series (auto-runs when step 2 is done)
    # ─────────────────────────────────────────────────────────────────────────
    if _s2_done and not _s3_done:
        _src_df_ex = st.session_state["source_df"]
        _plat_df_ex = st.session_state["platform_df"]
        _fmt_ex = st.session_state.get("source_format", "long")
        _cls_ex = st.session_state.get("source_column_classification", {})
        _plat_fmt_ex = st.session_state.get("platform_format", "long")
        _plat_cls_ex = st.session_state.get("platform_column_classification", {})

        with st.expander("🔄 PASO 3: Extraer Series", expanded=True):
            _ex_status = st.empty()
            _ex_status.info("Extrayendo series...")
            try:
                # — Source extraction —
                _src_ext = None
                if _fmt_ex == "long":
                    _src_df_composed, _date_col_ex = _apply_date_composition(
                        _src_df_ex, _cls_ex, st.session_state.get("source_period_type")
                    )
                    _dim_cols_ex = [c for c, t in _cls_ex.items() if t == "Dimensión"]
                    _val_col_ex = next(
                        (c for c, t in _cls_ex.items() if t == "Valor"), None
                    )
                    if not _date_col_ex or not _val_col_ex:
                        st.error("Clasificación Source incompleta: falta columna Fecha o Valor.")
                    else:
                        _src_ext = extract_series_long(_src_df_composed, _date_col_ex, _dim_cols_ex, _val_col_ex)

                elif _fmt_ex == "wide":
                    _src_df_composed, _date_col_ex = _apply_date_composition(
                        _src_df_ex, _cls_ex, st.session_state.get("source_period_type")
                    )
                    _val_cols_ex = [c for c, t in _cls_ex.items() if t == "Serie"]
                    if not _date_col_ex or not _val_cols_ex:
                        st.error("Clasificación Source incompleta: falta Fecha o columnas Serie.")
                    else:
                        _src_ext = extract_series_wide(_src_df_composed, _date_col_ex, _val_cols_ex)

                else:  # wide_transposed
                    _entity_col_ex = _cls_ex.get("entity_col")
                    _src_ext = extract_series_wide_transposed(_src_df_ex, _entity_col_ex)

                # — Platform extraction —
                _plat_ext = None
                _plat_df_composed, _plat_date_col_ex = _apply_date_composition(
                    _plat_df_ex, _plat_cls_ex, st.session_state.get("platform_period_type")
                )
                if _plat_fmt_ex == "wide":
                    _plat_val_cols_ex = [c for c, t in _plat_cls_ex.items() if t == "Valor"]
                    if not _plat_date_col_ex or not _plat_val_cols_ex:
                        st.error("Clasificación Platform incompleta: falta Fecha o columnas Valor.")
                    else:
                        _plat_ext = extract_series_wide(_plat_df_composed, _plat_date_col_ex, _plat_val_cols_ex)
                else:  # long
                    _plat_dim_cols_ex = [c for c, t in _plat_cls_ex.items() if t == "Dimensión"]
                    _plat_val_col_ex = next(
                        (c for c, t in _plat_cls_ex.items() if t == "Valor"), None
                    )
                    if not _plat_date_col_ex or not _plat_val_col_ex:
                        st.error("Clasificación Platform incompleta: falta Fecha o columna Valor.")
                    else:
                        _plat_ext = extract_series_long(
                            _plat_df_composed, _plat_date_col_ex, _plat_dim_cols_ex, _plat_val_col_ex
                        )

                if _src_ext is not None and _plat_ext is not None:
                    st.session_state["source_extraction"] = _src_ext
                    st.session_state["platform_extraction"] = _plat_ext
                    _s3_done = True
                    _ex_status.success(
                        f"Extraídas {_src_ext.total_series} series source y "
                        f"{_plat_ext.total_series} series platform."
                    )

            except Exception as _ex_err:
                st.error(f"Error en extracción: {_ex_err}")
                st.exception(_ex_err)

    if _s3_done:
        _src_ext = st.session_state["source_extraction"]
        _plat_ext = st.session_state["platform_extraction"]

        with st.expander(
            f"✅ PASO 3: Series Extraídas — {_src_ext.total_series} source · {_plat_ext.total_series} platform",
            expanded=False,
        ):
            _c3a, _c3b = st.columns(2)
            with _c3a:
                st.markdown(f"**Source: {_src_ext.total_series} series** ({_src_ext.dataset_format})")
                _src_tbl = [
                    {"Nombre": s.name, "Puntos": s.row_count,
                     "Desde": s.date_range[0], "Hasta": s.date_range[1]}
                    for s in _src_ext.series
                ]
                st.dataframe(pd.DataFrame(_src_tbl), hide_index=True,
                             use_container_width=True, height=250)
            with _c3b:
                st.markdown(f"**Platform: {_plat_ext.total_series} series** ({_plat_ext.dataset_format})")
                _plat_tbl = [
                    {"Nombre": s.name, "Puntos": s.row_count,
                     "Desde": s.date_range[0], "Hasta": s.date_range[1]}
                    for s in _plat_ext.series
                ]
                st.dataframe(pd.DataFrame(_plat_tbl), hide_index=True,
                             use_container_width=True, height=250)

            with st.expander("📋 Logs de extracción"):
                _lc1, _lc2 = st.columns(2)
                with _lc1:
                    st.markdown("**Source:**")
                    for _lg in _src_ext.extraction_log:
                        st.caption(_lg)
                with _lc2:
                    st.markdown("**Platform:**")
                    for _lg in _plat_ext.extraction_log:
                        st.caption(_lg)

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 3b: Seleccionar Series
    # ─────────────────────────────────────────────────────────────────────────
    if _s3_done:
        _src_ext_sel = st.session_state["source_extraction"]
        _plat_ext_sel = st.session_state["platform_extraction"]

        # ── Init / reset platform selection state when dataset identity changes
        _plat_id_3b = "|".join(s.name for s in _plat_ext_sel.series)
        if st.session_state.get("_sel_plat_id") != _plat_id_3b:
            st.session_state["_sel_plat_id"] = _plat_id_3b
            st.session_state["_sel_plat_ver"] = 0
            st.session_state["_sel_plat_defaults"] = {
                i: (_discard_reason(_s) is None)
                for i, _s in enumerate(_plat_ext_sel.series)
            }

        with st.expander(
            f"{_sicon(_s3_selection_done, not _s3_selection_done)} PASO 3b: Seleccionar Series",
            expanded=not _s3_selection_done,
        ):
            _c_sel_src, _c_sel_plat = st.columns(2)

            # ── Source: all series enter the pool automatically ───────────────
            with _c_sel_src:
                st.markdown(f"**📁 Source: {_src_ext_sel.total_series} series disponibles**")
                st.caption("Todas las series entran automáticamente al pool de matching.")
                with st.expander("Ver series"):
                    _src_info_df = pd.DataFrame([
                        {"Nombre": _s.name, "Puntos": _s.row_count,
                         "Desde": _s.date_range[0], "Hasta": _s.date_range[1]}
                        for _s in _src_ext_sel.series
                    ])
                    st.dataframe(_src_info_df, hide_index=True, use_container_width=True)

            # ── Platform: user selects which series to verify ─────────────────
            with _c_sel_plat:
                st.markdown(f"**📁 Platform: {_plat_ext_sel.total_series} series extraídas**")
                st.caption("Seleccioná qué series querés verificar contra la fuente:")
                _pb1, _pb2 = st.columns(2)
                with _pb1:
                    if st.button("Seleccionar todo", key="sel_plat_all"):
                        st.session_state["_sel_plat_defaults"] = {
                            i: True for i in range(len(_plat_ext_sel.series))
                        }
                        st.session_state["_sel_plat_ver"] += 1
                        st.rerun()
                with _pb2:
                    if st.button("Deseleccionar todo", key="sel_plat_none"):
                        st.session_state["_sel_plat_defaults"] = {
                            i: False for i in range(len(_plat_ext_sel.series))
                        }
                        st.session_state["_sel_plat_ver"] += 1
                        st.rerun()
                _plat_defs = st.session_state["_sel_plat_defaults"]
                _plat_editor_df = pd.DataFrame([
                    {
                        "Incluir": _plat_defs.get(i, _discard_reason(_s) is None),
                        "Nombre": _s.name,
                        "Puntos": _s.row_count,
                        "Descarte": _discard_reason(_s) or "",
                    }
                    for i, _s in enumerate(_plat_ext_sel.series)
                ])
                _edited_plat_df = st.data_editor(
                    _plat_editor_df,
                    key=f"de_plat_{st.session_state['_sel_plat_ver']}",
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Incluir": st.column_config.CheckboxColumn("Incluir", default=True),
                        "Nombre": st.column_config.TextColumn("Nombre", disabled=True),
                        "Puntos": st.column_config.NumberColumn("Puntos", disabled=True),
                        "Descarte": st.column_config.TextColumn("Motivo descarte", disabled=True),
                    },
                )
                _n_plat_sel = int(_edited_plat_df["Incluir"].sum())
                st.caption(
                    f"{len(_edited_plat_df)} series · "
                    f"**{_n_plat_sel} seleccionadas** · "
                    f"{len(_edited_plat_df) - _n_plat_sel} descartadas"
                )

            _selected_plat_names = list(
                _edited_plat_df.loc[_edited_plat_df["Incluir"] == True, "Nombre"]
            )

            if not _selected_plat_names:
                st.warning("Seleccioná al menos 1 serie Platform para continuar.")

            if st.button(
                "✅ Confirmar series", key="confirm_series_sel",
                type="primary", disabled=not bool(_selected_plat_names),
            ):
                # All source series enter the pool automatically
                st.session_state["source_selected_series"] = [
                    s.name for s in _src_ext_sel.series
                ]
                st.session_state["platform_selected_series"] = _selected_plat_names
                st.session_state["series_selection_confirmed"] = True
                for _k in ["matching_result", "matching_confirmed",
                            "_auto_matching_result", "compare_result"]:
                    st.session_state.pop(_k, None)
                st.rerun()

        if _s3_selection_done:
            _sel_plat_names = st.session_state["platform_selected_series"]
            _sel_src_all = st.session_state.get("source_selected_series", [])
            st.caption(
                f"Pool source: **{len(_sel_src_all)}** series · "
                f"Verificando: **{len(_sel_plat_names)}** series platform"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 4: Matchear Series
    # ─────────────────────────────────────────────────────────────────────────
    if _s3_done and _s3_selection_done:
        _src_ext4 = st.session_state["source_extraction"]
        _plat_ext4 = st.session_state["platform_extraction"]
        _sel_src4 = st.session_state.get("source_selected_series",
                                          [s.name for s in _src_ext4.series])
        _sel_plat4 = st.session_state.get("platform_selected_series",
                                           [s.name for s in _plat_ext4.series])

        _src_ext4_filtered = dataclasses.replace(
            _src_ext4,
            series=[s for s in _src_ext4.series if s.name in _sel_src4],
            total_series=len(_sel_src4),
        )
        _plat_ext4_filtered = dataclasses.replace(
            _plat_ext4,
            series=[s for s in _plat_ext4.series if s.name in _sel_plat4],
            total_series=len(_sel_plat4),
        )

        # Auto-match (cached)
        if "_auto_matching_result" not in st.session_state:
            _auto_mr = auto_match_series(_src_ext4_filtered, _plat_ext4_filtered)
            st.session_state["_auto_matching_result"] = _auto_mr
            # Build file map for main source series
            _src_fn = st.session_state.get("_src_filename", "source")
            st.session_state["source_series_file_map"] = {
                s.name: _src_fn for s in _src_ext4_filtered.series
            }
        _auto_mr = st.session_state["_auto_matching_result"]

        with st.expander(
            f"{_sicon(_s4_done, not _s4_done)} PASO 4: Matchear Series — "
            f"{_auto_mr.matched_count}/{_plat_ext4_filtered.total_series} platform matched",
            expanded=not _s4_done,
        ):
            st.markdown(
                f"**Matching automático: {_auto_mr.matched_count} de "
                f"{_plat_ext4_filtered.total_series} series platform matcheadas**"
            )

            # ── Auto-matches table with undo checkboxes (CAMBIO 4) ────────────
            _unmatched_src4 = _auto_mr.unmatched_source
            _unmatched_plat4 = _auto_mr.unmatched_platform

            if _auto_mr.matches:
                st.markdown(f"**Matches automáticos ({_auto_mr.matched_count}):**")
                _match_data4 = []
                for _m4 in _auto_mr.matches:
                    _match_data4.append({
                        "Incluir": True,
                        "Platform": _m4.platform_series.name,
                        "Source": _m4.source_series.name,
                        "Tipo": _m4.match_type,
                        "Corr.": _m4.similarity_details[:30] if _m4.similarity_details else "",
                    })
                _df_matches4 = pd.DataFrame(_match_data4)
                _edited_matches4 = st.data_editor(
                    _df_matches4,
                    column_config={
                        "Incluir": st.column_config.CheckboxColumn("✅", default=True, width="small"),
                        "Platform": st.column_config.TextColumn("Serie Platform", disabled=True),
                        "Source": st.column_config.TextColumn("Serie Source", disabled=True),
                        "Tipo": st.column_config.TextColumn("Tipo", disabled=True, width="small"),
                        "Corr.": st.column_config.TextColumn("Corr.", disabled=True, width="small"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    key=f"match_editor_{len(_auto_mr.matches)}",
                )
            else:
                _edited_matches4 = pd.DataFrame({"Incluir": []})

            # Determine which auto-matches were deselected
            _deselected_src_objs4: dict[str, object] = {}
            _deselected_plat_objs4: list[object] = []
            for _i4m in range(len(_auto_mr.matches)):
                if not _edited_matches4.iloc[_i4m]["Incluir"]:
                    _deselected_src_objs4[_auto_mr.matches[_i4m].source_series.name] = (
                        _auto_mr.matches[_i4m].source_series
                    )
                    _deselected_plat_objs4.append(_auto_mr.matches[_i4m].platform_series)

            # Effective unmatched platform = auto unmatched + deselected
            _all_unmatched_plat4 = list(_unmatched_plat4) + [
                UnmatchedSeries(s, "platform", "match deshecho manualmente")
                for s in _deselected_plat_objs4
            ]
            # Available source pool = auto unmatched + freed from deselected
            _avail_src_names4 = (
                [u.series.name for u in _unmatched_src4] + list(_deselected_src_objs4.keys())
            )
            _avail_src_objs4: dict[str, object] = {
                u.series.name: u.series for u in _unmatched_src4
            }
            _avail_src_objs4.update(_deselected_src_objs4)

            # ── Platform series sin match → manual source assignment (CAMBIO 5) ─
            if _all_unmatched_plat4:
                st.markdown(f"**⚠️ Series Platform sin match ({len(_all_unmatched_plat4)}):**")
                for _ip4, _up4 in enumerate(_all_unmatched_plat4):
                    _ua4, _ub4 = st.columns([2, 3])
                    with _ua4:
                        st.markdown(f"⚠️ **{_up4.series.name}**")
                        st.caption(
                            f"{_up4.series.row_count} pts · "
                            f"{_up4.series.date_range[0]} → {_up4.series.date_range[1]}"
                        )
                    with _ub4:
                        st.selectbox(
                            "Asignar source:",
                            ["🚫 Descartar"] + _avail_src_names4,
                            key=f"manual_match_plat_{_ip4}_{_up4.series.name}",
                        )
            elif _auto_mr.matches:
                st.success("Todas las series platform fueron matcheadas automáticamente.")

            # ── Unmatched platform (from auto) + additional sources ───────────
            _addl_sources = st.session_state.get("additional_sources", [])

            if _unmatched_plat4:

                # Summaries of already-matched additional sources
                _matched_addl = [a for a in _addl_sources if a.get("matched")]
                if _matched_addl:
                    st.markdown(f"**Sources adicionales ya procesados ({len(_matched_addl)}):**")
                    for _ma in _matched_addl:
                        st.caption(f"✅ {_ma['filename']} — {len(_ma.get('selected_series', []))} series")

                st.markdown("---")

                # Find current (unmatched) additional source being processed
                _cur_as_idx = next(
                    (i for i, a in enumerate(_addl_sources) if not a.get("matched")), None
                )

                if _cur_as_idx is not None:
                    # ── Current additional source workflow ────────────────────
                    _cur_as = _addl_sources[_cur_as_idx]
                    _akey = f"addl_{_cur_as_idx}"
                    st.markdown(f"**📁 Source adicional {_cur_as_idx + 1}: `{_cur_as['filename']}`**")

                    if "column_classification" not in _cur_as:
                        # Classification step
                        _adf = _cur_as["df"]
                        _afmt = _cur_as["format"]
                        if _afmt == "wide_transposed":
                            _a_entity_col = st.selectbox(
                                "Columna de entidades:", list(_adf.columns), key=f"{_akey}_entity"
                            )
                            if st.button("✅ Confirmar (Wide Transpuesto)", key=f"{_akey}_conf_trans"):
                                _addl_sources[_cur_as_idx]["column_classification"] = {"entity_col": _a_entity_col}
                                st.session_state["additional_sources"] = _addl_sources
                                st.rerun()
                        else:
                            _a_auto_cls = _auto_classify_columns(_adf, _afmt)
                            _a_cls_opts = (
                                ["Fecha", "Año", "Mes", "Trimestre", "Dimensión", "Valor", "Ignorar"] if _afmt == "long"
                                else ["Fecha", "Año", "Mes", "Trimestre", "Serie", "Ignorar"]
                            )
                            if _afmt == "long":
                                st.caption("**Fecha** · **Año+Mes/Trimestre** · **Dimensión** · **Valor** · **Ignorar**")
                            else:
                                st.caption("**Fecha** · **Año+Mes/Trimestre** · **Serie** (columna numérica) · **Ignorar**")
                            _a_cls_rows = []
                            for _col in _adf.columns:
                                _cur_t = _a_auto_cls.get(_col, _a_cls_opts[-1])
                                if _cur_t not in _a_cls_opts:
                                    _cur_t = _a_cls_opts[-1]
                                _a_cls_rows.append({
                                    "Columna": _col,
                                    "Tipo detectado": _detect_column_type(_adf, _col),
                                    "Tu clasificación": _cur_t,
                                })
                            _a_cls_edited = st.data_editor(
                                pd.DataFrame(_a_cls_rows),
                                column_config={
                                    "Columna": st.column_config.TextColumn(disabled=True),
                                    "Tipo detectado": st.column_config.TextColumn(disabled=True),
                                    "Tu clasificación": st.column_config.SelectboxColumn(
                                        options=_a_cls_opts, required=True,
                                    ),
                                },
                                hide_index=True, use_container_width=True,
                                key=f"{_akey}_cls_editor",
                            )
                            # Date composition panel for additional source
                            _aa_cls_dict = dict(zip(_a_cls_edited["Columna"], _a_cls_edited["Tu clasificación"]))
                            _aa_year_col = next((c for c, t in _aa_cls_dict.items() if t == "Año"), None)
                            _aa_period_col = next((c for c, t in _aa_cls_dict.items() if t in ("Mes", "Trimestre")), None)
                            _aa_period_kind = next((t for t in _aa_cls_dict.values() if t in ("Mes", "Trimestre")), None)
                            _aa_compose_valid = True
                            _aa_period_type = None
                            if _aa_year_col and _aa_period_col:
                                _aa_period_type, _aa_compose_valid = _show_date_compose_panel(
                                    _adf, _aa_year_col, _aa_period_col, _aa_period_kind,
                                    f"{_akey}_compose"
                                )

                            _a_n_fecha = (_a_cls_edited["Tu clasificación"] == "Fecha").sum()
                            _a_n_year = (_a_cls_edited["Tu clasificación"] == "Año").sum()
                            _a_n_period = (_a_cls_edited["Tu clasificación"].isin(["Mes", "Trimestre"])).sum()
                            _a_has_date = _a_n_fecha >= 1 or (_a_n_year >= 1 and _a_n_period >= 1)
                            _a_val_label = "Valor" if _afmt == "long" else "Serie"
                            _a_n_val = (_a_cls_edited["Tu clasificación"] == _a_val_label).sum()
                            _a_can_conf = _a_has_date and _a_n_val > 0 and _aa_compose_valid
                            if not _a_can_conf:
                                st.error("Necesitás al menos 1 Fecha (o Año+Mes/Trimestre) y 1 columna Valor/Serie.")
                            if st.button("✅ Confirmar clasificación", key=f"{_akey}_conf_cls",
                                         disabled=not _a_can_conf):
                                _addl_sources[_cur_as_idx]["column_classification"] = dict(
                                    zip(_a_cls_edited["Columna"], _a_cls_edited["Tu clasificación"])
                                )
                                _addl_sources[_cur_as_idx]["period_type"] = _aa_period_type
                                st.session_state["additional_sources"] = _addl_sources
                                st.rerun()

                    elif "extraction" not in _cur_as:
                        # Extraction step
                        st.info("Clasificación confirmada. Hacé clic en 'Extraer series'.")
                        if st.button("⚙️ Extraer series", key=f"{_akey}_extract", type="primary"):
                            _adf = _cur_as["df"]
                            _afmt = _cur_as["format"]
                            _acls = _cur_as["column_classification"]
                            try:
                                if _afmt == "long":
                                    _a_df_composed, _a_date = _apply_date_composition(
                                        _adf, _acls, _cur_as.get("period_type")
                                    )
                                    _a_dims = [c for c, t in _acls.items() if t == "Dimensión"]
                                    _a_val = next((c for c, t in _acls.items() if t == "Valor"), None)
                                    _a_ext_r = extract_series_long(_a_df_composed, _a_date, _a_dims, _a_val)
                                elif _afmt == "wide":
                                    _a_df_composed, _a_date = _apply_date_composition(
                                        _adf, _acls, _cur_as.get("period_type")
                                    )
                                    _a_vals = [c for c, t in _acls.items() if t == "Serie"]
                                    _a_ext_r = extract_series_wide(_a_df_composed, _a_date, _a_vals)
                                else:
                                    _a_ent = _acls.get("entity_col")
                                    _a_ext_r = extract_series_wide_transposed(_adf, _a_ent)
                                _addl_sources[_cur_as_idx]["extraction"] = _a_ext_r
                                st.session_state["additional_sources"] = _addl_sources
                                st.rerun()
                            except Exception as _ae:
                                st.error(f"Error en extracción: {_ae}")

                    elif "selected_series" not in _cur_as:
                        # Series selection step
                        _a_ext_r = _cur_as["extraction"]
                        st.markdown(f"Series extraídas: **{_a_ext_r.total_series}**")
                        _a_sel_checks: dict[str, bool] = {}
                        for _i, _s in enumerate(_a_ext_r.series):
                            _areason = _discard_reason(_s)
                            _adefault = _areason is None
                            _albl = f"{_s.name} ({_s.row_count} pts)"
                            if _areason:
                                _albl += f" ⚠️ {_areason}"
                            _ak = f"{_akey}_sel_{_i}_{_s.name}"
                            _a_sel_checks[_s.name] = st.checkbox(
                                _albl, value=st.session_state.get(_ak, _adefault), key=_ak
                            )
                        _a_selected = [n for n, v in _a_sel_checks.items() if v]
                        if not _a_selected:
                            st.warning("Seleccioná al menos 1 serie.")
                        if st.button("✅ Confirmar series adicionales", key=f"{_akey}_conf_sel",
                                     disabled=not bool(_a_selected)):
                            _addl_sources[_cur_as_idx]["selected_series"] = _a_selected
                            st.session_state["additional_sources"] = _addl_sources
                            st.rerun()

                    else:
                        # Matching step: auto-match against pending platform series
                        _a_ext_r = _cur_as["extraction"]
                        _a_selected = _cur_as["selected_series"]
                        _a_ext_filtered = dataclasses.replace(
                            _a_ext_r,
                            series=[s for s in _a_ext_r.series if s.name in _a_selected],
                            total_series=len(_a_selected),
                        )
                        _pending_names = [u.series.name for u in _unmatched_plat4]
                        _pending_plat_ext = dataclasses.replace(
                            _plat_ext4,
                            series=[s for s in _plat_ext4.series if s.name in _pending_names],
                            total_series=len(_pending_names),
                        )
                        _addl_mr_key = f"_addl_auto_mr_{_cur_as_idx}"
                        if _addl_mr_key not in st.session_state:
                            st.session_state[_addl_mr_key] = auto_match_series(
                                _a_ext_filtered, _pending_plat_ext
                            )
                        _addl_auto_mr = st.session_state[_addl_mr_key]

                        if _addl_auto_mr.matches:
                            st.markdown(f"**Auto-matches encontrados: {_addl_auto_mr.matched_count}**")
                            _am_rows = [
                                {
                                    "": "✅" if _am.confidence >= 0.9 else "⚠️",
                                    "Serie Source Adicional": _am.source_series.name,
                                    "Serie Platform": _am.platform_series.name,
                                    "Confianza": f"{_am.confidence:.0%}",
                                }
                                for _am in _addl_auto_mr.matches
                            ]
                            st.dataframe(pd.DataFrame(_am_rows), hide_index=True,
                                         use_container_width=True)
                        else:
                            st.info("No se encontraron matches automáticos para este source adicional.")

                        _addl_unmatched_plat = _addl_auto_mr.unmatched_platform
                        _addl_avail_src = _addl_auto_mr.unmatched_source
                        _addl_src_opts = ["🚫 Descartar"] + [u.series.name for u in _addl_avail_src]

                        if _addl_unmatched_plat:
                            st.markdown(f"**Sin match ({len(_addl_unmatched_plat)} series platform):**")
                            for _iap, _aup in enumerate(_addl_unmatched_plat):
                                _aa, _ab = st.columns([2, 3])
                                with _aa:
                                    st.caption(f"⚠️ **{_aup.series.name}**")
                                with _ab:
                                    st.selectbox(
                                        "Asignar source adicional:", _addl_src_opts,
                                        key=f"addl_manual_{_cur_as_idx}_{_iap}_{_aup.series.name}",
                                    )

                        if st.button("🔗 Aplicar matches del source adicional",
                                     key=f"{_akey}_apply", type="primary"):
                            _addl_manual_pairs = []
                            for _iap, _aup in enumerate(_addl_unmatched_plat):
                                _wk = f"addl_manual_{_cur_as_idx}_{_iap}_{_aup.series.name}"
                                _s_sel = st.session_state.get(_wk, "🚫 Descartar")
                                if _s_sel != "🚫 Descartar":
                                    _addl_manual_pairs.append((_s_sel, _aup.series.name))
                            _addl_final_mr = apply_manual_matches(_addl_auto_mr, _addl_manual_pairs)
                            _merged_mr = _merge_matching_results(_auto_mr, _addl_final_mr)
                            st.session_state["_auto_matching_result"] = _merged_mr
                            # Update file map
                            _fmap = dict(st.session_state.get("source_series_file_map", {}))
                            for _sn in _a_selected:
                                _fmap[_sn] = _cur_as["filename"]
                            st.session_state["source_series_file_map"] = _fmap
                            _addl_sources[_cur_as_idx]["matched"] = True
                            st.session_state["additional_sources"] = _addl_sources
                            for _k in ["matching_result", "matching_confirmed", "compare_result",
                                       _addl_mr_key]:
                                st.session_state.pop(_k, None)
                            st.rerun()

                else:
                    # No current unmatched source → show upload form for a new one
                    _new_idx = len(_addl_sources)
                    st.markdown("**📁 Agregar otro source para las series restantes**")
                    _a_fmt_radio = st.radio(
                        "Formato:", ["Long", "Wide", "Wide Transpuesto"],
                        horizontal=True, key=f"addl_fmt_{_new_idx}",
                    )
                    _a_fmt_map = {"Long": "long", "Wide": "wide", "Wide Transpuesto": "wide_transposed"}
                    _a_fmt_code = _a_fmt_map[_a_fmt_radio]
                    _addl_file = st.file_uploader(
                        "CSV o XLSX / XLS / XLSB", type=["csv", "xlsx", "xls", "xlsb"],
                        key=f"addl_file_{_new_idx}", label_visibility="collapsed",
                    )
                    if _addl_file:
                        _addl_df = load_file(_addl_file, f"addl_{_new_idx}")
                        if _addl_df is not None:
                            st.session_state["additional_sources"] = _addl_sources + [{
                                "filename": _addl_file.name,
                                "df": _addl_df,
                                "format": _a_fmt_code,
                            }]
                            st.rerun()

            st.markdown("---")

            # ── Confirm matching button ───────────────────────────────────────
            _n_kept4 = sum(
                1 for _i4m in range(len(_auto_mr.matches))
                if _edited_matches4.iloc[_i4m]["Incluir"]
            )
            _n_manual_assigned4 = sum(
                1 for _ip4, _up4 in enumerate(_all_unmatched_plat4)
                if st.session_state.get(
                    f"manual_match_plat_{_ip4}_{_up4.series.name}", "🚫 Descartar"
                ) != "🚫 Descartar"
            )
            _n_still_unmatched4 = len(_all_unmatched_plat4) - _n_manual_assigned4
            _has_any_match4 = _n_kept4 > 0 or _n_manual_assigned4 > 0
            _pending_discarded = st.session_state.get("platform_pending_discarded", False)
            _can_confirm_matching = _has_any_match4 and (
                _n_still_unmatched4 == 0 or _pending_discarded
            )

            if _n_still_unmatched4 > 0 and not _pending_discarded:
                st.warning(
                    f"⚠️ Quedan **{_n_still_unmatched4}** series Platform sin asignar. "
                    "Podés agregar otro source arriba o descartarlas para continuar."
                )
                if st.button("🚫 Descartar todas las pendientes", key="discard_pending"):
                    st.session_state["platform_pending_discarded"] = True
                    st.rerun()

            _btn4a, _btn4b = st.columns(2)
            with _btn4a:
                if st.button(
                    "✅ Confirmar matching", key="confirm_matching",
                    type="primary", disabled=not _can_confirm_matching,
                ):
                    # Kept auto-matches
                    _kept_matches4 = [
                        _auto_mr.matches[_i4m]
                        for _i4m in range(len(_auto_mr.matches))
                        if _edited_matches4.iloc[_i4m]["Incluir"]
                    ]
                    # Manual matches from unmatched platform selectboxes
                    _manual_matches4: list[SeriesMatch] = []
                    for _ip4, _up4 in enumerate(_all_unmatched_plat4):
                        _wk4 = f"manual_match_plat_{_ip4}_{_up4.series.name}"
                        _sel4 = st.session_state.get(_wk4, "🚫 Descartar")
                        if _sel4 != "🚫 Descartar" and _sel4 in _avail_src_objs4:
                            _manual_matches4.append(SeriesMatch(
                                source_series=_avail_src_objs4[_sel4],
                                platform_series=_up4.series,
                                match_type="manual",
                                confidence=1.0,
                                similarity_details=f"Manual: '{_sel4}' → '{_up4.series.name}'",
                            ))
                    _all_final4 = _kept_matches4 + _manual_matches4
                    _used_src_f4 = {m.source_series.name for m in _all_final4}
                    _used_plat_f4 = {m.platform_series.name for m in _all_final4}
                    _tc4 = {"exact":0,"normalized":0,"substring":0,
                             "correlation":0,"name_only":0,"manual":0}
                    for _mf in _all_final4:
                        _tc4[_mf.match_type] = _tc4.get(_mf.match_type, 0) + 1
                    _ns4 = len(_src_ext4_filtered.series)
                    _np4 = len(_plat_ext4_filtered.series)
                    _nm4 = len(_all_final4)
                    _final_mr4 = MatchingResult(
                        matches=_all_final4,
                        unmatched_source=[
                            UnmatchedSeries(s, "source", "No matching platform series found")
                            for s in _src_ext4_filtered.series
                            if s.name not in _used_src_f4
                        ],
                        unmatched_platform=[
                            UnmatchedSeries(s, "platform", "No matching source series found")
                            for s in _plat_ext4_filtered.series
                            if s.name not in _used_plat_f4
                        ],
                        total_source=_ns4,
                        total_platform=_np4,
                        matched_count=_nm4,
                        match_rate_source=_nm4/_ns4 if _ns4 > 0 else 0.0,
                        match_rate_platform=_nm4/_np4 if _np4 > 0 else 0.0,
                        matching_log=list(_auto_mr.matching_log),
                        exact_count=_tc4["exact"],
                        normalized_count=_tc4["normalized"],
                        substring_count=_tc4["substring"],
                        correlation_count=_tc4["correlation"],
                        name_only_count=_tc4["name_only"],
                        manual_count=_tc4["manual"],
                    )
                    st.session_state["matching_result"] = _final_mr4
                    st.session_state["matching_confirmed"] = True
                    st.session_state.pop("compare_result", None)
                    st.rerun()
            with _btn4b:
                if st.button("🔄 Re-run auto-match", key="rerun_automatch"):
                    for _k4 in ["_auto_matching_result", "matching_result",
                                 "matching_confirmed", "compare_result",
                                 "additional_sources", "platform_pending_discarded",
                                 "source_series_file_map"]:
                        st.session_state.pop(_k4, None)
                    st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 5: Comparar
    # ─────────────────────────────────────────────────────────────────────────
    if _s4_done:
        _mr5 = st.session_state["matching_result"]

        with st.expander(
            f"{_sicon(_s5_done, not _s5_done)} PASO 5: Comparar — "
            f"{_mr5.matched_count} pares matcheados",
            expanded=not _s5_done,
        ):
            st.markdown(f"**Listo para comparar {_mr5.matched_count} pares de series.**")

            _tol_rel5 = st.slider(
                "Tolerancia relativa %", 0.0, 5.0, 0.1, step=0.05, key="cmp_tol_rel5",
            )
            _tol_abs5 = st.slider(
                "Tolerancia absoluta", 0.0, 10.0, 0.01, step=0.001,
                format="%.3f", key="cmp_tol_abs5",
            )

            if st.button(
                "🚀 Comparar series matcheadas", type="primary", key="run_series_compare",
            ):
                _cmp5_config = {
                    "tolerance": _tol_rel5 / 100,
                    "source_name": "Source",
                    "platform_name": "Platform",
                }
                with st.spinner("Comparando series..."):
                    try:
                        _diff5 = compare_matched_series(_mr5, _cmp5_config)
                        st.session_state["compare_result"] = _diff5
                        st.session_state["compare_config"] = _cmp5_config
                        st.session_state["compare_ts"] = datetime.now(timezone.utc).isoformat()
                        _s5_done = True
                        st.rerun()
                    except Exception as _e5:
                        st.error(f"Error durante la comparación: {_e5}")
                        st.exception(_e5)

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 6: Resultados
    # ─────────────────────────────────────────────────────────────────────────
    if _s5_done:
        _diff6 = st.session_state["compare_result"]
        _tol6 = _diff6.config_used.get("tolerance", 0.01)
        _ss6 = _diff6.series_summary or {}
        _ps6 = _diff6.per_series_results or []
        _pct6 = _diff6.match_pct * 100

        with st.expander("✅ PASO 6: Resultados", expanded=True):
            # Header
            _hc6a, _hc6b = st.columns([3, 1])
            with _hc6a:
                _plat_label6 = _diff6.platform_name
                if st.session_state.get("platform_source_type") == "alphacast":
                    _ac_id6 = st.session_state.get("alphacast_dataset_id")
                    _ac_nm6 = st.session_state.get("alphacast_dataset_name")
                    if _ac_id6 and _ac_nm6:
                        _plat_label6 = f"Alphacast #{_ac_id6} — {_ac_nm6}"
                st.markdown(f"### {_diff6.source_name} vs {_plat_label6}")
            with _hc6b:
                if _pct6 >= 95:
                    st.success(f"{_pct6:.1f}% match")
                elif _pct6 >= 80:
                    st.warning(f"{_pct6:.1f}% match")
                else:
                    st.error(f"{_pct6:.1f}% match")

            # KPIs
            _mr6 = st.session_state.get("matching_result")
            _kp1, _kp2, _kp3, _kp4, _kp5 = st.columns(5)
            _kp1.metric("% Match Global", f"{_pct6:.1f}%")
            _kp2.metric("Series Matcheadas", _ss6.get("total_series_matched", 0))
            _kp3.metric("Series 100% Match", _ss6.get("series_perfect_match", 0))
            _kp4.metric("Series con Problemas", _ss6.get("series_with_diffs", 0))
            _kp5.metric("Max Diff %", f"{_diff6.max_rel_diff_pct:.1f}%")

            st.markdown("---")

            # Sub-tabs
            _rt_sum, _rt_missing, _rt_diffs, _rt_plots, _rt_disc, _rt_export = st.tabs([
                "📊 Resumen", "📋 Missing/Extra", "📊 Diffs por Serie",
                "📈 Plots", "📋 Descartadas", "📄 Export",
            ])

            # ── Resumen ───────────────────────────────────────────────────────
            with _rt_sum:
                st.progress(min(_diff6.match_pct, 1.0))
                st.markdown("")

                if _ps6:
                    st.markdown("**Resumen por serie:**")
                    _file_map6 = st.session_state.get("source_series_file_map", {})
                    _ps_rows6 = []
                    for _r6 in _ps6:
                        _mp6 = _r6.get("match_pct")
                        _sd6 = _r6.get("scale_detection") or {}
                        _scale_flag = ""
                        if _sd6.get("confidence") in ("high", "medium") and _sd6.get("scale_factor", 1.0) != 1.0:
                            _scale_flag = " ⚠️ escala"
                        _ps_rows6.append({
                            "Serie Source": _r6["source_name"],
                            "Archivo Source": _file_map6.get(_r6["source_name"], "—"),
                            "Serie Platform": _r6["platform_name"],
                            "Match %": round(_mp6 * 100, 1) if _mp6 is not None else None,
                            "Puntos": _r6.get("comparable_rows", 0),
                            "Max Diff": round(_r6.get("max_abs_diff") or 0.0, 4),
                            "Tipo match": _r6.get("match_type", "?") + _scale_flag,
                        })
                    _ps_df6 = pd.DataFrame(_ps_rows6)
                    st.dataframe(
                        _ps_df6, hide_index=True, use_container_width=True,
                        column_config={
                            "Match %": st.column_config.NumberColumn("Match %", format="%.1f%%"),
                        },
                    )

                    # Scale warnings
                    _scale_warns6 = [
                        _r6 for _r6 in _ps6
                        if (_r6.get("scale_detection") or {}).get("confidence") in ("high", "medium")
                        and (_r6.get("scale_detection") or {}).get("scale_factor", 1.0) != 1.0
                    ]
                    if _scale_warns6:
                        st.warning(
                            f"**Diferencia de escala en {len(_scale_warns6)} serie(s):**\n\n" +
                            "\n".join(
                                f"- **{_r6['source_name']}**: {(_r6['scale_detection'] or {}).get('label', '')} "
                                f"(confianza: {(_r6['scale_detection'] or {}).get('confidence', '?')})"
                                for _r6 in _scale_warns6
                            ),
                        )

                if _ss6:
                    with st.expander("ℹ️ Detalles del matching"):
                        _ms6 = {
                            k: v for k, v in _ss6.items()
                            if k not in ("matching_summary",)
                        }
                        st.json(_ms6)

            # ── Missing / Extra ───────────────────────────────────────────────
            with _rt_missing:
                _um_src6 = _ss6.get("unmatched_source_series", [])
                _um_plat6 = _ss6.get("unmatched_platform_series", [])
                _mc1_6, _mc2_6 = st.columns(2)
                with _mc1_6:
                    st.markdown(f"**Series Source sin match ({len(_um_src6)})**")
                    if _um_src6:
                        st.dataframe(
                            pd.DataFrame({"Serie Source": _um_src6}),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.success("Todas matcheadas")
                with _mc2_6:
                    st.markdown(f"**Series Platform sin match ({len(_um_plat6)})**")
                    if _um_plat6:
                        st.dataframe(
                            pd.DataFrame({"Serie Platform": _um_plat6}),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.success("Todas matcheadas")

            # ── Diffs por Serie ───────────────────────────────────────────────
            with _rt_diffs:
                if _diff6.all_diffs is None or len(_diff6.all_diffs) == 0:
                    st.success("Todos los valores coinciden dentro de la tolerancia.")
                else:
                    _mg6 = _diff6.all_diffs.copy()
                    _series_keys6 = sorted(_mg6["Key"].unique().tolist())

                    _fd1_6, _fd2_6 = st.columns(2)
                    with _fd1_6:
                        _sel_series6 = st.selectbox(
                            "Serie:", _series_keys6, key="diffs_series_sel6"
                        )
                    with _fd2_6:
                        _min_diff6 = st.slider(
                            "Mostrar diffs mayores a %",
                            0.0, 100.0, 0.0, step=0.5, key="diffs_min_pct6",
                        )

                    _show6 = _mg6[_mg6["Key"] == _sel_series6].copy()
                    if _min_diff6 > 0:
                        _show6 = _show6[_show6["rel_diff_pct"] >= _min_diff6]
                    _show6 = _show6.sort_values("abs_diff", ascending=False)

                    if "Date_str" not in _show6.columns:
                        _show6["Date_str"] = _show6["Date"].dt.strftime("%Y-%m-%d")

                    _disp6 = _show6[["Date_str", "Value_src", "Value_plat",
                                     "abs_diff", "rel_diff_pct"]].copy()
                    _disp6.columns = ["Fecha", "Source", "Platform", "Dif Abs", "Dif Rel %"]
                    _disp6["Dif Abs"] = _disp6["Dif Abs"].round(4)
                    _disp6["Dif Rel %"] = _disp6["Dif Rel %"].round(2)

                    st.markdown(f"**{len(_show6)} diffs para '{_sel_series6}'**")
                    st.dataframe(
                        _disp6, hide_index=True, use_container_width=True,
                        column_config={
                            "Dif Rel %": st.column_config.NumberColumn(format="%.2f%%"),
                        },
                    )

            # ── Plots ─────────────────────────────────────────────────────────
            with _rt_plots:
                try:
                    import plotly.graph_objects as go
                    _plotly_ok6 = True
                except ImportError:
                    _plotly_ok6 = False

                if not _plotly_ok6:
                    st.warning("Instalá plotly: `pip install plotly`")
                else:
                    _mr6_plot = st.session_state.get("matching_result")
                    if not _mr6_plot or not _mr6_plot.matches:
                        st.info("Sin datos de series para graficar.")
                    else:
                        _plot_opts6 = [
                            f"{_m6p.source_series.name} → {_m6p.platform_series.name}"
                            for _m6p in _mr6_plot.matches
                        ]
                        _plot_sel6 = st.selectbox(
                            "Serie a graficar:", _plot_opts6, key="plot_series_sel6"
                        )
                        _plot_idx6 = _plot_opts6.index(_plot_sel6)
                        _match6 = _mr6_plot.matches[_plot_idx6]
                        _src_s6 = _match6.source_series
                        _plat_s6 = _match6.platform_series

                        _src_pf = pd.DataFrame({
                            "Date": pd.to_datetime(_src_s6.dates.values),
                            "Value": _src_s6.values.values,
                        }).dropna()
                        _plat_pf = pd.DataFrame({
                            "Date": pd.to_datetime(_plat_s6.dates.values),
                            "Value": _plat_s6.values.values,
                        }).dropna()
                        _src_pf["Date_str"] = _src_pf["Date"].dt.strftime("%Y-%m-%d")
                        _plat_pf["Date_str"] = _plat_pf["Date"].dt.strftime("%Y-%m-%d")

                        _pfig6 = go.Figure()
                        _pfig6.add_trace(go.Scatter(
                            x=_src_pf["Date_str"], y=_src_pf["Value"],
                            name="Source", line=dict(color="#3b82f6", width=2),
                            hovertemplate="<b>%{x}</b><br>Source: %{y:.4f}<extra></extra>",
                        ))
                        _pfig6.add_trace(go.Scatter(
                            x=_plat_pf["Date_str"], y=_plat_pf["Value"],
                            name="Platform", line=dict(color="#f59e0b", width=2, dash="dash"),
                            hovertemplate="<b>%{x}</b><br>Platform: %{y:.4f}<extra></extra>",
                        ))

                        # Big diff markers
                        if _diff6.all_diffs is not None:
                            _sd6 = _diff6.all_diffs[
                                _diff6.all_diffs["Key"] == _src_s6.name
                            ].copy()
                            _big6 = _sd6[_sd6["rel_diff_pct"] > 5.0]
                            if len(_big6) > 0:
                                if "Date_str" not in _big6.columns:
                                    _big6 = _big6.copy()
                                    _big6["Date_str"] = _big6["Date"].dt.strftime("%Y-%m-%d")
                                _pfig6.add_trace(go.Scatter(
                                    x=_big6["Date_str"], y=_big6["Value_src"],
                                    mode="markers", name="Diff >5%",
                                    marker=dict(color="#ef4444", size=8, symbol="x"),
                                ))

                        _pfig6.update_layout(
                            title=f"{_src_s6.name} vs {_plat_s6.name}",
                            xaxis_title="Fecha", yaxis_title="Valor",
                            legend=dict(orientation="h", yanchor="bottom",
                                        y=1.02, xanchor="right", x=1),
                            height=420, plot_bgcolor="white", paper_bgcolor="white",
                            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                        )
                        st.plotly_chart(_pfig6, use_container_width=True)

                        # Diff bar chart for this series
                        if _diff6.all_diffs is not None:
                            _bar6 = _diff6.all_diffs[
                                _diff6.all_diffs["Key"] == _src_s6.name
                            ].sort_values("Date").copy()
                            if len(_bar6) > 0:
                                if "Date_str" not in _bar6.columns:
                                    _bar6["Date_str"] = _bar6["Date"].dt.strftime("%Y-%m-%d")
                                _colors6 = [
                                    "#ef4444" if not m else "#22c55e"
                                    for m in _bar6.get("matched", [True] * len(_bar6))
                                ]
                                _bfig6 = go.Figure()
                                _bfig6.add_trace(go.Bar(
                                    x=_bar6["Date_str"], y=_bar6["rel_diff_pct"],
                                    name="Dif Rel %", marker_color=_colors6,
                                ))
                                _bfig6.add_hline(
                                    y=_tol6 * 100, line_dash="dash", line_color="#6b7280",
                                    annotation_text=f"tolerancia {_tol6 * 100:.1f}%",
                                )
                                _bfig6.update_layout(
                                    title=f"Diferencia relativa % — {_src_s6.name}",
                                    height=280, plot_bgcolor="white", paper_bgcolor="white",
                                )
                                st.plotly_chart(_bfig6, use_container_width=True)

            # ── Descartadas ───────────────────────────────────────────────────
            with _rt_disc:
                _mr6d = st.session_state.get("matching_result")
                if _mr6d:
                    _um6d = _mr6d.unmatched_source
                    if _um6d:
                        st.markdown(f"**{len(_um6d)} series source descartadas o sin match:**")
                        st.dataframe(
                            pd.DataFrame([
                                {"Serie": u.series.name,
                                 "Puntos": u.series.row_count,
                                 "Motivo": u.reason}
                                for u in _um6d
                            ]),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.success("No hay series source descartadas.")

                    _um6dp = _mr6d.unmatched_platform
                    if _um6dp:
                        st.markdown(f"**{len(_um6dp)} series platform sin asignar:**")
                        st.dataframe(
                            pd.DataFrame([
                                {"Serie Platform": u.series.name, "Puntos": u.series.row_count}
                                for u in _um6dp
                            ]),
                            hide_index=True, use_container_width=True,
                        )

            # ── Export ────────────────────────────────────────────────────────
            with _rt_export:
                _d_exp6 = _diff6.to_serializable_dict()

                _ec1, _ec2, _ec3 = st.columns(3)
                with _ec1:
                    if st.button("💾 Export JSON", key="export_json_s6"):
                        _rdir6 = Path(__file__).parent / "reports"
                        _rdir6.mkdir(exist_ok=True)
                        _ts6 = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        _jp6 = _rdir6 / f"compare_series_{_ts6}.json"
                        with open(_jp6, "w", encoding="utf-8") as _jf6:
                            json.dump(_d_exp6, _jf6, indent=2, ensure_ascii=False, default=str)
                        st.success(f"Guardado: `{_jp6.name}`")

                with _ec2:
                    if st.button("📄 Export HTML", key="export_html_s6"):
                        _rdir6b = Path(__file__).parent / "reports"
                        _rdir6b.mkdir(exist_ok=True)
                        _ts6b = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        _hp6 = _rdir6b / f"compare_series_{_ts6b}.html"
                        with open(_hp6, "w", encoding="utf-8") as _hf6:
                            _hf6.write(_build_compare_html(_diff6))
                        st.success(f"Guardado: `{_hp6.name}`")

                with _ec3:
                    if st.button("💾 Config matching YAML", key="export_matching_yaml_s6"):
                        _mr6e = st.session_state.get("matching_result")
                        if _mr6e:
                            _ms6e = get_matching_summary(_mr6e)
                            _cdir6 = Path(__file__).parent / "configs"
                            _cdir6.mkdir(exist_ok=True)
                            _ts6e = datetime.now().strftime("%Y%m%d_%H%M%S")
                            _cp6 = _cdir6 / f"matching_{_ts6e}.yaml"
                            _my6 = {
                                "source_format": st.session_state.get("source_format"),
                                "tolerance": _diff6.config_used.get("tolerance", 0.01),
                                "matches": [
                                    {"source": _m6e["source"],
                                     "platform": _m6e["platform"],
                                     "match_type": _m6e["match_type"]}
                                    for _m6e in _ms6e["matches"]
                                ],
                            }
                            with open(_cp6, "w", encoding="utf-8") as _cyf6:
                                yaml.dump(_my6, _cyf6, allow_unicode=True,
                                          default_flow_style=False)
                            st.success(f"Guardado: `configs/{_cp6.name}`")

                st.markdown("---")
                _json_str6 = json.dumps(_d_exp6, indent=2, ensure_ascii=False, default=str)
                st.download_button(
                    "⬇ Descargar JSON",
                    data=_json_str6.encode("utf-8"),
                    file_name=f"compare_series_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json",
                    mime="application/json",
                    key="dl_json_s6",
                )
                st.json(_d_exp6, expanded=False)

        # Reset button
        st.markdown("---")
    if st.button("🔁 Reiniciar comparación", key="reset_compare"):
            for _kr in ["source_df", "platform_df", "source_format", "platform_format",
                        "classification_confirmed", "source_column_classification",
                        "platform_classification_confirmed", "platform_column_classification",
                        "source_extraction", "platform_extraction",
                        "_auto_cls", "_auto_cls_key", "_auto_cls_plat", "_auto_cls_plat_key",
                        "series_selection_confirmed", "source_selected_series", "platform_selected_series",
                        "_sel_plat_id", "_sel_plat_ver", "_sel_plat_defaults",
                        "matching_result", "matching_confirmed",
                        "_auto_matching_result", "compare_result", "compare_config",
                        "compare_ts", "_src_file_key", "_plat_file_key", "_src_filename",
                        "_last_source_format", "_last_platform_format",
                        "additional_sources", "platform_pending_discarded",
                        "source_series_file_map",
                        "source_period_type", "platform_period_type",
                        "platform_source_type", "_last_plat_origen",
                        "alphacast_dataset_id", "alphacast_dataset_name"]:
                st.session_state.pop(_kr, None)
            st.rerun()

    elif not _s1_done:
        st.info(
            "**Para comenzar:** Cargá los datasets Source y Platform en el **Paso 1** de arriba.",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — AUDIT
# ═════════════════════════════════════════════════════════════════════════════
with tab_audit:
    _audit_mode_tab = st.radio(
        "Auditar:",
        ["Source", "Platform", "Subir archivo"],
        horizontal=True,
        key="audit_tab_mode",
        help="Source/Platform usan el dataset cargado en Compare",
    )

    if "audit_result" not in st.session_state:
        st.markdown("## 🔍 Auditoría de Dataset")
        if _audit_mode_tab in ("Source", "Platform"):
            _label = "Source" if _audit_mode_tab == "Source" else "Platform"
            st.info(
                f"Configurá la auditoría en el panel izquierdo "
                f"(modo **{_label}** seleccionado) y hacé clic en **▶ Run Audit**.",
            )
        else:
            st.info("Subí un dataset en el panel izquierdo y hacé clic en **▶ Run Audit**.")
    else:
        report = st.session_state["audit_result"]

        col_title, col_badge = st.columns([3, 1])
        with col_title:
            st.markdown(f"## Auditoría: `{report.dataset_name}`")
        with col_badge:
            if report.blockers > 0:
                st.error("FAILED")
            elif report.warnings > 0:
                st.warning("WARNINGS")
            else:
                st.success("PASSED")
        st.caption(report.summary)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Checks", report.total_checks)
        m2.metric("Passed", report.passed)
        m3.metric("Blockers", report.blockers,
                  delta=None if report.blockers == 0 else f"-{report.blockers}",
                  delta_color="inverse")
        m4.metric("Warnings", report.warnings)
        m5.metric("Info", report.infos)
        st.markdown("---")

        a_sum, a_detail, a_json = st.tabs(["Resumen", "Detalle", "Raw JSON"])

        with a_sum:
            _apct = report.passed / report.total_checks if report.total_checks > 0 else 0
            st.markdown(f"**Checks pasados: {report.passed} / {report.total_checks}**")
            st.progress(_apct)
            st.markdown("")

            failed_checks = [c for c in report.checks if not c.passed]
            if not failed_checks:
                st.success("Todos los checks pasaron.")
            else:
                sev_order = {"BLOCKER": 0, "WARNING": 1, "INFO": 2}
                failed_checks = sorted(failed_checks, key=lambda c: sev_order.get(c.severity, 9))
                for sev_label, sev_key in [("BLOCKERS", "BLOCKER"), ("WARNINGS", "WARNING"), ("INFO", "INFO")]:
                    group = [c for c in failed_checks if c.severity == sev_key]
                    if not group:
                        continue
                    st.markdown(f"### {sev_label} ({len(group)})")
                    for chk in group:
                        css_cls = card_class(chk.severity)
                        st.markdown(
                            f"""<div class="check-card {css_cls}">
                            <span class="{severity_badge_class(chk.severity)}">{chk.severity}</span>
                            &nbsp;<strong>{chk.check_id}</strong> — {chk.check_name.replace('_', ' ').title()}<br>
                            <span style="font-size:0.9rem">{chk.message}</span>
                            {"<br><span style='font-size:0.82rem;color:#6b7280'>💡 " + chk.recommendation + "</span>" if chk.recommendation else ""}
                            </div>""",
                            unsafe_allow_html=True,
                        )

            passed_checks = [c for c in report.checks if c.passed]
            with st.expander(f"Checks aprobados ({len(passed_checks)})"):
                for chk in passed_checks:
                    st.markdown(
                        f"<span class='badge-ok'>OK</span> &nbsp;<strong>{chk.check_id}</strong> {chk.message}",
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

        with a_detail:
            rows = []
            for chk in report.checks:
                rows.append({
                    "Severity": chk.severity,
                    "ID": chk.check_id,
                    "Name": chk.check_name.replace("_", " ").title(),
                    "Category": chk.category.title(),
                    "Status": "PASS" if chk.passed else "FAIL",
                    "Message": chk.message,
                })
            df_checks = pd.DataFrame(rows)
            f1, f2 = st.columns(2)
            with f1:
                sev_filter = st.multiselect("Severidad", ["BLOCKER", "WARNING", "INFO"],
                                            default=["BLOCKER", "WARNING", "INFO"])
            with f2:
                cat_filter = st.multiselect("Categoria", sorted(df_checks["Category"].unique()),
                                            default=sorted(df_checks["Category"].unique()))
            mask = df_checks["Severity"].isin(sev_filter) & df_checks["Category"].isin(cat_filter)
            st.dataframe(df_checks[mask], use_container_width=True, hide_index=True,
                         column_config={
                             "Severity": st.column_config.TextColumn(width="small"),
                             "ID": st.column_config.TextColumn(width="small"),
                             "Status": st.column_config.TextColumn(width="small"),
                             "Message": st.column_config.TextColumn(width="large"),
                         })

        with a_json:
            report_dict = dataclasses.asdict(report)
            json_str = json.dumps(report_dict, indent=2, default=str, ensure_ascii=False)
            st.json(report_dict, expanded=False)
            st.code(json_str, language="json")
            st.download_button(
                "⬇ Descargar JSON", data=json_str.encode("utf-8"),
                file_name=f"{report.dataset_name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json",
                mime="application/json",
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.markdown("## ⚙️ Configuración")
    st.markdown("**Versión:** Dataset Auditor v2.0 — Series Compare + Audit")

    _base = Path(__file__).parent
    _reports_dir = _base / "reports"
    _configs_dir = _base / "configs"

    st.markdown("---")
    st.markdown("**Carpetas**")
    _s1, _s2 = st.columns(2)
    with _s1:
        st.markdown(f"📁 **Reportes:** `{_reports_dir}`")
        if st.button("📂 Abrir carpeta reportes", key="settings_open_reports"):
            _reports_dir.mkdir(exist_ok=True)
            reveal_directory(_reports_dir)
    with _s2:
        st.markdown(f"📁 **Configs:** `{_configs_dir}`")
        if st.button("📂 Abrir carpeta configs", key="settings_open_configs"):
            _configs_dir.mkdir(exist_ok=True)
            reveal_directory(_configs_dir)

    st.markdown("---")
    st.markdown("**🔑 Alphacast API**")
    _set_ac_key = st.session_state.get("alphacast_api_key")
    _set_ac_source = st.session_state.get("alphacast_api_key_source")
    _set_show_input = st.session_state.get("_set_ac_show_input", False)

    if _set_ac_key and not _set_show_input:
        if _set_ac_source == "streamlit_secrets":
            st.success(f"API Key configurada desde Streamlit secrets ({mask_api_key(_set_ac_key)})")
            st.caption("Cargala en Streamlit Community Cloud > App settings > Secrets con `ALPHACAST_API_KEY`.")
        elif _set_ac_source == "env":
            st.success(f"API Key configurada desde variable de entorno ({mask_api_key(_set_ac_key)})")
            st.caption(f"Variable usada: `{API_KEY_ENV_VAR}`")
        else:
            st.success(f"API Key configurada ({mask_api_key(_set_ac_key)})")
            st.caption(f"Archivo local: `{API_KEY_FILE}`")
            _sc1, _sc2 = st.columns(2)
            with _sc1:
                if st.button("Cambiar key", key="settings_ac_change"):
                    st.session_state["_set_ac_show_input"] = True
                    st.rerun()
            with _sc2:
                if st.button("Borrar key", key="settings_ac_delete"):
                    delete_api_key()
                    st.session_state.pop("alphacast_api_key", None)
                    st.session_state.pop("alphacast_api_key_source", None)
                    st.session_state["_set_ac_show_input"] = False
                    st.success("Key borrada.")
                    st.rerun()
    else:
        st.warning("No configurada")
        st.caption(f"Localmente se guardara en: `{API_KEY_FILE}`")
        st.caption("Para Streamlit Community Cloud, cargala en Secrets con `ALPHACAST_API_KEY`.")
        _set_new_key = st.text_input(
            "API Key:", type="password", placeholder="ak_xxxxxxxxxxxxxxxx",
            key="settings_ac_key_input",
            help="Encontrala en alphacast.io -> Settings",
        )
        if st.button("Guardar key", key="settings_ac_save"):
            if _set_new_key.strip():
                with st.spinner("Validando..."):
                    if validate_api_key(_set_new_key.strip()):
                        save_api_key(_set_new_key.strip())
                        st.session_state["alphacast_api_key"] = _set_new_key.strip()
                        st.session_state["alphacast_api_key_source"] = "file"
                        st.session_state["_set_ac_show_input"] = False
                        st.success("Key valida y guardada.")
                        st.rerun()
                    else:
                        st.error("Key invalida. Verifica que sea correcta.")
            else:
                st.warning("Ingresa una API key.")

    st.markdown("---")
    if st.button("🗑️ Limpiar datos de sesión", key="settings_clear_session"):
        for _k in [
            "source_df", "platform_df", "source_format",
            "classification_confirmed", "source_column_classification",
            "source_extraction", "platform_extraction",
            "_auto_cls", "_auto_cls_key",
            "matching_result", "matching_confirmed",
            "_auto_matching_result",
            "compare_result", "compare_config", "compare_ts",
            "_src_file_key", "_plat_file_key", "_last_source_format",
            "audit_result", "audit_df", "audit_config", "audit_ts",
            "_detected_config", "_detected_key",
            "loaded_matching_yaml",
            "alphacast_api_key_source",
        ]:
            st.session_state.pop(_k, None)
        st.success("Sesión limpiada.")
        st.rerun()

    st.markdown("---")
    st.markdown("**ℹ️ Archivos demo disponibles en `demo/`**")
    _demo_files = [
        ("demo_series_source_wide.csv", "Dataset fuente Wide (10 series económicas, 24 meses)"),
        ("demo_series_platform_long.csv", "Dataset plataforma Long Alphacast (10 series, diffs intencionales)"),
        ("demo_source.csv", "Dataset fuente para Compare clásico"),
        ("demo_platform.csv", "Dataset plataforma para Compare clásico"),
        ("demo_data_clean.csv", "Dataset limpio para Audit"),
        ("demo_data_dirty.csv", "Dataset con errores para Audit"),
        ("demo_config.yaml", "Config YAML para Audit"),
    ]
    st.dataframe(
        pd.DataFrame([{"Archivo": f, "Uso": u} for f, u in _demo_files]),
        hide_index=True, use_container_width=True,
    )

    st.markdown("---")
    st.markdown("""
**Flujo Compare por Series:**
1. **Paso 1** — Subí Source (seleccioná formato: Long/Wide/Wide Transpuesto) y Platform
2. **Paso 2** — Clasificá las columnas del Source (Fecha/Valor/Dimensión/Serie/Ignorar)
3. **Paso 3** — Extracción automática de series individuales
4. **Paso 4** — Matching automático + resolución manual de los sin match
5. **Paso 5** — Configurá tolerancias y ejecutá la comparación
6. **Paso 6** — Resultados: KPIs, tabla por serie, plots, export
""")


# ── Footer ────────────────────────────────────────────────────────────────────
_ts_parts = []
if "compare_ts" in st.session_state:
    _ts_parts.append(f"Compare: {st.session_state['compare_ts']}")
if "audit_ts" in st.session_state:
    _ts_parts.append(f"Audit: {st.session_state['audit_ts']}")
if _ts_parts:
    st.markdown(
        f"<div class='footer'>{' &nbsp;·&nbsp; '.join(_ts_parts)}</div>",
        unsafe_allow_html=True,
    )






