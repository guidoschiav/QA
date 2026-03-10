"""Load and validate dataset audit configuration from YAML."""
from __future__ import annotations

import yaml
from pathlib import Path


REQUIRED_FIELDS = ["dataset_name", "format", "date_column"]

DEFAULTS = {
    "format": "long",
    "date_column": "Date",
    "dimension_columns": [],
    "value_column": "Value",
    "value_columns": [],
    "expected_frequency": "D",
    "expected_start": None,
    "expected_end": None,
    "acceptable_lag_days": 3,
    "holidays_country": None,
    "timezone": None,
    "expected_dimensions": {},
    "max_null_pct": 0.05,
    "allow_zero_values": True,
    "value_range": {"min": None, "max": None},
    "outlier_zscore_threshold": 3.5,
    "pct_change_threshold": 0.5,
    "revision_whitelist": {
        "enabled": False,
        "max_revision_age_days": 90,
        "max_revision_pct": 0.10,
    },
    "severity_overrides": {},
}


def load_config(path: str | Path) -> dict:
    """Load YAML config file and fill in defaults."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw)}")

    config = dict(DEFAULTS)
    config.update(raw)

    # Ensure nested dicts are merged, not replaced
    for nested_key in ("value_range", "revision_whitelist"):
        if nested_key in raw:
            merged = dict(DEFAULTS[nested_key])
            merged.update(raw[nested_key] or {})
            config[nested_key] = merged

    _validate(config)
    return config


def config_from_dict(d: dict) -> dict:
    """Build a config dict from a plain dict, filling in defaults."""
    config = dict(DEFAULTS)
    config.update(d)
    for nested_key in ("value_range", "revision_whitelist"):
        if nested_key in d:
            merged = dict(DEFAULTS[nested_key])
            merged.update(d[nested_key] or {})
            config[nested_key] = merged
    return config


def _validate(config: dict) -> None:
    for field in REQUIRED_FIELDS:
        if field not in config or config[field] is None:
            raise ValueError(f"Config missing required field: '{field}'")

    fmt = config.get("format")
    if fmt not in ("long", "wide"):
        raise ValueError(f"'format' must be 'long' or 'wide', got '{fmt}'")

    if fmt == "long" and not config.get("value_column"):
        raise ValueError("Long format requires 'value_column'")

    if fmt == "wide" and not config.get("value_columns"):
        raise ValueError("Wide format requires 'value_columns' list")

    freq = config.get("expected_frequency")
    if freq not in ("D", "B", "W", "M", "Q", "A"):
        raise ValueError(f"'expected_frequency' must be one of D/B/W/M/Q/A, got '{freq}'")
