"""Alphacast platform profile — predefined canonicalization settings."""
from __future__ import annotations

ALPHACAST_PROFILE: dict = {
    "name": "Alphacast Standard",
    "header_row": 0,
    "date_formats": ["YYYY-MM-DD", "%Y-%m-%d"],
    "date_column_names": ["Date", "date", "DATE", "Fecha", "fecha"],
    "separator": ",",
    "decimal": ".",
    "thousands": ",",
    "scale": 1,
    "null_values": [],
    "skip_rows_top": 0,
    "skip_rows_bottom": 0,
    "encoding": "utf-8",
    "value_format": "standard",
}


def apply_alphacast_profile(config: dict) -> dict:
    """
    Take a compare config dict and fill in the 'platform' side with Alphacast
    profile defaults. Any value the user already specified has priority over
    the profile.

    Args:
        config: compare config dict with at least a 'source' key.

    Returns:
        A copy of config with 'platform' filled in from the Alphacast profile.

    Example config structure::

        {
            "source": { "date_column": "Date", "key_columns": [...], ... },
            "platform": { "date_column": "Date" },   # partial — rest is filled
            "tolerance": 0.01,
        }
    """
    result = dict(config)

    # Build the Alphacast-defaults for the platform side
    platform_defaults: dict = {
        "format": result.get("source", {}).get("format", "long"),
        "date_column": "Date",
        "date_format": "%Y-%m-%d",
        "dayfirst": False,
        "key_columns": result.get("source", {}).get("key_columns", []),
        "value_column": result.get("source", {}).get("value_column", "Value"),
        "value_columns": result.get("source", {}).get("value_columns", []),
        "header_row": ALPHACAST_PROFILE["header_row"],
        "skip_rows_bottom": ALPHACAST_PROFILE["skip_rows_bottom"],
        "number_format": {
            "decimal": ALPHACAST_PROFILE["decimal"],
            "thousands": ALPHACAST_PROFILE["thousands"],
            "currency": None,
            "is_percentage": False,
        },
        "scale": ALPHACAST_PROFILE["scale"],
        "normalizations": {
            "lowercase_keys": True,
            "strip_keys": True,
        },
    }

    existing_platform = result.get("platform", {})

    # Merge: user values take priority, fill missing keys from profile defaults
    merged: dict = dict(platform_defaults)
    merged.update(existing_platform)

    # Special case: number_format is a nested dict — merge at one level deeper
    if "number_format" in existing_platform and isinstance(existing_platform["number_format"], dict):
        merged["number_format"] = dict(platform_defaults["number_format"])
        merged["number_format"].update(existing_platform["number_format"])

    result["platform"] = merged
    return result
