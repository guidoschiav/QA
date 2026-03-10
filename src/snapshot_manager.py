"""Manage dataset snapshots stored as Parquet files."""
from __future__ import annotations

import pandas as pd
from pathlib import Path


SNAPSHOT_DIR = Path(__file__).parent.parent / "snapshots"


def get_snapshot_path(dataset_name: str, snapshot_dir: Path | None = None) -> Path:
    base = snapshot_dir or SNAPSHOT_DIR
    return base / f"{dataset_name}.parquet"


def load_snapshot(dataset_name: str, snapshot_dir: Path | None = None) -> pd.DataFrame | None:
    """Load the previous snapshot for a dataset. Returns None if not found."""
    path = get_snapshot_path(dataset_name, snapshot_dir)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def save_snapshot(
    df: pd.DataFrame,
    dataset_name: str,
    snapshot_dir: Path | None = None,
) -> Path:
    """Save current DataFrame as the new snapshot. Returns the saved path."""
    base = snapshot_dir or SNAPSHOT_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = get_snapshot_path(dataset_name, base)
    df.to_parquet(path, index=False)
    return path
