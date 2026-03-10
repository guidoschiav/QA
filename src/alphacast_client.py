"""Alphacast API client - lightweight, requests-only."""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

API_BASE_URL = "https://charts.alphacast.io/api/datasets"
API_KEY_ENV_VAR = "ALPHACAST_API_KEY"
API_KEY_FILE = Path.home() / ".dataset-auditor" / "alphacast_api_key.txt"


@dataclass
class AlphacastDataset:
    id: int
    name: str
    df: pd.DataFrame
    row_count: int
    col_count: int


def save_api_key(api_key: str) -> None:
    """Save API key to local file in ~/.dataset-auditor/."""
    API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    API_KEY_FILE.write_text(api_key.strip(), encoding="utf-8")


def load_api_key_with_source() -> tuple[str | None, str | None]:
    """Load API key from env var first, then local file."""
    env_key = os.getenv(API_KEY_ENV_VAR, "").strip()
    if env_key:
        return env_key, "env"

    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key, "file"

    return None, None


def load_api_key() -> str | None:
    """Load saved API key. Returns None if not set."""
    key, _source = load_api_key_with_source()
    return key


def delete_api_key() -> None:
    """Delete the saved API key file."""
    if API_KEY_FILE.exists():
        API_KEY_FILE.unlink()


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key by hitting the datasets list endpoint.
    Returns True if the key is accepted (HTTP 200), False otherwise.
    """
    try:
        r = requests.get(
            API_BASE_URL,
            auth=HTTPBasicAuth(api_key, ""),
            timeout=10,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


def fetch_dataset_metadata(api_key: str, dataset_id: int) -> dict | None:
    """
    Fetch metadata (name, id, etc.) for a dataset.
    Returns the parsed JSON dict or None on failure.
    """
    try:
        r = requests.get(
            f"{API_BASE_URL}/{dataset_id}",
            auth=HTTPBasicAuth(api_key, ""),
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None


def fetch_dataset(api_key: str, dataset_id: int) -> AlphacastDataset | None:
    """
    Download a dataset as CSV and return an AlphacastDataset.

    Steps:
    1. Fetch metadata to get the dataset name.
    2. Download .csv endpoint.
    3. Parse with pd.read_csv().

    Returns None on any failure.
    """
    try:
        metadata = fetch_dataset_metadata(api_key, dataset_id)
        name = (
            metadata.get("name", f"Dataset {dataset_id}")
            if metadata
            else f"Dataset {dataset_id}"
        )

        r = requests.get(
            f"{API_BASE_URL}/{dataset_id}.csv",
            auth=HTTPBasicAuth(api_key, ""),
            timeout=60,
        )
        if r.status_code != 200:
            return None

        df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
        df.columns = [str(c) for c in df.columns]
        return AlphacastDataset(
            id=dataset_id,
            name=name,
            df=df,
            row_count=len(df),
            col_count=len(df.columns),
        )
    except Exception:
        return None
