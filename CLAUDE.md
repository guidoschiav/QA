# Dataset Auditor

Automatic data quality auditing app with a Streamlit UI and CLI entrypoint.

## Stack
- Python 3.10+
- Streamlit, pandas, numpy, pyyaml, scipy, requests, plotly
- Snapshots in parquet, configs in YAML, reports in JSON

## Main Entry Points
- Streamlit app: `app.py`
- CLI audit runner: `run_audit.py`

## Core Conventions
- Check functions return `List[CheckResult]`
- Severities: `BLOCKER`, `WARNING`, `INFO`
- Per-dataset config lives in `configs/*.yaml`
- Date format should be `YYYY-MM-DD`
- Supported dataset layouts: `long`, `wide`, `wide_transposed`

## Commands
- Run app locally: `streamlit run app.py`
- Run CLI audit: `python run_audit.py --data archivo.csv --config configs/example_config.yaml`
- Run tests: `.venv\Scripts\python.exe -m pytest tests -q`

## Deploy Notes For Future Work
- Target deploy platform is Streamlit Community Cloud.
- Main file for deploy is `app.py`.
- Python dependencies are declared in `requirements.txt`.
- No `packages.txt` is currently needed.
- Secrets must not be committed.
- If Alphacast access is needed in Cloud, configure `ALPHACAST_API_KEY` in Streamlit Secrets.
- A template exists at `.streamlit/secrets.toml.example`.

## Changes Already Implemented
- Added Streamlit Community Cloud compatibility with minimal code changes.
- `app.py` now loads the Alphacast API key from Streamlit Secrets first, then falls back to env/local file for local development.
- `src/alphacast_client.py` supports loading the API key from the `ALPHACAST_API_KEY` environment variable.
- Local API-key storage remains acceptable for local dev; Cloud should use Secrets.
- Replaced folder-opening behavior with a Cloud-safe fallback through `reveal_directory(...)` so the app does not fail on Streamlit Cloud.
- Removed broken `icon=` parameters that were crashing Streamlit Cloud when mojibake emojis were present.
- Fixed the text encoding of `app.py` so labels and Spanish text render correctly in Cloud.
- Added support for campaign-style proxy dates in detection and parsing: `YYYY/YYYY`, `YYYY-YYYY`, and `YYYY/YY`, mapped to `YYYY-01-01`.
- Kept local behavior intact after deploy-related changes.
- `app.py` only shows `st.exception(exc)` when `DEBUG` is set in the environment.
- Removed the leftover debug `print()` from `src/auto_detect.py`.
- Added `.gitignore` entries for `.venv`, caches, generated reports/snapshots, and `.streamlit/secrets.toml`.
- Added `.streamlit/secrets.toml.example` as a non-secret template.
- Updated `requirements.txt` to reflect actual runtime dependencies used by the app.

## Validation Status
- `compileall app.py` passed after the Cloud fixes.
- Test suite passed after the Cloud fixes: `225 passed`.

## Practical Warnings
- `reports/` and `snapshots/` are runtime artifacts and should not be treated as persistent storage in Streamlit Community Cloud.
- If the UI ever shows mojibake again (`CargÃ¡`, `ðŸ...`), check file encoding first.
- If Cloud crashes on visual messages, inspect invalid `icon=` values or malformed emoji literals.
