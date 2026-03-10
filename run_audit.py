#!/usr/bin/env python3
"""
run_audit.py — CLI entry point for the Dataset Auditor.

Usage:
    python run_audit.py --data path/to/data.csv --config configs/my_config.yaml
    python run_audit.py --data path/to/data.parquet --config configs/my_config.yaml [--report-dir reports/]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full data quality audit on a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_audit.py --data sales.csv --config configs/example_config.yaml
  python run_audit.py --data sales.parquet --config configs/example_config.yaml --report-dir reports/
        """,
    )
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to save JSON report (default: reports/)",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="snapshots",
        help="Directory for dataset snapshots (default: snapshots/)",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip saving a new snapshot after audit",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Data file not found: {path}", file=sys.stderr)
        sys.exit(1)

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        # Try CSV by default
        try:
            return pd.read_csv(path)
        except Exception:
            print(f"[ERROR] Unsupported file format: {suffix}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    args = parse_args()

    # Add project root to path so `src` is importable
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from src.config_loader import load_config
    from src.snapshot_manager import load_snapshot, save_snapshot
    from src.auditor import audit_dataset
    from src.report import print_human_report, export_report_json

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    dataset_name = config["dataset_name"]
    snapshot_dir = Path(args.snapshot_dir)

    # Load data
    print(f"Loading data from: {args.data}")
    df = load_data(args.data)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Load previous snapshot
    prev_snapshot = load_snapshot(dataset_name, snapshot_dir)
    if prev_snapshot is not None:
        print(f"  Found previous snapshot: {len(prev_snapshot):,} rows")
    else:
        print("  No previous snapshot found — consistency checks will be skipped")

    # Run audit
    print(f"\nRunning audit for '{dataset_name}'...")
    report = audit_dataset(df, config, prev_snapshot=prev_snapshot)

    # Print human-readable report
    print_human_report(report)

    # Save snapshot
    if not args.no_snapshot:
        snap_path = save_snapshot(df, dataset_name, snapshot_dir)
        print(f"Snapshot saved: {snap_path}")

    # Export JSON report
    report_dir = Path(args.report_dir)
    ts = report.timestamp.replace(":", "-").replace(".", "-")
    report_path = report_dir / f"{dataset_name}_{ts}.json"
    export_report_json(report, report_path)
    print(f"JSON report saved: {report_path}")

    # Exit with non-zero code if there are blockers
    if report.blockers > 0:
        sys.exit(2)
    elif report.warnings > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
