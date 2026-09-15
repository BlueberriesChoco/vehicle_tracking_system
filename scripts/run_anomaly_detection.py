#!/usr/bin/env python
"""Run Phase 3 anomaly detection on behavior vector CSV files."""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import AnomalyDetector
from src.output import AlertWriter, VectorWriter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_groups(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_rows(input_path: str):
    path = Path(input_path)
    files = [str(path)] if path.is_file() else sorted(glob.glob(os.path.join(input_path, "**", "vec_*.csv"), recursive=True))
    rows = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_file"] = file_path
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 anomaly detection")
    parser.add_argument("--input", required=True, help="A vec_*.csv file or directory containing vector CSVs")
    parser.add_argument("--baseline", default=None, help="Baseline JSON path from build_baseline.py")
    parser.add_argument("--output", required=True, help="Output alert CSV path")
    parser.add_argument("--annotated-vectors", default=None, help="Optional output path for vectors with anomaly fields")
    parser.add_argument("--threshold", type=float, default=3.0, help="Baseline z-score anomaly threshold")
    parser.add_argument("--include-normal", action="store_true", help="Write normal rows into alert CSV too")
    parser.add_argument("--target-groups", default="ordinary", help="Comma separated vehicle groups included in anomaly detection")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        logger.error("No vector rows found: %s", args.input)
        sys.exit(1)

    detector = AnomalyDetector(
        baseline_path=args.baseline,
        baseline_threshold=args.threshold,
        target_groups=parse_groups(args.target_groups),
    )
    detected = detector.detect(rows)
    AlertWriter().write(detected, args.output, anomaly_only=not args.include_normal)

    if args.annotated_vectors:
        VectorWriter().write(detected, args.annotated_vectors)

    anomaly_count = sum(1 for row in detected if int(row.get("is_anomaly", 0)) == 1)
    target_count = sum(1 for row in detected if int(row.get("analysis_target", 0)) == 1)
    logger.info("Processed %s rows, target rows: %s, anomalies: %s", len(detected), target_count, anomaly_count)
    print(f"Alerts saved to: {args.output}")
    print(f"Rows: {len(detected)}, target rows: {target_count}, anomalies: {anomaly_count}")


if __name__ == "__main__":
    main()
