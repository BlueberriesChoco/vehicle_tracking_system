#!/usr/bin/env python
"""Run Phase 3 anomaly detection on behavior vector CSV files."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import AnomalyDetector, IsolationForestDetector, TrajectoryPredictionDetector
from src.anomaly.config import load_anomaly_config
from src.output import AlertWriter, VectorWriter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_groups(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


from src.utils.vectors_io import load_vector_rows as load_rows


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 anomaly detection")
    parser.add_argument("--input", required=True, help="A vec_*.csv file or directory containing vector CSVs")
    parser.add_argument("--baseline", default=None, help="Baseline JSON path from build_baseline.py")
    parser.add_argument("--iforest", default=None, help="Isolation Forest JSON path from train_iforest.py")
    parser.add_argument("--trajectory-model", default=None, help="Transformer trajectory model prefix from train_transformer.py")
    parser.add_argument("--output", required=True, help="Output alert CSV path")
    parser.add_argument("--annotated-vectors", default=None, help="Optional output path for vectors with anomaly fields")
    parser.add_argument("--threshold", type=float, default=3.0, help="Baseline z-score anomaly threshold")
    parser.add_argument("--include-normal", action="store_true", help="Write normal rows into alert CSV too")
    parser.add_argument("--target-groups", default="ordinary", help="Comma separated vehicle groups included in anomaly detection")
    parser.add_argument("--config", default="config/anomaly_config.yaml", help="Anomaly config (rules thresholds)")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        logger.error("No vector rows found: %s", args.input)
        sys.exit(1)

    iforest = IsolationForestDetector.load(args.iforest) if args.iforest else None
    trajectory_predictor = (
        TrajectoryPredictionDetector.load(args.trajectory_model)
        if args.trajectory_model
        else None
    )
    detector = AnomalyDetector(
        baseline_path=args.baseline,
        baseline_threshold=args.threshold,
        target_groups=parse_groups(args.target_groups),
        iforest=iforest,
        trajectory_predictor=trajectory_predictor,
        thresholds=load_anomaly_config(args.config).get("rules"),
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
