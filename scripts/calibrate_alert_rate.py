#!/usr/bin/env python
"""Calibrate the Isolation Forest / Transformer contamination to hit an alert-rate target.

For each candidate contamination value it trains both models on the
analysis-target rows and measures the resulting alert rate, then recommends the
largest contamination whose rate stays within the target.

Usage:
    python scripts/calibrate_alert_rate.py \
        --input data/outputs/full_rerun/alerts005/vectors.csv \
        --target-rate 0.15 \
        --grid 0.01,0.02,0.03,0.05,0.08,0.10 \
        --output data/outputs/full_rerun/calibration.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import AnomalyDetector, IsolationForestDetector, TrajectoryPredictionDetector
from src.anomaly.alert_budget import choose_contamination
from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.vectors_io import load_vector_rows as load_rows


def is_target(row) -> bool:
    return str(row.get("analysis_target", "")).strip() in {"1", "true", "yes"}


def count_alerts(rows) -> int:
    return sum(1 for row in rows if str(row.get("is_anomaly", 0)) in {"1", "True", "true"})


def main():
    parser = argparse.ArgumentParser(description="Calibrate alert rate")
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-rate", type=float, default=0.15)
    parser.add_argument("--grid", default="0.01,0.02,0.03,0.05,0.08,0.10")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = load_rows(args.input)
    targets = [row for row in rows if is_target(row)]
    if len(targets) < 3:
        logger.error("Not enough analysis targets: %s", len(targets))
        sys.exit(1)

    grid = [float(value) for value in args.grid.split(",") if value.strip()]
    trials = []
    for contamination in sorted(grid):
        iforest = IsolationForestDetector(contamination=contamination).fit(targets)
        predictor = TrajectoryPredictionDetector(
            past_len=5, future_len=5, epochs=args.epochs, contamination=contamination,
            device=args.device,
        ).fit(targets)
        detector = AnomalyDetector(iforest=iforest, trajectory_predictor=predictor)
        detected = detector.detect(rows)
        alerts = count_alerts(detected)
        rate = alerts / len(rows) if rows else 0.0
        trials.append({
            "contamination": contamination,
            "alert_count": alerts,
            "alert_rate": round(rate, 4),
            "within_target": rate <= args.target_rate,
        })
        logger.info("contamination=%.3f -> %s alerts (%.2f%%)", contamination, alerts, rate * 100)

    within = [trial for trial in trials if trial["within_target"]]
    recommended = choose_contamination(trials, args.target_rate)

    payload = {
        "target_rate": args.target_rate,
        "row_count": len(rows),
        "target_count": len(targets),
        "trials": trials,
        "recommended_contamination": recommended["contamination"],
        "recommended_alert_rate": recommended["alert_rate"],
        "recommended_within_target": recommended["within_target"],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Calibration saved to: %s", args.output)


if __name__ == "__main__":
    main()
