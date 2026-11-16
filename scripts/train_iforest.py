#!/usr/bin/env python
"""Train a Phase 3 Isolation Forest (unsupervised individual anomaly detector).

Reads behavior vector CSVs, trains only on normal-looking rows (``is_anomaly=0``),
and saves the model so ``run_anomaly_detection.py --iforest ...`` can use it.

The output ``iforest_score`` is the “行为异常指数 A” described in research point 2.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import IsolationForestDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_groups(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


from src.utils.vectors_io import load_vector_rows as load_rows


def main():
    parser = argparse.ArgumentParser(description="Train Phase 3 Isolation Forest detector")
    parser.add_argument("--input", required=True, help="A vec_*.csv file or directory containing vector CSVs")
    parser.add_argument("--output", required=True, help="Output Isolation Forest model JSON path")
    parser.add_argument("--target-groups", default="ordinary", help="Comma separated vehicle groups used for training")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of isolation trees")
    parser.add_argument("--max-samples", type=int, default=256, help="Samples per tree")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        logger.error("No vector rows found: %s", args.input)
        sys.exit(1)

    # 与 build_baseline.py 一致：优先只用“未被判为异常”的正常样本训练
    normal_rows = [
        row for row in rows
        if str(row.get("is_anomaly", "0")) in {"0", "", "False", "false"}
    ]
    training_rows = normal_rows or rows

    detector = IsolationForestDetector(
        target_groups=parse_groups(args.target_groups),
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        random_state=args.seed,
    )
    detector.fit(training_rows)
    detector.save(args.output)

    logger.info(
        "Trained Isolation Forest on %s/%s rows, saved to %s",
        len(training_rows), len(rows), args.output,
    )
    print(f"Isolation Forest saved to: {args.output}")
    print(f"Training rows: {len(training_rows)} (total: {len(rows)})")
    print(f"Target groups: {','.join(detector.builder.target_groups)}")


if __name__ == "__main__":
    main()
