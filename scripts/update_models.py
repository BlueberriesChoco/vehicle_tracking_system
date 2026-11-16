#!/usr/bin/env python
"""Incremental model update with new data.

Loads the existing Isolation Forest and Transformer trajectory models, updates
them with a new batch of vectors, and reports the elapsed time. This backs the
"retrain on one new week of data in under 4 hours" indicator.

Strategies:
- Isolation Forest: refit on the new data (fast, no warm start needed);
- Transformer: fine-tune from the existing weights (warm start).

Usage:
    python scripts/update_models.py \
        --new-vectors data/outputs/vectors/admin_in_new_week \
        --existing-iforest data/references/iforest_admin_in.json \
        --existing-traj data/references/traj_admin_in \
        --output-dir data/references/updated \
        --epochs 10
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import IsolationForestDetector, TrajectoryPredictionDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.vectors_io import load_vector_rows as load_rows


def main():
    parser = argparse.ArgumentParser(description="Incrementally update anomaly models")
    parser.add_argument("--new-vectors", required=True, help="New vector CSV or directory")
    parser.add_argument("--existing-iforest", default=None, help="Existing Isolation Forest JSON")
    parser.add_argument("--existing-traj", default=None, help="Existing Transformer model prefix")
    parser.add_argument("--output-dir", required=True, help="Directory for updated models")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tune epochs for Transformer")
    parser.add_argument("--full-retrain", action="store_true", help="Train Transformer from scratch")
    args = parser.parse_args()

    rows = load_rows(args.new_vectors)
    targets = [
        row for row in rows
        if str(row.get("analysis_target", "")).strip() in {"1", "true", "yes"}
    ]
    if not targets:
        logger.error("No analysis-target rows in %s", args.new_vectors)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    started = time.perf_counter()
    result = {"new_rows": len(rows), "new_targets": len(targets)}

    # ---- Isolation Forest ----
    iforest_started = time.perf_counter()
    if args.existing_iforest and os.path.exists(args.existing_iforest):
        iforest = IsolationForestDetector.load(args.existing_iforest)
    else:
        iforest = IsolationForestDetector()
    iforest.fit(targets)
    iforest_path = os.path.join(args.output_dir, "iforest.json")
    iforest.save(iforest_path)
    result["iforest"] = {
        "mode": "refit",
        "path": iforest_path,
        "elapsed_sec": round(time.perf_counter() - iforest_started, 2),
    }

    # ---- Transformer ----
    traj_started = time.perf_counter()
    if args.existing_traj and os.path.exists(args.existing_traj + ".pt") and not args.full_retrain:
        predictor = TrajectoryPredictionDetector.load(args.existing_traj)
        predictor.epochs = args.epochs
        predictor.fit(targets, reuse_existing=True)
        mode = "fine_tune"
    else:
        predictor = TrajectoryPredictionDetector(past_len=5, future_len=5, epochs=args.epochs)
        predictor.fit(targets)
        mode = "full_retrain"
    traj_prefix = os.path.join(args.output_dir, "traj")
    predictor.save(traj_prefix)
    result["transformer"] = {
        "mode": mode,
        "path": traj_prefix,
        "epochs": args.epochs,
        "elapsed_sec": round(time.perf_counter() - traj_started, 2),
    }

    result["total_elapsed_sec"] = round(time.perf_counter() - started, 2)
    result["under_4_hours"] = result["total_elapsed_sec"] < 4 * 3600

    output_path = os.path.join(args.output_dir, "update_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("Model update finished in %.2fs", result["total_elapsed_sec"])


if __name__ == "__main__":
    main()
