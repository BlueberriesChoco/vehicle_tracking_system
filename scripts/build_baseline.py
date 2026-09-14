#!/usr/bin/env python
"""Build a Phase 3 normal behavior baseline from behavior vector CSV files."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly.baseline import BaselineBuilder, BaselineStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build Phase 3 normal behavior baseline")
    parser.add_argument("--vector-dir", required=True, help="Directory containing vec_*.csv files")
    parser.add_argument("--camera", default="cam01", help="Camera ID for the baseline")
    parser.add_argument("--output", required=True, help="Output baseline JSON path")
    args = parser.parse_args()

    builder = BaselineBuilder()
    vectors = builder.load_vectors(args.vector_dir)
    if not vectors:
        logger.error("No vectors found. Run the video pipeline first.")
        sys.exit(1)

    normal_vectors = [row for row in vectors if str(row.get("is_anomaly", "0")) in {"0", "", "False", "false"}]
    training_rows = normal_vectors or vectors
    baseline = builder.build(training_rows, camera_id=args.camera)
    BaselineStore.save(baseline, args.output)

    logger.info("Loaded %s vectors, used %s rows for baseline", len(vectors), len(training_rows))
    logger.info("Baseline saved to: %s", args.output)
    print(f"Baseline saved to: {args.output}")
    print(f"Sample count: {baseline['sample_count']}")


if __name__ == "__main__":
    main()
