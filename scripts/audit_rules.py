#!/usr/bin/env python
"""Audit rule trigger rates on a vector dataset.

Prints, for every configured rule:
- how many vehicles triggered it;
- how many vehicles were *skipped* because the required geometry was unreliable;
- the distribution of the underlying field.

This makes "dead rules" visible: a rule that never fires is either
misconfigured, mis-gated, or simply has no violations in the data.

Usage:
    python scripts/audit_rules.py \
        --input data/outputs/full_rerun/alerts/vectors_admin_in.csv
"""

import argparse
import csv
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly.config import load_anomaly_config
from src.anomaly.rule_engine import RuleEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_rows(input_path: str):
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = sorted(glob.glob(os.path.join(input_path, "**", "vec_*.csv"), recursive=True))
    rows = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_source_file"] = file_path
                rows.append(row)
    return rows


def numeric(values):
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def describe(name, values):
    if not values:
        return f"{name}: <no numeric values>"
    values = sorted(values)
    count = len(values)
    return (
        f"{name}: n={count} min={values[0]:.2f} p50={values[count // 2]:.2f} "
        f"p95={values[min(count - 1, int(count * 0.95))]:.2f} max={values[-1]:.2f}"
    )


def flag(row, key):
    return str(row.get(key, "0")).strip() in {"1", "True", "true", "yes"}


def main():
    parser = argparse.ArgumentParser(description="Audit rule trigger rates")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config/anomaly_config.yaml")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        logger.error("No rows: %s", args.input)
        sys.exit(1)

    config = load_anomaly_config(args.config)
    engine = RuleEngine(thresholds=config.get("rules"))

    counts = Counter()
    hit_rows = 0
    for row in rows:
        reasons = engine.check(row)
        if reasons:
            hit_rows += 1
        for reason in reasons:
            counts[reason.split("(")[0]] += 1

    total = len(rows)
    print(f"vectors: {total}")
    print(f"rows with >=1 rule hit: {hit_rows} ({100.0 * hit_rows / total:.1f}%)")
    print("rule hit counts:")
    if counts:
        for name, value in counts.most_common():
            print(f"  {name}: {value}")
    else:
        print("  <none>")

    speed_reliable = sum(1 for row in rows if flag(row, "speed_reliable"))
    path_reliable = sum(1 for row in rows if flag(row, "path_reliable"))
    night = sum(1 for row in rows if flag(row, "is_night"))
    print("\nreliability flags:")
    print(f"  speed_reliable=1: {speed_reliable}/{total}  (超速规则只在为 1 时生效)")
    print(f"  path_reliable=1:  {path_reliable}/{total}  (路径偏离规则只在为 1 时生效)")
    print(f"  is_night=1:       {night}/{total}")

    print("\nfield distributions:")
    print("  " + describe("avg_speed_ms", numeric(row.get("avg_speed_ms") for row in rows)))
    print("  " + describe("max_dwell_sec", numeric(row.get("max_dwell_sec") for row in rows)))
    print("  " + describe("path_deviation", numeric(row.get("path_deviation") for row in rows)))
    print("  " + describe("path_smoothness", numeric(row.get("path_smoothness") for row in rows)))
    print("  " + describe("stop_count", numeric(row.get("stop_count") for row in rows)))
    print("  " + describe("freq_count_24h", numeric(row.get("freq_count_24h") for row in rows)))

    thresholds = config.get("rules", {})
    print("\nthresholds: " + ", ".join(f"{key}={value}" for key, value in thresholds.items()))


if __name__ == "__main__":
    main()
