#!/usr/bin/env python
"""生成「待核查清单」：按可疑度排名，不设异常阈值。

为什么不用告警率：违规夹带一年只有个位数，用 contamination=5% 切阈值会在
444 辆车上凭空造出 22 条「异常」。本脚本只做排序，由人按固定预算（每周 N 辆）
取前 N 条核查，输出的是可核查的线索，不是判定结论。

用法：
    python scripts/build_review_list.py \
        --risk data/outputs/e2e/risk/risk_admin_in.json \
        --budget 20 \
        --output data/outputs/review/review_list.csv
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.report.review_list import COLUMNS, build_rows, rank_vehicles
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build a ranked review list (no anomaly threshold)")
    parser.add_argument("--risk", required=True, help="risk_*.json from build_risk_report.py")
    parser.add_argument("--budget", type=int, default=20, help="How many vehicles to review (0 = all)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Optional score floor")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    with open(args.risk, encoding="utf-8") as handle:
        payload = json.load(handle)
    vehicles = payload.get("vehicles", []) if isinstance(payload, dict) else []

    selected = rank_vehicles(vehicles, budget=args.budget, min_score=args.min_score)
    rows = build_rows(selected)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    scores = [float(row["risk_score_100"]) for row in rows if row["risk_score_100"] != ""]
    logger.info(
        "Review list: %d / %d vehicles, score range %.1f-%.1f",
        len(rows), len(vehicles), min(scores) if scores else 0.0, max(scores) if scores else 0.0,
    )
    print(json.dumps({
        "vehicles_total": len(vehicles),
        "review_budget": len(rows),
        "score_min": round(min(scores), 1) if scores else None,
        "score_max": round(max(scores), 1) if scores else None,
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
