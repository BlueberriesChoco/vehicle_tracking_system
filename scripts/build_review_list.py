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

from src.utils.logger import get_logger

logger = get_logger(__name__)

COLUMNS = [
    "rank",
    "passage_id",
    "camera_id",
    "enter_time",
    "duration_sec",
    "risk_score_100",
    "evidence_count",
    "top_evidence",
    "plate_number",
]


def _evidence_summary(record: dict, limit: int = 2) -> str:
    items = record.get("evidence", []) or []
    summaries = []
    for item in items[:limit]:
        text = str(item.get("summary", "") or "").strip()
        if text:
            summaries.append(text)
    return "；".join(summaries)


def rank_vehicles(vehicles, budget: int = 20, min_score: float = 0.0):
    """按可疑度排序并截断到预算，不设「异常」布尔标记。"""
    eligible = [
        record for record in vehicles
        if float(record.get("risk_score_100", 0) or 0) >= min_score
    ]
    eligible.sort(key=lambda record: -float(record.get("risk_score_100", 0) or 0))
    if budget > 0:
        eligible = eligible[:budget]
    return eligible


def build_rows(vehicles) -> list:
    rows = []
    for rank, record in enumerate(vehicles, start=1):
        rows.append(
            {
                "rank": rank,
                "passage_id": record.get("passage_id", ""),
                "camera_id": record.get("camera_id", ""),
                "enter_time": record.get("enter_time", ""),
                "duration_sec": record.get("duration_sec", ""),
                "risk_score_100": record.get("risk_score_100", ""),
                "evidence_count": len(record.get("evidence", []) or []),
                "top_evidence": _evidence_summary(record),
                "plate_number": record.get("plate_number", ""),
            }
        )
    return rows


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
