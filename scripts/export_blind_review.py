#!/usr/bin/env python
"""导出盲评表：给专家看的抽样清单，不含系统分数。

没有查验记录时，唯一可行的评估方式是「专家盲评」：
让查验人员在不看系统评分的情况下判断哪些车可疑，再与系统排序比较
（重叠率、认可率）。盲评表里绝不能出现分数，否则会引导判断。

输出两个文件：
    blind_review.csv     给人填的表（无分数）
    blind_review_key.csv 对应的编号/分组/分数，评完再用

用法：
    python scripts/export_blind_review.py \
        --risk data/outputs/e2e/risk/risk_admin_in.json \
        --n-random 50 --n-top 50 --seed 42 \
        --out-dir data/outputs/review
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import get_logger

logger = get_logger(__name__)

BLIND_COLUMNS = [
    "case_id",
    "camera_id",
    "enter_time",
    "duration_sec",
    "vehicle_type",
    "vehicle_color",
    "plate_hint",
    "拦截意愿_1到5",
    "怀疑理由",
    "想查部位",
    "备注",
]


def sample_cases(vehicles, n_random=50, n_top=50, seed=42):
    """抽两组：随机样本（估计基线）+ 系统高分（看系统眼光）。

    两组可能重叠，返回时用 (record, group) 标记，由调用方去重。
    """
    ordered = sorted(vehicles, key=lambda row: -float(row.get("risk_score_100", 0) or 0))
    top = ordered[:n_top]
    top_keys = {row.get("passage_id") for row in top}

    pool = [row for row in vehicles if row.get("passage_id") not in top_keys]
    rng = random.Random(seed)
    random_pick = rng.sample(pool, min(n_random, len(pool)))

    cases = [(row, "system_top") for row in top]
    cases += [(row, "random") for row in random_pick]
    return cases


def to_blind_row(case_id: int, record: dict) -> dict:
    plate = str(record.get("plate_number", "") or "").strip()
    return {
        "case_id": case_id,
        "camera_id": record.get("camera_id", ""),
        "enter_time": record.get("enter_time", ""),
        "duration_sec": record.get("duration_sec", ""),
        "vehicle_type": record.get("vehicle_type", ""),
        "vehicle_color": record.get("vehicle_color", ""),
        # 只给末两位，避免把车牌直接暴露成"这辆肯定有问题"的暗示
        "plate_hint": f"**{plate[-2:]}" if len(plate) >= 2 else "",
        "拦截意愿_1到5": "",
        "怀疑理由": "",
        "想查部位": "",
        "备注": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Export a blind review sheet (no system scores)")
    parser.add_argument("--risk", required=True, help="risk_*.json")
    parser.add_argument("--n-random", type=int, default=50)
    parser.add_argument("--n-top", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.risk, encoding="utf-8") as handle:
        vehicles = json.load(handle).get("vehicles", [])

    cases = sample_cases(vehicles, n_random=args.n_random, n_top=args.n_top, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    blind_path = os.path.join(args.out_dir, "blind_review.csv")
    key_path = os.path.join(args.out_dir, "blind_review_key.csv")

    with open(blind_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_COLUMNS)
        writer.writeheader()
        for case_id, (record, _) in enumerate(cases, start=1):
            writer.writerow(to_blind_row(case_id, record))

    with open(key_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "passage_id", "group", "risk_score_100", "rank_in_system"])
        ranked = {row.get("passage_id"): index + 1
                  for index, row in enumerate(
                      sorted(vehicles, key=lambda r: -float(r.get("risk_score_100", 0) or 0)))}
        for case_id, (record, group) in enumerate(cases, start=1):
            pid = record.get("passage_id", "")
            writer.writerow([case_id, pid, group, record.get("risk_score_100", ""), ranked.get(pid, "")])

    logger.info("Blind review: %d cases (%d random + %d system top)", len(cases), args.n_random, args.n_top)
    print(json.dumps({
        "cases": len(cases),
        "blind_sheet": blind_path,
        "key_sheet": key_path,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
