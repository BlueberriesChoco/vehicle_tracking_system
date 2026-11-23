#!/usr/bin/env python
"""跨机位行程构建：把两路机位的通行记录按「ReID 相似度 + 时序约束」配成一对一行程。

为什么不用纯相似度匹配：
    `CrossCameraStitcher` 只按 ReID 余弦阈值连边，同一辆车可能同时连上多辆，
    实测 44 条出场记录里出现 3 对穿行时间 <20 秒（低于拓扑下限 20 秒），
    说明存在错配。这里改成全局最优的一对一分配（匈牙利算法），
    并把拓扑允许的时间窗做成硬约束。

输出 ``journeys.json``：
    [
      {
        "global_id": "G0001",
        "travel_sec": 132.0,
        "similarity": 0.96,
        "legs": [{"camera": "admin_in", "passage_id": "...", "enter_time": "...", "exit_time": "..."},
                 {"camera": "admin_out", ...}]
      }, ...
    ]

用法：
    python scripts/build_journeys.py \
        --in-a data/outputs/mc_admin_in/vectors/admin_in/20260901 \
        --in-b data/outputs/mc_admin_out2/vectors/admin_out/20260901 \
        --output data/outputs/journeys.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# 拓扑允许的穿行时间窗（秒）。config/site/topology.yaml 里 admin_in->admin_out 为 20s~3h。
MIN_TRAVEL_SEC = 20.0
MAX_TRAVEL_SEC = 3 * 3600.0
INVALID_COST = 1e6


def load_rows(directory: str) -> List[dict]:
    pattern = os.path.join(directory, "**", "*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    rows: List[dict] = []
    for path in files:
        with open(path, encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def parse_embedding(value: str) -> Optional[np.ndarray]:
    """解析 CSV 里的 ReID 向量。序列化格式是 ``[0.1, -0.2, ...]``（带逗号）。"""
    if not value:
        return None
    text = value.strip()
    try:
        vector = np.asarray(json.loads(text), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError):
        cleaned = text.strip("[]").replace(",", " ")
        vector = np.fromstring(cleaned, sep=" ", dtype=np.float64)
    if vector.size == 0:
        return None
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else None


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.dot(a, b))


def travel_seconds(row_a: dict, row_b: dict) -> Optional[float]:
    try:
        exit_a = datetime.fromisoformat(row_a["exit_time"])
        enter_b = datetime.fromisoformat(row_b["enter_time"])
    except (KeyError, ValueError):
        return None
    return (enter_b - exit_a).total_seconds()


def build_journeys(
    rows_a: List[dict],
    rows_b: List[dict],
    *,
    min_travel_sec: float = MIN_TRAVEL_SEC,
    max_travel_sec: float = MAX_TRAVEL_SEC,
    min_similarity: float = 0.5,
) -> List[dict]:
    """一对一行程分配。返回按进入时间排序的行程列表。"""
    if not rows_a or not rows_b:
        return []

    emb_a = [parse_embedding(r.get("reid_embedding", "")) for r in rows_a]
    emb_b = [parse_embedding(r.get("reid_embedding", "")) for r in rows_b]

    n, m = len(rows_a), len(rows_b)
    cost = np.full((n, m), INVALID_COST, dtype=np.float64)
    sims = np.zeros((n, m), dtype=np.float64)

    for i, row_a in enumerate(rows_a):
        for j, row_b in enumerate(rows_b):
            gap = travel_seconds(row_a, row_b)
            if gap is None or not (min_travel_sec <= gap <= max_travel_sec):
                continue
            similarity = cosine(emb_a[i], emb_b[j])
            if similarity is None or similarity < min_similarity:
                continue
            # 相似度越高、穿行时间越接近中位越优；时间惩罚只做轻微区分
            time_penalty = abs(gap - 180.0) / 3600.0 * 0.05
            cost[i, j] = 1.0 - similarity + time_penalty
            sims[i, j] = similarity

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy 是必需依赖
        logger.error("缺少 scipy，无法做一对一分配")
        return []

    rows_idx, cols_idx = linear_sum_assignment(cost)
    journeys: List[dict] = []
    for i, j in zip(rows_idx, cols_idx):
        if cost[i, j] >= INVALID_COST:
            continue
        row_a, row_b = rows_a[i], rows_b[j]
        journeys.append(
            {
                "travel_sec": round(travel_seconds(row_a, row_b) or 0.0, 1),
                "similarity": round(float(sims[i, j]), 4),
                "legs": [
                    {
                        "camera": row_a.get("camera_id", ""),
                        "passage_id": row_a.get("passage_id", ""),
                        "enter_time": row_a.get("enter_time", ""),
                        "exit_time": row_a.get("exit_time", ""),
                    },
                    {
                        "camera": row_b.get("camera_id", ""),
                        "passage_id": row_b.get("passage_id", ""),
                        "enter_time": row_b.get("enter_time", ""),
                        "exit_time": row_b.get("exit_time", ""),
                    },
                ],
            }
        )

    journeys.sort(key=lambda item: item["legs"][0]["enter_time"])
    for index, journey in enumerate(journeys, start=1):
        journey["global_id"] = f"G{index:04d}"
    return journeys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-a", required=True, help="上游机位向量目录")
    parser.add_argument("--in-b", required=True, help="下游机位向量目录")
    parser.add_argument("--output", required=True, help="输出 journeys.json")
    parser.add_argument("--min-similarity", type=float, default=0.5)
    parser.add_argument("--min-travel-sec", type=float, default=MIN_TRAVEL_SEC)
    parser.add_argument("--max-travel-sec", type=float, default=MAX_TRAVEL_SEC)
    args = parser.parse_args()

    rows_a = load_rows(args.in_a)
    rows_b = load_rows(args.in_b)
    journeys = build_journeys(
        rows_a, rows_b,
        min_travel_sec=args.min_travel_sec,
        max_travel_sec=args.max_travel_sec,
        min_similarity=args.min_similarity,
    )

    payload = {
        "cameras": {"a": len(rows_a), "b": len(rows_b)},
        "journey_count": len(journeys),
        "params": {
            "min_similarity": args.min_similarity,
            "min_travel_sec": args.min_travel_sec,
            "max_travel_sec": args.max_travel_sec,
        },
        "journeys": journeys,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    gaps = sorted(j["travel_sec"] for j in journeys)
    if gaps:
        logger.info(
            "Journeys: %d，穿行 %.1f~%.1f 分钟，相似度中位 %.3f",
            len(journeys), gaps[0] / 60, gaps[-1] / 60,
            sorted(j["similarity"] for j in journeys)[len(journeys) // 2],
        )
    else:
        logger.warning("没有匹配出任何行程")
    print(json.dumps({"journeys": len(journeys), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
