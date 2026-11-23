#!/usr/bin/env python
"""构建跨机位车辆行程，输出控制台可读取的 journeys.json。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger  # noqa: E402

from src.trajectory.stitching import CrossCameraStitcher
from src.utils.vectors_io import load_vector_rows as load_rows

logger = get_logger(__name__)

# 拓扑允许的穿行时间窗（秒）。config/site/topology.yaml 里 admin_in->admin_out 为 20s~3h。
MIN_TRAVEL_SEC = 20.0
MAX_TRAVEL_SEC = 3 * 3600.0

def build_journeys(rows_a, rows_b, **options) -> List[dict]:
    return sorted(
        build_chain_journeys({"a": rows_a, "b": rows_b}, [("a", "b")], **options),
        key=lambda journey: journey["legs"][0]["enter_time"],
    )


def assign_pairs(
    rows_a, rows_b, *, min_travel_sec=MIN_TRAVEL_SEC,
    max_travel_sec=MAX_TRAVEL_SEC, min_similarity=0.5,
) -> Dict[str, dict]:
    stitcher = CrossCameraStitcher(
        similarity_threshold=min_similarity, default_min_travel_sec=min_travel_sec,
        default_max_travel_sec=max_travel_sec,
    )
    return {
        match["from_key"]: {
            "to": match["to_key"], "similarity": match["similarity"],
            "gap": match["travel_sec"], "match_method": match["match_method"],
        }
        for match in stitcher.match(rows_a, rows_b, "a", "b")
    }


def build_chain_journeys(
    rows_by_camera: Dict[str, List[dict]], edges: List[tuple], *,
    min_travel_sec: float = MIN_TRAVEL_SEC, max_travel_sec: float = MAX_TRAVEL_SEC,
    min_similarity: float = 0.5, ambiguity_margin: float = 0.0,
) -> List[dict]:
    stitcher = CrossCameraStitcher(
        similarity_threshold=min_similarity, default_min_travel_sec=min_travel_sec,
        default_max_travel_sec=max_travel_sec, ambiguity_margin=ambiguity_margin,
    )
    by_id = {
        stitcher.passage_key(camera, row): (camera, row)
        for camera, rows in rows_by_camera.items() for row in rows
    }
    matches = stitcher.select_chain_matches(rows_by_camera, [
        match for a, b in edges
        for match in stitcher.match(rows_by_camera.get(a, []), rows_by_camera.get(b, []), a, b)
    ])
    successors = {match["from_key"]: match for match in matches}
    predecessors = {match["to_key"] for match in matches}
    journeys = []
    for start in successors.keys() - predecessors:
        keys = [start]
        steps = []
        while keys[-1] in successors:
            step = successors[keys[-1]]
            keys.append(step["to_key"])
            steps.append(step)
        legs = []
        for key in keys:
            camera, row = by_id[key]
            legs.append({
                "camera": row.get("camera_id") or camera,
                "passage_id": row.get("passage_id") or key,
                "enter_time": row.get("enter_time", ""),
                "exit_time": row.get("exit_time", ""),
            })
        similarities = [step["similarity"] for step in steps if step["match_method"] == "reid"]
        journeys.append({
            "travel_sec": round(sum(step["travel_sec"] for step in steps), 1),
            "similarity": round(sum(similarities) / len(similarities), 4) if similarities else 0.0,
            "min_similarity": min(similarities, default=0.0),
            "match_methods": [step["match_method"] for step in steps],
            "legs": legs,
        })
    journeys.sort(
        key=lambda journey: (journey["legs"][0]["enter_time"], journey["legs"][0]["passage_id"]),
        reverse=True,
    )
    for index, journey in enumerate(journeys, start=1):
        journey["global_id"] = f"G{index:04d}"
    return journeys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-a", required=True, help="上游机位向量目录")
    parser.add_argument("--in-b", required=True, help="下游机位向量目录")
    parser.add_argument("--output", required=True, help="输出 journeys.json")
    parser.add_argument("--min-similarity", type=float, default=0.78)
    parser.add_argument("--ambiguity-margin", type=float, default=0.035)
    parser.add_argument("--min-travel-sec", type=float, default=MIN_TRAVEL_SEC)
    parser.add_argument("--max-travel-sec", type=float, default=MAX_TRAVEL_SEC)
    args = parser.parse_args()

    rows_a = load_rows(args.in_a)
    rows_b = load_rows(args.in_b)
    journeys = build_journeys(
        rows_a, rows_b,
        min_travel_sec=args.min_travel_sec,
        max_travel_sec=args.max_travel_sec,
        min_similarity=args.min_similarity, ambiguity_margin=args.ambiguity_margin,
    )

    payload = {
        "cameras": {"a": len(rows_a), "b": len(rows_b)},
        "journey_count": len(journeys),
        "params": {
            "min_similarity": args.min_similarity,
            "ambiguity_margin": args.ambiguity_margin,
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
