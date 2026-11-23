#!/usr/bin/env python
"""由边表 CSV 生成 config/site/topology.yaml，并做连通性检查。

为什么用 CSV 而不是直接手写 YAML：
    拓扑的难点是"哪些机位相邻、车要开多久"这类领域知识，不是格式。
    CSV 可以在 Excel 里填、可以逐条 review、可以 diff，写完跑本脚本生成 YAML。

时间窗的推法（当 CSV 里没填 min/max 时）：
    min_travel_sec = distance_m / max_speed_kmh   （全速通过，留 0.8 的安全系数）
    max_travel_sec = distance_m / min_speed_kmh   （含场区内停留，再乘冗余）
    两侧上下限都取整到秒，并强制 max >= min * 2，避免"过窄时间窗把真实链路剪掉"。

用法：
    # 生成
    python scripts/build_topology.py \
        --edges config/site/topology_edges.csv \
        --cameras config/site/cameras.yaml \
        --output config/site/topology.yaml

    # 只做检查，不写文件
    python scripts/build_topology.py --edges ... --check
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 用于推算时间窗的默认车速（km/h）。闸口/场区内部车速很低。
DEFAULT_MAX_SPEED_KMH = 20.0
DEFAULT_MIN_SPEED_KMH = 3.0
MIN_WINDOW_FLOOR_SEC = 20
MAX_WINDOW_CEIL_SEC = 12 * 3600


def load_camera_ids(path: Optional[str]) -> set:
    """读取 cameras.yaml 里的 camera_id；文件缺失时返回空集合（跳过校验）。"""
    if not path or not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {str(item.get("camera_id")) for item in data.get("cameras", []) if item.get("camera_id")}


def parse_float(value: str) -> Optional[float]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def derive_window(
    distance_m: Optional[float],
    min_travel_sec: Optional[float],
    max_travel_sec: Optional[float],
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    min_speed_kmh: float = DEFAULT_MIN_SPEED_KMH,
) -> tuple:
    """返回 (min_sec, max_sec)。缺失值按距离与车速推算。"""
    low, high = min_travel_sec, max_travel_sec

    if low is None and distance_m:
        low = distance_m / (max_speed_kmh / 3.6) * 0.8
    if high is None and distance_m:
        high = distance_m / (min_speed_kmh / 3.6) * 1.5

    low = MIN_WINDOW_FLOOR_SEC if low is None else max(low, MIN_WINDOW_FLOOR_SEC)
    high = MAX_WINDOW_CEIL_SEC if high is None else min(high, MAX_WINDOW_CEIL_SEC)
    if high < low * 2:
        high = low * 2
    return int(round(low)), int(round(high))


def load_edges(path: str) -> List[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if (row.get("from_camera_id") or "").strip()]


def build_topology(rows: List[dict], camera_ids: set) -> tuple:
    """把边表转成 topology.yaml 结构，同时返回问题清单。"""
    edges: List[dict] = []
    problems: List[str] = []
    seen = set()

    for row in rows:
        source = row["from_camera_id"].strip()
        target = (row.get("to_camera_id") or "").strip()
        if not target:
            problems.append(f"{source}: to_camera_id 为空，跳过")
            continue
        key = (source, target)
        if key in seen:
            problems.append(f"{source}->{target}: 重复边，已跳过")
            continue
        seen.add(key)

        for camera in (source, target):
            if camera_ids and camera not in camera_ids:
                problems.append(f"{camera}: 不在 cameras.yaml 里")

        distance = parse_float(row.get("distance_m"))
        low = parse_float(row.get("min_travel_sec"))
        high = parse_float(row.get("max_travel_sec"))
        low, high = derive_window(distance, low, high)

        edge = {
            "from_camera_id": source,
            "to_camera_id": target,
            "min_travel_sec": low,
            "max_travel_sec": high,
        }
        if distance is not None:
            edge["distance_m"] = distance
        route_id = (row.get("route_id") or "").strip()
        if route_id:
            edge["route_id"] = route_id
        if (row.get("one_way") or "").strip().lower() in {"true", "1", "yes"}:
            edge["one_way"] = True
        notes = (row.get("notes") or "").strip()
        if notes:
            edge["notes"] = notes
        edges.append(edge)

    return edges, problems


def connectivity_report(edges: List[dict], camera_ids: set) -> dict:
    """检查孤立机位、双向缺口、以及从入口出发的可达性。"""
    forward = defaultdict(set)
    backward = defaultdict(set)
    for edge in edges:
        forward[edge["from_camera_id"]].add(edge["to_camera_id"])
        backward[edge["to_camera_id"]].add(edge["from_camera_id"])

    nodes = set(camera_ids) | set(forward) | set(backward)
    isolated = sorted(node for node in nodes if not forward[node] and not backward[node])

    # 有向边的反向缺口：A->B 存在但 B->A 不存在（单向通道是合理的，只作提示）
    one_way_gaps = sorted(
        f"{a}->{b}" for a in forward for b in forward[a] if a not in forward.get(b, set())
    )

    sources = sorted(node for node in nodes if not backward[node] and forward[node])
    sinks = sorted(node for node in nodes if forward[node] is not None and not forward[node] and backward[node])

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "isolated": isolated,
        "sources": sources,
        "sinks": sinks,
        "one_way_gaps": one_way_gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", default="config/site/topology_edges.csv")
    parser.add_argument("--cameras", default="config/site/cameras.yaml")
    parser.add_argument("--output", default="config/site/topology.yaml")
    parser.add_argument("--check", action="store_true", help="只检查，不写文件")
    args = parser.parse_args()

    camera_ids = load_camera_ids(args.cameras)
    rows = load_edges(args.edges)
    edges, problems = build_topology(rows, camera_ids)
    report = connectivity_report(edges, camera_ids)

    print(yaml.safe_dump({"connectivity": report}, allow_unicode=True, sort_keys=False))
    for problem in problems:
        print(f"[问题] {problem}")

    if args.check:
        return

    payload = {
        "version": 1,
        "generated_by": "scripts/build_topology.py",
        "edges": edges,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# 由 config/site/topology_edges.csv 生成，请勿手改（改 CSV 后重跑脚本）\n")
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    print(f"已写入 {args.output}：{len(edges)} 条边")


if __name__ == "__main__":
    main()
