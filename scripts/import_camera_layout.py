#!/usr/bin/env python
"""把 web/camera_placer 导出的 camera_layout.json 转成机位配置与拓扑边表。

做三件事：
    1. 生成 ``config/site/cameras.yaml``（机位 id / 名称 / 图上坐标）
    2. 沿每条道路把机位按行进顺序串起来，生成 ``config/site/topology_edges.csv`` 初稿
    3. 距离由道路弧长 × 米/像素 推算，时间窗交给 build_topology.py 补

注意：米/像素必须由你给（园区图没有真实比例尺）。两种给法：
    --meters-per-pixel 0.5          # 直接给比例
    --map-width-m 400               # 给整图宽度对应的真实米数

用法：
    python scripts/import_camera_layout.py \
        --layout camera_layout.json \
        --meters-per-pixel 0.5 \
        --cameras config/site/cameras.yaml \
        --edges config/site/topology_edges.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_layout(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def project_on_polyline(point: Tuple[float, float], polyline: List[Tuple[float, float]]) -> Tuple[float, float]:
    """返回 (弧长位置, 到折线距离)。"""
    best = (0.0, float("inf"))
    travelled = 0.0
    for index in range(1, len(polyline)):
        x1, y1 = polyline[index - 1]
        x2, y2 = polyline[index]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        t = ((point[0] - x1) * dx + (point[1] - y1) * dy) / (length * length)
        t = max(0.0, min(1.0, t))
        px, py = x1 + t * dx, y1 + t * dy
        distance = math.hypot(point[0] - px, point[1] - py)
        if distance < best[1]:
            best = (travelled + t * length, distance)
        travelled += length
    return best


def chain_road(
    cameras: List[dict],
    polyline: List[Tuple[float, float]],
    meters_per_pixel: float,
    max_offset_px: float,
) -> List[dict]:
    """把落在一条道路上的机位按行进顺序串成链，返回相邻边。"""
    entries = []
    for camera in cameras:
        arc, offset = project_on_polyline((camera["x"], camera["y"]), polyline)
        if offset <= max_offset_px:
            entries.append((arc, camera))
    entries.sort(key=lambda item: item[0])

    edges = []
    for index in range(1, len(entries)):
        arc_a, camera_a = entries[index - 1]
        arc_b, camera_b = entries[index]
        distance = (arc_b - arc_a) * meters_per_pixel
        if distance <= 0:
            continue
        edges.append({
            "from_camera_id": camera_a["id"],
            "to_camera_id": camera_b["id"],
            "distance_m": round(distance, 1),
        })
    return edges


def build(layout: dict, meters_per_pixel: float, max_offset_px: float = 60.0) -> Tuple[List[dict], List[dict]]:
    cameras = [
        {"id": item["id"], "name": item.get("name", item["id"]), "x": float(item["x"]), "y": float(item["y"])}
        for item in layout.get("cameras", [])
        if item.get("id")
    ]

    edges: List[dict] = []
    seen = set()
    for road in layout.get("roads", []):
        polyline = [(float(p["x"]), float(p["y"])) for p in road if "x" in p and "y" in p]
        if len(polyline) < 2:
            continue
        for edge in chain_road(cameras, polyline, meters_per_pixel, max_offset_px):
            key = (edge["from_camera_id"], edge["to_camera_id"])
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)
            # 双向补边：道路通常可双向通行
            reverse = (key[1], key[0])
            if reverse not in seen:
                seen.add(reverse)
                edges.append({
                    "from_camera_id": key[1],
                    "to_camera_id": key[0],
                    "distance_m": edge["distance_m"],
                })
    return cameras, edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", required=True, help="camera_layout.json")
    parser.add_argument("--meters-per-pixel", type=float, default=0.0)
    parser.add_argument("--map-width-m", type=float, default=0.0, help="整图宽度对应的真实米数")
    parser.add_argument("--cameras", default="config/site/cameras.yaml")
    parser.add_argument("--edges", default="config/site/topology_edges.csv")
    parser.add_argument("--max-offset-px", type=float, default=60.0, help="机位离道路多远仍算在这条路上")
    args = parser.parse_args()

    layout = load_layout(args.layout)
    meters_per_pixel = args.meters_per_pixel
    if meters_per_pixel <= 0 and args.map_width_m > 0:
        width = float(layout.get("map", {}).get("width", 1280))
        meters_per_pixel = args.map_width_m / width
    if meters_per_pixel <= 0:
        raise SystemExit("必须给 --meters-per-pixel 或 --map-width-m（园区图没有内置比例尺）")

    cameras, edges = build(layout, meters_per_pixel, args.max_offset_px)

    os.makedirs(os.path.dirname(os.path.abspath(args.cameras)) or ".", exist_ok=True)
    with open(args.cameras, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"cameras": [
                {"camera_id": c["id"], "camera_name": c["name"],
                 "map_xy": [int(round(c["x"])), int(round(c["y"]))]}
                for c in cameras
            ]},
            handle, allow_unicode=True, sort_keys=False,
        )

    with open(args.edges, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["from_camera_id", "to_camera_id", "distance_m", "min_travel_sec",
                        "max_travel_sec", "route_id", "one_way", "notes"],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow({**edge, "min_travel_sec": "", "max_travel_sec": "",
                             "route_id": "", "one_way": "false", "notes": "由道路链自动生成"})

    print(json.dumps({
        "cameras": len(cameras),
        "edges": len(edges),
        "meters_per_pixel": round(meters_per_pixel, 4),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
