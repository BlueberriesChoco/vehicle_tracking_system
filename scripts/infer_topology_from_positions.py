#!/usr/bin/env python
"""在只有机位坐标、没有道路折线时，按几何邻近关系推断一版拓扑草稿。

用途：
    落图工具里机位点好了但没画道路，可以先由本脚本给出一版"谁挨着谁"的边表，
    人工把明显错的边删掉，比从零画道路省时间。

规则（刻意保守）：
    1. 先按名称前缀分组（"围网41"->"围网"、"卡口1"->"卡口"），组内优先连；
    2. 组内按最近邻串链：每个机位只连"最近的、还没被连过"的那个，形成链而非网状；
    3. 跨组只在距离极近（< cross_threshold）时连，且标记为待复核。

产出：
    - 边表 CSV（可直接交给 build_topology.py）
    - 叠加图 PNG（机位点 + 推断的边），用于人工复核

用法：
    python scripts/infer_topology_from_positions.py \
        --layout camera_layout.json \
        --edges config/site/topology_edges.csv \
        --overlay data/outputs/probe/topology_draft.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def load_aliases(path: str) -> dict:
    """录像目录名 -> 管线 camera_id（如 行政车道进 -> admin_in）。"""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("folder_name") or "").strip(): (row.get("camera_id") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("folder_name") or "").strip() and (row.get("camera_id") or "").strip()
        }

def name_group(camera_id: str) -> str:
    """从机位名里取"路名/设施名"前缀，如 围网41 -> 围网，卡口1 -> 卡口。"""
    match = re.match(r"^(.*?)[0-9]+$", camera_id)
    return match.group(1) if match and match.group(1) else camera_id


def infer_edges(
    cameras: List[dict],
    *,
    same_group_threshold_px: float = 80.0,
    cross_group_threshold_px: float = 30.0,
) -> List[dict]:
    """返回边列表（无向，双向各一条）。"""
    groups: Dict[str, List[dict]] = {}
    for camera in cameras:
        groups.setdefault(name_group(camera["id"]), []).append(camera)

    edges: List[dict] = []
    seen = set()

    def add(a: dict, b: dict, distance_px: float, note: str) -> None:
        key = (a["id"], b["id"])
        if key in seen or a["id"] == b["id"]:
            return
        seen.add(key)
        for source, target in ((a, b), (b, a)):
            edges.append({
                "from_camera_id": source["id"],
                "to_camera_id": target["id"],
                "distance_px": round(distance_px, 1),
                "note": note,
            })

    # 组内：最近邻串链
    for members in groups.values():
        if len(members) < 2:
            continue
        connected = {members[0]["id"]}
        remaining = list(members[1:])
        while remaining:
            best = None
            for camera in remaining:
                for anchor_id in connected:
                    anchor = next(item for item in members if item["id"] == anchor_id)
                    distance = math.hypot(camera["x"] - anchor["x"], camera["y"] - anchor["y"])
                    if distance <= same_group_threshold_px and (best is None or distance < best[0]):
                        best = (distance, anchor, camera)
            if best is None:
                # 遇到断点：从剩余机位里另起一条链，而不是整体停止
                connected = {remaining[0]["id"]}
                remaining = remaining[1:]
                continue
            distance, anchor, camera = best
            add(anchor, camera, distance, "同组最近邻")
            connected.add(camera["id"])
            remaining.remove(camera)

    # 跨组：只连极近的点
    for index, camera_a in enumerate(cameras):
        for camera_b in cameras[index + 1:]:
            if name_group(camera_a["id"]) == name_group(camera_b["id"]):
                continue
            distance = math.hypot(camera_a["x"] - camera_b["x"], camera_a["y"] - camera_b["y"])
            if distance <= cross_group_threshold_px:
                add(camera_a, camera_b, distance, "跨组近邻(待复核)")

    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--overlay", default="")
    parser.add_argument("--same-group-px", type=float, default=80.0)
    parser.add_argument("--cross-group-px", type=float, default=30.0)
    parser.add_argument("--aliases", default="config/site/camera_aliases.csv")
    parser.add_argument("--meters-per-pixel", type=float, default=0.0, help="图上 1 px 对应多少米；给了才填 distance_m")
    args = parser.parse_args()

    with open(args.layout, encoding="utf-8") as handle:
        layout = json.load(handle)
    aliases = load_aliases(args.aliases)
    cameras = [c for c in layout.get("cameras", []) if c.get("id")]
    for camera in cameras:
        camera["id"] = aliases.get(camera["id"], camera["id"])
        camera["name"] = camera.get("name", camera["id"])

    edges = infer_edges(
        cameras,
        same_group_threshold_px=args.same_group_px,
        cross_group_threshold_px=args.cross_group_px,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.edges)) or ".", exist_ok=True)
    with open(args.edges, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["from_camera_id", "to_camera_id", "distance_m", "min_travel_sec",
                        "max_travel_sec", "route_id", "one_way", "notes"],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow({
                "from_camera_id": edge["from_camera_id"],
                "to_camera_id": edge["to_camera_id"],
                "distance_m": round(edge["distance_px"] * args.meters_per_pixel, 1) if args.meters_per_pixel > 0 else "",
                "min_travel_sec": "",
                "max_travel_sec": "",
                "route_id": "",
                "one_way": "false",
                "notes": f"{edge['note']}；图上 {edge['distance_px']} px，米制待比例尺",
            })

    print(json.dumps({
        "cameras": len(cameras),
        "edges": len(edges),
        "groups": len({name_group(c["id"]) for c in cameras}),
    }, ensure_ascii=False))

    if args.overlay:
        import cv2
        import numpy as np

        base = cv2.imread(str(layout.get("map", {}).get("source", "")) and
                          os.path.join("web/map_editor/assets", "zone-a-base-map.jpg"))
        if base is not None:
            for edge in edges:
                a = next(c for c in cameras if c["id"] == edge["from_camera_id"])
                b = next(c for c in cameras if c["id"] == edge["to_camera_id"])
                color = (0, 200, 255) if "待复核" in edge["note"] else (60, 200, 60)
                cv2.line(base, (int(a["x"]), int(a["y"])), (int(b["x"]), int(b["y"])), color, 2)
            for camera in cameras:
                cv2.circle(base, (int(camera["x"]), int(camera["y"])), 5, (30, 60, 220), -1)
                cv2.putText(base, camera["id"], (int(camera["x"]) + 6, int(camera["y"]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
            os.makedirs(os.path.dirname(os.path.abspath(args.overlay)) or ".", exist_ok=True)
            ok, buf = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                buf.tofile(args.overlay)
                print(json.dumps({"overlay": args.overlay}, ensure_ascii=False))


if __name__ == "__main__":
    main()
