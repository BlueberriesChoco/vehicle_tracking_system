#!/usr/bin/env python
"""把多份 camera_layout*.json 合并成一份（机位按 id 去重、道路累积）。

落图工具每次导出只包含"当前会话"的内容：有人一次只标机位、另一次只画道路，
所以需要把多份文件拼起来。

去重规则：
    - 机位按 id 去重，位置以后出现的文件为准（后导出的通常是最新修正）；
    - 位置冲突（同一 id 两次坐标差 > 2 px）会打印出来，便于确认是否误标；
    - 道路按"顶点序列去重"累积，完全相同的道路只保留一份。

用法：
    python scripts/merge_camera_layouts.py \
        --inputs "web/camera_placer/camera_layout.json" "web/camera_placer/camera_layout (4).json" \
        --output web/camera_placer/camera_layout_merged.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFLICT_PX = 2.0


def _road_key(road: List[dict]) -> Tuple:
    return tuple((int(round(p["x"])), int(round(p["y"]))) for p in road if "x" in p and "y" in p)


def merge_layouts(layouts: List[dict]) -> Tuple[dict, List[str]]:
    """返回 (合并后的 layout, 冲突/提示信息)。"""
    cameras: Dict[str, dict] = {}
    conflicts: List[str] = []
    roads: List[List[dict]] = []
    road_keys = set()
    map_meta = {}

    for layout in layouts:
        map_meta = layout.get("map", map_meta)
        for camera in layout.get("cameras", []):
            cid = camera.get("id")
            if not cid:
                continue
            previous = cameras.get(cid)
            if previous:
                dx = float(camera["x"]) - float(previous["x"])
                dy = float(camera["y"]) - float(previous["y"])
                if (dx * dx + dy * dy) ** 0.5 > CONFLICT_PX:
                    conflicts.append(
                        f"{cid}: ({previous['x']}, {previous['y']}) -> ({camera['x']}, {camera['y']})，"
                        f"位置冲突 {((dx*dx+dy*dy) ** 0.5):.0f} px，以后者为准"
                    )
            cameras[cid] = {
                "id": cid,
                "name": camera.get("name", cid),
                "x": int(round(float(camera["x"]))),
                "y": int(round(float(camera["y"]))),
            }
        for road in layout.get("roads", []):
            key = _road_key(road)
            if len(key) < 2 or key in road_keys:
                continue
            road_keys.add(key)
            roads.append([{"x": x, "y": y} for x, y in key])

    merged = {
        "map": map_meta or {"width": 1280, "height": 1809, "source": "zone-a-base-map.jpg"},
        "cameras": sorted(cameras.values(), key=lambda item: item["id"]),
        "roads": roads,
    }
    return merged, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    layouts = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as handle:
            layouts.append(json.load(handle))

    merged, conflicts = merge_layouts(layouts)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)

    for item in conflicts:
        print(f"[冲突] {item}")
    print(json.dumps({
        "cameras": len(merged["cameras"]),
        "roads": len(merged["roads"]),
        "conflicts": len(conflicts),
        "output": args.output,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
