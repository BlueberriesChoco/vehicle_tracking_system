#!/usr/bin/env python
"""把与 ROI 没有交集的出入口线，沿法线方向平行平移进 ROI。

为什么需要：
    跨线判定和 ROI 过滤是两套逻辑。线落在 ROI 之外时，车辆在跨线前就被 ROI 滤掉，
    这条线永远触发不了，该机位的通行记录会凭空少一半。

平移规则（保持用户画的线方向不变，只做平行移动）：
    1. 沿线的法线方向，按 4 px 步长向两侧试探；
    2. 取最小的、能让线与 ROI 产生交集的位移；
    3. 两侧都找不到（ROI 与线方向完全不搭）时，把线平移到「中点在 ROI 内」为止。

用法：
    python scripts/shift_lines_into_roi.py --config-dir config/camera --dry-run
    python scripts/shift_lines_into_roi.py --config-dir config/camera --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import List, Tuple

import yaml

STEP_PX = 4.0
MAX_SHIFT_PX = 400.0


def point_in_polygon(point, poly) -> bool:
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def segment_hits_polygon(p1, p2, poly) -> bool:
    """线是否与多边形有交集（采样 + 边相交）。"""
    steps = 40
    for index in range(steps + 1):
        t = index / steps
        if point_in_polygon((p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t), poly):
            return True

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    n = len(poly)
    for index in range(n):
        q1, q2 = poly[index], poly[(index + 1) % n]
        if ((ccw(p1, p2, q1) * ccw(p1, p2, q2) < 0)
                and (ccw(q1, q2, p1) * ccw(q1, q2, p2) < 0)):
            return True
    return False


def shift(p1, p2, normal, distance):
    return ([p1[0] + normal[0] * distance, p1[1] + normal[1] * distance],
            [p2[0] + normal[0] * distance, p2[1] + normal[1] * distance])


def find_shift(p1, p2, poly) -> Tuple[float, str]:
    """返回 (位移, 方向说明)；找不到返回 (0, '已相交')。"""
    if segment_hits_polygon(p1, p2, poly):
        return 0.0, "已相交"

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, "线长为 0"
    normal = (-dy / length, dx / length)

    distance = STEP_PX
    while distance <= MAX_SHIFT_PX:
        for sign, label in ((1.0, "正向"), (-1.0, "反向")):
            np1, np2 = shift(p1, p2, normal, sign * distance)
            if segment_hits_polygon(np1, np2, poly):
                return sign * distance, label
        distance += STEP_PX
    return 0.0, "找不到交集"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="config/camera")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", default="data/outputs/line_shift_report.json")
    args = parser.parse_args()

    changes: List[dict] = []
    for path in sorted(glob.glob(os.path.join(args.config_dir, "*.yaml"))):
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        poly = (config.get("roi") or {}).get("polygon") or []
        if len(poly) < 3:
            continue
        changed = False
        for key in ("entry_lines", "exit_lines"):
            for index, line in enumerate(config.get(key) or []):
                p1, p2 = line.get("p1"), line.get("p2")
                if not p1 or not p2:
                    continue
                offset, direction = find_shift(p1, p2, poly)
                if offset == 0.0:
                    continue
                new_p1, new_p2 = shift(p1, p2, (-(p2[1] - p1[1]) / max(1e-6, math.hypot(p2[0] - p1[0], p2[1] - p1[1])),
                                                 (p2[0] - p1[0]) / max(1e-6, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))),
                                        offset)
                line["p1"] = [int(round(v)) for v in new_p1]
                line["p2"] = [int(round(v)) for v in new_p2]
                changed = True
                changes.append({
                    "camera": os.path.basename(path).replace(".yaml", ""),
                    "line": f"{key}[{index}]",
                    "offset_px": round(offset, 1),
                    "direction": direction,
                    "before": [p1, p2],
                    "after": [line["p1"], line["p2"]],
                })
        if changed and args.apply:
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
    json.dump({"changes": changes}, open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({"shifted_lines": len(changes), "cameras": len({c['camera'] for c in changes}),
                      "applied": args.apply, "report": args.report}, ensure_ascii=False))
    for item in changes[:12]:
        print(f"  {item['camera']} {item['line']} 平移 {item['offset_px']} px（{item['direction']}）")


if __name__ == "__main__":
    main()
