#!/usr/bin/env python
"""静态校验人工标定结果：ROI 形状、出入口线位置是否会导致丢车。

不跑录像、不做跟踪，只用几何关系就能发现几类常见错误：

1. ROI 自相交——点选顺序错乱，pointPolygonTest 结果不可靠；
2. ROI 面积过小/过大——小于画面 1% 基本圈不到车，大于 95% 等于没圈；
3. 出入口线贴着画面边缘——车刚出现就跨线，轨迹不足 2 点，跨线事件被漏掉，
   最终"只进不出"整条被丢弃；
4. 线太短——没有横跨 ROI，部分车道的车永远不跨线；
5. 线落在 ROI 之外——跨线时车被 ROI 过滤掉，同样丢车；
6. 入口线与出口线重合或平行且极近——几乎不可能先跨一条再跨另一条。

用法：
    python scripts/validate_camera_marks.py --marks web/roi_editor/roi_marks.json \
        --manifest data/outputs/roi_frames/manifest.json --output data/outputs/marks_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import List, Tuple

EDGE_MARGIN = 0.04      # 线端点距画面边缘小于画面尺寸 4% 视为「贴边」
MIN_ROI_RATIO = 0.01
MAX_ROI_RATIO = 0.98
MIN_LINE_LEN_PX = 40


def _segments_cross(p1, p2, p3, p4) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p2, p3) * ccw(p1, p2, p4) < 0) and (ccw(p3, p4, p1) * ccw(p3, p4, p2) < 0)


def self_intersects(poly: List[List[float]]) -> bool:
    n = len(poly)
    if n < 4:
        return False
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                return True
    return False


def polygon_area(poly: List[List[float]]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


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


def validate_camera(entry: dict, camera: dict) -> List[str]:
    warnings = []
    width, height = camera.get("width") or 0, camera.get("height") or 0
    if not width or not height:
        return ["没有参考帧尺寸，无法校验"]

    roi = entry.get("roi") or []
    if len(roi) < 3:
        warnings.append("ROI 少于 3 点")
    else:
        if self_intersects(roi):
            warnings.append("ROI 自相交（顶点顺序乱了，建议重画）")
        ratio = polygon_area(roi) / float(width * height)
        if ratio < MIN_ROI_RATIO:
            warnings.append(f"ROI 面积过小（占画面 {ratio:.1%}）")
        elif ratio > MAX_ROI_RATIO:
            warnings.append(f"ROI 几乎覆盖全画面（{ratio:.1%}）")

    lines = []
    for key, label in (("entry_lines", "入口线"), ("entry_line", "入口线"),
                       ("exit_lines", "出口线"), ("exit_line", "出口线")):
        for line in entry.get(key) or []:
            # 两种格式都支持：{"p1": [...], "p2": [...]} 或 [[x1,y1], [x2,y2]]
            if isinstance(line, dict) and line.get("p1") and line.get("p2"):
                lines.append((label, line["p1"], line["p2"]))
            elif isinstance(line, (list, tuple)) and len(line) == 2:
                lines.append((label, line[0], line[1]))

    if not lines:
        warnings.append("没有出入口线（只能按轨迹起止统计）")
        return warnings

    margin_x, margin_y = width * EDGE_MARGIN, height * EDGE_MARGIN
    for label, p1, p2 in lines:
        for point in (p1, p2):
            x, y = point
            if x < margin_x or x > width - margin_x or y < margin_y or y > height - margin_y:
                warnings.append(
                    f"{label}端点 ({int(x)},{int(y)}) 贴着画面边缘，"
                    "车辆会在轨迹不足 2 点时跨线导致漏判；建议往画面内移 10–15%")
                break
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length < MIN_LINE_LEN_PX:
            warnings.append(f"{label}太短（{length:.0f} px），可能没横跨车道")
        if len(roi) >= 3:
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            if not point_in_polygon(mid, roi):
                warnings.append(f"{label}中点不在 ROI 内，跨线时车辆可能已被 ROI 过滤")

    entries = [item for item in lines if item[0] == "入口线"]
    exits = [item for item in lines if item[0] == "出口线"]
    if entries and exits:
        for _, e1, e2 in entries:
            for _, x1, x2 in exits:
                em = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
                xm = ((x1[0] + x2[0]) / 2, (x1[1] + x2[1]) / 2)
                if math.hypot(em[0] - xm[0], em[1] - xm[1]) < 20:
                    warnings.append("入口线与出口线几乎重合，无法先跨一条再跨另一条")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marks", required=True)
    parser.add_argument("--manifest", default="data/outputs/roi_frames/manifest.json")
    parser.add_argument("--output", default="data/outputs/marks_validation.json")
    args = parser.parse_args()

    marks = json.load(open(args.marks, encoding="utf-8"))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    sizes = {item["camera"]: item for item in manifest.get("cameras", [])}

    results = []
    for entry in marks.get("cameras", []):
        camera = entry.get("camera")
        warnings = validate_camera(entry, sizes.get(camera, {}))
        results.append({"camera": camera, "warnings": warnings, "ok": not warnings})

    ok = sum(1 for item in results if item["ok"])
    summary = {
        "cameras": len(results),
        "ok": ok,
        "with_warnings": len(results) - ok,
    }
    json.dump({"summary": summary, "results": results}, open(args.output, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))

    from collections import Counter
    counter = Counter()
    for item in results:
        for warning in item["warnings"]:
            counter[warning.split("，")[0].split("（")[0]] += 1
    for warning, count in counter.most_common(8):
        print(f"  {count:3d} 台：{warning}")


if __name__ == "__main__":
    main()
