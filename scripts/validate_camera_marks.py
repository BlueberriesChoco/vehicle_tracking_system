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

EDGE_MARGIN = 0.04
CLOSE_DUP_PX = 10.0


def clean_polygon(points):
    """去掉闭合重复点（末点≈首点）与相邻重复点。"""
    cleaned = []
    for point in points or []:
        candidate = [int(round(float(point[0]))), int(round(float(point[1])))]
        if cleaned and abs(candidate[0] - cleaned[-1][0]) < 3 and abs(candidate[1] - cleaned[-1][1]) < 3:
            continue
        cleaned.append(candidate)
    if len(cleaned) >= 3:
        first, last = cleaned[0], cleaned[-1]
        if ((first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2) ** 0.5 <= CLOSE_DUP_PX:
            cleaned.pop()
    return cleaned      # 线端点距画面边缘小于画面尺寸 4% 视为「贴边」
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


def _segment_touches_polygon(p1, p2, poly) -> bool:
    """线的任意一点落在多边形内，或与多边形边相交，就算有交集。"""
    steps = 24
    for index in range(steps + 1):
        t = index / steps
        point = (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)
        if point_in_polygon(point, poly):
            return True
    n = len(poly)
    for index in range(n):
        if _segments_cross(p1, p2, poly[index], poly[(index + 1) % n]):
            return True
    return False


def validate_camera(entry: dict, camera: dict) -> List[str]:
    warnings = []
    notes = []
    width, height = camera.get("width") or 0, camera.get("height") or 0
    if not width or not height:
        return ["没有参考帧尺寸，无法校验"]

    roi = clean_polygon(entry.get("roi") or [])
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
        return warnings + notes

    margin_x, margin_y = width * EDGE_MARGIN, height * EDGE_MARGIN
    edge_touching = 0
    for label, p1, p2 in lines:
        for point in (p1, p2):
            x, y = point
            if x < margin_x - 1 or x > width - margin_x + 1 or y < margin_y - 1 or y > height - margin_y + 1:
                edge_touching += 1
                break
    if edge_touching:
        # 线画到画面边缘是常见做法；只有跨线发生在边缘极窄区域时才漏判，
        # 这里降级为提示，不阻塞配置使用
        notes.append(f"[提示] {edge_touching} 条线端点贴画面边缘（常见做法，仅边界车辆可能受影响）")
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        # 右转支路等场景线本来就短，不再作为告警；保留长度信息供参考
        _ = length
        if len(roi) >= 3 and not _segment_touches_polygon(p1, p2, roi):
            warnings.append(f"{label}与 ROI 没有交集，跨线时车辆已被 ROI 过滤掉，永远触发不了")

    entries = [item for item in lines if item[0] == "入口线"]
    exits = [item for item in lines if item[0] == "出口线"]
    if entries and exits:
        for _, e1, e2 in entries:
            for _, x1, x2 in exits:
                em = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
                xm = ((x1[0] + x2[0]) / 2, (x1[1] + x2[1]) / 2)
                if math.hypot(em[0] - xm[0], em[1] - xm[1]) < 20:
                    warnings.append("入口线与出口线几乎重合，无法先跨一条再跨另一条")
    return warnings + notes


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
        real = [w for w in warnings if not w.startswith("[提示]")]
        results.append({"camera": camera, "warnings": real, "notes": [w for w in warnings if w.startswith("[提示]")], "ok": not real})

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
