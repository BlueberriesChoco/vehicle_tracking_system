#!/usr/bin/env python
"""从一段录像自动推导机位配置（ROI + 出入口线），供人工复核后写入 config/camera/。

为什么需要它：
    142 个机位里只有 2 个有配置。逐个手画 ROI 和出入口线，按 10 分钟/个算要 24 小时。
    这里用「车辆检测框的落地点聚类」自动给出初稿，人工只做微调。

推导逻辑：
    1. 均匀抽帧，跑 YOLO，只保留车辆类别的框；
    2. 取框底边中点（车辆接地位置）作为通行点；
    3. 通行点的凸包 + 外扩 → ROI（保证车体不贴边）；
    4. 对通行点做主方向（PCA），沿主方向取 8% / 92% 分位，作两条垂直于车流方向的出入口线。

用法：
    python scripts/derive_camera_config.py \
        --video data/outputs/h264/admin_out_xxx.h264 \
        --camera admin_out --name "综保A区行政车道出" \
        --output config/camera/admin_out_derived.yaml --overlay data/outputs/probe/derived.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_points(video: str, samples: int = 240, conf: float = 0.35) -> tuple:
    """均匀抽帧，返回 (框底边中点数组, 帧宽, 帧高, 采样帧数)。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    from src.detection.detector import VehicleDetector

    detector = VehicleDetector(model_path="models/yolov8s.pt")
    points = []
    step = max(1, total // samples)
    read = 0
    for frame_index in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        for item in detector.detect(frame):
            box = item.get("bbox")
            if not box:
                continue
            x1, y1, x2, y2 = box
            width_px = x2 - x1
            height_px = y2 - y1
            if width_px < 70 or height_px < 55:
                continue
            points.append(((x1 + x2) / 2.0, y2))
    cap.release()
    return np.asarray(points, dtype=np.float64), width, height, read


def simplify_polygon(points: np.ndarray, width: int, height: int, tolerance: float = 0.01) -> list:
    """把点集变成外扩后的凸包多边形（最多 10 个顶点，坐标取整）。"""
    if len(points) < 8:
        raise SystemExit("可用通行点太少，无法推导 ROI；请换一段有车的录像")
    # 密度过滤：剔除远离主通道的离群点（远处主路/路面误检）
    center = np.median(points, axis=0)
    distance = np.linalg.norm(points - center, axis=1)
    keep = distance <= np.percentile(distance, 90)
    points = points[keep]
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    center = hull.mean(axis=0)
    expanded = []
    for x, y in hull:
        direction = np.array([x, y]) - center
        norm = np.linalg.norm(direction)
        pad = max(24.0, 0.06 * norm)
        expanded.append([x + direction / (norm + 1e-6) * pad, y + direction / (norm + 1e-6) * pad])
    expanded = np.asarray(expanded, dtype=np.float32)
    perimeter = cv2.arcLength(expanded.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(expanded.reshape(-1, 1, 2), tolerance * perimeter, True).reshape(-1, 2)
    if len(approx) > 10:
        approx = approx[:10]
    if len(approx) < 3:
        # 近似把细长走廊压扁了：直接保留原始凸包，再均匀抽稀
        hull_points = hull.reshape(-1, 2)
        if len(hull_points) > 12:
            index = np.linspace(0, len(hull_points) - 1, 12).astype(int)
            hull_points = hull_points[index]
        approx = hull_points.astype(np.float32)
    polygon = []
    for x, y in approx:
        polygon.append([int(round(min(max(x, 0), width - 1))), int(round(min(max(y, 0), height - 1)))])
    return polygon


def derive_lines(points: np.ndarray, width: int, height: int) -> tuple:
    """沿主方向取 8%/92% 分位，生成两条垂直车流方向的线。"""
    center = points.mean(axis=0)
    centered = points - center
    # PCA 主方向 = 车流方向
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if abs(axis[0]) < 1e-6 and abs(axis[1]) < 1e-6:
        axis = np.array([0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    # 统一成"从上往下"为正向，便于阅读
    if axis[1] < 0:
        axis = -axis
    perpendicular = np.array([-axis[1], axis[0]])

    projection = centered @ axis
    low, high = np.percentile(projection, [8, 92])
    half_span = float(np.percentile(np.abs(centered @ perpendicular), 95)) + 60.0

    def make_line(offset: float) -> list:
        base = center + axis * offset
        p1 = base - perpendicular * half_span
        p2 = base + perpendicular * half_span
        return [
            [int(round(min(max(p1[0], 0), width - 1))), int(round(min(max(p1[1], 0), height - 1)))],
            [int(round(min(max(p2[0], 0), width - 1))), int(round(min(max(p2[1], 0), height - 1)))],
        ]

    return make_line(low), make_line(high), axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", required=True, help="机位 id（写进配置）")
    parser.add_argument("--name", default="", help="机位中文名")
    parser.add_argument("--samples", type=int, default=240, help="抽帧数量")
    parser.add_argument("--output", required=True, help="输出 yaml 路径")
    parser.add_argument("--overlay", default="", help="输出可视化 png（建议填写，便于人工复核）")
    args = parser.parse_args()

    points, width, height, read = collect_points(args.video, samples=args.samples)
    print(json.dumps({"frames_sampled": read, "vehicle_points": len(points), "size": [width, height]}, ensure_ascii=False))
    if len(points) < 8:
        raise SystemExit("检测到的车辆太少")

    polygon = simplify_polygon(points, width, height)
    entry_line, exit_line, axis = derive_lines(points, width, height)

    lines = [
        f"camera_id: {args.camera}",
        f"camera_name: {args.name or args.camera}",
        "# 由 scripts/derive_camera_config.py 自动推导，人工复核 ROI 与出入口线后再启用",
        "roi:",
        "  polygon:",
    ]
    for x, y in polygon:
        lines.append(f"  - [{x}, {y}]")
    lines.append("entry_line:")
    lines.append(f"  p1: [{entry_line[0][0]}, {entry_line[0][1]}]")
    lines.append(f"  p2: [{entry_line[1][0]}, {entry_line[1][1]}]")
    lines.append("exit_line:")
    lines.append(f"  p1: [{exit_line[0][0]}, {exit_line[0][1]}]")
    lines.append(f"  p2: [{exit_line[1][0]}, {exit_line[1][1]}]")
    lines.append("calibration:")
    lines.append("  # 未标定：速度/路径/聚集特征会被门控。现场量到 >=4 组控制点后填 homography。")
    lines.append("  px_per_meter: 0.0")
    lines.append("anomaly:")
    lines.append("  contamination: 0.05")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"yaml": args.output, "roi_vertices": len(polygon),
                      "flow_axis": [round(float(axis[0]), 3), round(float(axis[1]), 3)]}, ensure_ascii=False))

    if args.overlay:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4))
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.polylines(frame, [np.asarray(polygon, np.int32)], True, (0, 200, 255), 3)
            for line, color, label in ((entry_line, (0, 255, 0), "entry"), (exit_line, (0, 0, 255), "exit")):
                cv2.line(frame, tuple(line[0]), tuple(line[1]), color, 3)
                mid = ((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2)
                cv2.putText(frame, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            for x, y in points[:: max(1, len(points) // 60)]:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 255), -1)
            os.makedirs(os.path.dirname(os.path.abspath(args.overlay)) or ".", exist_ok=True)
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok2:
                buf.tofile(args.overlay)
                print(json.dumps({"overlay": args.overlay}, ensure_ascii=False))


if __name__ == "__main__":
    main()
