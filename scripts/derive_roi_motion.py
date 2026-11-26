#!/usr/bin/env python
"""用「运动目标」推导机位 ROI：剔除停放车辆，只圈真正有车流通过的区域。

为什么不能直接用检测框的凸包：
    路侧停放的车辆每帧都在同一位置被检出，凸包会把停车带一起圈进 ROI，
    导致 ROI 覆盖到人行道/路肩（实测「广兴路西」因此把停放车算成了通行车，
    记录的"通行时长"是 889–1289 秒）。

做法：
    1. 均匀抽帧，逐帧检测车辆，取框底边中点（接地位置）；
    2. 把画面切成网格，统计每个网格被"多少帧"命中（命中率）；
    3. 命中率高于阈值的网格视为静止目标（停放车辆），剔除；
    4. 剩余点的凸包向外扩一圈，得到 ROI 多边形。

只产出 ROI。entry_line / exit_line 属于硬门控，自动推导误差会让整条车道的车
都不产生向量，必须人工标定（scripts/calibrate_camera.py）。

用法：
    python scripts/derive_roi_motion.py \
        --video data/outputs/h264/cl_north/xxx.h264 \
        --camera cl_north --name 中央路北 \
        --output config/camera/cl_north.yaml \
        --overlay data/outputs/probe/cl_north_roi.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRID_PX = 48          # 网格边长（像素）
STATIC_RATIO = 0.55   # 命中率高于该值视为静止目标
MIN_POINTS = 8        # 少于该数量视为推导失败


def collect_points(video: str, samples: int = 240, min_box: int = 24) -> Tuple[np.ndarray, int, int, int]:
    """返回 (接地坐标数组, 宽, 高, 实际采样帧数)。"""
    from src.detection.detector import VehicleDetector

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = VehicleDetector(model_path="models/yolov8s.pt")
    step = max(1, total // max(1, samples))
    points: List[Tuple[float, float]] = []
    frames = 0
    for index in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        for item in detector.detect(frame):
            box = item.get("bbox")
            if not box:
                continue
            if (box[2] - box[0]) < min_box or (box[3] - box[1]) < min_box:
                continue
            points.append(((box[0] + box[2]) / 2.0, float(box[3])))
    cap.release()
    return np.asarray(points, dtype=np.float64), width, height, frames


def drop_static(points: np.ndarray, frames: int, grid: int = GRID_PX,
                ratio: float = STATIC_RATIO) -> Tuple[np.ndarray, int]:
    """按网格命中率剔除静止目标，返回 (保留点, 被剔除的静止网格数)。"""
    if len(points) == 0 or frames <= 0:
        return points, 0

    # 用「检出次数 / 采样帧数」近似命中率：停放车辆几乎每帧都被检出，
    # 过路车辆只在少数帧出现，两者区分度足够。
    counts: dict = {}
    for x, y in points:
        key = (int(x // grid), int(y // grid))
        counts[key] = counts.get(key, 0) + 1

    static_cells = {key for key, count in counts.items() if count / frames >= ratio}
    keep = np.array([
        (x, y) for x, y in points
        if (int(x // grid), int(y // grid)) not in static_cells
    ], dtype=np.float64)
    return keep, len(static_cells)


def polygon_from_points(points: np.ndarray, width: int, height: int,
                        pad_ratio: float = 0.12, max_vertices: int = 12) -> List[List[int]]:
    """凸包 + 外扩，返回多边形顶点。"""
    if len(points) < 3:
        return [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    center = hull.mean(axis=0)
    expanded = []
    for x, y in hull:
        direction = np.array([x, y], dtype=np.float64) - center
        norm = float(np.linalg.norm(direction))
        pad = max(24.0, pad_ratio * norm)
        expanded.append(direction / (norm + 1e-6) * pad + np.array([x, y]))
    expanded = np.asarray(expanded, dtype=np.float32)

    perimeter = cv2.arcLength(expanded.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(expanded.reshape(-1, 1, 2), 0.01 * perimeter, True).reshape(-1, 2)
    if len(approx) < 3:
        approx = expanded.reshape(-1, 2)
    if len(approx) > max_vertices:
        index = np.linspace(0, len(approx) - 1, max_vertices).astype(int)
        approx = approx[index]
    return [
        [int(round(min(max(x, 0), width - 1))), int(round(min(max(y, 0), height - 1)))]
        for x, y in approx
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", default="")
    args = parser.parse_args()

    points, width, height, frames = collect_points(args.video, samples=args.samples)
    keep, static_cells = drop_static(points, frames)
    if len(keep) < MIN_POINTS:
        print(json.dumps({"camera": args.camera, "ok": False,
                          "points": len(points), "kept": len(keep),
                          "hint": "运动点太少，退回全画幅"}, ensure_ascii=False))
        polygon = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        mode = "full_frame"
    else:
        polygon = polygon_from_points(keep, width, height)
        mode = "motion"

    lines = [
        f"camera_id: {args.camera}",
        f"camera_name: {args.name or args.camera}",
        "# 由 scripts/derive_roi_motion.py 推导：按网格命中率剔除停放车辆后取凸包。",
        "# 只写 ROI，不写进出线——出入口线是硬门控，自动推导会造成大量丢车，",
        "# 需要时用 scripts/calibrate_camera.py 人工补 entry_line / exit_line。",
        "roi:",
        "  polygon:",
    ]
    for x, y in polygon:
        lines.append(f"  - [{x}, {y}]")
    lines.append("calibration:")
    lines.append("  # 未标定：速度/路径特征会被门控")
    lines.append("  px_per_meter: 0.0")
    lines.append("anomaly:")
    lines.append("  contamination: 0.05")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(json.dumps({
        "camera": args.camera, "ok": mode == "motion",
        "frames": frames, "points": len(points), "kept": len(keep),
        "static_cells": static_cells, "vertices": len(polygon), "output": args.output,
    }, ensure_ascii=False))

    if args.overlay:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4))
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.polylines(frame, [np.asarray(polygon, np.int32)], True, (0, 200, 255), 3)
            for x, y in points[:: max(1, len(points) // 80)]:
                cv2.circle(frame, (int(x), int(y)), 2, (200, 0, 200), -1)
            for x, y in keep[:: max(1, len(keep) // 80)]:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            os.makedirs(os.path.dirname(os.path.abspath(args.overlay)) or ".", exist_ok=True)
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok2:
                buf.tofile(args.overlay)


if __name__ == "__main__":
    main()
