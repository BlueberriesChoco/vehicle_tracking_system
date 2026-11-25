#!/usr/bin/env python
"""机位普查：用每个机位录像的开头几秒，测分辨率与实际车流量。

为什么需要：
    142 个机位的画质和流量差异极大——有的 1080p 主路，有的 352x288 且一天没几辆车。
    不先测一遍就开跑，等于拿 GPU 去算空画面（实测内卡口4 在 20 分钟里 0 辆车）。

做法：
    只读 .mbf 的前几个块（约 8 MB），还原出几秒钟的 H.264，解码若干帧跑检测，
    统计分辨率与"每帧平均检出车辆数"。这是流量代理指标，不是精确计数。

用法：
    python scripts/survey_cameras.py --root "F:\\A区录像 新" --output data/outputs/camera_survey.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.mbf_extract import MBF_BLOCK_HEADER_SIZE, MBF_BLOCK_MARKER, MBF_DEFAULT_BLOCK_SIZE, parse_mbf_header

# 每个机位最多读这么多块（约 8 MB ≈ 数秒视频）
MAX_BLOCKS = 4
SAMPLE_FRAMES = 12


def read_partial_stream(path: str, max_blocks: int = MAX_BLOCKS):
    """读取 .mbf 开头若干块，返回 h264 字节与块大小。"""
    with open(path, "rb") as handle:
        head = handle.read(12)
        info = parse_mbf_header(head)
        block_size = info["block_size"] or MBF_DEFAULT_BLOCK_SIZE
        handle.seek(0)
        data = handle.read(block_size * (max_blocks + 1))

    out = bytearray()
    for offset in range(block_size, len(data), block_size):
        segment = data[offset:offset + block_size]
        if segment.startswith(MBF_BLOCK_MARKER):
            out.extend(segment[MBF_BLOCK_HEADER_SIZE:])
        else:
            out.extend(segment)
    return bytes(out)


def survey_camera(folder: str, detector) -> dict:
    files = sorted(glob(os.path.join(folder, "*.mbf")))
    if not files:
        return {"camera": os.path.basename(folder), "error": "无 mbf"}

    stream = read_partial_stream(files[0])
    if not stream:
        return {"camera": os.path.basename(folder), "error": "空流"}

    import cv2

    with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as handle:
        handle.write(stream)
        temp_path = handle.name
    try:
        cap = cv2.VideoCapture(temp_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = 0
        frames = 0
        while frames < SAMPLE_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            if frames % 3 != 0:
                continue
            for item in detector.detect(frame):
                box = item.get("bbox")
                if not box:
                    continue
                if (box[2] - box[0]) < 24 or (box[3] - box[1]) < 24:
                    continue
                total += 1
        cap.release()
        return {
            "camera": os.path.basename(folder),
            "width": width,
            "height": height,
            "fps": round(fps, 1),
            "frames": frames,
            "detections": total,
            "veh_per_frame": round(total / max(1, frames // 3), 2),
            "file": os.path.basename(files[0]),
        }
    finally:
        os.unlink(temp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"F:\A区录像 新")
    parser.add_argument("--output", default="data/outputs/camera_survey.csv")
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 个机位（调试用）")
    args = parser.parse_args()

    from src.detection.detector import VehicleDetector

    detector = VehicleDetector(model_path="models/yolov8s.pt")
    folders = sorted(p for p in glob(os.path.join(args.root, "*")) if os.path.isdir(p))
    folders = [p for p in folders if not os.path.basename(p).startswith("_")]
    if args.limit:
        folders = folders[: args.limit]

    rows = []
    for index, folder in enumerate(folders, start=1):
        row = survey_camera(folder, detector)
        rows.append(row)
        print(f"[{index}/{len(folders)}] {row.get('camera')}: "
              f"{row.get('width')}x{row.get('height')} 检出{row.get('detections')} {row.get('error', '')}",
              flush=True)

    keys = ["camera", "width", "height", "fps", "frames", "detections", "veh_per_frame", "file", "error"]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
