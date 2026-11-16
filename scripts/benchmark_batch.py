#!/usr/bin/env python
"""Benchmark batched vs per-frame detection throughput.

Measures detection frames-per-second for different batch sizes on the same
pre-decoded frames, isolating the per-call Python overhead.

Usage:
    python scripts/benchmark_batch.py \
        --video data/outputs/h264/admin_in/xxx.h264 \
        --frames 64 \
        --batch-sizes 1,2,4,8,16,32
"""

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.detector import VehicleDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_frames(video_path: str, count: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Benchmark batched detection")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="models/yolov8s.pt")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    detector = VehicleDetector(model_path=args.model, device=args.device)
    frames = load_frames(args.video, args.frames)
    if len(frames) < max(batch_sizes):
        logger.error("Not enough frames: %s", len(frames))
        sys.exit(1)

    # 预热
    detector.detect_batch(frames[: max(batch_sizes)])

    results = []
    for batch_size in batch_sizes:
        start = time.perf_counter()
        for index in range(0, len(frames), batch_size):
            detector.detect_batch(frames[index:index + batch_size])
        elapsed = time.perf_counter() - start
        fps = len(frames) / elapsed if elapsed > 0 else 0.0
        results.append({"batch_size": batch_size, "fps": round(fps, 2), "elapsed_sec": round(elapsed, 3)})
        print(f"batch={batch_size:<3} fps={fps:6.2f}")

    payload = {"video": args.video, "frames": len(frames), "model": args.model, "results": results}
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Benchmark saved to: %s", args.output)


if __name__ == "__main__":
    main()
