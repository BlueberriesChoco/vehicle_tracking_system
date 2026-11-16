#!/usr/bin/env python
"""Throughput benchmark for the video pipeline.

Measures:
- detection-only FPS (YOLO inference on decoded frames);
- end-to-end pipeline FPS (decode + detection + tracking + crop + vectorization);
- estimated number of 1080p streams sustainable at 15 FPS on the measured GPU.

Usage:
    python scripts/benchmark_throughput.py \
        --video data/outputs/h264/admin_in/xxx.h264 \
        --camera admin_in \
        --frames 200 \
        --skip 1
"""

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.batch_processor import BatchProcessor
from src.utils.threaded_reader import ThreadedFrameReader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def measure_detection(video_path, detector, frames, skip=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    processed = 0
    start = None
    for index in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if index % skip != 0:
            continue
        if start is None:
            start = time.perf_counter()
        detector.detect(frame)
        processed += 1

    elapsed = time.perf_counter() - start if start else 0.0
    cap.release()
    return processed, elapsed


def measure_pipeline(video_path, processor, frames, skip=1, batch_size=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    pipeline = processor.pipeline
    pipeline.reset()
    # 与生产链路一致：启用解码预取，否则基准会低估吞吐（见 PERFORMANCE.md P1）
    reader = None
    prefetch = getattr(processor, "prefetch_frames", 0)
    if prefetch > 0:
        reader = ThreadedFrameReader(cap, queue_size=prefetch)
    processed = 0
    start = None
    buffer = []
    for index in range(frames):
        ok, frame = reader.read() if reader is not None else cap.read()
        if not ok:
            break
        if index % skip != 0:
            continue
        if start is None:
            start = time.perf_counter()
        buffer.append(frame)
        if len(buffer) >= batch_size:
            pipeline.process_frames(buffer, [None] * len(buffer))
            processed += len(buffer)
            buffer.clear()

    if buffer:
        pipeline.process_frames(buffer, [None] * len(buffer))
        processed += len(buffer)

    elapsed = time.perf_counter() - start if start else 0.0
    if reader is not None:
        reader.stop()
    cap.release()
    return processed, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline throughput")
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", default="admin_in")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--frames", type=int, default=200, help="Frames to measure")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup frames (model lazy init)")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for detection")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    processor = BatchProcessor(
        config_path=args.config,
        camera_config_path=f"config/camera/{args.camera}.yaml",
    )
    processor.detector.device = args.device

    # 预热：加载模型、初始化显存
    measure_detection(args.video, processor.detector, args.warmup, skip=args.skip)
    measure_pipeline(args.video, processor, args.warmup, skip=args.skip)

    det_frames, det_elapsed = measure_detection(args.video, processor.detector, args.frames, skip=args.skip)
    pipe_frames, pipe_elapsed = measure_pipeline(
        args.video, processor, args.frames, skip=args.skip, batch_size=args.batch_size
    )

    det_fps = det_frames / det_elapsed if det_elapsed > 0 else 0.0
    pipe_fps = pipe_frames / pipe_elapsed if pipe_elapsed > 0 else 0.0
    streams = pipe_fps / args.target_fps if args.target_fps > 0 else 0.0

    result = {
        "video": args.video,
        "device": args.device,
        "frames": pipe_frames,
        "skip": args.skip,
        "detection_fps": round(det_fps, 2),
        "pipeline_fps": round(pipe_fps, 2),
        "target_fps_per_stream": args.target_fps,
        "estimated_streams_at_target_fps": round(streams, 2),
        "detection_elapsed_sec": round(det_elapsed, 3),
        "pipeline_elapsed_sec": round(pipe_elapsed, 3),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Benchmark saved to: %s", args.output)


if __name__ == "__main__":
    main()
