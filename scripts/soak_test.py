#!/usr/bin/env python
"""Soak test: run the pipeline repeatedly in one process and watch for leaks / FPS drift.

This is a short proxy for the "7x24 无宕机" indicator: it keeps the process alive
across many videos and measures resident memory and throughput per iteration.

Usage:
    python scripts/soak_test.py \
        --video data/outputs/h264/admin_in/clip_08_short.h264 \
        --iterations 10 --max-frames 1000 \
        --output data/outputs/soak --output-json data/outputs/soak/summary.json
"""

import argparse
import json
import os
import sys
import time

try:
    import psutil
except ImportError:  # pragma: no cover - 环境缺 psutil 时给出明确提示
    raise SystemExit("soak_test.py 需要 psutil（pip install psutil）") from None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.batch_processor import BatchProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pipeline soak test")
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", default="admin_in")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200, help="Warmup frames (not measured)")
    parser.add_argument("--output", default="data/outputs/soak")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-rss-growth-mb", type=float, default=200.0,
                        help="Fail if RSS grows more than this across iterations")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        logger.error("Video not found: %s", args.video)
        sys.exit(2)

    processor = BatchProcessor(
        config_path=args.config,
        camera_config_path=f"config/camera/{args.camera}.yaml",
    )
    process = psutil.Process()

    # 预热：加载模型、初始化显存
    processor.max_frames = args.warmup
    processor.process_video(args.video, os.path.join(args.output, "warmup"))

    processor.max_frames = args.max_frames
    results = []
    for index in range(args.iterations):
        rss_before = process.memory_info().rss / 1e6
        started = time.perf_counter()
        processor.process_video(args.video, os.path.join(args.output, f"iter_{index}"))
        elapsed = time.perf_counter() - started
        rss_after = process.memory_info().rss / 1e6
        fps = args.max_frames / elapsed if elapsed > 0 else 0.0

        results.append({
            "iteration": index,
            "fps": round(fps, 2),
            "elapsed_sec": round(elapsed, 2),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_mb": round(rss_after, 1),
        })
        print(
            f"[{index + 1}/{args.iterations}] fps={fps:6.1f} "
            f"elapsed={elapsed:6.2f}s rss={rss_after:7.1f}MB",
            flush=True,
        )

    first = results[0]
    last = results[-1]
    rss_growth = last["rss_after_mb"] - first["rss_after_mb"]
    fps_drift = ((last["fps"] - first["fps"]) / first["fps"] * 100.0) if first["fps"] else 0.0
    crashes = 0  # 能跑到这里说明没有崩溃

    payload = {
        "video": args.video,
        "iterations": args.iterations,
        "max_frames": args.max_frames,
        "results": results,
        "rss_growth_mb": round(rss_growth, 1),
        "fps_drift_percent": round(fps_drift, 2),
        "crashes": crashes,
        "rss_growth_ok": rss_growth <= args.max_rss_growth_mb,
        "fps_stable": abs(fps_drift) <= 20.0,
    }

    print("\n" + "=" * 56)
    print(f"iterations      : {args.iterations}")
    print(f"RSS 增长         : {rss_growth:+.1f} MB (阈值 {args.max_rss_growth_mb} MB)")
    print(f"FPS 漂移         : {fps_drift:+.1f}%")
    print(f"崩溃次数         : {crashes}")
    print("=" * 56)

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Soak summary saved to: %s", args.output_json)

    ok = payload["rss_growth_ok"] and payload["fps_stable"]
    print("\nSOAK TEST PASSED" if ok else "\nSOAK TEST WARNING")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
