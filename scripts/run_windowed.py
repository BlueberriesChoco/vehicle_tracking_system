#!/usr/bin/env python
"""Windowed processing: bound per-vehicle reporting latency on long videos.

Splits a long recording into fixed-size windows, runs the full chain
(vectorize -> detect -> graph -> risk report) per window, and writes one report
per window. Tracks stay active across windows and enter reports after completion.

Usage:
    python scripts/run_windowed.py \
        --video data/outputs/h264/admin_in/clip_08_short.h264 \
        --camera admin_in \
        --window-sec 60 --max-windows 2 \
        --output data/outputs/windowed
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.batch_processor import BatchProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_command(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        logger.error("Command failed: %s", " ".join(command))
        logger.error(completed.stderr[-1500:])
        raise RuntimeError(f"exit code {completed.returncode}")


def count_vector_rows(csv_path):
    """统计向量 CSV 的数据行数（不含表头）。

    窗口没有车时 VectorWriter 仍会写出只有表头的 CSV，所以“文件存在”
    不等于“有样本”，训练前必须看行数。
    """
    if not csv_path or not os.path.exists(csv_path):
        return 0
    with open(csv_path, encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def main():
    parser = argparse.ArgumentParser(description="Windowed processing for bounded latency")
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", default="admin_in")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--window-sec", type=float, default=120.0)
    parser.add_argument("--max-windows", type=int, default=0, help="0 = all")
    parser.add_argument("--output", default="data/outputs/windowed")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iforest", help="Pretrained Isolation Forest JSON")
    parser.add_argument("--trajectory-model", help="Pretrained Transformer checkpoint prefix")
    parser.add_argument("--gae-model", help="Pretrained GAE checkpoint prefix")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        logger.error("Video not found: %s", args.video)
        sys.exit(2)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    window_frames = max(1, int(args.window_sec * fps))
    logger.info("Video: %s frames @ %.1f fps, window=%s frames", total_frames, fps, window_frames)

    processor = BatchProcessor(
        config_path=args.config,
        camera_config_path=f"config/camera/{args.camera}.yaml",
    )
    processor.detector.device = args.device
    python = sys.executable
    results = []

    started = time.perf_counter()
    windows = processor.process_video_windows(args.video, args.output, window_frames, args.max_windows)
    for window_index, (start, vector_csv) in enumerate(windows):
        window_dir = os.path.join(args.output, f"window_{window_index:03d}")
        row_count = count_vector_rows(vector_csv)
        if row_count == 0:
            logger.warning("Window %s produced no vehicles; skipping", window_index)
            results.append({
                "window": window_index,
                "start_frame": start,
                "start_sec": round(start / fps, 1),
                "vehicles": 0,
                "latency_sec": round(time.perf_counter() - started, 1),
            })
            started = time.perf_counter()
            continue

        alerts_dir = os.path.join(window_dir, "alerts")
        os.makedirs(alerts_dir, exist_ok=True)
        detection_cmd = [
            python, "scripts/run_anomaly_detection.py",
            "--input", vector_csv,
            "--output", os.path.join(alerts_dir, "alerts.csv"),
            "--annotated-vectors", os.path.join(alerts_dir, "vectors.csv"),
        ]
        if args.iforest:
            detection_cmd += ["--iforest", args.iforest]
        if args.trajectory_model:
            detection_cmd += ["--trajectory-model", args.trajectory_model]
        run_command(detection_cmd)
        graph_path = os.path.join(window_dir, "graph.json")
        graph_cmd = [python, "scripts/build_graph.py", "--input", os.path.join(alerts_dir, "vectors.csv"),
                     "--output", graph_path, "--top-k", "3"]
        if args.gae_model:
            graph_cmd += ["--gae-model", args.gae_model]
        run_command(graph_cmd)
        risk_json = os.path.join(window_dir, "risk.json")
        risk_md = os.path.join(window_dir, "risk.md")
        run_command([python, "scripts/build_risk_report.py",
                     "--input", os.path.join(alerts_dir, "vectors.csv"),
                     "--graph", graph_path, "--output", risk_json, "--markdown", risk_md])

        elapsed = time.perf_counter() - started
        with open(risk_json, encoding="utf-8") as f:
            risk = json.load(f)
        results.append({
            "window": window_index,
            "start_frame": start,
            "start_sec": round(start / fps, 1),
            "vehicles": risk.get("vehicle_count"),
            "medium_risk": risk.get("medium_risk"),
            "latency_sec": round(elapsed, 1),
            "report": risk_md,
        })
        print(f"[window {window_index}] frames {start}-{start + window_frames} "
              f"vehicles={risk.get('vehicle_count')} latency={elapsed:.1f}s", flush=True)
        started = time.perf_counter()

    summary = {
        "video": args.video,
        "window_sec": args.window_sec,
        "window_frames": window_frames,
        "window_count": len(results),
        "max_latency_sec": max((item["latency_sec"] for item in results), default=0.0),
        "windows": results,
    }
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 56)
    print(f"windows          : {len(results)}")
    print(f"max latency/window: {summary['max_latency_sec']}s")
    print("=" * 56)
    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
