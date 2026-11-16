#!/usr/bin/env python
"""Windowed processing: bound per-vehicle reporting latency on long videos.

Splits a long recording into fixed-size windows, runs the full chain
(vectorize -> detect -> graph -> risk report) per window, and writes one report
per window. A vehicle is reported at the end of its window instead of waiting
for the whole recording, which keeps latency bounded.

Usage:
    python scripts/run_windowed.py \
        --video data/outputs/h264/admin_in/clip_08_short.h264 \
        --camera admin_in \
        --window-sec 60 --max-windows 2 \
        --output data/outputs/windowed
"""

import argparse
import glob
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


def find_vectors(window_dir, camera):
    pattern = os.path.join(window_dir, "vectors", camera, "**", "vec_*.csv")
    # 必须排序：glob 的返回顺序依赖文件系统，否则“取第一个”会不稳定
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[0] if matches else None


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
    processor.max_frames = window_frames

    references = os.path.join(args.output, "references")
    os.makedirs(references, exist_ok=True)
    iforest_path = os.path.join(references, "iforest.json")
    traj_prefix = os.path.join(references, "traj")

    python = sys.executable
    results = []
    window_index = 0

    for start in range(0, total_frames, window_frames):
        if args.max_windows and window_index >= args.max_windows:
            break

        window_dir = os.path.join(args.output, f"window_{window_index:03d}")
        os.makedirs(window_dir, exist_ok=True)
        started = time.perf_counter()

        processor.process_video(args.video, window_dir, start_frame=start)
        vector_csv = find_vectors(window_dir, args.camera)
        row_count = count_vector_rows(vector_csv)
        if row_count == 0:
            logger.warning("Window %s produced no vehicles; skipping", window_index)
            results.append({"window": window_index, "start_frame": start, "vehicles": 0})
            window_index += 1
            continue

        # 首次窗口顺带训练模型，后续窗口复用。
        # 只有 1–2 辆车时样本不足，训练会失败——此时不训练，本窗及后续窗口
        # 只用规则引擎与统计基线，而不是让整段录像崩掉。
        if window_index == 0:
            if row_count >= 3:
                try:
                    run_command([python, "scripts/train_iforest.py", "--input", vector_csv, "--output", iforest_path])
                except RuntimeError as exc:
                    logger.warning("Isolation Forest 训练失败（%s），本段仅用规则检测", exc)
                if os.path.exists(iforest_path):
                    try:
                        run_command([
                            python, "scripts/train_transformer.py", "--input", vector_csv,
                            "--output", traj_prefix, "--epochs", "20", "--device", "cpu",
                        ])
                    except RuntimeError as exc:
                        logger.warning("Transformer 训练失败（%s），仅用 IF 与规则", exc)
            else:
                logger.warning(
                    "First window has only %s vehicles; skipping model training (rule engine only)",
                    row_count,
                )

        alerts_dir = os.path.join(window_dir, "alerts")
        os.makedirs(alerts_dir, exist_ok=True)
        detection_cmd = [
            python, "scripts/run_anomaly_detection.py",
            "--input", vector_csv,
            "--output", os.path.join(alerts_dir, "alerts.csv"),
            "--annotated-vectors", os.path.join(alerts_dir, "vectors.csv"),
        ]
        if os.path.exists(iforest_path):
            detection_cmd += ["--iforest", iforest_path]
        if os.path.exists(traj_prefix + ".pt"):
            detection_cmd += ["--trajectory-model", traj_prefix]
        run_command(detection_cmd)
        graph_path = os.path.join(window_dir, "graph.json")
        run_command([python, "scripts/build_graph.py", "--input", os.path.join(alerts_dir, "vectors.csv"),
                     "--output", graph_path, "--top-k", "3"])
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
        window_index += 1

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
