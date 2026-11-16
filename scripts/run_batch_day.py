#!/usr/bin/env python
"""Batch-process a full day of camera recordings with resume support.

For every ``.mbf`` in the input directory it:
  1. extracts ``.h264`` (skipped if present),
  2. runs the behavior-vector pipeline (skipped if the vector CSV already exists),
  3. records progress so an interrupted run can resume without redoing work.

Usage:
    python scripts/run_batch_day.py \
        --mbf-dir "F:/A区录像 新/行政车道进" \
        --camera admin_in \
        --output data/outputs \
        --max-frames 3000

    # only the first 3 segments, for a quick check
    python scripts/run_batch_day.py --mbf-dir ... --camera admin_in --limit 3 --max-frames 500
"""

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import get_logger
from src.utils.mbf_extract import extract_mbf_file

logger = get_logger(__name__)


def parse_start_timestamp(filename: str) -> str:
    """返回录像开始时间戳（14 位），例如 20260901001941。"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    for part in stem.split("_"):
        if len(part) >= 14 and part[:14].isdigit():
            return part[:14]
    return ""


def parse_date_hour(filename: str):
    """从录像文件名解析日期和小时（与 BatchProcessor 保持一致）。"""
    start = parse_start_timestamp(filename)
    if start:
        return start[:8], start[8:10]
    return "unknown", "00"


def main():
    parser = argparse.ArgumentParser(description="Batch-process a full day of .mbf recordings")
    parser.add_argument("--mbf-dir", required=True, help="Directory containing .mbf files")
    parser.add_argument("--camera", default="admin_in", help="Camera config id")
    parser.add_argument("--config", default="config/default.yaml", help="Global config")
    parser.add_argument("--output", default="data/outputs", help="Output root")
    parser.add_argument("--h264-dir", default="data/outputs/h264/admin_in", help="Directory for extracted .h264")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per segment (0 = all)")
    parser.add_argument("--device", type=int, default=0, help="GPU id")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N segments (0 = all)")
    parser.add_argument("--only", default="", help="Comma-separated 1-based segment indices, e.g. 8,10,12")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already-processed segments (default)")
    parser.add_argument("--force", action="store_true", help="Reprocess even if outputs exist")
    args = parser.parse_args()

    os.makedirs(args.h264_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.mbf_dir, "*.mbf")))
    if not all_files:
        logger.error("No .mbf files found in %s", args.mbf_dir)
        sys.exit(1)

    files = all_files
    if args.only:
        wanted = {int(token) for token in args.only.split(",") if token.strip()}
        files = [path for position, path in enumerate(all_files, start=1) if position in wanted]
        if not files:
            logger.error("No segments matched --only %s", args.only)
            sys.exit(1)
    if args.limit > 0:
        files = files[: args.limit]
    logger.info("Segments: %s", len(files))

    processed, skipped, failed = 0, 0, 0
    results = []

    for index, mbf in enumerate(files, start=1):
        stem = os.path.splitext(os.path.basename(mbf))[0]
        date_str, hour_str = parse_date_hour(stem)
        start = parse_start_timestamp(stem) or "00000000000000"
        # 同一小时内可能有多个分段，用开始时间后缀避免互相覆盖
        suffix = start[8:]
        h264 = os.path.join(args.h264_dir, stem + ".h264")
        vector_dir = os.path.join(args.output, "vectors", args.camera, date_str)
        vector = os.path.join(vector_dir, f"vec_{args.camera}_{date_str}_{hour_str}.csv")
        vector_final = os.path.join(vector_dir, f"vec_{args.camera}_{date_str}_{hour_str}_{suffix}.csv")
        event_dir = os.path.join(args.output, "events", args.camera, date_str)
        event = os.path.join(event_dir, f"events_{args.camera}_{date_str}_{hour_str}.csv")
        event_final = os.path.join(event_dir, f"events_{args.camera}_{date_str}_{hour_str}_{suffix}.csv")

        if not args.force and os.path.exists(h264) and os.path.exists(vector_final):
            skipped += 1
            logger.info("[%s/%s] skip (already processed): %s", index, len(files), stem)
            results.append({"segment": stem, "status": "skipped"})
            continue

        try:
            if not os.path.exists(h264):
                extract_mbf_file(mbf, h264)
                logger.info("[%s/%s] extracted: %s", index, len(files), h264)

            command = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "run_pipeline.py"),
                "--video", h264,
                "--camera", args.camera,
                "--config", args.config,
                "--output", args.output,
                "--device", str(args.device),
            ]
            if args.max_frames > 0:
                command += ["--max-frames", str(args.max_frames)]

            subprocess.run(command, check=True)

            # 重命名为本段唯一文件名，避免同小时分段覆盖
            if os.path.exists(vector) and vector != vector_final:
                os.replace(vector, vector_final)
            if os.path.exists(event) and event != event_final:
                os.replace(event, event_final)

            processed += 1
            results.append({"segment": stem, "status": "processed", "vectors": vector_final})
            logger.info("[%s/%s] done: %s", index, len(files), stem)
        except Exception as exc:  # noqa: BLE001 - 批处理需要继续跑完剩余段
            failed += 1
            results.append({"segment": stem, "status": "failed", "error": str(exc)})
            logger.error("[%s/%s] failed: %s (%s)", index, len(files), stem, exc)

    summary = {
        "mbf_dir": args.mbf_dir,
        "camera": args.camera,
        "segments": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }
    summary_path = os.path.join(args.output, "batch_day_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Segments: {len(files)}, processed: {processed}, skipped: {skipped}, failed: {failed}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
