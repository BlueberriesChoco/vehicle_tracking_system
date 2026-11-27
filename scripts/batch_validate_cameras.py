#!/usr/bin/env python
"""批量验证机位标定：每个机位取同一天的同一时段，跑一段真实录像，统计通行量。

为什么需要：
    静态校验只能发现几何问题，"标了但一条通行都出不来"必须靠真实数据才能暴露。
    这个脚本把「取片段 -> 跑流水线 -> 统计通行数」串起来，一次跑完所有在范围内的机位。

时间对齐的近似做法：
    直接按 .mbf 文件的时间范围与大小估算字节偏移，从目标时段附近读取一段连续数据块，
    还原成 h264 后从第 0 帧开始跑。误差通常在几十秒内，对「有没有车/能不能出通行」足够。

用法：
    python scripts/batch_validate_cameras.py \
        --root "F:\\A区录像 新" \
        --start "2026-09-01T09:00:00" --minutes 5 \
        --output data/outputs/camera_validation.csv

    # 先挑 5 台试跑
    python scripts/batch_validate_cameras.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.mbf_extract import MBF_BLOCK_HEADER_SIZE, MBF_BLOCK_MARKER, MBF_DEFAULT_BLOCK_SIZE, parse_mbf_header

READ_BLOCKS = 60          # 约 120 MB，够跑 5 分钟
FPS = 25


def parse_name_times(path: str):
    stamps = re.findall(r"20\d{12}", os.path.basename(path))
    if len(stamps) < 2:
        return None
    try:
        return datetime.strptime(stamps[0], "%Y%m%d%H%M%S"), datetime.strptime(stamps[1], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def pick_clip(camera_dir: str, moment: datetime):
    """找出覆盖目标时刻的文件；找不到就取时间最接近的一个。"""
    best = None
    for path in sorted(glob(os.path.join(camera_dir, "*.mbf"))):
        span = parse_name_times(path)
        if not span:
            continue
        start, end = span
        if start <= moment <= end:
            return path, span
        distance = min(abs((moment - start).total_seconds()), abs((moment - end).total_seconds()))
        if best is None or distance < best[0]:
            best = (distance, path, span)
    return (best[1], best[2]) if best else (None, None)


def read_chunk(path: str, moment: datetime, span) -> bytes:
    """按时间比例估算块偏移，读取一段连续数据并还原成 h264 字节流。"""
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        info = parse_mbf_header(handle.read(12))
    block_size = info["block_size"] or MBF_DEFAULT_BLOCK_SIZE
    total_blocks = max(1, size // block_size)

    start, end = span
    duration = max(1.0, (end - start).total_seconds())
    ratio = min(1.0, max(0.0, (moment - start).total_seconds() / duration))
    first_block = int(total_blocks * ratio)
    # 往前退一点，保证能解到关键帧
    first_block = max(1, first_block - 2)

    with open(path, "rb") as handle:
        handle.seek(first_block * block_size)
        data = handle.read(block_size * READ_BLOCKS)

    out = bytearray()
    for offset in range(0, len(data), block_size):
        segment = data[offset:offset + block_size]
        if segment.startswith(MBF_BLOCK_MARKER):
            out.extend(segment[MBF_BLOCK_HEADER_SIZE:])
        else:
            out.extend(segment)
    return bytes(out)


def count_events(csv_path: str) -> int:
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_one(camera: str, root: str, moment: datetime, minutes: int, out_root: str) -> dict:
    camera_dir = os.path.join(root, camera)
    clip, span = pick_clip(camera_dir, moment)
    if not clip:
        return {"camera": camera, "passages": 0, "error": "没有覆盖目标时段的录像"}

    stream = read_chunk(clip, moment, span)
    if not stream:
        return {"camera": camera, "passages": 0, "error": "读取数据块为空"}

    temp_dir = tempfile.mkdtemp(prefix="camcheck_")
    try:
        # 文件名里带上时间，流水线据此解析日期/小时
        stamp = (span[0] + (span[1] - span[0]) * 0.0).strftime("%Y%m%d%H%M%S")
        video = os.path.join(temp_dir, f"{camera}_{stamp}_20990101000000.h264")
        with open(video, "wb") as handle:
            handle.write(stream)

        run_dir = os.path.join(out_root, f"check_{camera}")
        shutil.rmtree(run_dir, ignore_errors=True)
        command = [
            sys.executable, "scripts/run_full_pipeline.py",
            "--input", video, "--camera", camera,
            "--output", run_dir,
            "--max-frames", str(minutes * 60 * FPS),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=1800)
        event_files = glob(os.path.join(run_dir, "events", "**", "*.csv"), recursive=True)
        passages = count_events(event_files[0]) if event_files else 0
        return {
            "camera": camera,
            "passages": passages,
            "exit_code": proc.returncode,
            "error": "" if proc.returncode == 0 else (proc.stderr or "")[-200:].replace("\n", " "),
        }
    except subprocess.TimeoutExpired:
        return {"camera": camera, "passages": 0, "error": "流水线超时"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_scope(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("camera_id") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("in_scope") or "").strip().lower() in {"false", "0", "no"}
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"F:\A区录像 新")
    parser.add_argument("--start", default="2026-09-01T09:00:00")
    parser.add_argument("--minutes", type=int, default=5)
    parser.add_argument("--scope", default="config/site/camera_scope.csv")
    parser.add_argument("--config-dir", default="config/camera")
    parser.add_argument("--output", default="data/outputs/camera_validation.csv")
    parser.add_argument("--run-root", default="data/outputs/validation_runs")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    moment = datetime.fromisoformat(args.start)
    excluded = load_scope(args.scope)
    cameras = [os.path.basename(p) for p in sorted(glob(os.path.join(args.root, "*")))
               if os.path.isdir(p) and not os.path.basename(p).startswith("_")]
    cameras = [c for c in cameras if c not in excluded]
    cameras = [c for c in cameras if os.path.isfile(os.path.join(args.config_dir, f"{c}.yaml"))]
    if args.limit:
        cameras = cameras[: args.limit]

    os.makedirs(args.run_root, exist_ok=True)
    rows = []
    for index, camera in enumerate(cameras, start=1):
        row = run_one(camera, args.root, moment, args.minutes, args.run_root)
        rows.append(row)
        print(f"[{index}/{len(cameras)}] {camera}: 通行 {row['passages']} {row.get('error', '')}", flush=True)

    keys = ["camera", "passages", "exit_code", "error"]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

    zero = [r["camera"] for r in rows if not r["passages"]]
    print(json.dumps({"cameras": len(rows), "with_passages": len(rows) - len(zero), "zero": len(zero)},
                     ensure_ascii=False))
    if zero:
        print("零通行的机位：", "、".join(zero[:20]))


if __name__ == "__main__":
    main()
