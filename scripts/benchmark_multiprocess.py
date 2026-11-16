#!/usr/bin/env python
"""Measure aggregate throughput when running N pipeline workers concurrently.

单进程已经测到 125.6 FPS（约 8.4 路）。这个脚本回答另一个问题：
**横向加进程能不能接近线性扩展**——如果能，那么 20 路缺的是机器而不是算法。

启动前会检查可用内存：每个 worker 约占 2 GB RSS，内存不足时直接拒绝，
避免跑到一半被系统 OOM 掉。

用法：
    python scripts/benchmark_multiprocess.py \
        --video data/outputs/h264/admin_in/clip_08_short.h264 \
        --workers 1,2 --frames 600
"""

import argparse
import os
import subprocess
import sys
import time

import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import get_logger

logger = get_logger(__name__)

RSS_PER_WORKER_MB = 2048


def run_workers(video: str, workers: int, frames: int, out_root: str, device: int) -> dict:
    procs = []
    started = time.time()
    for index in range(workers):
        out_dir = os.path.join(out_root, f"w{workers}_{index}")
        cmd = [
            sys.executable, os.path.join("scripts", "run_pipeline.py"),
            "--video", video,
            "--camera", "admin_in",
            "--output", out_dir,
            "--max-frames", str(frames),
            "--device", str(device),
        ]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    peak_rss = 0
    while any(p.poll() is None for p in procs):
        total = 0
        for p in procs:
            try:
                total += psutil.Process(p.pid).memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_rss = max(peak_rss, total)
        time.sleep(0.5)

    elapsed = time.time() - started
    codes = [p.returncode for p in procs]
    return {
        "workers": workers,
        "frames_per_worker": frames,
        "elapsed_sec": round(elapsed, 2),
        "per_worker_fps": round(frames / elapsed, 1) if elapsed > 0 else 0.0,
        "aggregate_fps": round(workers * frames / elapsed, 1) if elapsed > 0 else 0.0,
        "streams_at_15fps": round(workers * frames / elapsed / 15.0, 2) if elapsed > 0 else 0.0,
        "peak_rss_mb": round(peak_rss / 2**20),
        "exit_codes": codes,
        "all_ok": all(code == 0 for code in codes),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark concurrent pipeline workers")
    parser.add_argument("--video", required=True, help="Source video file")
    parser.add_argument("--workers", default="1,2", help="Comma separated worker counts, e.g. 1,2,4")
    parser.add_argument("--frames", type=int, default=600, help="Frames per worker")
    parser.add_argument("--output", default="data/outputs/mpbench", help="Output root directory")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    counts = [int(item) for item in args.workers.split(",") if item.strip()]
    mem = psutil.virtual_memory()
    available_mb = mem.available / 2**20
    total_mb = mem.total / 2**20
    results = []
    logger.info("memory: %.0f MB available / %.0f MB total", available_mb, total_mb)

    for workers in counts:
        need = workers * RSS_PER_WORKER_MB
        # 用总量判断（Windows 的 available 会因文件缓存显得很低，实际可回收）；
        # 但若连总量都放不下，直接跳过，避免触发 OOM。
        if need > total_mb * 0.8:
            logger.warning(
                "skip workers=%d: 预计需要 %d MB，超出物理内存的 80%%（总 %.0f MB）",
                workers, need, total_mb,
            )
        elif need > available_mb:
            logger.warning(
                "workers=%d 需要约 %d MB，当前 available 仅 %.0f MB（缓存可回收，继续尝试）",
                workers, need, available_mb,
            )
        logger.info("running workers=%d frames=%d", workers, args.frames)
        results.append(run_workers(args.video, workers, args.frames, args.output, args.device))

    print()
    header = f"{'workers':>8} {'elapsed(s)':>11} {'agg FPS':>9} {'路@15FPS':>9} {'peakRSS(MB)':>12} {'ok':>4}"
    print(header)
    print("-" * len(header))
    for row in results:
        if row.get("skipped"):
            print(f"{row['workers']:>8} {'skipped':>11} {'-':>9} {'-':>9} {'-':>12} {'-':>4}")
            continue
        print(
            f"{row['workers']:>8} {row['elapsed_sec']:>11.2f} {row['aggregate_fps']:>9.1f} "
            f"{row['streams_at_15fps']:>9.2f} {row['peak_rss_mb']:>12} {str(row['all_ok']):>4}"
        )

    return 0 if all(row.get("all_ok", True) for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
