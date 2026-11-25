#!/usr/bin/env python
"""白天 / 夜晚车牌识别率对照评测。

背景：车牌识别长期被记为「0.45%，全景相机车牌只有 20-30 像素」，
2026-11-23 定位到真实原因是 Windows 上模型目录解析错盘导致 hyperlpr3 未启动。
修复后需要回答一个更实际的问题：**夜间光照下还能不能识别**。

选段原则：
- 白天取正午到下午（光照均匀、无逆光），时长 1 小时；
- 夜晚取天黑后（9 月江苏约 18:30 日落）。夜间车流稀疏且集中在前半段之后，
  只截 1 小时会几乎没有样本，因此夜晚跑**完整片段**，在报告中如实标注时长差异。

用法：
    python scripts/eval_plate_day_night.py --output-json tmp/plate_daynight.json
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.batch_processor import BatchProcessor

MBF_ROOT = r"F:\A区录像 新"

#: 段配置。max_frames=0 表示整段。
SEGMENTS = [
    {
        "key": "in_day",
        "label": "入口·白天",
        "camera": "admin_in",
        "mbf": os.path.join(MBF_ROOT, "行政车道进",
                            "Download_南京南通海关综保A区行政车道进_20260901134815_20260901151852.mbf"),
        "max_frames": 90000,
        "note": "13:48–15:18，取前 1 小时",
    },
    {
        "key": "in_night",
        "label": "入口·夜晚",
        "camera": "admin_in",
        "mbf": os.path.join(MBF_ROOT, "行政车道进",
                            "Download_南京南通海关综保A区行政车道进_20260901205732_20260901221656.mbf"),
        "max_frames": 0,
        "note": "20:57–22:16，完整 79 分钟（前 44 分钟无车）",
    },
    {
        "key": "out_day",
        "label": "出口·白天",
        "camera": "admin_out",
        "mbf": os.path.join(MBF_ROOT, "行政车道出",
                            "Download_南京南通海关综保A区行政车道出_20260901120032_20260901140021.mbf"),
        "max_frames": 90000,
        "note": "12:00–14:00，取前 1 小时",
    },
    {
        "key": "out_night",
        "label": "出口·夜晚",
        "camera": "admin_out",
        "mbf": os.path.join(MBF_ROOT, "行政车道出",
                            "Download_南京南通海关综保A区行政车道出_20260901204818_20260901224807.mbf"),
        "max_frames": 0,
        "note": "20:48–22:48，完整片段",
    },
]


def h264_path_for(mbf_path: str, h264_dir: str) -> str:
    return os.path.join(h264_dir, os.path.splitext(os.path.basename(mbf_path))[0] + ".h264")


def ensure_h264(mbf_path: str, h264_dir: str) -> str:
    target = h264_path_for(mbf_path, h264_dir)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"    h264 已存在，跳过转码: {os.path.basename(target)}", flush=True)
        return target
    os.makedirs(h264_dir, exist_ok=True)
    print(f"    转码 {os.path.basename(mbf_path)} ...", flush=True)
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "scripts/extract_mbf.py", "--input", mbf_path, "--output", target],
        check=True,
    )
    print(f"    转码完成 {time.perf_counter() - started:.0f}s", flush=True)
    return target


def summarize_vectors(run_dir: str) -> dict:
    files = glob.glob(os.path.join(run_dir, "**", "*vec*.csv"), recursive=True)
    rows = []
    for path in files:
        with open(path, encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        return {"vehicles": 0, "with_plate": 0, "rate": 0.0, "plates": [], "confidence": {}}

    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    with_plate = [r for r in rows if (r.get("plate_number") or "").strip()]
    confs = [to_float(r.get("plate_confidence")) for r in with_plate]
    confs = [c for c in confs if c is not None]
    confs.sort()
    return {
        "vehicles": len(rows),
        "with_plate": len(with_plate),
        "rate": round(len(with_plate) / len(rows) * 100, 1),
        "plates": [r["plate_number"] for r in with_plate],
        "confidence": {
            "n": len(confs),
            "min": round(confs[0], 3) if confs else None,
            "p50": round(confs[len(confs) // 2], 3) if confs else None,
            "max": round(confs[-1], 3) if confs else None,
        } if confs else {},
        "files": len(files),
    }


def main():
    parser = argparse.ArgumentParser(description="Day/night plate OCR evaluation")
    parser.add_argument("--output-root", default="data/outputs/plate_daynight")
    parser.add_argument("--h264-dir", default=os.path.join(MBF_ROOT, "_h264_eval"))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--only", default="", help="逗号分隔的 key 子集，如 in_day,in_night")
    args = parser.parse_args()

    selected = SEGMENTS
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        selected = [seg for seg in SEGMENTS if seg["key"] in wanted]
        if not selected:
            raise SystemExit(f"未匹配到任何段: {args.only}")

    results = []
    for seg in selected:
        print(f"\n{'=' * 66}\n[{seg['label']}] {seg['note']}\n{'=' * 66}", flush=True)
        if not os.path.exists(seg["mbf"]):
            print(f"    源文件缺失，跳过: {seg['mbf']}", flush=True)
            results.append({**seg, "error": "mbf not found"})
            continue

        started = time.perf_counter()
        video = ensure_h264(seg["mbf"], args.h264_dir)

        processor = BatchProcessor(
            config_path="config/default.yaml",
            camera_config_path=f"config/camera/{seg['camera']}.yaml",
        )
        processor.max_frames = seg["max_frames"]
        run_dir = os.path.join(args.output_root, seg["key"])
        print(f"    处理中（max_frames={seg['max_frames'] or '全部'}）...", flush=True)
        processor.process_video(video, run_dir)
        elapsed = time.perf_counter() - started

        stats = summarize_vectors(run_dir)
        record = {**seg, "elapsed_sec": round(elapsed, 1), **stats}
        results.append(record)
        print(
            f"    车辆 {stats['vehicles']} | 识别出车牌 {stats['with_plate']} "
            f"({stats['rate']}%) | 耗时 {elapsed:.0f}s",
            flush=True,
        )
        if stats["plates"]:
            print(f"    样例: {', '.join(stats['plates'][:8])}", flush=True)

    print(f"\n{'=' * 66}\n对照汇总\n{'=' * 66}")
    print(f"{'时段':<12}{'车辆':>6}{'识别':>6}{'识别率':>9}{'置信度p50':>11}{'耗时':>8}")
    for item in results:
        if "error" in item:
            print(f"{item['label']:<12}{'—':>6}{'—':>6}{item['error']:>9}")
            continue
        p50 = (item.get("confidence") or {}).get("p50")
        print(
            f"{item['label']:<12}{item['vehicles']:>6}{item['with_plate']:>6}"
            f"{item['rate']:>8.1f}%{(p50 if p50 is not None else '—'):>11}"
            f"{item['elapsed_sec']:>7.0f}s"
        )

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"\n已写入: {args.output_json}")


if __name__ == "__main__":
    main()
