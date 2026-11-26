#!/usr/bin/env python
"""批量推导机位 ROI，并把每个机位的检查图汇总成一个 Word，供人工逐台复核。

流程（每台）：
    1. 从该机位的一段 .mbf 里只读前若干块（约 16 MB），还原成临时 .h264；
    2. 调 derive_roi_motion 的算法推导 ROI（剔除停放车辆）；
    3. 更新 config/camera/<机位>.yaml —— **只替换 roi，保留已有的
       entry_line / exit_line / calibration**（手工标定的成果不能被覆盖）；
    4. 生成叠加图（黄线 ROI、绿点保留的运动点、紫点被剔除的静止点）；
    5. 把图与统计写进 Word。

用法：
    python scripts/batch_roi_and_review.py \
        --root "F:\\A区录像 新" \
        --docx docs/roi_review.docx \
        --overlay-dir data/outputs/roi_review
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from glob import glob
from typing import List, Optional

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.mbf_extract import MBF_BLOCK_HEADER_SIZE, MBF_BLOCK_MARKER, MBF_DEFAULT_BLOCK_SIZE, parse_mbf_header
from scripts.derive_roi_motion import collect_points, drop_static, polygon_from_points

READ_BLOCKS = 24       # 约 48 MB/段；3 段合计约 144 MB
SAMPLES = 100          # 每段抽帧数量
MIN_BOX = 24

# 这两台有现场标定与手工进出线，整份配置都不动
PROTECTED = {"admin_in", "admin_out"}


def read_partial(path: str, blocks: int = READ_BLOCKS) -> bytes:
    with open(path, "rb") as handle:
        info = parse_mbf_header(handle.read(12))
        block_size = info["block_size"] or MBF_DEFAULT_BLOCK_SIZE
        handle.seek(0)
        data = handle.read(block_size * (blocks + 1))
    out = bytearray()
    for offset in range(block_size, len(data), block_size):
        segment = data[offset:offset + block_size]
        if segment.startswith(MBF_BLOCK_MARKER):
            out.extend(segment[MBF_BLOCK_HEADER_SIZE:])
        else:
            out.extend(segment)
    return bytes(out)


def pick_files(camera_dir: str) -> list:
    """返回候选源文件：白天段优先，再按一天中 25%/50%/75% 的位置各取一段。

    只在文件开头采样会碰到夜间/凌晨的低流量时段，导致「没车 -> 无法推导 ROI」。
    分散到一天的不同时段，等价于换 3 个时间段重试。
    """
    files = sorted(glob(os.path.join(camera_dir, "*.mbf")))
    if not files:
        return []
    daytime = [p for p in files if any(k in os.path.basename(p) for k in ("2026090107", "2026090108", "2026090109"))]
    count = len(files)
    spread = [files[int(count * ratio)] for ratio in (0.25, 0.5, 0.75)]
    candidates = daytime + spread + files
    seen, ordered = set(), []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def make_overlay(video: str, polygon: List[List[int]], points: np.ndarray, keep: np.ndarray, out_path: str) -> bool:
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    cv2.polylines(frame, [np.asarray(polygon, np.int32)], True, (0, 200, 255), 3)
    for x, y in points[:: max(1, len(points) // 100)]:
        cv2.circle(frame, (int(x), int(y)), 2, (200, 0, 200), -1)
    for x, y in keep[:: max(1, len(keep) // 100)]:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 210, 0), -1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if ok:
        buf.tofile(out_path)
    return bool(ok)


def update_config(config_path: str, camera: str, name: str, polygon: List[List[int]]) -> dict:
    """只替换 roi，其余键原样保留。"""
    existing = {}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            existing = yaml.safe_load(handle) or {}
    config = dict(existing)
    config["camera_id"] = existing.get("camera_id", camera)
    config["camera_name"] = existing.get("camera_name", name)
    config["roi"] = {"polygon": polygon}
    if "calibration" not in config:
        config["calibration"] = {"px_per_meter": 0.0}
    if "anomaly" not in config:
        config["anomaly"] = {"contamination": 0.05}
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return {
        "kept_lines": "entry_line" in existing and "exit_line" in existing,
        "kept_calibration": bool((existing.get("calibration") or {}).get("homography")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"F:\A区录像 新")
    parser.add_argument("--config-dir", default="config/camera")
    parser.add_argument("--overlay-dir", default="data/outputs/roi_review")
    parser.add_argument("--docx", default="docs/roi_review.docx")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些机位（调试用）")
    parser.add_argument("--only-file", default=None, help="从文件读机位名单（每行一个，避免命令行中文编码问题）")
    args = parser.parse_args()

    from docx import Document
    from docx.shared import Cm, Pt

    folders = [p for p in sorted(glob(os.path.join(args.root, "*")))
               if os.path.isdir(p) and not os.path.basename(p).startswith("_")]
    wanted = set(args.only or [])
    if args.only_file and os.path.isfile(args.only_file):
        with open(args.only_file, encoding="utf-8") as handle:
            wanted |= {line.strip() for line in handle if line.strip()}
    if wanted:
        folders = [p for p in folders if os.path.basename(p) in wanted]
    if args.limit:
        folders = folders[: args.limit]

    os.makedirs(args.overlay_dir, exist_ok=True)
    document = Document()
    document.add_heading("机位 ROI 复核单", level=0)
    document.add_paragraph(
        "每页一台机位。黄线为推导出的 ROI，绿点为参与推导的运动目标，紫点为被判为静止、"
        "已剔除的停放车辆。只替换了 ROI，已有的人工进出线与现场标定均保留。"
    )

    records = []
    for index, folder in enumerate(folders, start=1):
        camera = os.path.basename(folder)
        sources = pick_files(folder)
        entry = {"camera": camera, "ok": False}
        if not sources:
            entry["error"] = "无 mbf"
        else:
            try:
                # 依次尝试多个源文件：某一段没车不代表这个机位推导不出 ROI
                best = None
                temps = []
                for candidate in sources[:3]:
                    stream = read_partial(candidate)
                    if not stream:
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as handle:
                        handle.write(stream)
                        temp_path = handle.name
                    temps.append(temp_path)
                    try:
                        points, width, height, frames = collect_points(temp_path, samples=SAMPLES, min_box=MIN_BOX)
                        keep, static_cells = drop_static(points, frames)
                        motion = len(keep) >= 8
                        candidate_result = {
                            "polygon": polygon_from_points(keep, width, height) if motion else
                                       [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                            "mode": "motion" if motion else "full_frame",
                            "frames": frames, "points": len(points), "kept": len(keep),
                            "static_cells": static_cells, "width": width, "height": height,
                            "source": os.path.basename(candidate), "temp": temp_path,
                            "points_arr": points, "keep_arr": keep, "attempts": len(temps),
                        }
                        # 拿到运动 ROI 就用它；否则先留住全画幅结果继续试下一段
                        if motion or best is None:
                            best = candidate_result
                        if motion:
                            break
                    except Exception:
                        continue

                if best is None:
                    raise RuntimeError("所有候选段都推导失败")

                entry.update({k: best[k] for k in ("mode", "frames", "points", "kept",
                                                   "static_cells", "width", "height", "source", "attempts")})
                entry["ok"] = True
                polygon = best["polygon"]
                config_path = os.path.join(args.config_dir, f"{camera}.yaml")
                if camera not in PROTECTED and best["mode"] == "motion":
                    entry.update(update_config(config_path, camera, camera, polygon))
                elif camera not in PROTECTED:
                    # 推导失败：不要用全画幅覆盖已有的有效 ROI，只在复核单里标出来
                    existing_roi = False
                    if os.path.isfile(config_path):
                        with open(config_path, encoding="utf-8") as handle:
                            existing_roi = bool((yaml.safe_load(handle) or {}).get("roi", {}).get("polygon"))
                    entry["kept_original_roi"] = existing_roi
                overlay = os.path.join(args.overlay_dir, f"{camera}.jpg")
                if make_overlay(best["temp"], polygon, best["points_arr"], best["keep_arr"], overlay):
                    entry["overlay"] = overlay
                for temp_path in temps:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
            except Exception as error:  # 单台失败不影响整体
                entry["error"] = f"{type(error).__name__}: {error}"

        records.append(entry)
        print(f"[{index}/{len(folders)}] {camera}: {entry.get('mode', entry.get('error'))} "
              f"kept={entry.get('kept')}", flush=True)

        document.add_heading(f"{index}. {camera}", level=2)
        if entry.get("ok"):
            lines = (f"来源：{entry.get('source', '')}　分辨率：{entry['width']}x{entry['height']}　"
                     f"采样 {entry.get('frames')} 帧，检出点 {entry.get('points')}，保留运动点 {entry.get('kept')}，"
                     f"剔除静止网格 {entry.get('static_cells')}")
            document.add_paragraph(lines)
            if entry.get("mode") == "full_frame":
                if entry.get("kept_original_roi"):
                    document.add_paragraph("⚠ 运动点不足：未覆盖原 ROI，请人工确认原 ROI 是否仍然正确。")
                else:
                    document.add_paragraph("⚠ 运动点不足，已退回全画幅，需要人工圈定 ROI。")
            if entry.get("kept_lines"):
                document.add_paragraph("已有手工进出线：保留（未覆盖）。")
            if entry.get("overlay") and os.path.isfile(entry["overlay"]):
                document.add_picture(entry["overlay"], width=Cm(16))
        else:
            document.add_paragraph(f"⚠ 推导失败：{entry.get('error')}")

    os.makedirs(os.path.dirname(os.path.abspath(args.docx)) or ".", exist_ok=True)
    document.save(args.docx)

    ok_count = sum(1 for r in records if r.get("ok"))
    motion = sum(1 for r in records if r.get("mode") == "motion")
    summary = {
        "cameras": len(records),
        "ok": ok_count,
        "motion_roi": motion,
        "full_frame_fallback": ok_count - motion,
        "failed": len(records) - ok_count,
        "docx": args.docx,
    }
    with open(os.path.join(args.overlay_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
