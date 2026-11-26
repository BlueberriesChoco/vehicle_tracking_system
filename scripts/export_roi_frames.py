#!/usr/bin/env python
"""导出每个机位的一帧干净参考图，并生成人工标定用的清单 manifest.json。

产物：
    data/outputs/roi_frames/<机位>.jpg     干净帧（无任何叠加线）
    data/outputs/roi_frames/manifest.json  机位清单 + 尺寸 + 已有机位配置

manifest 里带出已有配置，标定工具可以在此基础上微调而不是从零画。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.mbf_extract import MBF_BLOCK_HEADER_SIZE, MBF_BLOCK_MARKER, MBF_DEFAULT_BLOCK_SIZE, parse_mbf_header

READ_BLOCKS = 12  # 约 24 MB，足够解出若干帧


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


def pick_file(camera_dir: str) -> str | None:
    files = sorted(glob(os.path.join(camera_dir, "*.mbf")))
    if not files:
        return None
    for path in files:
        if any(k in os.path.basename(path) for k in ("2026090107", "2026090108", "2026090109")):
            return path
    return files[len(files) // 2]


def existing_config(camera: str, config_dir: str) -> dict:
    path = os.path.join(config_dir, f"{camera}.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    result = {}
    polygon = (data.get("roi") or {}).get("polygon")
    if polygon:
        result["roi"] = polygon
    for key in ("entry_line", "exit_line"):
        line = data.get(key) or {}
        if line.get("p1") and line.get("p2"):
            result[key] = [line["p1"], line["p2"]]
    calibration = data.get("calibration") or {}
    if calibration:
        result["calibration"] = calibration
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"F:\A区录像 新")
    parser.add_argument("--out-dir", default="data/outputs/roi_frames")
    parser.add_argument("--config-dir", default="config/camera")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strip-existing", action="store_true", help="manifest 里不带已有 roi/进出线（人工全新标定时用）")
    parser.add_argument("--manifest-only", action="store_true", help="不重新取帧，只用已有 jpg 重建 manifest")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    folders = [p for p in sorted(glob(os.path.join(args.root, "*")))
               if os.path.isdir(p) and not os.path.basename(p).startswith("_")]
    if args.limit:
        folders = folders[: args.limit]

    manifest = []
    for index, folder in enumerate(folders, start=1):
        camera = os.path.basename(folder)
        source = pick_file(folder)
        existing = {} if args.strip_existing else existing_config(camera, args.config_dir)
        entry = {"camera": camera, "frame": None, "width": 0, "height": 0, **existing}
        if args.manifest_only:
            frame_path = os.path.join(args.out_dir, f"{camera}.jpg")
            if os.path.isfile(frame_path):
                import cv2 as _cv2
                import numpy as _np
                image = _cv2.imdecode(_np.fromfile(frame_path, dtype=_np.uint8), _cv2.IMREAD_COLOR)
                if image is not None:
                    entry.update({"frame": f"{camera}.jpg", "height": image.shape[0], "width": image.shape[1]})
            manifest.append(entry)
            continue
        if not source:
            entry["error"] = "无 mbf"
            manifest.append(entry)
            continue
        try:
            stream = read_partial(source)
            temp = os.path.join(args.out_dir, f".{camera}.tmp.h264")
            with open(temp, "wb") as handle:
                handle.write(stream)
            cap = cv2.VideoCapture(temp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 取中间帧，避开片头黑屏
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
            ok, frame = cap.read()
            cap.release()
            os.unlink(temp)
            if not ok:
                entry["error"] = "取帧失败"
            else:
                height, width = frame.shape[:2]
                frame_path = os.path.join(args.out_dir, f"{camera}.jpg")
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok2:
                    buf.tofile(frame_path)
                    entry.update({"frame": f"{camera}.jpg", "width": width, "height": height})
                else:
                    entry["error"] = "编码失败"
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        manifest.append(entry)
        print(f"[{index}/{len(folders)}] {camera}: {entry.get('width')}x{entry.get('height')} {entry.get('error', '')}",
              flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"cameras": manifest}, handle, ensure_ascii=False, indent=2)
    ok_count = sum(1 for item in manifest if item.get("frame"))
    print(json.dumps({"cameras": len(manifest), "frames": ok_count, "out": args.out_dir}, ensure_ascii=False))


if __name__ == "__main__":
    main()
