#!/usr/bin/env python
"""Render a camera config (ROI / entry line / exit line) onto a real frame.

This is a fast visual check that the configured geometry actually matches the
scene before trusting passage direction and ROI filtering.

Usage:
    python scripts/render_camera_config.py \
        --video data/outputs/h264/admin_in/clip.h264 \
        --camera admin_in \
        --frame 1500 \
        --output data/outputs/calib_check.png
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="Render camera config on a frame")
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", default="admin_in")
    parser.add_argument("--config", default=None, help="Camera YAML (default config/camera/<id>.yaml)")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = args.config or os.path.join("config", "camera", f"{args.camera}.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        sys.exit(1)
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Cannot read frame")
        sys.exit(1)

    roi = (config.get("roi") or {}).get("polygon") or []
    if roi:
        points = np.array(roi, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=4)
        cv2.putText(frame, "ROI", tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    for key, color, label in (
        ("entry_line", (0, 0, 255), "ENTRY"),
        ("exit_line", (255, 0, 0), "EXIT"),
    ):
        line = config.get(key) or {}
        p1, p2 = line.get("p1"), line.get("p2")
        if p1 and p2:
            cv2.line(frame, tuple(p1), tuple(p2), color, 4)
            cv2.putText(frame, label, (p1[0], max(30, p1[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, frame)
    print(f"Saved: {args.output}")
    print(f"ROI points: {len(roi)}")


if __name__ == "__main__":
    main()
