#!/usr/bin/env python
"""Convert MBF surveillance recordings to decodable H.264 streams.

The .mbf container is a proprietary wrapper around a standard H.264 Annex B
stream. This tool strips the container headers so OpenCV / FFmpeg can decode
the video directly.

Usage:
    python scripts/extract_mbf.py --input "F:/A区录像 新/行政车道进" --output data/outputs/h264/admin_in
    python scripts/extract_mbf.py --input one_file.mbf --output one_file.h264 --verify
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.mbf_extract import extract_mbf_file
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert .mbf recordings to .h264")
    parser.add_argument("--input", required=True, help="A .mbf file or a directory containing .mbf files")
    parser.add_argument("--output", default=None, help="Output .h264 file or output directory")
    parser.add_argument("--verify", action="store_true", help="Open the extracted stream with OpenCV to verify")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error("Input path does not exist: %s", args.input)
        sys.exit(2)

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.mbf")))
        if not files:
            logger.error("No .mbf files found in %s", args.input)
            sys.exit(1)
        out_dir = args.output or os.path.join(args.input, "_h264")
        os.makedirs(out_dir, exist_ok=True)
    else:
        files = [args.input]
        # 单文件输入时，--output 既可以是文件路径，也可以是已存在的目录
        if args.output and (os.path.isdir(args.output) or args.output.endswith(("/", "\\"))):
            out_dir = args.output
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = None

    converted = 0
    for path in files:
        if out_dir is not None:
            name = os.path.splitext(os.path.basename(path))[0] + ".h264"
            target = os.path.join(out_dir, name)
        else:
            target = args.output or (os.path.splitext(path)[0] + ".h264")

        extract_mbf_file(path, target)
        converted += 1
        size_mb = round(os.path.getsize(target) / 1024 / 1024, 1)
        print(f"[{converted}/{len(files)}] {os.path.basename(path)} -> {target} ({size_mb} MB)")

        if args.verify:
            import cv2

            cap = cv2.VideoCapture(target)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok = cap.isOpened() and frames > 0
            cap.release()
            print(f"    verify: opened={ok}, {width}x{height}, frames={frames}")
            if not ok:
                logger.error("Extracted stream failed to open: %s", target)

    logger.info("Converted %s file(s)", converted)


if __name__ == "__main__":
    main()
