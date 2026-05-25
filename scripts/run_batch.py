#!/usr/bin/env python
"""批量处理脚本（Phase 4 全量部署用）。

对摄像头24小时分段视频批量执行流水线处理。

用法:
    python scripts/run_batch.py \\
        --camera cam01 \\
        --date 20260520 \\
        --video-dir data/videos/camera_01 \\
        --output data/outputs
"""

import argparse
import sys
import os
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.batch_processor import BatchProcessor
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description="Batch video processing")
    parser.add_argument("--camera", required=True, help="Camera ID")
    parser.add_argument("--date", required=True, help="Date string (YYYYMMDD)")
    parser.add_argument("--video-dir", required=True, help="Directory containing hourly video segments")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="data/outputs")
    args = parser.parse_args()

    logger = get_logger("run_batch")

    # 查找该日期下的所有视频段
    pattern = os.path.join(args.video_dir, f"*{args.date}*.mp4")
    video_files = sorted(glob.glob(pattern))

    if not video_files:
        logger.error(f"No videos found matching: {pattern}")
        sys.exit(1)

    logger.info(f"Found {len(video_files)} video segments for {args.camera} on {args.date}")

    # 初始化处理器
    camera_config = f"config/camera/{args.camera}.yaml"
    processor = BatchProcessor(
        config_path=args.config,
        camera_config_path=camera_config if os.path.exists(camera_config) else None,
    )

    # 逐段处理
    for video_path in video_files:
        logger.info(f">>> {os.path.basename(video_path)}")
        try:
            csv_path = processor.process_video(
                video_path=video_path,
                output_dir=args.output,
            )
            logger.info(f"    Output: {csv_path}")
        except Exception as e:
            logger.error(f"    Failed: {e}")

    logger.info("Batch processing complete.")


if __name__ == "__main__":
    main()
