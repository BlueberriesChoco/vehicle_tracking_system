#!/usr/bin/env python
"""车辆行为向量化主运行脚本。

对输入视频执行检测→跟踪→轨迹提取→行为向量构建流水线，
输出标准化的行为向量 CSV 文件。

用法:
    python scripts/run_pipeline.py \\
        --video data/test_videos/test_cam01_5min.mp4 \\
        --camera cam01 \\
        --output data/outputs

    python scripts/run_pipeline.py \\
        --video data/videos/camera_01/ch01_20260520_080000_090000.mp4 \\
        --camera cam01 \\
        --output data/outputs \\
        --output-video data/outputs/annotated.mp4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.batch_processor import BatchProcessor
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(
        description="车辆行为向量化流水线"
    )
    parser.add_argument(
        "--video", required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--camera", default="cam01",
        help="摄像头编号 (对应 config/camera/camera_XX.yaml)",
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="全局配置文件路径",
    )
    parser.add_argument(
        "--output", default="data/outputs",
        help="输出根目录",
    )
    parser.add_argument(
        "--output-video", default=None,
        help="标注视频输出路径（可选）",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="最大处理帧数（0=不限制，调试用）",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU 设备编号（-1=CPU）",
    )
    args = parser.parse_args()

    logger = get_logger("run_pipeline")

    # 摄像头配置文件
    camera_config = f"config/camera/{args.camera}.yaml"
    if not os.path.exists(camera_config):
        logger.warning(f"Camera config not found: {camera_config}, using defaults")

    # 构建处理器
    processor = BatchProcessor(
        config_path=args.config,
        camera_config_path=camera_config if os.path.exists(camera_config) else None,
    )

    # 应用命令行覆盖
    if args.max_frames > 0:
        processor.max_frames = args.max_frames
    if args.device != 0:
        processor.detector.device = args.device

    logger.info(f"Camera: {args.camera}")
    logger.info(f"Video: {args.video}")
    logger.info(f"Output dir: {args.output}")

    # 运行
    csv_path = processor.process_video(
        video_path=args.video,
        output_dir=args.output,
        output_video_path=args.output_video,
    )

    logger.info(f"Done. Behavior vectors saved to: {csv_path}")
    print(f"\nOutput: {csv_path}")


if __name__ == "__main__":
    main()
