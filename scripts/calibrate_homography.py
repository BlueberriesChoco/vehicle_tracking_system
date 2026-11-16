#!/usr/bin/env python
"""Compute a camera homography from ground control points.

Provide at least 4 pixel/world point pairs; the script prints the 3x3 matrix and
optionally writes it into ``config/camera/<camera>.yaml`` under
``calibration.homography``. The pipeline then treats speed / path / aggregation
as reliable (``geometry_level = calibrated``).

Usage:
    python scripts/calibrate_homography.py \
        --pixels "100,200;800,200;800,900;100,900" \
        --world  "0,0;10,0;10,15;0,15" \
        --camera admin_in \
        --update-config
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trajectory.homography import compute_homography, parse_point_list
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute camera homography from control points")
    parser.add_argument("--pixels", required=True, help='Pixel points, e.g. "100,200;800,200;800,900;100,900"')
    parser.add_argument("--world", required=True, help='World points in metres, e.g. "0,0;10,0;10,15;0,15"')
    parser.add_argument("--camera", default="admin_in", help="Camera id for config update")
    parser.add_argument("--config", default=None, help="Camera config path (default config/camera/<camera>.yaml)")
    parser.add_argument("--update-config", action="store_true", help="Write homography into the camera config")
    args = parser.parse_args()

    pixels = parse_point_list(args.pixels)
    world = parse_point_list(args.world)
    if len(pixels) != len(world):
        logger.error("像素点与世界点数量不一致: %s vs %s", len(pixels), len(world))
        sys.exit(1)

    matrix = compute_homography(pixels, world)
    print("Homography (3x3):")
    print(json.dumps(matrix.tolist(), ensure_ascii=False, indent=2))

    if args.update_config:
        config_path = args.config or f"config/camera/{args.camera}.yaml"
        if not os.path.exists(config_path):
            logger.error("Camera config not found: %s", config_path)
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        calibration = config.setdefault("calibration", {})
        calibration["homography"] = matrix.tolist()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        logger.info("Homography written to %s", config_path)
        print(f"Updated: {config_path}")


if __name__ == "__main__":
    main()
