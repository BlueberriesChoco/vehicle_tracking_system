#!/usr/bin/env python
"""把现场填好的标定表转换成单应矩阵。

现场量的是 5 个距离（W/g/d3/d4/d5），而 `calibrate_homography.py` 需要
「像素坐标 ↔ 地面米制坐标」的成对点。本脚本按固定几何约定换算，避免手工拼参数出错。

几何约定（与 docs/calibration_guide.png 一致）：
    原点 P0 = 路缘石轴（Y 轴）与减速带轴（X 轴）的交点
    减速带轴向 = +X，路缘石向远处 = +Y
    P1 = (g, 0)          P1 在减速带轴上、距路缘石 g
    P2 = (g + W, 0)      P2 沿减速带再走 W
    P3 = (0, d3)  P4 = (0, d4)  P5 = (0, d5)

用法：
    先填 config/calibration_measurements.csv，然后：
    python scripts/calibration_from_measurements.py --dry-run     # 只看换算结果
    python scripts/calibration_from_measurements.py --apply       # 写入摄像头配置

注意：该换算假设「减速带大致垂直于路缘石」。写入后请检查重投影误差，
误差明显偏大（> 2%）时说明现场几何与此假设不符，需要现场补测一个角度或改用更多点。
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TABLE = os.path.join(ROOT, "config", "calibration_measurements.csv")

PIXELS = {
    "P1": (496, 676),
    "P2": (1712, 845),
    "P3": (435, 646),
    "P4": (626, 568),
    "P5": (764, 511),
}

KEYS = ["W", "g", "d3", "d4", "d5"]


def parse_table(path):
    """从 Markdown 表格里读出已填的数值。返回 {key: float}。"""
    values = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip().startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            key = cells[0].strip("`").strip()
            if key not in KEYS and not key.startswith("check_"):
                continue
            raw = cells[-1]
            match = re.search(r"-?\d+(?:\.\d+)?", raw)
            if match:
                values[key] = float(match.group(0))
    return values


def build_points(values):
    """按几何约定生成 (pixels, world) 两组字符串。"""
    missing = [key for key in KEYS if key not in values]
    if missing:
        raise ValueError(f"标定表缺少未填项：{', '.join(missing)}")

    world = {
        "P1": (values["g"], 0.0),
        "P2": (values["g"] + values["W"], 0.0),
        "P3": (0.0, values["d3"]),
        "P4": (0.0, values["d4"]),
        "P5": (0.0, values["d5"]),
    }
    pixels_str = ";".join(f"{PIXELS[name][0]},{PIXELS[name][1]}" for name in ("P1", "P2", "P3", "P4", "P5"))
    world_str = ";".join(f"{world[name][0]:.2f},{world[name][1]:.2f}" for name in ("P1", "P2", "P3", "P4", "P5"))
    return pixels_str, world_str, world


def main():
    parser = argparse.ArgumentParser(description="Convert field measurements into a homography")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--camera", default="admin_in")
    parser.add_argument("--apply", action="store_true", help="Write homography into the camera config")
    parser.add_argument("--dry-run", action="store_true", help="Only print the derived points")
    args = parser.parse_args()

    values = parse_table(args.table)
    pixels_str, world_str, world = build_points(values)

    print("读取到的实测值（米）：", {k: values[k] for k in KEYS})
    print("\n世界坐标（按 P0 为原点、减速带为 X、路缘石为 Y）：")
    for name, (x, y) in world.items():
        print(f"  {name}: ({x:.2f}, {y:.2f})")
    print("\n--pixels:", pixels_str)
    print("--world :", world_str)

    if values.get("check_tape") and values.get("d5"):
        diff = abs(values["check_tape"] - values["d5"])
        print(f"\n自检：check_tape 与 d5 相差 {diff:.2f} 米", "（正常）" if diff <= 0.1 else "（偏大，建议复测）")

    if args.dry_run or not args.apply:
        print("\n（未写入配置。确认无误后加 --apply）")
        return

    command = [
        sys.executable,
        os.path.join(ROOT, "scripts", "calibrate_homography.py"),
        "--pixels", pixels_str,
        "--world", world_str,
        "--camera", args.camera,
        "--update-config",
    ]
    print("\n执行：", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
