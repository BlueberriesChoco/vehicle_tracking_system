"""单应矩阵（homography）求解工具。

现场标定时，测量 4 个以上地面控制点的像素坐标和米制坐标，
即可求出把像素坐标映射到地面平面米制坐标的单应矩阵。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def parse_point_list(values: Iterable) -> List[List[float]]:
    """解析 ``"x1,y1;x2,y2;..."`` 或嵌套列表为点列表。"""
    if isinstance(values, str):
        points = []
        for token in values.split(";"):
            token = token.strip()
            if not token:
                continue
            parts = token.split(",")
            if len(parts) != 2:
                raise ValueError(f"无法解析点: {token}")
            points.append([float(parts[0]), float(parts[1])])
        return points

    points = []
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            points.append([float(item[0]), float(item[1])])
        else:
            raise ValueError(f"无法解析点: {item}")
    return points


def compute_homography(pixel_points: Sequence, world_points: Sequence) -> np.ndarray:
    """由像素坐标和地面米制坐标求解 3x3 单应矩阵。"""
    import cv2

    pixels = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    world = np.asarray(world_points, dtype=np.float32).reshape(-1, 1, 2)
    if len(pixels) < 4 or len(pixels) != len(world):
        raise ValueError("至少需要 4 组一一对应的点")

    matrix, _ = cv2.findHomography(pixels, world, 0)
    if matrix is None:
        raise ValueError("无法求解单应矩阵，请检查点是否共线或重复")
    return matrix.astype(np.float64)


def apply_homography(matrix: np.ndarray, x: float, y: float):
    """把像素坐标 (x, y) 映射到地面米制坐标。"""
    h = np.asarray(matrix, dtype=np.float64)
    projected = h @ np.array([float(x), float(y), 1.0], dtype=np.float64)
    if abs(projected[2]) < 1e-9:
        raise ValueError("点映射到无穷远，单应矩阵可能不适用")
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])
