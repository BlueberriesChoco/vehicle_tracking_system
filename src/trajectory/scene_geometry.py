from typing import List, Tuple, Optional
import numpy as np
import math


class SceneGeometry:
    """场景几何工具类。

    管理行政通道的几何定义：入口线、出口线、ROI边界、像素/世界坐标映射。
    提供线段交叉判断、点到线距离等基础几何运算。
    """

    def __init__(
        self,
        entry_line: Tuple[Tuple[float, float], Tuple[float, float]],
        exit_line: Tuple[Tuple[float, float], Tuple[float, float]],
        px_per_meter: float = 20.0,
    ):
        """
        Args:
            entry_line: 入口线段 ((x1,y1), (x2,y2))
            exit_line: 出口线段 ((x1,y1), (x2,y2))
            px_per_meter: 像素/米换算比例
        """
        self.entry_p1 = np.array(entry_line[0], dtype=np.float64)
        self.entry_p2 = np.array(entry_line[1], dtype=np.float64)
        self.exit_p1 = np.array(exit_line[0], dtype=np.float64)
        self.exit_p2 = np.array(exit_line[1], dtype=np.float64)
        self.px_per_meter = px_per_meter

    def check_line_crossing(
        self,
        p_prev: Tuple[float, float],
        p_curr: Tuple[float, float],
        line: str = "entry",
    ) -> bool:
        """判断从 p_prev 到 p_curr 是否跨越了指定线段（entry/exit）。

        使用跨立实验（叉积符号判断）。
        """
        a, b = (self.entry_p1, self.entry_p2) if line == "entry" else (self.exit_p1, self.exit_p2)
        c = np.array(p_prev, dtype=np.float64)
        d = np.array(p_curr, dtype=np.float64)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(c, a, b)
        d2 = cross(d, a, b)
        d3 = cross(a, c, d)
        d4 = cross(b, c, d)

        if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
            return True
        return False

    def has_valid_entry_line(self) -> bool:
        """入口线是否有效（两端点不同）。"""
        return not np.array_equal(self.entry_p1, self.entry_p2)

    def has_valid_exit_line(self) -> bool:
        """出口线是否有效（两端点不同）。"""
        return not np.array_equal(self.exit_p1, self.exit_p2)

    def pixel_to_world(self, px_distance: float) -> float:
        """像素距离 → 世界距离（米）。"""
        return px_distance / self.px_per_meter if self.px_per_meter > 0 else 0.0

    def world_to_pixel(self, meters: float) -> float:
        """世界距离 → 像素距离。"""
        return meters * self.px_per_meter

    def trajectory_length_meters(
        self,
        centers: List[Tuple[float, float]],
    ) -> float:
        """计算轨迹总长度（米）。"""
        if len(centers) < 2:
            return 0.0
        total_px = 0.0
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            total_px += math.sqrt(dx * dx + dy * dy)
        return self.pixel_to_world(total_px)

    @staticmethod
    def point_to_line_distance(
        point: Tuple[float, float],
        line_p1: Tuple[float, float],
        line_p2: Tuple[float, float],
    ) -> float:
        """点到线段的最短距离。"""
        px, py = point
        x1, y1 = line_p1
        x2, y2 = line_p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
