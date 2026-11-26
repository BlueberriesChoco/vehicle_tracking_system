from typing import List, Tuple
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
        geometry_level: str = "none",
        has_roi: bool = False,
        homography=None,
        bidirectional: bool = False,
    ):
        """
        Args:
            entry_line: 入口线段 ((x1,y1), (x2,y2))，也接受多条线的列表 [[p1,p2], [p1,p2], ...]
            exit_line: 出口线段 ((x1,y1), (x2,y2))，同上；跨任意一条即算通过
            px_per_meter: 像素/米换算比例（无单应矩阵时的近似）
            homography: 可选 3x3 单应矩阵，把像素坐标映射到地面米制坐标
            bidirectional: 双向车道。True 时「先跨任一条线记为进入、再跨另一条记为离开」，
                这样两个方向的车都能形成通行记录；False 时要求先入口后出口（单向）
        """
        self.entry_lines = self._normalize_lines(entry_line)
        self.exit_lines = self._normalize_lines(exit_line)
        self.px_per_meter = px_per_meter
        self.geometry_level = geometry_level
        self.has_roi = has_roi
        self.bidirectional = bool(bidirectional)

        self.homography = None
        if homography is not None:
            matrix = np.asarray(homography, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError("homography 必须是 3x3 矩阵")
            self.homography = matrix

    @staticmethod
    def _normalize_lines(raw) -> list:
        """把「单条线」「多条线的列表」「{p1,p2} 字典列表」统一成 [((x1,y1),(x2,y2)), ...]。"""
        if not raw:
            return []
        if isinstance(raw, dict):
            raw = [raw]

        def _is_point(value) -> bool:
            return (isinstance(value, (list, tuple)) and len(value) == 2
                    and all(isinstance(item, (int, float)) for item in value))

        # 单条线：((x1,y1), (x2,y2))
        if (isinstance(raw, (list, tuple)) and len(raw) == 2
                and _is_point(raw[0]) and _is_point(raw[1])):
            return [(tuple(raw[0]), tuple(raw[1]))]

        lines = []
        for item in raw:
            if isinstance(item, dict):
                p1, p2 = item.get("p1"), item.get("p2")
                if _is_point(p1) and _is_point(p2):
                    lines.append((tuple(p1), tuple(p2)))
            elif (isinstance(item, (list, tuple)) and len(item) == 2
                  and _is_point(item[0]) and _is_point(item[1])):
                # [[x1,y1], [x2,y2]]
                lines.append((tuple(item[0]), tuple(item[1])))
        return lines

    # 兼容旧代码：取第一条线
    @property
    def entry_p1(self):
        return np.array(self.entry_lines[0][0], dtype=np.float64) if self.entry_lines else np.zeros(2)

    @property
    def entry_p2(self):
        return np.array(self.entry_lines[0][1], dtype=np.float64) if self.entry_lines else np.zeros(2)

    @property
    def exit_p1(self):
        return np.array(self.exit_lines[0][0], dtype=np.float64) if self.exit_lines else np.zeros(2)

    @property
    def exit_p2(self):
        return np.array(self.exit_lines[0][1], dtype=np.float64) if self.exit_lines else np.zeros(2)

    @property
    def has_homography(self) -> bool:
        return self.homography is not None

    def pixel_to_ground(self, x: float, y: float) -> Tuple[float, float]:
        """像素坐标 → 地面米制坐标。无单应矩阵时退化为常数比例。"""
        if self.homography is None:
            scale = self.px_per_meter if self.px_per_meter > 0 else 1.0
            return float(x) / scale, float(y) / scale

        projected = self.homography @ np.array([float(x), float(y), 1.0], dtype=np.float64)
        if abs(projected[2]) < 1e-9:
            return 0.0, 0.0
        return float(projected[0] / projected[2]), float(projected[1] / projected[2])

    def ground_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """两点之间的地面距离（米）。"""
        if self.homography is None:
            if self.px_per_meter <= 0:
                return 0.0
            return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]) / self.px_per_meter)

        x1, y1 = self.pixel_to_ground(*p1)
        x2, y2 = self.pixel_to_ground(*p2)
        return float(math.hypot(x2 - x1, y2 - y1))

    def check_line_crossing(
        self,
        p_prev: Tuple[float, float],
        p_curr: Tuple[float, float],
        line: str = "entry",
    ) -> bool:
        """判断从 p_prev 到 p_curr 是否跨越了入口/出口线（跨任意一条即算）。

        使用跨立实验（叉积符号判断）。
        """
        lines = self.entry_lines if line == "entry" else self.exit_lines
        c = np.array(p_prev, dtype=np.float64)
        d = np.array(p_curr, dtype=np.float64)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        for p1, p2 in lines:
            a = np.array(p1, dtype=np.float64)
            b = np.array(p2, dtype=np.float64)
            d1 = cross(c, a, b)
            d2 = cross(d, a, b)
            d3 = cross(a, c, d)
            d4 = cross(b, c, d)
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                return True
        return False

    def has_valid_entry_line(self) -> bool:
        """入口线是否有效（至少一条线两端点不同）。"""
        return any(not np.array_equal(np.array(p1), np.array(p2)) for p1, p2 in self.entry_lines)

    def has_valid_exit_line(self) -> bool:
        """出口线是否有效（至少一条线两端点不同）。"""
        return any(not np.array_equal(np.array(p1), np.array(p2)) for p1, p2 in self.exit_lines)

    @property
    def speed_reliable(self) -> int:
        return int(self.geometry_level == "calibrated")

    @property
    def path_reliable(self) -> int:
        return int(self.geometry_level == "calibrated")

    @property
    def aggregation_reliable(self) -> int:
        return int(self.geometry_level == "calibrated")

    @property
    def passage_reliable(self) -> int:
        return int(
            self.has_roi
            and self.has_valid_entry_line()
            and self.has_valid_exit_line()
        )

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
        total = 0.0
        for i in range(1, len(centers)):
            total += self.ground_distance(centers[i - 1], centers[i])
        return total

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
