from typing import List, Tuple, Optional
import numpy as np
import math
import json


class PathReference:
    """通道正常路径参考线。

    用于计算车辆的路径偏离度。通过统计多条正常轨迹构建参考曲线，
    后续对每条目标轨迹计算其到参考路径的偏离距离。
    """

    def __init__(self, reference_path: Optional[np.ndarray] = None):
        """
        Args:
            reference_path: shape (N, 2) 参考路径点序列，None 则后续通过 build 构建
        """
        self.reference_path = reference_path
        self._segments = None  # 预计算的线段列表

    def build_from_trajectories(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]],
        num_points: int = 100,
    ):
        """从多条轨迹构建参考路径（取中位数线）。

        Args:
            trajectories: [(xs, ys), ...] 每条轨迹的坐标数组
            num_points: 参考路径的采样点数
        """
        if not trajectories:
            return

        # 对所有轨迹做线性插值到统一长度
        resampled = []
        for xs, ys in trajectories:
            if len(xs) < 2:
                continue
            # 计算累积弧长
            dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
            cum_dist = np.insert(np.cumsum(dists), 0, 0)
            total_len = cum_dist[-1]
            if total_len == 0:
                continue
            # 等距采样
            sample_dists = np.linspace(0, total_len, num_points)
            interp_x = np.interp(sample_dists, cum_dist, xs)
            interp_y = np.interp(sample_dists, cum_dist, ys)
            resampled.append(np.column_stack([interp_x, interp_y]))

        if not resampled:
            return

        stacked = np.stack(resampled, axis=0)  # (N, num_points, 2)
        median_path = np.median(stacked, axis=0)  # (num_points, 2)
        self.reference_path = median_path
        self._build_segments()

    def _build_segments(self):
        """预计算参考路径的线段列表，加速后续点到路径的距离计算。"""
        if self.reference_path is None or len(self.reference_path) < 2:
            self._segments = []
            return
        pts = self.reference_path
        self._segments = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    def compute_deviation(
        self,
        centers: List[Tuple[float, float]],
    ) -> Tuple[float, float, float]:
        """计算给定轨迹到参考路径的偏离度。

        Returns:
            mean_deviation: 平均法向偏离距离（像素）
            max_deviation: 最大法向偏离距离（像素）
            dtw_distance: DTW 距离（归一化后）
        """
        if self._segments is None or len(self._segments) == 0:
            return 0.0, 0.0, 0.0

        deviations = []
        for cx, cy in centers:
            min_dist = float("inf")
            for (ax, ay), (bx, by) in self._segments:
                dist = self._point_to_segment(cx, cy, ax, ay, bx, by)
                if dist < min_dist:
                    min_dist = dist
            deviations.append(min_dist)

        mean_dev = float(np.mean(deviations)) if deviations else 0.0
        max_dev = float(np.max(deviations)) if deviations else 0.0

        # 简化 DTW：直接用平均偏离 + 方差近似
        std_dev = float(np.std(deviations)) if len(deviations) > 1 else 0.0
        dtw_approx = mean_dev + std_dev

        return mean_dev, max_dev, dtw_approx

    @staticmethod
    def _point_to_segment(px, py, ax, ay, bx, by) -> float:
        """点到线段的最短距离。"""
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def compute_smoothness(
        self,
        centers: List[Tuple[float, float]],
    ) -> float:
        """计算路径平滑度（曲率累积）。

        值越小越平滑（沿直线行驶），越大越"犹豫"。
        """
        if len(centers) < 3:
            return 0.0
        total_curvature = 0.0
        for i in range(1, len(centers) - 1):
            total_curvature += self._angle_between(
                centers[i - 1], centers[i], centers[i + 1]
            )
        # 归一化：除以轨迹点数
        return total_curvature / (len(centers) - 2)

    @staticmethod
    def _angle_between(p0, p1, p2) -> float:
        """计算三个连续点之间的转向角（弧度）。"""
        v1 = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        v2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(abs(np.arccos(cos_theta)))

    def save(self, filepath: str):
        """保存参考路径到 JSON。"""
        if self.reference_path is not None:
            data = {"path": self.reference_path.tolist()}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f)

    def load(self, filepath: str):
        """从 JSON 加载参考路径。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.reference_path = np.array(data["path"])
        self._build_segments()
