from typing import List, Tuple
import numpy as np
from scipy.signal import savgol_filter


class TrajectorySmoother:
    """轨迹平滑器。

    对轨迹坐标序列做平滑处理，去除检测框抖动造成的飞点，
    为后续速度计算和行为分析提供稳定的位置序列。
    """

    def __init__(self, window_size: int = 5, method: str = "savgol"):
        """
        Args:
            window_size: 平滑窗口大小（必须为奇数）
            method: 平滑方法 ("savgol" / "moving_avg" / "median")
        """
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.method = method

    def smooth(
        self,
        centers: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """对轨迹中心点坐标序列进行平滑。

        Args:
            centers: 底部中心点坐标列表 [(cx, cy), ...]

        Returns:
            平滑后的坐标列表
        """
        if len(centers) < self.window_size:
            return list(centers)

        xs = np.array([c[0] for c in centers], dtype=np.float64)
        ys = np.array([c[1] for c in centers], dtype=np.float64)

        if self.method == "savgol":
            poly_order = min(3, self.window_size - 1)
            xs_smooth = savgol_filter(xs, self.window_size, poly_order)
            ys_smooth = savgol_filter(ys, self.window_size, poly_order)
        elif self.method == "moving_avg":
            kernel = np.ones(self.window_size) / self.window_size
            xs_smooth = np.convolve(xs, kernel, mode="same")
            ys_smooth = np.convolve(ys, kernel, mode="same")
        elif self.method == "median":
            from scipy.ndimage import median_filter
            xs_smooth = median_filter(xs, size=self.window_size)
            ys_smooth = median_filter(ys, size=self.window_size)
        else:
            return list(centers)

        return [(float(xs_smooth[i]), float(ys_smooth[i])) for i in range(len(centers))]

    def smooth_single_track(
        self,
        centers: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """平滑并返回 numpy 数组格式。"""
        smoothed = self.smooth(centers)
        xs = np.array([s[0] for s in smoothed])
        ys = np.array([s[1] for s in smoothed])
        return xs, ys
