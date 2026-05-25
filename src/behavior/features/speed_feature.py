from typing import List, Tuple
import numpy as np

from ...trajectory.scene_geometry import SceneGeometry


class SpeedFeatureExtractor:
    """速度特征提取器。

    从平滑轨迹中计算瞬时速度序列，提取平均速度、最大速度、速度方差。
    速度从像素/帧转换为米/秒。
    """

    def __init__(
        self,
        scene_geometry: SceneGeometry,
        frame_rate: int = 25,
        dwell_threshold_ms: float = 0.5,
    ):
        self.scene = scene_geometry
        self.frame_rate = frame_rate
        self.dwell_threshold_ms = dwell_threshold_ms

    def compute_instant_speeds(
        self,
        xs_smooth: np.ndarray,
        ys_smooth: np.ndarray,
    ) -> np.ndarray:
        """计算瞬时速度序列 (m/s)。"""
        if len(xs_smooth) < 2:
            return np.array([0.0])

        speeds = []
        for i in range(1, len(xs_smooth)):
            dx = xs_smooth[i] - xs_smooth[i - 1]
            dy = ys_smooth[i] - ys_smooth[i - 1]
            dist_px = np.sqrt(dx * dx + dy * dy)
            dist_m = self.scene.pixel_to_world(dist_px)
            # 速度 = 距离(米) / 时间(秒), 时间 = 1/frame_rate
            speed_ms = dist_m * self.frame_rate
            speeds.append(speed_ms)

        return np.array(speeds)

    def extract(
        self,
        xs_smooth: np.ndarray,
        ys_smooth: np.ndarray,
    ) -> dict:
        """提取速度相关特征。

        Returns:
            dict with avg_speed_ms, max_speed_ms, speed_variance
        """
        speeds = self.compute_instant_speeds(xs_smooth, ys_smooth)

        if len(speeds) == 0:
            return {
                "avg_speed_ms": 0.0,
                "max_speed_ms": 0.0,
                "speed_variance": 0.0,
                "instant_speeds": [],
            }

        return {
            "avg_speed_ms": round(float(np.mean(speeds)), 3),
            "max_speed_ms": round(float(np.max(speeds)), 3),
            "speed_variance": round(float(np.var(speeds)), 3),
            "instant_speeds": speeds.tolist(),
        }
