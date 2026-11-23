from typing import List, Optional
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
    ):
        self.scene = scene_geometry
        self.frame_rate = frame_rate
        # 最近一次计算中被判定为跟踪跳变、并被剔除的帧数（供审计）
        self.speed_outlier_count = 0

    def _reject_spikes(self, speeds: np.ndarray) -> np.ndarray:
        """剔除跟踪跳变造成的异常瞬时速度。

        单帧内检测框跳变（换 ID、遮挡后重新关联）会在轨迹上留下几十到上百像素的
        位移，换算成速度可达 100 km/h 以上，足以让 max_speed 和超速规则全部失真。
        真实车辆在闸口车道的瞬时速度不会超过中位速度的 3 倍，超出者按中位数回填。
        """
        self.speed_outlier_count = 0
        if len(speeds) < 5:
            return speeds
        median = float(np.median(speeds))
        if median <= 0:
            return speeds
        limit = max(3.0 * median, 1.0)  # m/s
        mask = speeds > limit
        self.speed_outlier_count = int(np.count_nonzero(mask))
        if self.speed_outlier_count == 0:
            return speeds
        cleaned = speeds.copy()
        cleaned[mask] = median
        return cleaned

    def compute_instant_speeds(
        self,
        xs_smooth: np.ndarray,
        ys_smooth: np.ndarray,
        timestamps: Optional[List] = None,
    ) -> np.ndarray:
        """计算瞬时速度序列 (m/s)。"""
        if len(xs_smooth) < 2:
            return np.array([0.0])

        speeds = []
        for i in range(1, len(xs_smooth)):
            # 距离（米）：有单应矩阵时按地面坐标计算，否则按常数 px_per_meter 近似
            dist_m = self.scene.ground_distance(
                (xs_smooth[i - 1], ys_smooth[i - 1]),
                (xs_smooth[i], ys_smooth[i]),
            )
            # 速度 = 距离(米) / 时间(秒), 时间 = 1/frame_rate
            delta_sec = 1.0 / self.frame_rate
            if timestamps is not None and len(timestamps) == len(xs_smooth):
                measured_delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
                if measured_delta > 0:
                    delta_sec = measured_delta
            speed_ms = dist_m / delta_sec
            speeds.append(speed_ms)

        return self._reject_spikes(np.array(speeds))

    def extract(
        self,
        xs_smooth: np.ndarray,
        ys_smooth: np.ndarray,
        timestamps: Optional[List] = None,
    ) -> dict:
        """提取速度相关特征。

        Returns:
            dict with avg_speed_ms, max_speed_ms, speed_variance
        """
        speeds = self.compute_instant_speeds(xs_smooth, ys_smooth, timestamps)

        if len(speeds) == 0:
            return {
                "avg_speed_ms": 0.0,
                "max_speed_ms": 0.0,
                "speed_variance": 0.0,
                "speed_cv": 0.0,
                "instant_speeds": [],
            }

        return {
            "avg_speed_ms": round(float(np.mean(speeds)), 3),
            "max_speed_ms": round(float(np.max(speeds)), 3),
            "speed_variance": round(float(np.var(speeds)), 3),
            # 速度变异系数 CV = 标准差 / 均值，衡量车速稳定性（研究要点一要求的特征）
            "speed_cv": round(
                float(np.std(speeds)) / float(np.mean(speeds)), 4
            ) if float(np.mean(speeds)) > 1e-6 else 0.0,
            "speed_outlier_count": int(self.speed_outlier_count),
            "instant_speeds": speeds.tolist(),
        }
