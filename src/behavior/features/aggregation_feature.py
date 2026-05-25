from typing import List, Tuple, Dict
import numpy as np

from ...trajectory.scene_geometry import SceneGeometry


class AggregationFeatureExtractor:
    """聚集行为特征提取器。

    分析车辆在通道内行驶时与其他车辆的时空邻近度，
    计算聚集行为指数（车辆成群结队通过的程度）。
    """

    def __init__(self, scene_geometry: SceneGeometry, aggregation_radius_m: float = 5.0):
        self.scene = scene_geometry
        self.aggregation_radius_m = aggregation_radius_m
        self.aggregation_radius_px = scene_geometry.world_to_pixel(aggregation_radius_m)

    def extract(
        self,
        centers: List[Tuple[float, float]],
        all_active_positions: Dict[int, Tuple[float, float]],
    ) -> dict:
        """计算聚集行为指数。

        Args:
            centers: 当前车辆的轨迹中心点
            all_active_positions: 同时段其他车辆的 {track_id: (cx, cy)}

        Returns:
            dict with aggregation_index, nearest_vehicle_m, max_concurrent_vehicles
        """
        if not centers or len(all_active_positions) <= 1:
            return {
                "aggregation_index": 0.0,
                "nearest_vehicle_m": -1.0,
                "max_concurrent_vehicles": max(1, len(all_active_positions)),
            }

        # 对每帧计算当前车辆与其他车辆的最小距离
        aggregation_frames = 0
        valid_frames = 0

        for cx, cy in centers:
            valid_frames += 1
            min_dist_px = float("inf")
            for other_id, (ox, oy) in all_active_positions.items():
                dist_px = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                if dist_px < min_dist_px:
                    min_dist_px = dist_px

            if min_dist_px < self.aggregation_radius_px:
                aggregation_frames += 1

        aggregation_index = aggregation_frames / valid_frames if valid_frames > 0 else 0.0

        # 全局最近距离
        all_distances = []
        for cx, cy in centers:
            for other_id, (ox, oy) in all_active_positions.items():
                dist_px = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                all_distances.append(dist_px)

        nearest_m = -1.0
        if all_distances:
            nearest_px = min(all_distances)
            nearest_m = self.scene.pixel_to_world(nearest_px)

        return {
            "aggregation_index": round(aggregation_index, 4),
            "nearest_vehicle_m": round(nearest_m, 2),
            "max_concurrent_vehicles": max(1, len(all_active_positions)),
        }
