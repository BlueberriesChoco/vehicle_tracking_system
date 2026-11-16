from typing import Dict, List, Optional, Tuple

from ...trajectory.scene_geometry import SceneGeometry


class AggregationFeatureExtractor:
    """Extract proximity features from vehicles visible at the same time."""

    def __init__(self, scene_geometry: SceneGeometry, aggregation_radius_m: float = 5.0):
        self.scene = scene_geometry
        self.aggregation_radius_m = aggregation_radius_m

    def extract(
        self,
        centers: List[Tuple[float, float]],
        all_active_positions: Dict[int, Tuple[float, float]],
        current_track_id: int = None,
        frame_indices: Optional[List[int]] = None,
        position_histories: Optional[Dict[int, Dict[int, Tuple[float, float]]]] = None,
    ) -> dict:
        """Calculate proximity using matching frame indices when available."""
        snapshot_positions = {
            track_id: position
            for track_id, position in all_active_positions.items()
            if track_id != current_track_id
        }
        if not centers:
            return self._empty_result()

        aggregation_frames = 0
        all_distances = []
        max_concurrent = 1

        for index, (cx, cy) in enumerate(centers):
            concurrent_positions = snapshot_positions
            if (
                frame_indices is not None
                and position_histories is not None
                and index < len(frame_indices)
            ):
                frame_idx = frame_indices[index]
                concurrent_positions = {
                    track_id: history[frame_idx]
                    for track_id, history in position_histories.items()
                    if track_id != current_track_id and frame_idx in history
                }

            max_concurrent = max(max_concurrent, len(concurrent_positions) + 1)
            # 地面距离（米）：有单应矩阵时按透视关系计算
            distances_m = [
                self.scene.ground_distance((cx, cy), (ox, oy))
                for ox, oy in concurrent_positions.values()
            ]
            if not distances_m:
                continue

            min_dist_m = min(distances_m)
            all_distances.append(min_dist_m)
            if min_dist_m < self.aggregation_radius_m:
                aggregation_frames += 1

        nearest_m = -1.0
        if all_distances:
            nearest_m = min(all_distances)

        return {
            "aggregation_index": round(aggregation_frames / len(centers), 4),
            "nearest_vehicle_m": round(nearest_m, 2),
            "max_concurrent_vehicles": max_concurrent,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "aggregation_index": 0.0,
            "nearest_vehicle_m": -1.0,
            "max_concurrent_vehicles": 1,
        }
