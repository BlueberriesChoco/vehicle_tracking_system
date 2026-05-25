from typing import List, Tuple
from ...trajectory.path_reference import PathReference
from ...trajectory.scene_geometry import SceneGeometry


class PathFeatureExtractor:
    """路径特征提取器。

    结合参考路径计算车辆的路径偏离度和路径平滑度。
    """

    def __init__(self, path_reference: PathReference, scene_geometry: SceneGeometry):
        self.path_reference = path_reference
        self.scene = scene_geometry

    def extract(
        self,
        centers: List[Tuple[float, float]],
    ) -> dict:
        """提取路径相关特征。

        Args:
            centers: 平滑后的轨迹中心点列表

        Returns:
            dict with path_deviation, path_smoothness
        """
        mean_dev_px, max_dev_px, dtw_approx_px = self.path_reference.compute_deviation(centers)

        path_deviation = self.scene.pixel_to_world(mean_dev_px)
        path_smoothness = self.path_reference.compute_smoothness(centers)

        return {
            "path_deviation": round(path_deviation, 3),
            "path_deviation_max": round(self.scene.pixel_to_world(max_dev_px), 3),
            "path_smoothness": round(path_smoothness, 4),
            "dtw_approximation": round(self.scene.pixel_to_world(dtw_approx_px), 3),
        }
