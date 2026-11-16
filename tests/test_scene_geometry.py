"""透视标定（单应矩阵）与速度/聚集换算的回归测试。"""

from datetime import datetime

import numpy as np

from src.behavior.features.aggregation_feature import AggregationFeatureExtractor
from src.behavior.features.speed_feature import SpeedFeatureExtractor
from src.trajectory.scene_geometry import SceneGeometry


def _homography_scene(geometry_level="calibrated"):
    # 1 px = 0.1 m (x), 1 px = 0.2 m (y)
    return SceneGeometry(
        ((0, 0), (0, 0)),
        ((0, 0), (0, 0)),
        px_per_meter=10.0,
        geometry_level=geometry_level,
        has_roi=True,
        homography=[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]],
    )


def test_pixel_to_ground_uses_homography():
    scene = _homography_scene()

    assert scene.has_homography
    assert scene.pixel_to_ground(100.0, 50.0) == (10.0, 10.0)


def test_ground_distance_uses_homography():
    scene = _homography_scene()

    assert scene.ground_distance((0.0, 0.0), (100.0, 0.0)) == 10.0
    assert scene.ground_distance((0.0, 0.0), (0.0, 100.0)) == 20.0


def test_trajectory_length_uses_ground_distance():
    scene = _homography_scene()

    length = scene.trajectory_length_meters([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])

    assert length == 30.0


def test_homography_marks_geometry_reliable():
    scene = _homography_scene()

    assert scene.speed_reliable == 1
    assert scene.path_reliable == 1
    assert scene.aggregation_reliable == 1


def test_without_homography_falls_back_to_px_per_meter():
    scene = SceneGeometry(((0, 0), (0, 0)), ((0, 0), (0, 0)), px_per_meter=10.0)

    assert not scene.has_homography
    assert scene.ground_distance((0.0, 0.0), (100.0, 0.0)) == 10.0


def test_speed_feature_with_homography():
    scene = _homography_scene()
    extractor = SpeedFeatureExtractor(scene, frame_rate=25)

    xs = np.array([0.0, 25.0])  # 25 px = 2.5 m
    ys = np.array([0.0, 0.0])
    timestamps = [datetime(2026, 6, 1, 8, 0, 0), datetime(2026, 6, 1, 8, 0, 1)]

    result = extractor.extract(xs, ys, timestamps)

    assert result["avg_speed_ms"] == 2.5  # 2.5 m / 1 s


def test_aggregation_nearest_distance_with_homography():
    scene = _homography_scene()
    extractor = AggregationFeatureExtractor(scene, aggregation_radius_m=5.0)

    result = extractor.extract([(0.0, 0.0)], {2: (100.0, 0.0)}, current_track_id=1)

    assert result["nearest_vehicle_m"] == 10.0


def test_batch_processor_parses_homography():
    from src.pipeline.batch_processor import BatchProcessor

    matrix = BatchProcessor._parse_homography(
        {"homography": [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]]}
    )

    assert matrix is not None
    assert matrix.shape == (3, 3)
    assert BatchProcessor._parse_homography({}) is None
    assert BatchProcessor._parse_homography({"homography": [1, 2, 3]}) is None


def test_geometry_level_becomes_calibrated_with_homography():
    from src.pipeline.batch_processor import BatchProcessor

    level = BatchProcessor._infer_geometry_level(
        has_configured_roi=True,
        entry_line={},
        exit_line={},
        px_per_meter=50.0,
        calibration={},
        has_homography=True,
    )

    assert level == "calibrated"


def test_ground_distance_same_point_is_zero():
    scene = _homography_scene()

    assert scene.ground_distance((10.0, 10.0), (10.0, 10.0)) == 0.0


def test_trajectory_length_handles_empty_and_single_point():
    scene = _homography_scene()

    assert scene.trajectory_length_meters([]) == 0.0
    assert scene.trajectory_length_meters([(0.0, 0.0)]) == 0.0


def test_trajectory_length_zero_for_duplicate_points():
    scene = _homography_scene()

    assert scene.trajectory_length_meters([(5.0, 5.0)] * 6) == 0.0
