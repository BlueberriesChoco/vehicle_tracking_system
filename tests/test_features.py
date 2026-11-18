"""行为特征提取器（速度 / 停留 / 聚集）的边界测试。"""

import numpy as np

from src.behavior.features.aggregation_feature import AggregationFeatureExtractor
from src.behavior.features.dwell_feature import DwellFeatureExtractor


def test_dwell_counts_sample_time_and_excludes_missing_frames():
    from datetime import datetime, timedelta

    origin = datetime(2026, 1, 1)
    timestamps = [origin + timedelta(seconds=t) for t in (0, 1, 3.25, 10, 11)]
    result = DwellFeatureExtractor(frame_rate=25).extract(
        [0.1, 0.1, 0.1, 0.1], timestamps, [0, 1, 2, 5, 6]
    )
    assert result["max_dwell_sec"] == 3.25
    assert result["stop_count"] == 1
    assert result["dwell_ratio"] == 0.7647
from src.behavior.features.speed_feature import SpeedFeatureExtractor
from src.trajectory.scene_geometry import SceneGeometry


def _scene():
    # 1 px = 0.1 m（x），1 px = 0.2 m（y）
    return SceneGeometry(
        ((0, 0), (0, 0)),
        ((0, 0), (0, 0)),
        px_per_meter=10.0,
        geometry_level="calibrated",
        has_roi=True,
        homography=[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]],
    )


def test_constant_speed_is_computed_in_meters_per_second():
    scene = _scene()
    extractor = SpeedFeatureExtractor(scene, frame_rate=25)
    xs = np.arange(0.0, 110.0, 10.0)  # 每帧 10px = 1m
    ys = np.zeros_like(xs)

    features = extractor.extract(xs, ys)

    # 1m / (1/25 s) = 25 m/s
    assert abs(features["avg_speed_ms"] - 25.0) < 1e-6
    assert features["speed_variance"] == 0.0


def test_single_point_speed_is_zero():
    extractor = SpeedFeatureExtractor(_scene())

    features = extractor.extract(np.array([5.0]), np.array([5.0]))

    assert features["avg_speed_ms"] == 0.0


def test_dwell_detected_for_long_stop():
    extractor = DwellFeatureExtractor(frame_rate=25, min_dwell_duration_sec=3.0)
    speeds = [0.0] * 80 + [5.0] * 20  # 80 帧 = 3.2 秒停留

    features = extractor.extract(speeds)

    assert features["stop_count"] == 1
    assert abs(features["max_dwell_sec"] - 3.2) < 1e-6


def test_short_stop_is_ignored():
    extractor = DwellFeatureExtractor(frame_rate=25, min_dwell_duration_sec=3.0)
    speeds = [0.0] * 10 + [5.0] * 90  # 只有 0.4 秒

    features = extractor.extract(speeds)

    assert features["stop_count"] == 0
    assert features["max_dwell_sec"] == 0.0


def test_dwell_at_trajectory_end_is_counted():
    extractor = DwellFeatureExtractor(frame_rate=25, min_dwell_duration_sec=3.0)
    speeds = [5.0] * 10 + [0.0] * 100

    features = extractor.extract(speeds)

    assert features["stop_count"] == 1


def test_empty_speed_sequence_returns_zeros():
    extractor = DwellFeatureExtractor()

    features = extractor.extract([])

    assert features["stop_count"] == 0
    assert features["max_dwell_sec"] == 0.0


def test_aggregation_detects_nearby_vehicle():
    extractor = AggregationFeatureExtractor(_scene(), aggregation_radius_m=5.0)

    features = extractor.extract(
        centers=[(0.0, 0.0)],
        all_active_positions={1: (0.0, 0.0), 2: (20.0, 0.0)},  # 20px = 2m
        current_track_id=1,
    )

    assert features["aggregation_index"] == 1.0
    assert features["nearest_vehicle_m"] == 2.0
    assert features["max_concurrent_vehicles"] == 2


def test_aggregation_ignores_far_vehicle():
    extractor = AggregationFeatureExtractor(_scene(), aggregation_radius_m=5.0)

    features = extractor.extract(
        centers=[(0.0, 0.0)],
        all_active_positions={1: (0.0, 0.0), 2: (100.0, 0.0)},  # 10m
        current_track_id=1,
    )

    assert features["aggregation_index"] == 0.0
    assert features["nearest_vehicle_m"] == 10.0


def test_aggregation_without_other_vehicles():
    extractor = AggregationFeatureExtractor(_scene(), aggregation_radius_m=5.0)

    features = extractor.extract(
        centers=[(0.0, 0.0)],
        all_active_positions={1: (0.0, 0.0)},
        current_track_id=1,
    )

    assert features["aggregation_index"] == 0.0
    assert features["nearest_vehicle_m"] == -1.0
    assert features["max_concurrent_vehicles"] == 1


def test_speed_cv_is_std_over_mean():
    """研究要点一要求的“速度变异系数”：CV = 标准差 / 均值。"""
    scene = _scene()
    extractor = SpeedFeatureExtractor(scene, frame_rate=25)
    # 每帧步长在 10px(1m) 与 20px(2m) 之间交替 -> 速度 25 / 50 m/s
    xs = np.array([0.0, 10.0, 30.0, 40.0, 60.0, 70.0])
    ys = np.zeros_like(xs)

    features = extractor.extract(xs, ys)

    speeds = np.array(features["instant_speeds"])
    expected = float(np.std(speeds)) / float(np.mean(speeds))
    assert abs(features["speed_cv"] - expected) < 1e-3
    assert features["speed_cv"] > 0


def test_speed_cv_is_zero_for_constant_speed():
    extractor = SpeedFeatureExtractor(_scene())
    xs = np.arange(0.0, 110.0, 10.0)
    ys = np.zeros_like(xs)

    features = extractor.extract(xs, ys)

    assert features["speed_cv"] == 0.0
