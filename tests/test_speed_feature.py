import numpy as np

from src.behavior.features.speed_feature import SpeedFeatureExtractor
from src.trajectory.scene_geometry import SceneGeometry


def _extractor() -> SpeedFeatureExtractor:
    geometry = SceneGeometry(((0, 0), (1, 0)), ((0, 1), (1, 1)), px_per_meter=50.0)
    return SpeedFeatureExtractor(geometry, frame_rate=25)


def test_spike_rejected_from_max_speed():
    """单帧跳变（换 ID / 遮挡重关联）不应把 max_speed 顶到 100 km/h 以上。"""
    xs = np.array([100, 100, 100, 100, 100, 300, 100, 100, 100, 100], dtype=float)
    ys = np.array([500, 505, 510, 515, 520, 520, 530, 535, 540, 545], dtype=float)

    extractor = _extractor()
    speeds = extractor.compute_instant_speeds(xs, ys)

    assert extractor.speed_outlier_count >= 1
    assert speeds.max() < 10.0


def test_no_spike_keeps_values():
    """匀速轨迹不应被误判为跳变。"""
    xs = np.arange(100, 200, 10, dtype=float)
    ys = np.arange(500, 600, 10, dtype=float)

    extractor = _extractor()
    speeds = extractor.compute_instant_speeds(xs, ys)

    assert extractor.speed_outlier_count == 0
    assert np.allclose(speeds, speeds[0])


def test_extract_reports_outlier_count():
    """extract() 的返回里要带上剔除计数，便于审计。"""
    xs = np.array([100, 100, 100, 100, 100, 300, 100, 100, 100, 100], dtype=float)
    ys = np.array([500, 505, 510, 515, 520, 520, 530, 535, 540, 545], dtype=float)

    result = _extractor().extract(xs, ys)

    assert result["speed_outlier_count"] >= 1
    assert result["max_speed_ms"] < 10.0
