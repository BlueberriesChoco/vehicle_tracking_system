"""BehaviorVectorizer 的构造与最小端到端测试。

这类测试用于防止“特征提取器签名改了、调用点没改”的回归——上一轮就出现过
SpeedFeatureExtractor 删除参数后 BehaviorVectorizer 仍按位置传参的问题。
"""

import numpy as np

from datetime import datetime, timedelta

from src.behavior.behavior_vector import BehaviorVectorizer
from src.trajectory.path_reference import PathReference
from src.trajectory.scene_geometry import SceneGeometry
from src.trajectory.trajectory_extractor import TrajectoryExtractor
from src.trajectory.trajectory_smoother import TrajectorySmoother
from src.tracking.tracklet import Tracklet

T0 = None


def _scene():
    return SceneGeometry(
        ((0.0, 0.0), (100.0, 0.0)),
        ((0.0, 100.0), (100.0, 100.0)),
        px_per_meter=10.0,
        geometry_level="calibrated",
        has_roi=True,
    )


def test_vectorizer_can_be_constructed():
    scene = _scene()
    smoother = TrajectorySmoother()
    extractor = TrajectoryExtractor(scene, smoother)

    vectorizer = BehaviorVectorizer(
        scene_geometry=scene,
        path_reference=PathReference(),
        trajectory_extractor=extractor,
        smoother=smoother,
    )

    assert vectorizer.speed_extractor is not None
    assert vectorizer.dwell_extractor is not None


def test_vectorizer_produces_expected_fields():
    from datetime import datetime, timedelta

    scene = _scene()
    smoother = TrajectorySmoother()
    extractor = TrajectoryExtractor(scene, smoother)
    vectorizer = BehaviorVectorizer(
        scene_geometry=scene,
        path_reference=PathReference(),
        trajectory_extractor=extractor,
        smoother=smoother,
    )

    tracklet = Tracklet(track_id=1, camera_id="cam01", class_id=2, class_name="car")
    start = datetime(2026, 9, 1, 8, 0, 0)
    for frame_idx in range(30):
        tracklet.add_detection(
            frame_idx,
            start + timedelta(seconds=frame_idx / 25.0),
            [40, 10 + frame_idx, 60, 20 + frame_idx],
            (50.0, float(10 + frame_idx)),
            0.9,
        )
    tracklet.mark_entered(0)
    tracklet.mark_exited(29)

    vector = vectorizer.vectorize(tracklet, {}, {})

    for key in ("avg_speed_ms", "max_dwell_sec", "stop_count", "path_smoothness",
                "is_night", "freq_count_24h", "max_concurrent_vehicles"):
        assert key in vector
    assert vector["track_id"] == 1
    assert np.isfinite(vector["avg_speed_ms"])


def test_missing_tracklet_time_uses_fixed_placeholder():
    """时间缺失时必须用固定占位时间，不能用 datetime.now()（否则结果不可复现）。"""
    from src.behavior.behavior_vector import BehaviorVectorizer
    from src.trajectory.path_reference import PathReference
    from src.trajectory.scene_geometry import SceneGeometry
    from src.trajectory.trajectory_extractor import TrajectoryExtractor
    from src.trajectory.trajectory_smoother import TrajectorySmoother
    from src.tracking.tracklet import Tracklet

    scene = SceneGeometry(
        ((0.0, 0.0), (100.0, 0.0)),
        ((0.0, 100.0), (100.0, 100.0)),
        px_per_meter=10.0,
        geometry_level="roi",
        has_roi=True,
    )
    smoother = TrajectorySmoother()
    vectorizer = BehaviorVectorizer(
        scene_geometry=scene,
        path_reference=PathReference(),
        trajectory_extractor=TrajectoryExtractor(scene, smoother),
        smoother=smoother,
    )

    tracklet = Tracklet(track_id=9, camera_id="cam01", class_id=2, class_name="car")
    start = datetime(2026, 9, 1, 8, 0, 0)
    for frame_idx in range(12):
        tracklet.add_detection(
            frame_idx,
            start + timedelta(seconds=frame_idx / 25.0),
            [40, 10 + frame_idx, 60, 20 + frame_idx],
            (50.0, float(10 + frame_idx)),
            0.9,
        )
    # 模拟时间字段缺失（防御性路径）
    tracklet.first_seen_time = None
    tracklet.last_seen_time = None

    vector = vectorizer.vectorize(tracklet, {}, {})

    # 正午占位 -> 不是夜间
    assert vector["is_night"] == 0
    assert vector["night_ratio"] == 0.0
