"""轨迹提取与进出判定测试。"""

from datetime import datetime, timedelta

from src.trajectory.scene_geometry import SceneGeometry
from src.trajectory.trajectory_extractor import TrajectoryExtractor
from src.trajectory.trajectory_smoother import TrajectorySmoother
from src.tracking.tracklet import Tracklet

T0 = datetime(2026, 9, 1, 9, 0, 0)


def _scene_with_lines():
    return SceneGeometry(
        ((0.0, 0.0), (100.0, 0.0)),      # entry at y=0
        ((0.0, 100.0), (100.0, 100.0)),  # exit at y=100
        px_per_meter=10.0,
        geometry_level="calibrated",
        has_roi=True,
    )


def _scene_without_lines():
    return SceneGeometry(
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0)),
        px_per_meter=10.0,
        geometry_level="roi",
        has_roi=True,
    )


def _tracklet(track_id=1):
    return Tracklet(track_id=track_id, camera_id="cam01", class_id=2, class_name="car")


def _add(tracklet, frame_idx, x, y, t0=T0):
    timestamp = t0 + timedelta(seconds=frame_idx / 25.0)
    tracklet.add_detection(frame_idx, timestamp, [int(x) - 5, int(y) - 5, int(x) + 5, int(y) + 5], (x, y), 0.9)


def test_line_crossing_detects_passage():
    scene = _scene_with_lines()

    assert scene.check_line_crossing((50, -10), (50, 10), "entry")
    assert not scene.check_line_crossing((50, 10), (50, 20), "entry")
    # 线段只覆盖 x∈[0,100]，x=200 处经过不应判定为穿越
    assert not scene.check_line_crossing((200, -10), (200, 10), "entry")


def test_enter_and_exit_by_lines():
    extractor = TrajectoryExtractor(_scene_with_lines(), TrajectorySmoother())
    tracklet = _tracklet()
    ys = [-10, -8, -6, -4, -2, 2, 20, 40, 60, 80, 100, 110]
    exited = []
    for frame_idx, y in enumerate(ys):
        _add(tracklet, frame_idx, 50, y)
        exited.extend(extractor.process_frame({1: tracklet}, frame_idx))

    assert tracklet.has_entered
    assert tracklet.has_exited
    assert len(exited) == 1
    assert tracklet.is_complete


def test_fallback_entry_and_exit_without_lines():
    extractor = TrajectoryExtractor(_scene_without_lines(), TrajectorySmoother(), enter_min_frames=5)
    tracklet = _tracklet()
    exited = []
    for frame_idx in range(5):
        _add(tracklet, frame_idx, 50, 10 + frame_idx)
        exited.extend(extractor.process_frame({1: tracklet}, frame_idx))

    assert tracklet.has_entered
    assert not tracklet.has_exited

    tracklet.mark_finished()
    exited.extend(extractor.process_frame({1: tracklet}, 5))
    assert tracklet.has_exited
    assert len(exited) == 1


def test_vehicle_that_never_crosses_is_not_valid():
    extractor = TrajectoryExtractor(_scene_with_lines(), TrajectorySmoother(), min_trajectory_length=3)
    tracklet = _tracklet()
    # 一直在 y=50 附近，从未越过 y=0 的入口线
    for frame_idx in range(10):
        _add(tracklet, frame_idx, 50, 50 + frame_idx)
    extractor.process_frame({1: tracklet}, 9)

    assert not tracklet.has_entered
    assert extractor.extract_valid_trajectories([tracklet]) == []


def test_short_tracklet_is_filtered_out():
    extractor = TrajectoryExtractor(_scene_without_lines(), TrajectorySmoother(), min_trajectory_length=10)
    tracklet = _tracklet()
    for frame_idx in range(5):
        _add(tracklet, frame_idx, 50, 10 + frame_idx)
    extractor.process_frame({1: tracklet}, 4)

    assert tracklet.has_entered
    assert extractor.extract_valid_trajectories([tracklet]) == []
