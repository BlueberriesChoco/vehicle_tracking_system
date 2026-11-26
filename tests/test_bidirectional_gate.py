"""双向车道通行判定。

一对出入口线在单向模式下只能覆盖一个方向：反向行驶的车辆会先跨出口线，
此时还没「进入」，跨入口线后才算进入，之后再也不跨出口线 -> 永远不产生通行记录。

双向模式（camera yaml 里写 `bidirectional: true`）改为：
先跨任意一条线记为进入、再跨另一条线记为离开。
"""

from datetime import datetime, timedelta

from src.trajectory.scene_geometry import SceneGeometry
from src.trajectory.trajectory_extractor import TrajectoryExtractor
from src.trajectory.trajectory_smoother import TrajectorySmoother
from src.tracking.tracklet import Tracklet

# 入口线在 x=800，出口线在 x=200，均为竖直、横跨 y=0..100
ENTRY = ((800, 0), (800, 100))
EXIT = ((200, 0), (200, 100))


def _run(points, bidirectional: bool):
    """把一串 x 坐标逐帧喂给同一个 tracklet，返回 (完成的通行记录, tracklet)。"""
    scene = SceneGeometry(ENTRY, EXIT, px_per_meter=10.0, bidirectional=bidirectional)
    extractor = TrajectoryExtractor(scene, TrajectorySmoother(), min_trajectory_length=1)
    tracklet = Tracklet(track_id=1, camera_id="cam", class_id=2, class_name="car")
    start = datetime(2026, 9, 1, 8, 0, 0)

    exited = []
    for index, x in enumerate(points):
        tracklet.add_detection(
            frame_idx=index,
            timestamp=start + timedelta(seconds=index / 25),
            bbox=[int(x - 20), 0, int(x + 20), 40],
            center=(float(x), 50.0),
            confidence=0.9,
        )
        if index >= 1:
            exited.extend(extractor.process_frame({1: tracklet}, index))
    return exited, tracklet


def test_forward_direction_works_only_in_bidirectional_mode():
    """从左到右：先跨出口线(200)再跨入口线(800)。"""
    points = [100, 150, 250, 400, 600, 750, 850, 900]

    one_way, _ = _run(points, bidirectional=False)
    both_way, tracklet = _run(points, bidirectional=True)

    # 单向模式下这是「反向车」，不该产生通行记录
    assert one_way == []
    # 双向模式应当产生一条，且记录先跨的是出口线
    assert len(both_way) == 1
    assert tracklet.entry_side == "exit"


def test_one_way_mode_still_accepts_entry_first():
    """从右到左：先跨入口线(800)再跨出口线(200)，单向模式必须仍然有效。"""
    points = [900, 850, 750, 600, 400, 250, 150, 100]

    one_way, tracklet = _run(points, bidirectional=False)

    assert len(one_way) == 1
    assert tracklet.entry_side == "entry"


def test_bidirectional_mode_does_not_double_count():
    """双向模式下每条轨迹只应完成一次，不能同一条线跨两次就算离开。"""
    points = [100, 250, 400, 600, 750, 850, 900, 850, 750, 600, 400, 250, 150]

    both_way, _ = _run(points, bidirectional=True)

    assert len(both_way) == 1
