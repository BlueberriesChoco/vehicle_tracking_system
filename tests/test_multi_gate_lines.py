"""多条出入口线：跨任意一条即算通过。

场景：一个机位可能有多条出路（宽路口、两个闸道、场地有左右两个出口），
只画一条线会漏掉另一侧的车。
"""

from datetime import datetime, timedelta

from src.pipeline.batch_processor import BatchProcessor
from src.trajectory.scene_geometry import SceneGeometry
from src.trajectory.trajectory_extractor import TrajectoryExtractor
from src.trajectory.trajectory_smoother import TrajectorySmoother
from src.tracking.tracklet import Tracklet


def _run(points, entry_lines, exit_lines):
    scene = SceneGeometry(entry_lines, exit_lines, px_per_meter=10.0)
    extractor = TrajectoryExtractor(scene, TrajectorySmoother(), min_trajectory_length=1)
    tracklet = Tracklet(track_id=1, camera_id="cam", class_id=2, class_name="car")
    start = datetime(2026, 9, 1, 8, 0, 0)
    exited = []
    for index, (x, y) in enumerate(points):
        tracklet.add_detection(
            frame_idx=index,
            timestamp=start + timedelta(seconds=index / 25),
            bbox=[int(x - 20), int(y - 20), int(x + 20), int(y + 20)],
            center=(float(x), float(y)),
            confidence=0.9,
        )
        if index >= 1:
            exited.extend(extractor.process_frame({1: tracklet}, index))
    return exited


def test_crossing_second_exit_line_counts():
    """出口有两条线，车从第二条线出去也要算通过。"""
    entry_lines = [((100, 600), (100, 700))]
    exit_lines = [((400, 0), (400, 100)), ((400, 600), (400, 700))]

    # 车走在 y=650：先跨入口线，再跨第二条出口线
    points = [(50, 650), (150, 650), (250, 650), (380, 650), (450, 650)]
    exited = _run(points, entry_lines, exit_lines)

    assert len(exited) == 1


def test_geometry_accepts_plural_lines():
    scene = SceneGeometry(
        [((0, 0), (0, 10))],
        [((5, 0), (5, 10)), ((9, 0), (9, 10))],
    )

    assert scene.has_valid_entry_line()
    assert scene.has_valid_exit_line()
    assert scene.check_line_crossing((4, 5), (6, 5), "exit")
    assert scene.check_line_crossing((8, 5), (10, 5), "exit")
    assert not scene.check_line_crossing((6, 5), (8, 5), "exit")


def test_batch_processor_parses_single_and_plural():
    single = BatchProcessor._parse_gate_lines(
        {"entry_line": {"p1": [1, 2], "p2": [3, 4]}}, "entry_line", "entry_lines")
    plural = BatchProcessor._parse_gate_lines(
        {"exit_lines": [{"p1": [5, 6], "p2": [7, 8]}, {"p1": [9, 10], "p2": [11, 12]}]},
        "exit_line", "exit_lines")

    assert single == [{"p1": [1, 2], "p2": [3, 4]}]
    assert len(plural) == 2
    assert plural[1]["p1"] == [9, 10]


def test_plural_and_single_are_merged_without_duplicates():
    merged = BatchProcessor._parse_gate_lines(
        {
            "exit_line": {"p1": [1, 2], "p2": [3, 4]},
            "exit_lines": [{"p1": [1, 2], "p2": [3, 4]}, {"p1": [5, 6], "p2": [7, 8]}],
        },
        "exit_line", "exit_lines")

    assert len(merged) == 2
