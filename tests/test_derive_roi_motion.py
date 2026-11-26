import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.derive_roi_motion import drop_static, polygon_from_points  # noqa: E402


def test_static_targets_are_dropped():
    """每帧都出现的点（停放车辆）应被剔除，只留少量帧出现的点（过路车辆）。"""
    frames = 20
    # 停放车：同一位置出现 20 次（命中率 100%）
    parked = [(100.0, 500.0)] * frames
    # 过路车：两个位置各出现 2 次（命中率 10%）
    moving = [(600.0, 400.0)] * 2 + [(700.0, 380.0)] * 2
    points = np.asarray(parked + moving, dtype=np.float64)

    keep, static_cells = drop_static(points, frames)

    assert static_cells == 1
    assert len(keep) == 4
    assert all(x > 500 for x, _ in keep)


def test_no_parked_car_keeps_everything():
    frames = 10
    points = np.asarray([(600.0, 400.0), (640.0, 410.0)], dtype=np.float64)

    keep, static_cells = drop_static(points, frames)

    assert static_cells == 0
    assert len(keep) == 2


def test_polygon_is_within_frame():
    points = np.asarray([[100, 100], [900, 120], [950, 600], [150, 650]], dtype=np.float64)

    polygon = polygon_from_points(points, width=1280, height=720)

    assert len(polygon) >= 3
    for x, y in polygon:
        assert 0 <= x <= 1279
        assert 0 <= y <= 719


def test_polygon_falls_back_to_full_frame_when_too_few_points():
    points = np.asarray([[10.0, 10.0]], dtype=np.float64)

    polygon = polygon_from_points(points, width=640, height=360)

    assert len(polygon) == 4
    assert [0, 0] in polygon
