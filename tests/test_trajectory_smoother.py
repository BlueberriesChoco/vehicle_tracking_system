"""轨迹平滑器的边界与稳定性测试。"""

import numpy as np

from src.trajectory.trajectory_smoother import TrajectorySmoother


def test_empty_trajectory():
    assert TrajectorySmoother().smooth([]) == []


def test_single_point_unchanged():
    assert TrajectorySmoother().smooth([(5.0, 7.0)]) == [(5.0, 7.0)]


def test_short_trajectory_shorter_than_window_unchanged():
    centers = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]

    assert TrajectorySmoother(window_size=5).smooth(centers) == centers


def test_duplicate_points_stay_constant():
    centers = [(10.0, 20.0)] * 7

    smoothed = TrajectorySmoother(window_size=5).smooth(centers)

    assert len(smoothed) == 7
    for x, y in smoothed:
        assert abs(x - 10.0) < 1e-6
        assert abs(y - 20.0) < 1e-6


def test_smoothing_reduces_outlier_deviation():
    # 匀速直线 + 一个飞点（index=4 处 x=100，理论值 40）
    centers = [(index * 10.0, 0.0) for index in range(10)]
    centers[4] = (100.0, 0.0)

    smoothed = TrajectorySmoother(window_size=5).smooth(centers)

    raw_deviation = max(abs(x - index * 10.0) for index, (x, _) in enumerate(centers))
    smooth_deviation = max(abs(x - index * 10.0) for index, (x, _) in enumerate(smoothed))
    assert smooth_deviation < raw_deviation


def test_even_window_size_is_rounded_up():
    smoother = TrajectorySmoother(window_size=4)
    assert smoother.window_size == 5


def test_moving_average_and_median_methods_run():
    centers = [(float(index), float(index % 3)) for index in range(9)]

    for method in ("moving_avg", "median"):
        smoothed = TrajectorySmoother(window_size=5, method=method).smooth(centers)
        assert len(smoothed) == len(centers)


def test_smooth_single_track_returns_numpy_arrays():
    centers = [(float(index), float(index) * 2) for index in range(8)]

    xs, ys = TrajectorySmoother(window_size=5).smooth_single_track(centers)

    assert isinstance(xs, np.ndarray)
    assert xs.shape == (8,)
    assert ys.shape == (8,)


def test_smooth_single_track_empty():
    xs, ys = TrajectorySmoother().smooth_single_track([])

    assert xs.size == 0
    assert ys.size == 0
