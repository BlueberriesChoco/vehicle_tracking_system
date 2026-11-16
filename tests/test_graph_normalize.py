"""图分数归一化的单元测试（原 _minmax 在 anomaly.py 与 gae.py 各有一份）。"""

import numpy as np

from src.graph.normalize import minmax_normalize


def test_minmax_normalize_maps_to_unit_range():
    result = minmax_normalize(np.array([2.0, 4.0, 6.0]))

    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_minmax_normalize_handles_empty_input():
    result = minmax_normalize(np.array([]))

    assert result.size == 0


def test_minmax_normalize_handles_constant_input():
    result = minmax_normalize(np.array([3.0, 3.0, 3.0]))

    assert np.array_equal(result, np.zeros(3))


def test_both_graph_modules_share_the_same_helper():
    import src.graph.anomaly as anomaly
    import src.graph.gae as gae

    assert anomaly.minmax_normalize is gae.minmax_normalize
