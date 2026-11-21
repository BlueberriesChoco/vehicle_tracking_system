"""图分数标定的单元测试。

历史：``anomaly.py`` 与 ``gae.py`` 曾各自维护一份 min-max 归一化，并各有一份
逐行等价的聚类系数实现（本项目曾有测试断言两个模块共享同一个 ``minmax_normalize``）。

2026-11-20 起口径改为 **分位锚点标定**，原因见 ``src/graph/calibration.py``：
min-max 会让每张图必然产出一个 1.0 和一个 0.0，导致"本批最异常者恒得满分"，
且无法跨批次比较。本文件相应改为验证锚点标定的契约。
"""

import numpy as np

from src.graph.calibration import (
    anchor_normalize,
    fit_and_calibrate,
    quantile_anchors,
)
from src.graph.normalize import minmax_normalize


# -- minmax_normalize 仍保留（供旧调用方），但不再是图分数的主口径 -------------


def test_minmax_normalize_maps_to_unit_range():
    result = minmax_normalize(np.array([2.0, 4.0, 6.0]))

    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_minmax_normalize_handles_empty_input():
    result = minmax_normalize(np.array([]))

    assert result.size == 0


def test_minmax_normalize_handles_constant_input():
    result = minmax_normalize(np.array([3.0, 3.0, 3.0]))

    assert np.array_equal(result, np.zeros(3))


# -- 锚点标定的核心契约 -------------------------------------------------------


def test_minmax_always_produces_a_one_that_is_the_problem():
    """记录 min-max 被替换的原因：任何输入都必然产出一个满分。

    这不是抱怨实现有 bug，而是说明它的语义不适合"异常指数"：
    一批全部正常的图，排名第一的节点照样得 1.0。
    """
    all_normal = np.array([0.10, 0.11, 0.12, 0.13])

    result = minmax_normalize(all_normal)

    assert result.max() == 1.0  # 全是正常值，却有一个满分


def test_anchor_normalize_keeps_new_normal_batch_low():
    """用基线锚点给**另一批**正常数据打分，顶格样本应只占约 1%（即 P99 的尾部）。

    精确边界：P99 锚点的定义就是"1% 的正常样本会达到或超过它并顶格"，
    所以断言不能是"绝不顶格"，而是"顶格样本极少"。
    对比 min-max：它把本批最大值**必然**抬到 1.0，且与本批是否异常完全无关。
    """
    rng = np.random.RandomState(0)
    baseline = rng.normal(0.12, 0.01, size=2000)
    anchors = quantile_anchors(baseline, 50, 99)

    fresh = rng.normal(0.12, 0.01, size=2000)  # 同分布的正常数据
    result = anchor_normalize(fresh, anchors)

    at_ceiling = float((result >= 1.0).mean())
    assert at_ceiling <= 0.03, f"顶格比例 {at_ceiling:.3f} 过高"
    # 多数正常样本落在低分段：中位数以下一律 0
    assert float((result == 0.0).mean()) >= 0.45

    # 对照：min-max 下最大值必然是 1.0，且与"这批数据有没有异常"无关
    assert minmax_normalize(np.array([0.1, 0.11, 0.12])).max() == 1.0


def test_anchor_normalize_is_monotonic():
    """标定必须单调，否则同批次排序会被打乱。"""
    anchors = {"low": 0.0, "high": 2.0}
    values = np.array([0.1, 0.5, 1.0, 1.5, 3.0, -1.0])

    result = anchor_normalize(values, anchors)

    order_before = np.argsort(values)
    order_after = np.argsort(result)
    assert np.array_equal(order_before, order_after)


def test_anchor_normalize_clips_out_of_range():
    anchors = {"low": 0.0, "high": 1.0}

    result = anchor_normalize(np.array([-5.0, 0.5, 5.0]), anchors)

    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_anchor_normalize_returns_zeros_when_anchors_degenerate():
    """锚点退化时返回全 0，而不是把噪声放大成满分。"""
    result = anchor_normalize(np.array([1.0, 2.0, 3.0]), {"low": 0.5, "high": 0.5})

    assert np.array_equal(result, np.zeros(3))


def test_anchor_normalize_returns_zeros_without_anchors():
    result = anchor_normalize(np.array([1.0, 2.0, 3.0]), None)

    assert np.array_equal(result, np.zeros(3))


def test_anchor_normalize_is_comparable_across_batches():
    """同一锚点下，两批数据得到的是同一把尺子的读数——这是跨批次可比的定义。"""
    anchors = {"low": 0.0, "high": 10.0}

    batch_a = anchor_normalize(np.array([0.0, 5.0, 10.0]), anchors)
    batch_b = anchor_normalize(np.array([2.0, 8.0]), anchors)

    assert np.allclose(batch_a, [0.0, 0.5, 1.0])
    assert np.allclose(batch_b, [0.2, 0.8])
    # 关键：两批都没出现"必然的 1.0"
    assert batch_b.max() < 1.0


def test_fit_and_calibrate_returns_anchors():
    values = np.arange(0.0, 100.0, 1.0)

    calibrated, anchors = fit_and_calibrate(values, 50, 99)

    assert set(anchors) == {"low", "high", "n"}
    assert anchors["n"] == 100
    assert calibrated.shape == values.shape
    assert calibrated.min() == 0.0
    assert calibrated.max() == 1.0


def test_fit_and_calibrate_handles_empty():
    calibrated, anchors = fit_and_calibrate(np.array([]))

    assert calibrated.size == 0
    assert anchors["low"] == 0.0 and anchors["high"] == 0.0


# -- 聚类系数实现已收敛为单一来源 ---------------------------------------------


def test_both_graph_modules_share_the_same_clustering_helper():
    """原测试校验"两个模块共享 minmax_normalize"；现在共享的是聚类系数。

    min-max 已从图打分路径移除，重复的聚类系数实现是当时遗留的技术债，
    已于本次收敛到 ``src.graph.features.clustering_coefficient``。
    """
    import src.graph.anomaly as anomaly
    import src.graph.features as features
    import src.graph.gae as gae

    assert anomaly.clustering_coefficient is features.clustering_coefficient
    assert gae._clustering is features.clustering_coefficient
