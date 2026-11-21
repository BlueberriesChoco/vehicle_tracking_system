"""异常分量的标定（calibration）：把原始分数映射成可跨批次比较的 [0,1] 指数。

## 为什么需要它

原先 A/G/T 三个分量各自用不同方式落到 [0,1]：

- A（Isolation Forest）本身就是 [0,1] 的路径长度指数，值域天然被压缩在 0.4~0.8；
- G 用 ``minmax_normalize``，于是**每张图必然产出一个 1.0 和一个 0.0**；
- T 在单窗口运行中恒为 0。

前两条合起来的后果是：融合分 R = αA + βG + γT 的绝对值既没有物理含义，
也不能跨批次比较——换一批数据，G 的 1.0 依然会出现在某个"本批最异常"的车身上，
哪怕这批车全都正常。

## 做法：分位锚点（anchor normalization）

标定只依赖两个锚点 ``low`` / ``high``：

    calibrated = clip((raw - low) / (high - low), 0, 1)

- ``low`` = 正常样本的分位点（默认 P50，中位数）；
- ``high`` = 极端异常的分位点（默认 P99）。

语义变成"**相对于该机位常态分布的偏离程度**"：
低于常态中位数 → 0，达到常态 P99 → 1，超过 → 仍为 1（clip 截断，不放大）。

这与 min-max 的关键差别是**锚点来自参考分布、而不是本批数据**，
因此同一个分数在不同批次、不同节点数的图上得到同一个标定值，跨批次可比。

## 与排序的关系

分位锚定是单调变换，**同一批次内部排序完全不变**（clip 只压平尾部）。
换句话说：标定只改绝对分与档位归属，不重新洗牌。

## 保守回退

锚点退化（``high - low`` 过小）时返回全 0，而不是放大噪声——
宁可说"本批没有可区分的异常"，也不要把 0.001 的抖动放大成满分。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


#: 默认低位锚点分位（正常样本的中位数）。
DEFAULT_LOW_PERCENTILE: float = 50.0

#: 默认高位锚点分位（极端异常的起点）。
DEFAULT_HIGH_PERCENTILE: float = 99.0

#: 锚点跨度小于该值时视为退化，不再放大。
_MIN_SPAN: float = 1e-9


def quantile_anchors(
    values: np.ndarray,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> Dict[str, float]:
    """从参考分布里提取分位锚点。

    Args:
        values: 参考样本的原始分数（通常是一个"正常"批次）。
        low_percentile: 低位锚点分位，标定后映射到 0。
        high_percentile: 高位锚点分位，标定后映射到 1。

    Returns:
        ``{"low": ..., "high": ..., "n": ...}``。空输入返回 low=high=0。
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"low": 0.0, "high": 0.0, "n": 0}

    low = float(np.percentile(array, low_percentile))
    high = float(np.percentile(array, high_percentile))
    return {"low": low, "high": high, "n": int(array.size)}


def anchor_normalize(
    values: np.ndarray,
    anchors: Optional[Dict[str, float]],
    clip: bool = True,
) -> np.ndarray:
    """用给定锚点把原始分数映射到 [0,1]。

    Args:
        values: 待标定的原始分数。
        anchors: ``quantile_anchors`` 的产物，至少含 ``low`` / ``high``。
        clip: 是否把超出锚点范围的值截断到 [0,1]。默认 True——
            超过 P99 就是"和 P99 一样异常"，不做无上界的放大。

    Returns:
        与 ``values`` 同形状的 float64 数组。锚点缺失或退化时返回全 0。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros_like(array, dtype=np.float64)

    if not anchors:
        return np.zeros_like(array, dtype=np.float64)

    low = float(anchors.get("low", 0.0) or 0.0)
    high = float(anchors.get("high", 0.0) or 0.0)
    span = high - low
    if not np.isfinite(span) or span <= _MIN_SPAN:
        # 锚点退化：本批分数没有区分度，返回全 0 而不是放大噪声
        return np.zeros_like(array, dtype=np.float64)

    scaled = (array - low) / span
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)
    return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)


def fit_and_calibrate(
    values: np.ndarray,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> tuple:
    """一步完成"从本批提取锚点 + 标定"，返回 ``(calibrated, anchors)``。

    适用于没有独立参考批次、只能以本批分布自标定的场景（例如 SVD 分支）。
    有独立基线时应先用 ``quantile_anchors`` 在基线上取锚点，
    再对目标批次调用 ``anchor_normalize``——那样才真正跨批次可比。
    """
    anchors = quantile_anchors(values, low_percentile, high_percentile)
    return anchor_normalize(values, anchors), anchors
