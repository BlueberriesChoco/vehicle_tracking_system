"""图异常分数用到的归一化工具。"""

from __future__ import annotations

import numpy as np


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """把分数线性映射到 [0, 1]。

    - 空输入原样返回；
    - 所有值相同（high - low ≈ 0）时返回全 0，避免除零。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)
