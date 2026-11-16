"""时间频率异常指数 T（研究要点四）。

T 关注“什么时候、多频繁”，与个体行为异常 A、图结构异常 G 互补：

- 高频通行：24 小时内通行次数超过阈值；
- 夜间通行：夜间通行占比；
- 夜间标记：是否发生在夜间。

输出 ``time_freq_index`` ∈ [0, 1]，越大越异常。

频次分量按“超出基线（单次通行）”归一化：freq=1 → 0，freq≥freq_threshold → 1。
这样单次正常通行不会被误标为时间频率异常。
"""

from __future__ import annotations

from typing import List, Sequence


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class TimeFrequencyScorer:
    """把频次、夜间占比、夜间标记合成为时间频率异常指数 T。"""

    def __init__(
        self,
        freq_threshold: float = 6.0,
        weight_frequency: float = 0.5,
        weight_night_ratio: float = 0.3,
        weight_night_flag: float = 0.2,
    ):
        self.freq_threshold = max(float(freq_threshold), 1e-6)
        self.weight_frequency = weight_frequency
        self.weight_night_ratio = weight_night_ratio
        self.weight_night_flag = weight_night_flag

    def score_one(self, vector: dict) -> dict:
        frequency = _to_float(vector.get("freq_count_24h"))
        night_ratio = _clip(_to_float(vector.get("night_ratio")))
        night_flag = 1.0 if _to_float(vector.get("is_night")) > 0 else 0.0

        # 频次异常是“超出基线”的部分：单次通行（freq=1）是正常基线，不应算异常。
        # 旧实现用 freq / threshold，freq=1 也会得到 0.083，导致每辆正常车
        # 都被写入一条“时间频率异常”证据。
        excess = max(frequency - 1.0, 0.0)
        frequency_component = _clip(excess / max(self.freq_threshold - 1.0, 1e-6))
        index = (
            self.weight_frequency * frequency_component
            + self.weight_night_ratio * night_ratio
            + self.weight_night_flag * night_flag
        )
        index = _clip(index)

        record = dict(vector)
        record["time_freq_index"] = round(index, 4)
        record["time_freq_components"] = {
            "frequency": round(frequency_component, 4),
            "night_ratio": round(night_ratio, 4),
            "night_flag": night_flag,
            "freq_count_24h": frequency,
        }
        return record

    def score(self, vectors: Sequence[dict]) -> List[dict]:
        return [self.score_one(vector) for vector in vectors]
