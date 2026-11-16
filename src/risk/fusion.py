"""多维度风险融合 R = αA + βG + γT（研究要点四）。

- A：行为异常指数（Isolation Forest 的 ``iforest_score``）；
- G：图异常指数（图谱节点的 ``graph_anomaly_index``）；
- T：时间频率异常指数（``time_freq_index``）。

三者都在 [0, 1]，融合后输出 ``risk_score``（0~1）与 ``risk_score_100``（0~100），
并按 80/50 分档为 high / medium / low。
"""

from __future__ import annotations

from typing import List, Sequence


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class RiskFusion:
    """加权融合 A/G/T。

    ``corroboration=True`` 时启用「多信号印证」规则：当同一辆车被 ≥2 个**独立**
    信号命中（Isolation Forest / Transformer 残差 / 预设规则），即使加权分不到中档，
    也提升到 ``medium_threshold``。

    为什么需要它：默认权重下 A 的最大贡献只有 ``0.4 x 0.822 = 0.329``，
    个体异常再强也够不到 50 分；而中风险主要靠图异常 G 撑起（G 恒能到 1.0）。
    实测 5 辆被 ≥2 信号命中的车全部落在低风险。这是有意的默认行为，
    但业务若要"多信号 = 高优先级"，打开本开关即可。
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
        high_threshold: float = 80.0,
        medium_threshold: float = 50.0,
        corroboration: bool = False,
    ):
        weights = [max(float(alpha), 0.0), max(float(beta), 0.0), max(float(gamma), 0.0)]
        total = sum(weights)
        if total <= 0:
            raise ValueError("α/β/γ 不能全为 0")
        self.alpha, self.beta, self.gamma = (w / total for w in weights)
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.corroboration = bool(corroboration)

    @staticmethod
    def count_independent_signals(record: dict) -> int:
        """统计命中的独立信号数（IF / Transformer / 预设规则）。"""
        signals = 0
        try:
            if int(record.get("iforest_anomaly", 0) or 0) == 1:
                signals += 1
        except (TypeError, ValueError):
            pass
        try:
            if int(record.get("traj_anomaly", 0) or 0) == 1:
                signals += 1
        except (TypeError, ValueError):
            pass
        # alert_reason 里形如 "iforest_score=0.71" 的是检测器分数，不是规则
        for token in str(record.get("alert_reason", "") or "").split(";"):
            token = token.strip()
            if token and "=" not in token.split("(")[0]:
                signals += 1
                break
        return signals

    def fuse_one(self, record: dict) -> dict:
        a = _clip(_to_float(record.get("iforest_score")))
        g = _clip(_to_float(record.get("graph_anomaly_index")))
        t = _clip(_to_float(record.get("time_freq_index")))

        risk = self.alpha * a + self.beta * g + self.gamma * t
        risk_100 = round(100.0 * risk, 1)

        corroborated = 0
        if self.corroboration:
            corroborated = self.count_independent_signals(record)
            if corroborated >= 2 and risk_100 < self.medium_threshold:
                risk_100 = self.medium_threshold
                risk = self.medium_threshold / 100.0

        if risk_100 >= self.high_threshold:
            level = "high"
        elif risk_100 >= self.medium_threshold:
            level = "medium"
        else:
            level = "low"

        fused = dict(record)
        fused.update(
            {
                "risk_score": round(risk, 4),
                "risk_score_100": risk_100,
                "risk_level": level,
                "corroboration_signals": corroborated,
                "risk_components": {
                    "A": round(a, 4),
                    "G": round(g, 4),
                    "T": round(t, 4),
                    "alpha": round(self.alpha, 4),
                    "beta": round(self.beta, 4),
                    "gamma": round(self.gamma, 4),
                },
            }
        )
        return fused

    def fuse(self, records: Sequence[dict]) -> List[dict]:
        return [self.fuse_one(record) for record in records]
