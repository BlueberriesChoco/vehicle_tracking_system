"""多维度风险融合 R = αA + βG + γT（研究要点四）。

- A：行为异常指数（Isolation Forest 的 ``iforest_score``）；
- G：图异常指数（图谱节点的 ``graph_anomaly_index``）；
- T：时间频率异常指数（``time_freq_index``）。

三者都在 [0, 1]，融合后输出 ``risk_score``（0~1）与 ``risk_score_100``（0~100），
并按 80/50 分档为 high / medium / low。
"""

from __future__ import annotations

from typing import List, Optional, Sequence


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

    ## 分量可用性与权重重归一化（2026-11-20 修正）

    原先三个分量的权重是固定的 α=0.4 / β=0.35 / γ=0.25。单窗口运行时
    ``freq_count_24h`` 恒为 1、夜间标记恒为 0，于是 **T 恒为 0**，γ 成为死权重：

        R_max = 100 × (0.4 + 0.35) = 75.0

    80 分档在结构上不可达，与数据量无关。这不是"参数没调好"，而是把
    "该分量无数据"错当成了"该分量值为 0"。

    现在按批次检测每个分量是否真的携带信息（``A_available`` / ``G_available`` /
    ``T_available``），不可用的分量从加权和中剔除，剩余权重按比例放大到 1。
    这样"只有 A 和 G 可用"时 R 的上限回到 100 分，档位语义恢复。

    ## 单路证据封顶（2026-11-20 二次修正）

    重归一化本身有个副作用：**只剩一路分量时，它的权重被放大到 1.0**，
    于是 ``R`` 退化成该分量的原值，一辆 ``G=1.0`` 的车直接得 100 分、判高风险。
    冒烟测试（15 辆车，A 与 T 均不可用）实测出现 2 辆伪 100 分。

    问题不在数学而在语义：100 分声称的是"最大异常"，而单源信号没有旁证。
    因此引入**证据封顶**——只用一路分量时分数上限压到 75 分（低于 80 分档），
    即"高风险必须至少两路独立证据"。两路及以上不折扣：此时缺的第三路
    本来就常有结构性缺失，折扣会反过来压低真实高分（实测会让本批 80 分重新变 72 分）。

    注意：重归一化只改变量纲，**不改变同批次内的排序**（各分量同向），
    但会改变绝对分，因此历史报数需要按新口径重算。
    """

    #: 分量名 → (可用性字段, 取值字段)
    COMPONENTS = {
        "A": ("iforest_available", "iforest_score"),
        "G": ("graph_available", "graph_anomaly_index"),
        "T": ("time_freq_available", "time_freq_index"),
    }

    #: 批次内可用记录占比达到该阈值才认为该分量整批可用。
    availability_threshold = 0.5

    #: 可用分量数 → 分数上限（0~100）。单路证据封顶在 80 分档之下。
    evidence_ceiling = {3: 100.0, 2: 100.0, 1: 75.0}

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
        high_threshold: float = 80.0,
        medium_threshold: float = 50.0,
        corroboration: bool = False,
        renormalize: bool = True,
    ):
        weights = [max(float(alpha), 0.0), max(float(beta), 0.0), max(float(gamma), 0.0)]
        total = sum(weights)
        if total <= 0:
            raise ValueError("α/β/γ 不能全为 0")
        self.alpha, self.beta, self.gamma = (w / total for w in weights)
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.corroboration = bool(corroboration)
        self.renormalize = bool(renormalize)

    @staticmethod
    def _available(record: dict, field: str, value_field: str) -> bool:
        """判断某分量在这条记录上是否真的算了。

        显式标记优先（``*_available``）；没有标记时按"字段是否存在且非空"判断，
        以兼容旧报告与外部调用方。
        """
        if field in record:
            return str(record.get(field)).strip() not in {"0", "False", "false", ""}
        return record.get(value_field) not in (None, "")

    def available_components(self, records: Sequence[dict]) -> dict:
        """按批次统计哪些分量可用。

        判定用"批次内至少有一定比例记录标记为可用"，而不是"任一记录可用"——
        个别记录缺能会把整批权重拉偏。默认阈值 50%。
        """
        result = {}
        total = len(records) or 1
        for name, (field, value_field) in self.COMPONENTS.items():
            hits = sum(1 for record in records if self._available(record, field, value_field))
            result[name] = hits / total >= self.availability_threshold
        return result

    def effective_weights(self, available: dict) -> dict:
        """把权重按可用分量重归一化；全部不可用时退回原始权重。"""
        nominal = {"A": self.alpha, "G": self.beta, "T": self.gamma}
        if not self.renormalize:
            return dict(nominal)

        kept = {name: weight for name, weight in nominal.items() if available.get(name, True)}
        total = sum(kept.values())
        if total <= 0:
            # 三路都没数据：退回名义权重，由调用方从 *_available 判断可信度
            return dict(nominal)

        weights = {name: 0.0 for name in nominal}
        for name, weight in kept.items():
            weights[name] = weight / total
        return weights

    def score_ceiling(self, available: dict) -> float:
        """按可用分量数给出本次运行的分数上限（0~100）。

        单路证据下把上限压到 75 分（低于 80 分档）：**高风险必须至少两路独立证据**。
        理由是一路分数的"100 分"只是该分量自身的最大值，没有任何旁证，
        把它当成最高异常等级是过度声称。两路及以上不折扣——
        缺失的第三路在本项目是结构性缺失（无车牌即无 T），折扣真实高分得不偿失。
        """
        kept = [name for name in self.COMPONENTS if available.get(name, False)]
        return self.evidence_ceiling.get(len(kept), 100.0)

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

    def fuse_one(self, record: dict, weights: Optional[dict] = None) -> dict:
        return self.fuse([record], weights=weights)[0]

    def fuse(self, records: Sequence[dict], weights: Optional[dict] = None) -> List[dict]:
        """批量融合。

        权重按**整批**的分量可用性确定（同一批用同一把尺子），
        避免同一次运行内不同车辆用不同权重导致分数不可比。
        """
        records = list(records)
        if weights is None:
            available = self.available_components(records)
            weights = self.effective_weights(available)
        else:
            available = {name: weights.get(name, 0.0) > 0 for name in self.COMPONENTS}

        ceiling = self.score_ceiling(available) if self.renormalize else 100.0

        output = []
        for record in records:
            a = _clip(_to_float(record.get("iforest_score")))
            g = _clip(_to_float(record.get("graph_anomaly_index")))
            t = _clip(_to_float(record.get("time_freq_index")))

            risk = (
                weights.get("A", 0.0) * a
                + weights.get("G", 0.0) * g
                + weights.get("T", 0.0) * t
            )
            risk_100 = round(100.0 * risk, 1)
            raw_100 = risk_100

            # 证据封顶：单路证据不足以支撑最高异常等级，但原始分要留下来可追溯
            if risk_100 > ceiling:
                risk_100 = round(ceiling, 1)
                risk = ceiling / 100.0

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
                        "alpha": round(weights.get("A", 0.0), 4),
                        "beta": round(weights.get("G", 0.0), 4),
                        "gamma": round(weights.get("T", 0.0), 4),
                    },
                    "risk_component_available": {
                        "A": 1 if available.get("A") else 0,
                        "G": 1 if available.get("G") else 0,
                        "T": 1 if available.get("T") else 0,
                    },
                    "risk_score_ceiling": ceiling,
                    "risk_score_raw_100": round(raw_100, 1),
                    "risk_score_capped": 1 if raw_100 > ceiling else 0,
                }
            )
            output.append(fused)
        return output
