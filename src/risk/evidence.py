"""结构化证据链（研究要点四）。

每条证据都要求可回溯到具体的行为向量记录（camera_id + track_id + 源文件），
这是“结构化证据链生成准确率 ≥ 90%”的基础。
"""

from __future__ import annotations

from typing import List


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class EvidenceBuilder:
    """从风险记录中抽取可解释证据。"""

    def __init__(self, min_signal: float = 0.05):
        self.min_signal = min_signal

    # -- 证据条目 -------------------------------------------------------

    @staticmethod
    def _trace(record: dict) -> dict:
        return {
            "camera_id": record.get("camera_id", ""),
            "track_id": record.get("track_id", ""),
            "source_file": record.get("_source_file", ""),
        }

    def _item(self, record: dict, category: str, signal: str, field: str, value, description: str) -> dict:
        return {
            "category": category,
            "signal": signal,
            "field": field,
            "value": value,
            "description": description,
            "trace": self._trace(record),
        }

    # -- 构建 -----------------------------------------------------------

    def build(self, record: dict) -> List[dict]:
        items: List[dict] = []

        # 1) 行为异常指数 A
        a = _to_float(record.get("iforest_score"))
        if a >= self.min_signal:
            items.append(
                self._item(
                    record,
                    "behavior",
                    "isolation_forest",
                    "iforest_score",
                    round(a, 4),
                    f"无监督行为异常指数 A={a:.2f}",
                )
            )

        # 2) 轨迹预测残差
        residual = _to_float(record.get("traj_residual"))
        if residual > 0:
            items.append(
                self._item(
                    record,
                    "trajectory",
                    "transformer_residual",
                    "traj_residual",
                    round(residual, 2),
                    f"Transformer 轨迹预测残差 {residual:.1f}px",
                )
            )

        # 3) 图异常指数 G 与社区
        g = _to_float(record.get("graph_anomaly_index"))
        if g >= self.min_signal:
            community = record.get("community_id", "")
            size = record.get("community_size", "")
            description = f"图异常指数 G={g:.2f}"
            if community != "":
                description += f"，所属社区 {community}"
            if size != "":
                description += f"（{size} 个节点）"
            items.append(
                self._item(record, "graph", "graph_anomaly", "graph_anomaly_index", round(g, 4), description)
            )

        # 4) 时间频率 T
        t = _to_float(record.get("time_freq_index"))
        if t >= self.min_signal:
            frequency = _to_float(record.get("freq_count_24h"))
            night_ratio = _to_float(record.get("night_ratio"))
            items.append(
                self._item(
                    record,
                    "temporal",
                    "time_frequency",
                    "time_freq_index",
                    round(t, 4),
                    f"时间频率异常指数 T={t:.2f}（24h 通行 {frequency:.0f} 次，夜间占比 {night_ratio:.2f}）",
                )
            )

        # 5) 规则告警
        # alert_reason 里混有两类内容：真正的规则命中（如“蛇形行驶(平均转角0.55)”）
        # 和检测器写入的分数（如“iforest_score=0.71”“traj_residual=177.4px”）。
        # 后者不是规则，不能写成“命中预设监管规则”，否则会误导一线人员。
        reason = str(record.get("alert_reason", "") or "")
        for token in [part.strip() for part in reason.split(";") if part.strip()]:
            if "=" in token.split("(")[0]:
                continue
            items.append(
                self._item(record, "rule", "rule_engine", "alert_reason", token, f"规则命中：{token}")
            )

        # 6) 跨机位同车关联
        partners = record.get("cross_camera_partners") or []
        if partners:
            names = "、".join(
                f"{item.get('camera')}:{item.get('track')}（"
                + ("车牌一致）" if item.get("match_method") == "plate"
                   else f"外观相似度 {float(item.get('similarity', 0)):.2f}）")
                for item in partners
            )
            items.append(
                self._item(
                    record,
                    "cross_camera",
                    "stitching",
                    "cross_camera_partners",
                    len(partners),
                    f"跨机位同车关联：{names}",
                )
            )

        return items

    def build_many(self, records: List[dict]) -> List[dict]:
        results = []
        for record in records:
            enriched = dict(record)
            enriched["evidence"] = self.build(record)
            results.append(enriched)
        return results
