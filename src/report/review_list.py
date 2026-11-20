"""待核查清单：按可疑度排名，不设异常阈值。

脚本（`scripts/build_review_list.py`）与控制台后端共用这一份逻辑，
避免两处各写一套排序规则。

为什么不做「异常判定」：违规夹带一年只有个位数，用分位数切阈值会在几百辆车上
凭空造出几十条「异常」。这里只排序，由人按预算取前 N 条核查。
"""

from __future__ import annotations

from typing import Iterable, List

COLUMNS = [
    "rank",
    "passage_id",
    "display_name",
    "plate_number",
    "camera_id",
    "enter_time",
    "duration_sec",
    "risk_score_100",
    "evidence_count",
    "reason",
    "top_evidence",
]

# 证据强弱排序：图异常最能解释"为什么排前面"，其次是轨迹、行为、规则
SIGNAL_PRIORITY = {
    "graph_anomaly": 0,
    "transformer_residual": 1,
    "isolation_forest": 2,
    "time_frequency": 3,
    "rule": 4,
}

SIGNAL_LABEL = {
    "graph_anomaly": "车辆关系异常",
    "transformer_residual": "轨迹偏离预测",
    "isolation_forest": "行为组合偏离常态",
    "time_frequency": "通行频次异常",
    "rule": "触发监管规则",
    "cross_camera": "跨机位重复出现",
}


def display_name(record: dict) -> str:
    """给人看的车名：优先车牌，其次「未识别车牌 · 轨迹号」。"""
    plate = str(record.get("plate_number", "") or "").strip()
    if plate:
        return plate
    track = str(record.get("track_id", "") or "").strip()
    return f"未识别车牌 · {track}" if track else "未识别车牌"


def _sorted_evidence(record: dict) -> List[dict]:
    items = list(record.get("evidence", []) or [])

    def key(item):
        signal = str(item.get("signal", "") or item.get("category", ""))
        return SIGNAL_PRIORITY.get(signal, 9)

    return sorted(items, key=key)


def reasons(record: dict) -> List[str]:
    """把结构化证据转成人能读的「为什么可疑」。"""
    texts = []
    for item in _sorted_evidence(record):
        text = str(item.get("description") or item.get("summary") or "").strip()
        if text and text not in texts:
            texts.append(text)
    return texts


def reasons_with_labels(record: dict) -> List[dict]:
    """带类别的原因，供控制台详情页分条展示。"""
    output = []
    for item in _sorted_evidence(record):
        text = str(item.get("description") or item.get("summary") or "").strip()
        if not text:
            continue
        signal = str(item.get("signal", "") or "")
        label = SIGNAL_LABEL.get(signal) or str(item.get("category", "") or "证据")
        output.append({"label": label, "text": text, "signal": signal})
    return output


def evidence_summary(record: dict, limit: int = 2) -> str:
    items = _sorted_evidence(record)
    summaries = []
    for item in items[:limit]:
        # 证据项的文本字段是 description；保留 summary 以兼容旧格式
        text = str(item.get("description") or item.get("summary") or "").strip()
        if text:
            summaries.append(text)
    return "；".join(summaries)


def rank_vehicles(vehicles: Iterable[dict], budget: int = 20, min_score: float = 0.0) -> List[dict]:
    """按可疑度排序并截断到预算。"""
    eligible = [
        record for record in vehicles
        if float(record.get("risk_score_100", 0) or 0) >= min_score
    ]
    eligible.sort(key=lambda record: -float(record.get("risk_score_100", 0) or 0))
    if budget > 0:
        eligible = eligible[:budget]
    return eligible


def build_rows(vehicles: Iterable[dict]) -> List[dict]:
    rows = []
    for rank, record in enumerate(vehicles, start=1):
        texts = reasons(record)
        rows.append(
            {
                "rank": rank,
                "passage_id": record.get("passage_id", ""),
                "display_name": display_name(record),
                "plate_number": record.get("plate_number", "") or "",
                "camera_id": record.get("camera_id", ""),
                "enter_time": record.get("enter_time", ""),
                "duration_sec": record.get("duration_sec", ""),
                "risk_score_100": record.get("risk_score_100", ""),
                "evidence_count": len(record.get("evidence", []) or []),
                "reason": texts[0] if texts else "",
                "top_evidence": evidence_summary(record),
            }
        )
    return rows
