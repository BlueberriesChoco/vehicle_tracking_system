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
    "camera_id",
    "enter_time",
    "duration_sec",
    "risk_score_100",
    "evidence_count",
    "top_evidence",
    "plate_number",
]


def evidence_summary(record: dict, limit: int = 2) -> str:
    items = record.get("evidence", []) or []
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
        rows.append(
            {
                "rank": rank,
                "passage_id": record.get("passage_id", ""),
                "camera_id": record.get("camera_id", ""),
                "enter_time": record.get("enter_time", ""),
                "duration_sec": record.get("duration_sec", ""),
                "risk_score_100": record.get("risk_score_100", ""),
                "evidence_count": len(record.get("evidence", []) or []),
                "top_evidence": evidence_summary(record),
                "plate_number": record.get("plate_number", ""),
            }
        )
    return rows
