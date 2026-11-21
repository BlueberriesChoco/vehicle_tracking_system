"""从全链路产物中汇总关键指标（供文档引用，避免手工抄写过期数字）。"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from typing import Optional


def _read_csv(path: str):
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: str):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first(pattern: str) -> str:
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[0] if matches else ""


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(row: dict, key: str) -> bool:
    return str(row.get(key, "0")).strip() in {"1", "True", "true", "yes"}


def collect_metrics(root: str, benchmark_path: str = "") -> dict:
    """读取 ``root`` 下的向量/告警/图谱/风险产物并汇总。

    ``benchmark_path`` 可选：传入 ``benchmark_throughput.py --output`` 生成的 JSON，
    汇总结果会额外带上吞吐与可承载路数。
    """
    # 兼容两种产物命名：`vectors.csv`（smoke_test）与 `vectors_admin_in.csv`（run_full_pipeline）
    vectors = _read_csv(_first(os.path.join(root, "alerts", "vectors*.csv")))
    alerts = _read_csv(_first(os.path.join(root, "alerts", "alerts*.csv")))
    graph = _read_json(_first(os.path.join(root, "graphs", "graph*.json")))
    risk = _read_json(_first(os.path.join(root, "risk", "risk*.json")))
    benchmark = _read_json(benchmark_path) if benchmark_path else {}

    risk_rows = risk.get("vehicles", []) if isinstance(risk, dict) else []
    risk_scores = [_to_float(row.get("risk_score_100")) for row in risk_rows]

    # 图谱层健康度证据：只看"有图谱节点"的车辆（graph_available != 0），
    # 否则非监管对象/键不匹配的 0 分会把最大值也拉平。
    graph_scores = [
        _to_float(row.get("graph_anomaly_index"))
        for row in risk_rows
        if _to_float(row.get("graph_available", 1)) != 0.0
    ]
    graph_available_nodes = sum(
        1 for row in risk_rows if _to_float(row.get("graph_available", 1)) != 0.0
    )

    def median_component(key: str) -> Optional[float]:
        values = [_to_float(row.get(key)) for row in risk_rows]
        return round(statistics.median(values), 3) if values else None

    evidence_total = sum(len(row.get("evidence", []) or []) for row in risk_rows)
    evidence_traced = sum(
        1
        for row in risk_rows
        for item in (row.get("evidence", []) or [])
        if (item.get("trace", {}) or {}).get("source_file")
    )

    anomaly_rows = [row for row in alerts if _flag(row, "is_anomaly")]
    # 分母必须始终是“车辆总数”。run_anomaly_detection 默认 anomaly_only=True，
    # 此时 alerts.csv 只有告警行，用它作分母会把告警率算成 100%。
    denominator = len(vectors) or len(alerts)
    reasons = {}
    for row in anomaly_rows:
        for part in str(row.get("alert_reason", "") or "").split(";"):
            # 归一化：去掉 "iforest_score=0.66" 的取值、"(平均转角0.55)" 的括号说明
            key = part.split("(")[0].split("=")[0].strip()
            if key:
                reasons[key] = reasons.get(key, 0) + 1

    return {
        "root": root,
        "vehicles": len(vectors) or len(risk_rows),
        "alerts": len(anomaly_rows),
        "alert_rate_percent": round(100.0 * len(anomaly_rows) / denominator, 2) if denominator else None,
        "alert_reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])),
        "risk_high": risk.get("high_risk") if isinstance(risk, dict) else None,
        "risk_medium": risk.get("medium_risk") if isinstance(risk, dict) else None,
        "risk_min": round(min(risk_scores), 1) if risk_scores else None,
        "risk_median": round(statistics.median(risk_scores), 1) if risk_scores else None,
        "risk_max": round(max(risk_scores), 1) if risk_scores else None,
        "component_medians": {
            "A": median_component("iforest_score"),
            "G": median_component("graph_anomaly_index"),
            "T": median_component("time_freq_index"),
        },
        # G 改为分位锚点标定后（P50→0、P99→1），中位数**按定义**就是 0，
        # 不能再拿"中位数为 0"判定图谱层失效。真正能区分"失效"与"正常"的是
        # 分布是否有区分度：最大值和可用节点数。
        "graph_score_max": round(max(graph_scores), 4) if graph_scores else None,
        "graph_available_nodes": graph_available_nodes,
        "graph_calibration": (graph.get("graph_anomaly_calibration") if isinstance(graph, dict) else None),
        "evidence_total": evidence_total,
        "evidence_traced": evidence_traced,
        "evidence_trace_rate_percent": (
            round(100.0 * evidence_traced / evidence_total, 2) if evidence_total else None
        ),
        "graph_nodes": graph.get("node_count") if isinstance(graph, dict) else None,
        "graph_edges": graph.get("edge_count") if isinstance(graph, dict) else None,
        "graph_communities": len(graph.get("communities", [])) if isinstance(graph, dict) else None,
        "graph_modularity": round(_to_float(graph.get("modularity")), 4) if isinstance(graph, dict) else None,
        "pipeline_fps": benchmark.get("pipeline_fps"),
        "streams": benchmark.get("estimated_streams_at_target_fps"),
    }


def render_markdown(metrics: dict) -> str:
    """把指标渲染成可直接粘进文档的 Markdown 表格。"""

    def show(value, suffix: str = "") -> str:
        return "—" if value is None else f"{value}{suffix}"

    medians = metrics.get("component_medians", {})
    lines = [
        "<!-- generated by scripts/report_metrics.py; do not edit by hand -->",
        "| 指标 | 实测值 |",
        "| --- | --- |",
        f"| 车辆数 | **{show(metrics['vehicles'])}** |",
        f"| 告警数 / 告警率 | **{show(metrics['alerts'])} / {show(metrics['alert_rate_percent'], '%')}** |",
        f"| 中风险 / 高风险 | **{show(metrics['risk_medium'])} / {show(metrics['risk_high'])}** |",
        f"| R 范围（min / median / max） | **{show(metrics['risk_min'])} / {show(metrics['risk_median'])} / {show(metrics['risk_max'])}** |",
        f"| A / G / T 中位数 | {show(medians.get('A'))} / {show(medians.get('G'))} / {show(medians.get('T'))} |",
        f"| 证据数 / 可追溯率 | **{show(metrics['evidence_total'])} / {show(metrics['evidence_trace_rate_percent'], '%')}** |",
        f"| 图谱（节点 / 边 / 社区 / Q） | {show(metrics['graph_nodes'])} / {show(metrics['graph_edges'])} / {show(metrics['graph_communities'])} / {show(metrics['graph_modularity'])} |",
    ]
    if metrics.get("pipeline_fps") is not None:
        lines.append(
            "| 吞吐 / 可承载路数（纯处理，不含 CSV 写出） | **"
            + show(metrics["pipeline_fps"])
            + " FPS / "
            + show(metrics.get("streams"))
            + " 路（按 15 FPS/路）** |"
        )
    reasons = metrics.get("alert_reasons") or {}
    if reasons:
        detail = "、".join(f"{name} {count}" for name, count in reasons.items())
        lines.append(f"| 告警构成 | {detail} |")
    return "\n".join(lines) + "\n"


def graph_layer_is_dead(metrics: dict) -> bool:
    """判断图谱层是否真的没有贡献任何信号。

    **口径变更（2026-11-20）**：G 已从 min-max 改为分位锚点标定（P50→0、P99→1），
    因此中位数**按定义就是 0**，旧判据（``median == 0`` 即失效）会恒定误报。
    新判据看分布是否有区分度：

    - 没有任何可用图谱节点 → 失效；
    - 最大值也是 0（所有车的 G 都归零）→ 失效；
    - 否则正常（中位数为 0 是标定的预期结果，不是故障）。

    兼容旧产物：没有 ``graph_score_max`` 字段时退回按中位数判断。
    """
    medians = metrics.get("component_medians") or {}
    if "graph_score_max" in metrics:
        if not metrics.get("graph_available_nodes"):
            return True
        top = metrics.get("graph_score_max")
        return not top  # None 或 0.0 都视为失效
    return not medians.get("G")


def pick_top_risk(risk_payload: dict) -> dict:
    """从风险报告载荷中挑出综合风险最高的一辆车。"""
    rows = risk_payload.get("vehicles", []) if isinstance(risk_payload, dict) else []
    if not rows:
        return {}
    return max(rows, key=lambda row: float(row.get("risk_score_100", 0) or 0))
