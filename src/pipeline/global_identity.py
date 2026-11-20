"""跨机位全局身份回灌。

跨机位拼接（`CrossCameraStitcher`）产出的是 `passage_key -> global_vehicle_id`
的映射，但此前只写进 `run_multi_camera.py` 的 JSON，没有回流到向量、图谱和风险报告：

- 向量里没有 `global_vehicle_id`，下游频次统计仍按单次通行算；
- 图谱默认按 `passage_id` 建节点，同一辆车在两个机位会被算成两个节点；
- 风险报告拿不到"同一辆车在其他机位出现"的证据。

本模块把这条链路接上，顺序与 `build_risk_report.py` 保持一致：

    apply_stitch → 按身份合并 → 全局图 → 回填 G → T → R → 证据 → 按身份合并报告

约定：**一个全局身份 = 一个图节点 = 一条风险记录**；未拼接上的通行仍按各自
`passage_id` 独立存在，不会被强行并成一辆车。
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence

from ..graph.payload import build_graph_payload
from ..graph.relations import RelationConfig
from ..risk.merge import apply_stitch

GLOBAL_PREFIX = "gv:"


def identity_key(row: dict) -> str:
    """风险/图节点使用的身份键：优先全局身份，其次单次通行。"""
    global_id = str(row.get("global_vehicle_id", "") or "").strip()
    if global_id:
        return f"{GLOBAL_PREFIX}{global_id}"
    passage_id = str(row.get("passage_id", "") or "").strip()
    if passage_id:
        return passage_id
    return f"{row.get('camera_id', '')}:{row.get('track_id', '')}"


def annotate_global_identity(rows_by_camera: Dict[str, List[dict]], stitch: Optional[dict]) -> Dict[str, List[dict]]:
    """把拼接结果写回每个机位的向量行（只标注真正匹配上的车辆）。"""
    if not stitch:
        return {camera: [dict(row) for row in rows] for camera, rows in rows_by_camera.items()}
    return {camera: apply_stitch([dict(row) for row in rows], stitch) for camera, rows in rows_by_camera.items()}


def _max_float(rows: Sequence[dict], field: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(field, 0) or 0))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def _longest_trajectory(rows: Sequence[dict]) -> str:
    best = ""
    for row in rows:
        points = str(row.get("trajectory_points", "") or "")
        if len(points) > len(best):
            best = points
    return best


def merge_passages(rows: Iterable[dict]) -> List[dict]:
    """把多次通行合并为「一车一行」，供建图使用。

    时间取并集（最早进入、最晚离开），时长相加，其余数值字段取最大。
    """
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(identity_key(row), []).append(row)

    merged: List[dict] = []
    for key, group in groups.items():
        representative = dict(group[0])
        entries = sorted(str(row.get("enter_time", "") or "") for row in group if row.get("enter_time"))
        exits = sorted(str(row.get("exit_time", "") or "") for row in group if row.get("exit_time"))
        cameras = sorted({str(row.get("camera_id", "") or "") for row in group if row.get("camera_id")})

        representative["identity_key"] = key
        representative["passage_count"] = len(group)
        representative["camera_ids"] = ",".join(cameras)
        if entries:
            representative["enter_time"] = entries[0]
        if exits:
            representative["exit_time"] = exits[-1]
        if len(group) > 1:
            representative["duration_sec"] = round(
                sum(float(row.get("duration_sec") or 0) for row in group), 3
            )
        representative["trajectory_points"] = _longest_trajectory(group)
        for field in ("iforest_score", "graph_anomaly_index", "time_freq_index"):
            if any(field in row for row in group):
                representative[field] = round(_max_float(group, field), 4)
        merged.append(representative)
    return merged


def build_global_graph(rows: Sequence[dict], top_k: int = 3) -> Optional[dict]:
    """按全局身份建图：同一辆车跨机位只算一个节点。"""
    config = RelationConfig(top_k=top_k, use_global_vehicle_id=True)
    return build_graph_payload(list(rows), config=config)


def apply_graph_by_identity(rows: Sequence[dict], graph: Optional[dict]) -> List[dict]:
    """把全局图的 G 回填到**每次通行**上（图谱节点是身份，记录是通行）。"""
    if not graph:
        return [dict(row) for row in rows]

    nodes = {str(node["node"]): node for node in graph.get("nodes", [])}
    communities = {item["community_id"]: item for item in graph.get("communities", [])}
    enriched: List[dict] = []
    matched = 0
    for row in rows:
        record = dict(row)
        node = nodes.get(identity_key(record))
        if node is None:
            record["graph_available"] = 0
            record.setdefault("graph_anomaly_index", 0.0)
        else:
            matched += 1
            record["graph_available"] = 1
            record["graph_anomaly_index"] = node.get("graph_anomaly_index", 0.0)
            record["community_id"] = node.get("community_id", "")
            community = communities.get(node.get("community_id"))
            if community:
                record["community_size"] = community.get("size", "")
                record["community_density"] = community.get("density", "")
        enriched.append(record)
    if rows and matched == 0:
        raise ValueError("全局图节点与记录身份完全不匹配，graph_anomaly_index 会全为 0")
    return enriched


def merge_risk_records(records: Iterable[dict]) -> List[dict]:
    """按身份合并风险记录：一辆车一条，证据与伙伴取并集。"""
    groups: Dict[str, List[dict]] = {}
    for record in records:
        groups.setdefault(identity_key(record), []).append(record)

    merged: List[dict] = []
    for key, group in groups.items():
        top = max(group, key=lambda row: float(row.get("risk_score_100", 0) or 0))
        record = dict(top)
        record["identity_key"] = key
        record["passage_count"] = len(group)
        record["camera_ids"] = ",".join(
            sorted({str(row.get("camera_id", "") or "") for row in group if row.get("camera_id")})
        )
        record["risk_score_100"] = max(float(row.get("risk_score_100", 0) or 0) for row in group)

        evidence: List[dict] = []
        seen = set()
        for row in group:
            for item in row.get("evidence", []) or []:
                signature = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                if signature in seen:
                    continue
                seen.add(signature)
                evidence.append(item)
        record["evidence"] = evidence

        partners: List[dict] = []
        seen_partners = set()
        for row in group:
            for partner in row.get("cross_camera_partners", []) or []:
                marker = (partner.get("camera"), partner.get("track"))
                if marker in seen_partners:
                    continue
                seen_partners.add(marker)
                partners.append(partner)
        if partners:
            record["cross_camera_partners"] = partners
        merged.append(record)

    merged.sort(key=lambda row: -float(row.get("risk_score_100", 0) or 0))
    return merged


def identity_summary(rows_by_camera: Dict[str, List[dict]], merged_records: Sequence[dict]) -> dict:
    """回灌覆盖情况：有多少通行拿到了全局身份、合并成多少辆车。"""
    total = sum(len(rows) for rows in rows_by_camera.values())
    with_identity = sum(
        1
        for rows in rows_by_camera.values()
        for row in rows
        if str(row.get("global_vehicle_id", "") or "").strip()
    )
    multi_camera = sum(
        1 for record in merged_records if len(str(record.get("camera_ids", "")).split(",")) > 1
    )
    return {
        "passages": total,
        "passages_with_global_id": with_identity,
        "global_identity_rate": round(with_identity / total, 4) if total else 0.0,
        "vehicles_after_merge": len(merged_records),
        "multi_camera_vehicles": multi_camera,
        "cameras": {camera: len(rows) for camera, rows in rows_by_camera.items()},
    }
