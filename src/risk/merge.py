"""风险报告的数据合并工具（从脚本抽出，便于单元测试）。

这些函数负责把图谱结果与跨机位拼接结果合并进风险记录。它们此前以
脚本内部函数的形式存在，出过 `passage_id`/伙伴匹配相关的 bug，因此
抽到模块里并加了回归测试。
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


def node_key(row: dict) -> str:
    """与图谱模块保持一致的节点键：优先 passage_id。"""
    passage_id = str(row.get("passage_id", "") or "").strip()
    if passage_id:
        return passage_id
    return f"{row.get('camera_id', '')}:{row.get('track_id', '')}"


def apply_graph(rows: List[dict], graph: Optional[dict]) -> List[dict]:
    """把图谱异常指数与社区信息补充到风险记录。"""
    if graph is None:
        return rows

    nodes = {node["node"]: node for node in graph.get("nodes", [])}
    communities = {item["community_id"]: item for item in graph.get("communities", [])}
    enriched = []
    matched = 0
    for row in rows:
        record = dict(row)
        node = nodes.get(node_key(row))
        if node is not None:
            matched += 1
            record["graph_available"] = 1
            record["graph_anomaly_index"] = node.get("graph_anomaly_index", 0.0)
            record["community_id"] = node.get("community_id", "")
            community = communities.get(node.get("community_id"))
            if community:
                record["community_size"] = community.get("size", "")
                record["community_density"] = community.get("density", "")
        else:
            # 没有图谱节点 ≠ 图异常低：用 graph_available 区分“无数据”和“值为 0”
            record["graph_available"] = 0
            record.setdefault("graph_anomaly_index", 0.0)
        enriched.append(record)

    # 键完全不匹配时 G 会静默变成 0（曾因旧版图谱用 camera:track 键踩过坑）
    if rows and matched == 0:
        logger.warning(
            "apply_graph: 图谱节点键与记录键完全不匹配（节点 %d 个，记录 %d 条），"
            "graph_anomaly_index 将全为 0；请确认图谱由当前版本生成",
            len(nodes),
            len(rows),
        )
    return enriched


def apply_stitch(rows: List[dict], stitch: Optional[dict]) -> List[dict]:
    """把跨机位匹配结果补充到风险记录。

    - 优先用 ``passage_id`` 匹配，其次 ``camera:源文件:track``，最后 ``camera:track``；
    - 只有真正匹配上的车辆才标注 ``global_vehicle_id``，避免给每辆车打号造成噪声。
    """
    if not stitch:
        return rows

    global_ids = stitch.get("global_vehicle_ids", {})
    # 兼容 "camera:源文件:track" 与 "camera:track" 两种键
    short_global_ids = {}
    for key, gid in global_ids.items():
        camera, _, remainder = str(key).partition(":")
        short_global_ids[f"{camera}:{remainder.rsplit(':', 1)[-1]}"] = gid

    partner_index = {}
    for data in stitch.get("stitching", {}).values():
        for match in data.get("matches", []):
            a_full = match.get("from_key") or f"{match['from_camera']}:{match['from_track']}"
            b_full = match.get("to_key") or f"{match['to_camera']}:{match['to_track']}"
            a_short = f"{match['from_camera']}:{match['from_track']}"
            b_short = f"{match['to_camera']}:{match['to_track']}"
            to_partner = {"camera": match["to_camera"], "track": match["to_track"], "similarity": match["similarity"]}
            from_partner = {"camera": match["from_camera"], "track": match["from_track"], "similarity": match["similarity"]}
            if match.get("match_method"):
                to_partner["match_method"] = from_partner["match_method"] = match["match_method"]
            partner_index.setdefault(a_full, []).append(to_partner)
            partner_index.setdefault(b_full, []).append(from_partner)

    enriched = []
    for row in rows:
        record = dict(row)
        camera = row.get("camera_id", "")
        track = str(row.get("track_id", ""))
        source = os.path.basename(str(row.get("_source_file", "") or ""))
        passage_id = str(row.get("passage_id", "") or "").strip()
        short_key = f"{camera}:{track}"
        full_key = f"{camera}:{source}:{track}" if source else short_key

        partners = []
        if passage_id:
            partners = partner_index.get(passage_id, []) or []
        if not partners and (not passage_id or full_key in global_ids):
            partners = partner_index.get(full_key, []) or []
        if not partners and ((not passage_id and not source) or full_key in global_ids):
            partners = partner_index.get(short_key, []) or []
        # 去重（同一伙伴可能同时命中 passage_id 与短键）
        seen = set()
        unique_partners = []
        for partner in partners:
            key = (partner["camera"], partner["track"])
            if key in seen:
                continue
            seen.add(key)
            unique_partners.append(partner)

        if unique_partners:
            record["cross_camera_partners"] = unique_partners
            global_id = global_ids.get(passage_id) if passage_id else None
            if global_id is None:
                global_id = global_ids.get(full_key)
            if global_id is None:
                global_id = short_global_ids.get(short_key)
            if global_id is not None:
                record["global_vehicle_id"] = global_id

        enriched.append(record)
    return enriched
