#!/usr/bin/env python
"""Export the densest high-anomaly community as a gang ("蚂蚁搬家") case report.

Rebuilds the vehicle-relation graph from the vector CSV, ranks communities by
size and mean graph anomaly, then writes a Markdown case report that a reviewer
can trace back to the recording segments.

Usage:
    python scripts/export_gang_case.py \
        --vectors data/outputs/e2e/alerts/vectors_admin_in.csv \
        --risk data/outputs/e2e/risk/risk_admin_in.json \
        --output demo/gang_sample.md
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph.community_case import rank_communities
from src.graph.payload import build_graph_payload
from src.graph.relations import RelationConfig
from src.risk.merge import node_key
from src.utils.logger import get_logger
from src.utils.vectors_io import load_vector_rows

logger = get_logger(__name__)

RELATION_LABELS = {
    "co_occurrence": "共现",
    "aggregation": "聚集",
    "following": "跟随",
    "path_similarity": "路径相似",
}



def _render(payload, community_id, members, mean_g, internal_edges, risk_index, top_n):
    member_nodes = {item["node"] for item in members}
    relation_counts = Counter(edge["relation"] for edge in internal_edges)
    global_mean_g = sum(node["graph_anomaly_index"] for node in payload["nodes"]) / len(payload["nodes"])

    lines = [
        f"# 团伙案例：社区 #{community_id}（{len(members)} 辆车）",
        "",
        "> 由 `scripts/export_gang_case.py` 自动生成，可复现。",
        f"> 数据来源：图谱 `{payload['graph_anomaly_method']}`，节点 {payload['node_count']}、边 {payload['edge_count']}、"
        f"全图模块度 {payload['modularity']}。",
        "",
        "## 一、社区概况",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 社区 ID | {community_id} |",
        f"| 规模 | **{len(members)} 辆车** |",
        f"| 社区内边数 | {len(internal_edges)} |",
        f"| 平均图异常指数 G | **{mean_g:.3f}**（全图均值 {global_mean_g:.3f}） |",
        "",
        "## 二、关系构成",
        "",
        "| 关系类型 | 条数 |",
        "|---|---|",
    ]
    for relation, count in relation_counts.most_common():
        lines.append(f"| {RELATION_LABELS.get(relation, relation)} | {count} |")
    if not relation_counts:
        lines.append("| — | 0 |")

    lines += [
        "",
        "## 三、成员清单（按 G 降序）",
        "",
        "| # | 节点 | G | R | 度 | 通行时间 | 来源文件 |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, item in enumerate(sorted(members, key=lambda node: -node["graph_anomaly_index"]), start=1):
        record = risk_index.get(item["node"], {})
        risk = record.get("risk_score_100", "")
        enter = str(record.get("enter_time", "") or "")[:19]
        source = os.path.basename(str(record.get("_source_file", "") or ""))
        lines.append(
            f"| {index} | `{item['node']}` | {item['graph_anomaly_index']:.3f} | "
            f"{risk} | {item['degree']:.0f} | {enter or '—'} | {source or '—'} |"
        )

    busiest = sorted(members, key=lambda node: -node["graph_anomaly_index"])[:3]

    # 统计通行时间跨度：短时间内的密集通行也可能是高峰车流，必须标注出来
    times = []
    for item in members:
        value = str(risk_index.get(item["node"], {}).get("enter_time", "") or "")
        try:
            times.append(datetime.fromisoformat(value))
        except ValueError:
            pass
    span_text = ""
    crowded = False
    if len(times) >= 2:
        span_sec = (max(times) - min(times)).total_seconds()
        crowded = span_sec < 120
        span_text = (
            f"本社区 {len(members)} 辆车的通行集中在 **{span_sec:.0f} 秒内**"
            f"（{min(times).strftime('%H:%M:%S')}–{max(times).strftime('%H:%M:%S')}）"
        )

    lines += [
        "",
        "## 四、为什么值得核查",
        "",
        f"1. **拓扑上紧密**：这 {len(members)} 辆车之间形成 {len(internal_edges)} 条边，"
        f"关系类型以「{RELATION_LABELS.get(relation_counts.most_common(1)[0][0], '—') if relation_counts else '—'}」为主，"
        "不是随机相邻；",
        f"2. **结构上异常**：社区平均 G={mean_g:.3f}，高于全图均值 {global_mean_g:.3f}，"
        "说明该子图的连接模式偏离常态；",
        f"3. **成员上可疑**：G 最高的 `{busiest[0]['node']}`、`{busiest[1]['node']}`、`{busiest[2]['node']}` "
        "同时出现在同一社区，符合“多车接力、短时集中通行”的特征。",
    ]

    if crowded:
        lines += [
            "",
            "## 五、需要注意（可能是误报）",
            "",
            f"{span_text}，属于**同一波连续车流**。高峰时段的正常排队同样会形成高共现、高路径相似，"
            "**不能仅凭本社区直接判定为团伙**。",
            "",
            "要排除高峰车流，需进一步确认：",
            "1. 这些车是否属于同一企业、同一司机或同一批货物；",
            "2. 是否在**非高峰时段**也重复出现（真正的“蚂蚁搬家”通常跨多日、多时段）；",
            "3. 单次载量是否明显偏低而频次偏高。",
        ]

    lines += [
        "",
        "## 六、研判建议",
        "",
        "1. 调取该社区成员的通行时段录像，核对是否来自同一企业、同一司机或同一批货物；",
        "2. 比对申报单据与实际载货，重点看“单次载量小、频次密”的模式；",
        "3. 若同一批车辆在次日或次周再次集中出现，应升级为团伙线索并移交稽查。",
        "",
        "## 七、证据可追溯",
        "",
        "上表每行都带有来源文件与通行时间，可直接定位到录像分段；"
        "完整证据链（含 camera/track/源文件）见对应风险报告中各车的 `evidence` 字段。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export a gang case report from the vehicle graph")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--risk", default="", help="Optional risk JSON for R / enter_time")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3, help="Keep top-k strongest neighbors per node")
    parser.add_argument("--top", type=int, default=1, help="How many communities to export")
    args = parser.parse_args()

    rows = load_vector_rows(args.vectors)
    if not rows:
        logger.error("No vectors in %s", args.vectors)
        sys.exit(1)

    # top_k=0 会退化成近似完全图（上万条边、模块度接近 0），必须与 build_graph.py 保持一致
    config = RelationConfig(top_k=args.top_k)
    payload = build_graph_payload(rows, config, resolution=args.resolution, method="svd")
    if payload is None:
        logger.error("Graph is empty")
        sys.exit(1)

    ranked = rank_communities(payload["nodes"], args.min_size)
    if not ranked:
        logger.error("No community with at least %d nodes", args.min_size)
        sys.exit(1)

    risk_index = {}
    if args.risk and os.path.isfile(args.risk):
        with open(args.risk, "r", encoding="utf-8") as f:
            risk_payload = json.load(f)
        for record in risk_payload.get("vehicles", []):
            risk_index[node_key(record)] = record

    sections = []
    for community_id, members, mean_g in ranked[: args.top]:
        member_nodes = {item["node"] for item in members}
        internal_edges = [
            edge
            for edge in payload["edges"]
            if edge["source"] in member_nodes and edge["target"] in member_nodes
        ]
        sections.append(
            _render(payload, community_id, members, mean_g, internal_edges, risk_index, args.top)
        )
        logger.info(
            "community %s: %d vehicles, %d internal edges, mean G=%.3f",
            community_id,
            len(members),
            len(internal_edges),
            mean_g,
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(sections))
    logger.info("Gang case written to: %s", args.output)


if __name__ == "__main__":
    main()
