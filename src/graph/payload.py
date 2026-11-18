"""车辆关系图谱载荷构建（脚本共用）。

`build_graph.py` 与 `run_full_pipeline.py` 此前各写了一份节点/社区/边的组装逻辑，
容易出现格式分叉。这里收敛为唯一实现。
"""

from __future__ import annotations

from typing import Optional

from .anomaly import graph_anomaly_index
from .community import community_summary, louvain_partition, modularity
from .relations import RelationConfig, build_vehicle_graph


def build_graph_payload(
    rows,
    config: Optional[RelationConfig] = None,
    method: str = "svd",
    resolution: float = 1.0,
    gae_epochs: int = 150,
    only_analysis_target: bool = True,
    gae_model: Optional[str] = None,
    save_gae_model: Optional[str] = None,
) -> Optional[dict]:
    """构建图谱 JSON 载荷；没有可分析车辆时返回 None。"""
    config = config or RelationConfig()
    graph = build_vehicle_graph(rows, config=config, only_analysis_target=only_analysis_target)
    if len(graph.nodes) == 0:
        return None

    membership = louvain_partition(graph.adjacency, resolution=resolution)
    if method == "gae" or gae_model:
        from .gae import GraphAutoencoderDetector

        if gae_model:
            detector = GraphAutoencoderDetector.load(gae_model)
        else:
            detector = GraphAutoencoderDetector(epochs=gae_epochs).fit(graph.adjacency)
            if save_gae_model:
                detector.save(save_gae_model)
        scores = detector.score(graph.adjacency)
        method_name = "gae_baseline" if gae_model else "gae"
    else:
        scores = graph_anomaly_index(graph.adjacency, community=membership)
        method_name = "low_rank_svd"

    q = modularity(graph.adjacency, membership, resolution=resolution)
    summaries = community_summary(graph.adjacency, membership)

    nodes = []
    for position, key in enumerate(graph.nodes):
        nodes.append(
            {
                "node": key,
                "community_id": int(membership[position]),
                "graph_anomaly_index": round(float(scores["graph_anomaly_index"][position]), 4),
                "reconstruction_error": round(float(scores["reconstruction_error"][position]), 4),
                "degree": round(float(scores["degree"][position]), 4),
                "clustering": round(float(scores["clustering"][position]), 4),
            }
        )
    nodes.sort(key=lambda item: -item["graph_anomaly_index"])

    return {
        "version": 1,
        "kind": "vehicle_relation_graph",
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "graph_anomaly_method": method_name,
        "modularity": round(float(q), 4),
        "communities": summaries,
        "nodes": nodes,
        "edges": [
            {"source": source, "target": target, "relation": relation, "weight": weight}
            for source, target, relation, weight in graph.edge_list()
        ],
    }
