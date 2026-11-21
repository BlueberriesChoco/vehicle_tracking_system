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
    calibration: dict = {}
    # 评分图是否与训练图是同一张图。这决定 G 能不能跨批次比较：
    # 同图训练时误差带被压缩，锚点自参照，标定会把窄带放大。
    self_referential = False
    if method == "gae" or gae_model:
        from .gae import GraphAutoencoderDetector

        if gae_model:
            detector = GraphAutoencoderDetector.load(gae_model)
        else:
            detector = GraphAutoencoderDetector(epochs=gae_epochs).fit(graph.adjacency)
            # 训练图 == 评分图：锚点来自本图自身，不是独立基线
            self_referential = True
            if save_gae_model:
                detector.save(save_gae_model)
        scores = detector.score(graph.adjacency)
        calibration = dict(scores.get("calibration") or {})
        method_name = "gae_baseline" if gae_model else "gae"
    else:
        scores = graph_anomaly_index(graph.adjacency, community=membership)
        calibration = dict(scores.get("calibration") or {})
        method_name = "low_rank_svd"
        # SVD 分支没有独立基线概念，锚点恒由本图自算
        self_referential = calibration.get("source") != "baseline"

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
        "version": 2,
        "kind": "vehicle_relation_graph",
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "graph_anomaly_method": method_name,
        "graph_anomaly_calibration": calibration,
        # 标定锚点的来源决定分数能不能"当真"：
        # - 独立基线（加载了 --gae-model）：编码器在别的图上训练，锚点来自那张图，
        #   跨批次可比，G 可横向比较；
        # - 自参照（同图训练 / SVD）：锚点由本图自算，误差带被压缩
        #   （实测 GAE 同图 p99/p50=1.053 vs SVD 1.376），标定会把窄带放大，
        #   高分节点数偏多，只能当本批内部相对排序看。
        #
        # 注意：判据是"训练图是否就是评分图"，**不能**只看 calibration.source——
        # GAE 的 fit 总会记录锚点，score 随之走 baseline 分支，
        # 同图训练时 source 也是 "baseline"，用它判断会把自参照误标成可比。
        "graph_anomaly_comparable": not self_referential,
        "graph_anomaly_self_referential": self_referential,
        "modularity": round(float(q), 4),
        "communities": summaries,
        "nodes": nodes,
        "edges": [
            {"source": source, "target": target, "relation": relation, "weight": weight}
            for source, target, relation, weight in graph.edge_list()
        ],
    }
