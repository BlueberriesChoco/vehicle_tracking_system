"""车辆关系图谱（研究要点三）的回归测试。

覆盖：轨迹解析/相似度、时间重叠、关系建边、分析对象过滤、
Louvain 社区发现、图异常指数 G、社区摘要。
"""

import numpy as np

from src.graph import (
    RelationConfig,
    build_vehicle_graph,
    community_summary,
    graph_anomaly_index,
    louvain_partition,
    modularity,
    parse_trajectory,
    time_overlap_seconds,
    trajectory_similarity,
)


def _vec(track, camera="cam01", enter="2026-06-01T08:00:00", exit_="2026-06-01T08:00:20",
         points="0,0;5,0;10,0;15,0;20,0;25,0;30,0;35,0;40,0;45,0", target=1):
    return {
        "track_id": track,
        "camera_id": camera,
        "enter_time": enter,
        "exit_time": exit_,
        "duration_sec": 20.0,
        "trajectory_points": points,
        "analysis_target": target,
        "vehicle_group": "ordinary",
    }


def test_parse_trajectory_roundtrip():
    points = parse_trajectory("0,0;10,5;20,10")
    assert points.shape == (3, 2)
    assert points[1].tolist() == [10.0, 5.0]


def test_trajectory_similarity_identical_high_and_reversed_low():
    a = parse_trajectory("0,0;10,0;20,0;30,0")
    identical = parse_trajectory("0,0;10,0;20,0;30,0")
    reversed_ = parse_trajectory("30,0;20,0;10,0;0,0")
    perpendicular = parse_trajectory("0,0;0,10;0,20;0,30")

    assert trajectory_similarity(a, identical) > 0.99
    assert trajectory_similarity(a, reversed_) < 0.3
    assert trajectory_similarity(a, perpendicular) < 0.5


def test_time_overlap_seconds():
    from datetime import datetime

    a = (datetime(2026, 6, 1, 8, 0, 0), datetime(2026, 6, 1, 8, 0, 20))
    b = (datetime(2026, 6, 1, 8, 0, 5), datetime(2026, 6, 1, 8, 0, 25))
    assert time_overlap_seconds(a[0], a[1], b[0], b[1]) == 15.0


def test_build_graph_creates_three_relation_types():
    v1 = _vec(1)
    v2 = _vec(
        2,
        enter="2026-06-01T08:00:05",
        points="5,0;10,0;15,0;20,0;25,0;30,0;35,0;40,0;45,0;50,0",
    )

    graph = build_vehicle_graph([v1, v2], config=RelationConfig(path_similarity_threshold=0.9))

    relations = {edge.relation for edge in graph.edges}
    assert {"co_occurrence", "following", "path_similarity"} <= relations
    assert graph.adjacency.shape == (2, 2)
    assert graph.adjacency[0, 1] == graph.adjacency[1, 0]
    assert graph.adjacency[0, 1] > 0


def test_build_graph_skips_non_analysis_target():
    v1 = _vec(1)
    freight = _vec(2, target=0)

    graph = build_vehicle_graph([v1, freight])

    assert graph.nodes == ["cam01:1"]
    assert graph.edges == []


def test_build_graph_uses_passage_id_to_avoid_track_collision():
    """同一机位不同录像段的 track_id 会重复，passage_id 能区分它们。"""
    first = _vec(1)
    first["passage_id"] = "cam01_2026090108_1"
    second = _vec(1)
    second["passage_id"] = "cam01_2026090118_1"

    graph = build_vehicle_graph([first, second])

    assert len(graph.nodes) == 2
    assert "cam01_2026090108_1" in graph.nodes
    assert "cam01_2026090118_1" in graph.nodes


def test_build_graph_same_camera_only():
    v1 = _vec(1, camera="cam01")
    v2 = _vec(2, camera="cam02")

    graph = build_vehicle_graph([v1, v2], config=RelationConfig(same_camera_only=True))

    assert len(graph.nodes) == 2
    assert graph.edges == []


def test_build_graph_top_k_limits_degree():
    # 4 辆车轨迹形状相同、时间都错开，默认会两两连边
    vectors = [_vec(index) for index in range(4)]

    full = build_vehicle_graph(vectors, config=RelationConfig(path_similarity_threshold=0.9))
    trimmed = build_vehicle_graph(
        vectors, config=RelationConfig(path_similarity_threshold=0.9, top_k=1)
    )

    assert len(full.edges) >= 6
    # top-k 取“每个节点最强 k 个邻居”的并集，边数应显著减少
    assert len(trimmed.edges) > 0
    assert len(trimmed.edges) < len(full.edges)


def test_louvain_finds_two_clusters_in_barbell():
    n = 6
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        adjacency[i, j] = adjacency[j, i] = 1.0

    membership = louvain_partition(adjacency, resolution=1.0, random_state=0)

    assert len(np.unique(membership)) == 2
    assert len(set(membership[:3].tolist())) == 1
    assert len(set(membership[3:].tolist())) == 1
    assert modularity(adjacency, membership) > 0.2


def test_graph_anomaly_index_range_and_shape():
    adjacency = np.array([
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    result = graph_anomaly_index(adjacency)

    assert result["graph_anomaly_index"].shape == (4,)
    assert np.all(result["graph_anomaly_index"] >= 0.0)
    assert np.all(result["graph_anomaly_index"] <= 1.0)
    assert result["degree"].tolist() == [2.0, 2.0, 3.0, 1.0]


def test_graph_anomaly_index_flags_deviant_node():
    # 0..3 构成 K4，节点 4 只与 0 相连：其连接模式明显偏离整体
    n = 5
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            adjacency[i, j] = adjacency[j, i] = 1.0
    adjacency[0, 4] = adjacency[4, 0] = 1.0

    result = graph_anomaly_index(adjacency)
    recon = result["reconstruction_error"]

    assert recon[4] > recon[0]


def test_community_summary_covers_all_nodes():
    adjacency = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            adjacency[i, j] = adjacency[j, i] = 1.0
    membership = np.zeros(4, dtype=np.int64)

    summary = community_summary(adjacency, membership)

    assert len(summary) == 1
    assert summary[0]["size"] == 4
    assert summary[0]["density"] == 1.0


def test_community_density_stays_at_most_one_with_multi_relation_weights():
    """多条关系累加会让边权重 > 1，但密度必须仍在 [0,1]。"""
    adjacency = np.full((3, 3), 3.0)
    np.fill_diagonal(adjacency, 0.0)
    membership = np.zeros(3, dtype=np.int64)

    summary = community_summary(adjacency, membership)

    assert summary[0]["density"] == 1.0
    assert summary[0]["internal_weight"] == 9.0


def test_graph_anomaly_index_empty_graph():
    result = graph_anomaly_index(np.zeros((0, 0)))

    assert result["graph_anomaly_index"].size == 0
    assert result["degree"].size == 0


def test_graph_anomaly_index_no_edges_returns_zeros():
    """没有任何边时，低秩重构残差全为 0，不应出现除零或 NaN。"""
    result = graph_anomaly_index(np.zeros((4, 4)))

    assert result["graph_anomaly_index"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert not np.isnan(result["graph_anomaly_index"]).any()


def test_graph_anomaly_index_single_node():
    result = graph_anomaly_index(np.zeros((1, 1)))

    assert result["graph_anomaly_index"].tolist() == [0.0]


def test_low_rank_reconstruction_zero_graph():
    from src.graph.anomaly import low_rank_reconstruction

    approximation = low_rank_reconstruction(np.zeros((3, 3)))

    assert np.allclose(approximation, 0.0)


def test_low_rank_reconstruction_symmetric_for_symmetric_input():
    from src.graph.anomaly import low_rank_reconstruction

    adjacency = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    approximation = low_rank_reconstruction(adjacency, rank=1)

    assert np.allclose(approximation, approximation.T, atol=1e-8)
