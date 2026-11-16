"""团伙案例排序逻辑的测试。"""

from src.graph.community_case import rank_communities


def _node(name, community, g):
    return {"node": name, "community_id": community, "graph_anomaly_index": g}


def test_rank_prefers_large_high_anomaly_community():
    nodes = [
        _node("a", 0, 0.9),
        _node("b", 0, 0.9),
        _node("c", 0, 0.9),
        _node("d", 1, 0.2),
        _node("e", 1, 0.2),
        _node("f", 1, 0.2),
    ]

    ranked = rank_communities(nodes, min_size=3)

    assert ranked[0][0] == 0
    assert abs(ranked[0][2] - 0.9) < 1e-9


def test_small_communities_are_excluded():
    nodes = [_node("a", 0, 0.9), _node("b", 1, 0.8), _node("c", 2, 0.7)]

    assert rank_communities(nodes, min_size=3) == []


def test_community_mean_is_computed_over_members():
    nodes = [_node("a", 0, 0.6), _node("b", 0, 0.2), _node("c", 0, 1.0)]

    ranked = rank_communities(nodes, min_size=3)

    assert abs(ranked[0][2] - 0.6) < 1e-9
