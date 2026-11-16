"""Louvain 社区发现（研究要点三）。

实现标准的 Louvain 流程：
1. 局部移动（local moving）：逐节点尝试移动到使模块度增益最大的邻居社区；
2. 社区聚合（aggregation）：把同一社区合并为超级节点，重复第 1 步；
3. 直到模块度不再提升或达到最大层数。

纯 NumPy 实现，无第三方依赖。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def modularity(adjacency: np.ndarray, membership, resolution: float = 1.0) -> float:
    """计算加权无向图的模块度 Q。"""
    A = np.asarray(adjacency, dtype=np.float64)
    if A.shape[0] == 0:
        return 0.0
    total = float(A.sum())
    if total <= 0:
        return 0.0

    membership = np.asarray(membership)
    score = 0.0
    for community in np.unique(membership):
        nodes = np.nonzero(membership == community)[0]
        if nodes.size == 0:
            continue
        internal = float(A[np.ix_(nodes, nodes)].sum())
        degree_sum = float(A[nodes].sum())
        score += internal / total - resolution * (degree_sum / total) ** 2
    return float(score)


def _local_moving(
    adjacency: np.ndarray,
    resolution: float,
    rng: np.random.RandomState,
    max_iter: int = 100,
) -> np.ndarray:
    """Louvain 第一阶段：逐节点局部移动。"""
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    degrees = A.sum(axis=1)
    total = float(A.sum())
    if total <= 0:
        return np.arange(n, dtype=np.int64)

    membership = np.arange(n, dtype=np.int64)
    weights = degrees.copy()

    for _ in range(max_iter):
        moved = False
        for node in rng.permutation(n):
            community = int(membership[node])
            degree = float(degrees[node])
            if degree <= 0:
                continue

            neighbor_weights: Dict[int, float] = {}
            for neighbor in np.nonzero(A[node])[0]:
                target = int(membership[neighbor])
                neighbor_weights[target] = neighbor_weights.get(target, 0.0) + float(A[node, neighbor])

            other_weight = weights[community] - degree
            stay = neighbor_weights.get(community, 0.0) / total - resolution * degree * other_weight / (total * total)

            best_community = community
            best_gain = 0.0
            for target, edge_weight in neighbor_weights.items():
                if target == community:
                    continue
                gain = (
                    edge_weight / total
                    - resolution * degree * weights[target] / (total * total)
                    - stay
                )
                if gain > best_gain:
                    best_gain = gain
                    best_community = target

            if best_community != community:
                weights[community] -= degree
                weights[best_community] += degree
                membership[node] = best_community
                moved = True

        if not moved:
            break

    return membership


def _aggregate(adjacency: np.ndarray, membership: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把同一社区合并为超级节点，返回 (新邻接矩阵, 节点→新索引)。"""
    labels = np.unique(membership)
    index = {int(label): position for position, label in enumerate(labels)}
    mapped = np.array([index[int(label)] for label in membership], dtype=np.int64)
    size = len(labels)

    A = np.asarray(adjacency, dtype=np.float64)
    new_adjacency = np.zeros((size, size), dtype=np.float64)
    for i in range(A.shape[0]):
        ci = int(mapped[i])
        for j in np.nonzero(A[i])[0]:
            new_adjacency[ci, int(mapped[j])] += float(A[i, j])
    return new_adjacency, mapped


def louvain_partition(
    adjacency: np.ndarray,
    resolution: float = 1.0,
    max_levels: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """返回每个节点所属的社区编号（0..k-1）。"""
    A = np.array(adjacency, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.RandomState(random_state)
    fine = np.arange(n, dtype=np.int64)
    current = A

    for _ in range(max_levels):
        local = _local_moving(current, resolution, rng)
        if len(np.unique(local)) == current.shape[0]:
            break
        new_current, mapped = _aggregate(current, local)
        fine = mapped[fine]
        current = new_current

    # 重新编号为 0..k-1
    unique = np.unique(fine)
    remap = {int(label): position for position, label in enumerate(unique)}
    return np.array([remap[int(label)] for label in fine], dtype=np.int64)


def community_summary(adjacency: np.ndarray, membership: np.ndarray) -> List[dict]:
    """输出每个社区的规模、内部边权重和密度，便于人工复核团伙。"""
    A = np.asarray(adjacency, dtype=np.float64)
    membership = np.asarray(membership)
    summary = []
    for community in np.unique(membership):
        nodes = np.nonzero(membership == community)[0]
        if nodes.size == 0:
            continue
        size = int(nodes.size)
        subgraph = A[np.ix_(nodes, nodes)]
        internal_weight = float(subgraph.sum()) / 2.0
        # 密度按“是否存在边”计算，避免多条关系累加的权重使密度大于 1
        internal_edges = float((subgraph > 0).sum()) / 2.0
        max_internal = size * (size - 1) / 2.0
        summary.append(
            {
                "community_id": int(community),
                "size": size,
                "internal_weight": round(internal_weight, 4),
                "density": round(internal_edges / max_internal, 4) if max_internal > 0 else 0.0,
                "node_indices": [int(node) for node in nodes],
            }
        )
    summary.sort(key=lambda item: (-item["size"], -item["density"]))
    return summary
