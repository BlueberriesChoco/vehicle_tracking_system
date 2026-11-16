"""图异常指数 G（研究要点三）。

当前实现采用**线性图自编码器（低秩 SVD 重构）**作为基线：
用邻接矩阵的低秩近似 Â 重构正常图结构，节点重构误差越大，说明其连接模式
越偏离整体结构，图异常指数 G 越高。

说明：真正的神经图自编码器（GAE/VGAE）需要更多带标注的图数据，属于后续工作；
低秩重构是 GAE 的线性特例，可作为可解释、无需训练的基线。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .normalize import minmax_normalize


def _clustering_coefficient(binary_adjacency: np.ndarray) -> np.ndarray:
    n = binary_adjacency.shape[0]
    coefficients = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neighbors = np.nonzero(binary_adjacency[i])[0]
        degree = int(neighbors.size)
        if degree < 2:
            continue
        subgraph = binary_adjacency[np.ix_(neighbors, neighbors)]
        links = float(subgraph.sum()) / 2.0
        coefficients[i] = links / (degree * (degree - 1) / 2.0)
    return coefficients


def low_rank_reconstruction(adjacency: np.ndarray, rank: Optional[int] = None) -> np.ndarray:
    """邻接矩阵的低秩逼近 Â = U_k S_k V_k^T。"""
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n >= 2 and A.sum() > 0:
        u, s, vt = np.linalg.svd(A)
        if rank is None:
            rank = max(1, min(3, n - 1))
        rank = int(min(max(rank, 1), len(s)))
        approximation = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        # 对称图（邻接矩阵对称）在奇异值退化时 U 与 V 不一定平行，
        # 截断结果会轻微不对称；对称化后节点的重构误差才可比。
        if np.allclose(A, A.T, atol=1e-9):
            approximation = 0.5 * (approximation + approximation.T)
        return approximation
    return np.zeros_like(A)


def graph_anomaly_index(
    adjacency: np.ndarray,
    community: Optional[np.ndarray] = None,
    rank: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """计算每个节点的图异常指数 G 及支撑证据。

    Returns:
        dict，包含:
        - ``graph_anomaly_index``: [0,1] 图异常指数 G
        - ``reconstruction_error``: 低秩重构残差（G 的原始值）
        - ``degree``: 加权度
        - ``clustering``: 聚类系数
    """
    A = np.array(adjacency, dtype=np.float64, copy=True)
    n = A.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "graph_anomaly_index": empty,
            "reconstruction_error": empty,
            "degree": empty,
            "clustering": empty,
        }

    np.fill_diagonal(A, 0.0)
    approximation = low_rank_reconstruction(A, rank)
    reconstruction_error = np.linalg.norm(A - approximation, axis=1)
    degree = A.sum(axis=1)
    binary = (A > 0).astype(np.float64)
    clustering = _clustering_coefficient(binary)

    return {
        "graph_anomaly_index": minmax_normalize(reconstruction_error),
        "reconstruction_error": reconstruction_error,
        "degree": degree,
        "clustering": clustering,
    }
