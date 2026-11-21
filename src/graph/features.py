"""图谱结构特征：节点度、聚类系数、邻居度。

原先 ``anomaly.py`` 与 ``gae.py`` 各写了一份逐行等价的聚类系数实现
（``_clustering_coefficient`` 与 ``_clustering``），属于复制粘贴分叉风险。
这里收敛为唯一实现，两处调用方都改为引用本模块。
"""

from __future__ import annotations

import numpy as np


def clustering_coefficient(binary_adjacency: np.ndarray) -> np.ndarray:
    """无向二值图的局部聚类系数。

    ``C_i = 2 * links_i / (d_i * (d_i - 1))``，其中 ``links_i`` 是节点 i 的邻居之间
    实际相连的边数，``d_i`` 是节点 i 的度。度为 0 或 1 时聚类系数无定义，取 0。

    注意本函数不做对角线清理：调用方应保证传入的是**无自环**的二值矩阵。
    """
    adjacency = np.asarray(binary_adjacency)
    n = adjacency.shape[0]
    coefficients = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neighbors = np.nonzero(adjacency[i])[0]
        degree = int(neighbors.size)
        if degree < 2:
            continue
        subgraph = adjacency[np.ix_(neighbors, neighbors)]
        links = float(subgraph.sum()) / 2.0
        coefficients[i] = links / (degree * (degree - 1) / 2.0)
    return coefficients


def binarize(adjacency: np.ndarray, *, clear_diagonal: bool = True) -> np.ndarray:
    """把加权邻接矩阵转成二值矩阵（可选清理对角线自环）。"""
    matrix = np.asarray(adjacency, dtype=np.float64)
    binary = (matrix > 0).astype(np.float64)
    if clear_diagonal and binary.size:
        np.fill_diagonal(binary, 0.0)
    return binary
