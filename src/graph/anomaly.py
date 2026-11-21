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

from .calibration import fit_and_calibrate
from .features import clustering_coefficient


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
    anchors: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """计算每个节点的图异常指数 G 及支撑证据。

    G 的标定口径（2026-11-20 修正）：**不再用 min-max**。
    ``minmax_normalize`` 会让每张图必然产出一个 1.0 和一个 0.0，
    于是"本批最异常的车"恒得满分，与这批车是否真有异常无关，
    也无法跨批次比较。改为分位锚点标定：

    - 传入 ``anchors``（来自基线图 ``quantile_anchors``）→ 用基线锚点标定，跨批次可比；
    - 不传 ``anchors`` → 以本图 P50/P99 自标定，仅保证档位语义、不保证跨批可比。

    Args:
        adjacency: 加权邻接矩阵。
        community: 社区划分（当前仅作接口保留，未参与打分）。
        rank: 低秩截断秩。
        anchors: 分位锚点 ``{"low":..., "high":...}``。

    Returns:
        dict，包含:
        - ``graph_anomaly_index``: [0,1] 图异常指数 G（锚点标定后）
        - ``reconstruction_error``: 低秩重构残差（G 的原始值）
        - ``degree``: 加权度
        - ``clustering``: 聚类系数
        - ``calibration``: 本次使用的锚点，便于回溯
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
            "calibration": {"low": 0.0, "high": 0.0, "n": 0},
        }

    np.fill_diagonal(A, 0.0)
    approximation = low_rank_reconstruction(A, rank)
    reconstruction_error = np.linalg.norm(A - approximation, axis=1)
    degree = A.sum(axis=1)
    binary = (A > 0).astype(np.float64)
    clustering = clustering_coefficient(binary)

    if anchors:
        from .calibration import anchor_normalize

        calibrated = anchor_normalize(reconstruction_error, anchors)
        used = dict(anchors)
    else:
        calibrated, used = fit_and_calibrate(reconstruction_error)

    return {
        "graph_anomaly_index": calibrated,
        "reconstruction_error": reconstruction_error,
        "degree": degree,
        "clustering": clustering,
        "calibration": used,
    }
