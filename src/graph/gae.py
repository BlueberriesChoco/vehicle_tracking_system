"""图自编码器（GAE）与图异常指数 G（研究要点三）。

实现要点：
- **特征驱动的 GCN 编码器**：节点输入是该节点的结构特征向量（度、加权度、
  聚类系数、邻居平均度），而不是“节点数 × 节点数”的展开矩阵。
  因此模型参数与图的节点数解耦，可以在**基线图**上训练、在**目标图**上评分。
- **边掩码 + 负采样训练**：随机遮住一部分真实边，同时抽取等量的非边作为负样本，
  只在这两部分上算 BCE。只重建正边时模型会退化成“全输出 1”，重构误差失去区分度；
  加入负样本后重构值才有实际含义。
- 节点异常分 = 漏判真实边与幻觉非边两个均值中**较大的一个**：
  ``max( mean_{A_ij>0}(1 - recon_ij), mean_{A_ij=0}(recon_ij) )``。
  两侧都取均值，避免退化成“度检测器”（实测与度相关系数在两次时间切分下为
  −0.33 与 +0.07，没有稳定偏置，因此也不宣称与度无关）。
- 输出 [0,1] 的图异常指数 G（对节点异常分做 min-max 归一化）。

为什么改：原实现的编码器是 ``nn.Linear(num_nodes, hidden)``，权重形状绑定节点数，
换一张节点数不同的图就必须重新训练，只能“单图自监督”——训练图与评分图重合，
重构误差偏乐观，也无法跨天/跨机位迁移。指标 11（GAE 识别异常节点）卡在这里。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration import anchor_normalize, quantile_anchors
from .features import clustering_coefficient as _clustering
from .normalize import minmax_normalize

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


#: 节点特征列名，顺序与 ``node_features`` 的列一致。
NODE_FEATURE_NAMES: Tuple[str, ...] = (
    "degree",
    "weighted_degree",
    "clustering",
    "neighbor_degree",
)


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """对称归一化：D^-1/2 (A + I) D^-1/2。"""
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    A = A + np.eye(n)
    degrees = A.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    return inv_sqrt[:, None] * A * inv_sqrt[None, :]


def node_features(adjacency: np.ndarray) -> np.ndarray:
    """把邻接矩阵转换成 (n, 4) 的结构特征，供 GCN 编码器输入。

    - ``degree``：度 / 图内最大度；
    - ``weighted_degree``：加权度 / 图内最大加权度，反映关系强度之和；
    - ``clustering``：聚类系数；
    - ``neighbor_degree``：邻居平均度 / 图内最大度，反映局部区域稠密程度；

    归一化用**本图最大值**：单车道图上度只有 3~8，若除以 (n-1) 所有节点的度特征
    会挤在 0.01 附近，模型无法区分；除以图内最大度后取值铺满 (0, 1]。

    实测取舍：也试过"用基线图的尺度做固定参照，让超范围节点的特征大于 1"，
    在 66 分钟数据上反而更差且不稳定（hub 型注入 AUC 0.83±0.24 vs 本方案 0.98±0.04），
    因此保留图内归一。
    """
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, len(NODE_FEATURE_NAMES)), dtype=np.float32)

    A = A.copy()
    np.fill_diagonal(A, 0.0)
    binary = (A > 0).astype(np.float64)

    degree = binary.sum(axis=1)
    weighted = A.sum(axis=1)
    clustering = _clustering(binary)

    weighted_max = float(weighted.max()) if weighted.size and weighted.max() > 0 else 1.0
    degree_max = float(degree.max()) if degree.size and degree.max() > 0 else 1.0

    neighbor_sum = binary @ degree
    neighbor_mean = np.divide(
        neighbor_sum, degree, out=np.zeros_like(neighbor_sum), where=degree > 0
    )

    features = np.stack(
        [
            degree / degree_max,
            weighted / weighted_max,
            clustering,
            neighbor_mean / degree_max,
        ],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class GraphAutoencoder(nn.Module):
    """两层 GCN 编码器 + 内积解码器；参数与节点数无关，只看特征维度。"""

    def __init__(self, in_dim: int, hidden_dim: int = 32, latent_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.encoder1 = nn.Linear(in_dim, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, features: "torch.Tensor", adjacency_norm: "torch.Tensor") -> "torch.Tensor":
        hidden = torch.relu(adjacency_norm @ self.encoder1(features))
        hidden = self.dropout(hidden)
        return adjacency_norm @ self.encoder2(hidden)

    def decode(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(embeddings @ embeddings.t())

    def forward(self, features: "torch.Tensor", adjacency_norm: "torch.Tensor"):
        embeddings = self.encode(features, adjacency_norm)
        return self.decode(embeddings), embeddings


class GraphAutoencoderDetector:
    """边掩码 + 负采样训练的 GAE，用节点重构偏差输出图异常指数 G。

    训练与评分解耦：``fit`` 只吃一张参考图（正常基线），``score`` 可以对任意
    节点数的另一张图打分。两者节点数可以不同。
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        epochs: int = 150,
        learning_rate: float = 0.02,
        weight_decay: float = 1e-3,
        dropout: float = 0.2,
        mask_fraction: float = 0.2,
        negative_ratio: float = 1.0,
        device: str = "cpu",
        random_state: int = 42,
    ):
        if torch is None:
            raise RuntimeError("GraphAutoencoderDetector 需要 PyTorch")
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.mask_fraction = mask_fraction
        self.negative_ratio = negative_ratio
        self.device = device
        self.random_state = random_state

        self.model: Optional[GraphAutoencoder] = None
        self.masked_edges_: List[Tuple[int, int]] = []
        self.negative_edges_: List[Tuple[int, int]] = []
        self.training_history: List[dict] = []
        #: 基线图上的分位锚点，用于把目标图的原始残差标定成跨批次可比的 G。
        self.calibration_anchors_: Optional[Dict[str, float]] = None
        self.calibration_percentiles: Tuple[float, float] = (50.0, 99.0)

    # -- 训练 -----------------------------------------------------------

    def fit(self, adjacency: np.ndarray) -> "GraphAutoencoderDetector":
        """在参考图（正常基线）上训练。

        只用到邻接矩阵的结构信息，因此参考图的节点数、评分的图节点数互不影响。
        """
        A = np.asarray(adjacency, dtype=np.float64)
        n = A.shape[0]
        if n < 3:
            raise ValueError("节点太少，无法训练 GAE")

        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        binary = (A > 0).astype(np.float64)
        np.fill_diagonal(binary, 0.0)

        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if binary[i, j] > 0]
        rng.shuffle(edges)
        mask_count = max(1, int(len(edges) * self.mask_fraction)) if edges else 0
        self.masked_edges_ = edges[:mask_count]

        self.negative_edges_ = self._sample_negatives(binary, len(self.masked_edges_), rng)

        corrupted = binary.copy()
        for i, j in self.masked_edges_:
            corrupted[i, j] = corrupted[j, i] = 0.0

        features = torch.tensor(node_features(A), dtype=torch.float32, device=self.device)
        input_norm = torch.tensor(normalize_adjacency(corrupted), dtype=torch.float32, device=self.device)
        target = torch.tensor(binary, dtype=torch.float32, device=self.device)

        loss_mask = torch.zeros_like(target)
        for i, j in self.masked_edges_:
            loss_mask[i, j] = loss_mask[j, i] = 1.0
        for i, j in self.negative_edges_:
            loss_mask[i, j] = loss_mask[j, i] = 1.0
        if float(loss_mask.sum()) == 0:
            # 没有边可遮时退化为全图重构
            loss_mask = torch.ones_like(target) - torch.eye(n, device=self.device)

        self.model = GraphAutoencoder(
            in_dim=features.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)
        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCELoss(reduction="none")

        self.training_history = []
        for epoch in range(self.epochs):
            self.model.train()
            optimiser.zero_grad()
            reconstruction, _ = self.model(features, input_norm)
            loss_matrix = loss_fn(reconstruction, target) * loss_mask
            loss = loss_matrix.sum() / loss_mask.sum()
            loss.backward()
            optimiser.step()
            self.training_history.append({"epoch": epoch + 1, "loss": round(float(loss.item()), 6)})

        # 在训练图（= 正常基线）上取分位锚点：目标图上超过 P99 的残差才记 1.0。
        # 关键点是锚点来自基线而不是目标批次，否则"本批最异常者恒得满分"。
        baseline_error = self.node_error(A)
        self.calibration_anchors_ = quantile_anchors(
            baseline_error,
            self.calibration_percentiles[0],
            self.calibration_percentiles[1],
        )
        return self

    def _sample_negatives(self, binary: np.ndarray, count: int, rng: np.random.RandomState):
        """抽取与正样本等量（可调比例）的非边作为负样本。"""
        n = binary.shape[0]
        wanted = int(round(count * max(self.negative_ratio, 0.0)))
        if wanted <= 0:
            return []
        existing = {(i, j) for i, j in zip(*np.nonzero(np.triu(binary, 1)))}
        total_possible = n * (n - 1) // 2
        if len(existing) >= total_possible:
            return []
        sampled: List[Tuple[int, int]] = []
        attempts = 0
        max_attempts = wanted * 50 + 100
        while len(sampled) < wanted and attempts < max_attempts:
            attempts += 1
            i = int(rng.randint(0, n))
            j = int(rng.randint(0, n))
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in existing or pair in sampled:
                continue
            sampled.append(pair)
        return sampled

    # -- 推理 -----------------------------------------------------------

    def _reconstruction(self, adjacency: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GAE 尚未训练")
        A = np.asarray(adjacency, dtype=np.float64)
        binary = (A > 0).astype(np.float64)
        np.fill_diagonal(binary, 0.0)

        features = torch.tensor(node_features(A), dtype=torch.float32, device=self.device)
        input_norm = torch.tensor(
            normalize_adjacency(binary), dtype=torch.float32, device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            reconstruction, _ = self.model(features, input_norm)
        return reconstruction.cpu().numpy().astype(np.float64)

    def node_error(self, adjacency: np.ndarray, reconstruction: Optional[np.ndarray] = None) -> np.ndarray:
        """节点级重构偏差：取"漏判真实边"与"幻觉非边"中较大的一侧。

        两侧都取均值（而不是求和），避免退化成"度检测器"；实测与度的相关系数在
        两次时间切分下为 −0.33 与 +0.07，既没有稳定偏置，也不等于与度无关。

        为什么取 ``max`` 而不是两侧平均：实测在本项目 66 分钟数据上（基线图训练、
        目标图评分、注入 25 个异常节点），两侧平均会把强信号稀释掉——hub 型注入
        的 AUC 只有 0.75，改为取 ``max`` 后同一实验 AUC 0.98。
        代价是分数对单侧强偏离更敏感，需要在应用层用分位数阈值控制误报。
        """
        A = np.asarray(adjacency, dtype=np.float64)
        binary = (A > 0).astype(np.float64)
        np.fill_diagonal(binary, 0.0)
        n = A.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        if reconstruction is None:
            reconstruction = self._reconstruction(A)
        np.fill_diagonal(reconstruction, 0.0)

        off_diagonal = ~np.eye(n, dtype=bool)
        absent = np.where(binary > 0, 0.0, reconstruction)
        present = np.where(binary > 0, 1.0 - reconstruction, 0.0)

        present_count = binary.sum(axis=1)
        absent_count = off_diagonal.sum(axis=1) - present_count

        present_mean = np.divide(
            present.sum(axis=1),
            present_count,
            out=np.full(n, np.nan, dtype=np.float64),
            where=present_count > 0,
        )
        absent_mean = np.divide(
            absent.sum(axis=1),
            absent_count,
            out=np.full(n, np.nan, dtype=np.float64),
            where=absent_count > 0,
        )

        error = np.where(
            np.isnan(present_mean),
            absent_mean,
            np.where(np.isnan(absent_mean), present_mean, np.maximum(present_mean, absent_mean)),
        )
        return np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)

    def reconstruction_error(self, adjacency: np.ndarray) -> np.ndarray:
        """节点级重构误差（``node_error`` 的别名，保持既有调用方不变）。"""
        return self.node_error(adjacency)

    def score(self, adjacency: np.ndarray) -> Dict[str, np.ndarray]:
        """对目标图打分。

        G 用 **fit 阶段在基线图上记录的分位锚点** 标定，而不是对本图做 min-max。
        差别在于：min-max 会让每张图必然产出一个 1.0，于是"本批最异常者恒得满分"，
        与这批数据是否真有异常无关；锚点标定下，只有残差超过基线 P99 才会到 1.0，
        本批全部正常时 G 会整体接近 0——这才是可解释的。

        锚点缺失（例如旧模型）时回退到本图自标定，并在 ``calibration["source"]``
        标注来源，便于排查口径。
        """
        A = np.asarray(adjacency, dtype=np.float64).copy()
        np.fill_diagonal(A, 0.0)
        reconstruction = self._reconstruction(A)
        error = self.node_error(A, reconstruction=reconstruction)
        binary = (A > 0).astype(np.float64)

        if self.calibration_anchors_:
            calibrated = anchor_normalize(error, self.calibration_anchors_)
            calibration = {**self.calibration_anchors_, "source": "baseline"}
        else:
            from .calibration import fit_and_calibrate

            calibrated, anchors = fit_and_calibrate(error)
            calibration = {**anchors, "source": "self"}

        return {
            "graph_anomaly_index": calibrated,
            "reconstruction_error": error,
            "degree": A.sum(axis=1),
            "clustering": _clustering(binary),
            "calibration": calibration,
        }

    # -- 持久化 ---------------------------------------------------------

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("GAE 尚未训练")
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "kind": "phase3_graph_autoencoder",
                    "encoder": "gcn_features",
                    "in_dim": int(self.model.in_dim),
                    "node_feature_names": list(NODE_FEATURE_NAMES),
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "dropout": self.dropout,
                    "mask_fraction": self.mask_fraction,
                    "negative_ratio": self.negative_ratio,
                    "calibration_anchors": self.calibration_anchors_,
                    "calibration_percentiles": list(self.calibration_percentiles),
                    "masked_edges": [list(edge) for edge in self.masked_edges_],
                    "negative_edge_count": len(self.negative_edges_),
                    "training_history": self.training_history,
                },
                f,
                ensure_ascii=False,
            )
        torch.save(self.model.state_dict(), path + ".pt")

    @classmethod
    def load(cls, path: str, num_nodes: Optional[int] = None) -> "GraphAutoencoderDetector":
        """加载已训练的检测器。

        ``num_nodes`` 仅为兼容旧调用保留：新编码器的参数与节点数无关，
        训练图与评分的图节点数可以不同。
        """
        with open(path + ".json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        detector = cls(
            hidden_dim=payload.get("hidden_dim", 32),
            latent_dim=payload.get("latent_dim", 16),
            epochs=payload.get("epochs", 150),
            learning_rate=payload.get("learning_rate", 0.02),
            weight_decay=payload.get("weight_decay", 1e-3),
            dropout=payload.get("dropout", 0.2),
            mask_fraction=payload.get("mask_fraction", 0.2),
            negative_ratio=payload.get("negative_ratio", 1.0),
        )
        detector.masked_edges_ = [tuple(edge) for edge in payload.get("masked_edges", [])]
        detector.training_history = payload.get("training_history", [])
        detector.calibration_anchors_ = payload.get("calibration_anchors")
        percentiles = payload.get("calibration_percentiles")
        if isinstance(percentiles, (list, tuple)) and len(percentiles) == 2:
            detector.calibration_percentiles = (float(percentiles[0]), float(percentiles[1]))
        in_dim = payload.get("in_dim") or len(NODE_FEATURE_NAMES)
        detector.model = GraphAutoencoder(
            in_dim=int(in_dim),
            hidden_dim=detector.hidden_dim,
            latent_dim=detector.latent_dim,
            dropout=detector.dropout,
        )
        detector.model.load_state_dict(torch.load(path + ".pt", map_location="cpu", weights_only=True))
        detector.model.eval()
        return detector
