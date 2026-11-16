"""图自编码器（GAE）与图异常指数 G（研究要点三）。

实现要点：
- 两层 GCN 编码器 + 内积解码器；
- **边掩码（去噪）训练**：随机遮住一部分边，只用剩余结构去重建被遮住的边，
  避免“训练图=评分图”时模型直接记住邻接矩阵、重构误差退化为 0；
- 节点级重构误差 = 该节点被遮住的边被判为“存在”的偏差，越大越异常；
- 输出 [0,1] 的图异常指数 G。

说明：单张图上的 GAE 仍然是自监督的；更严谨的做法是“正常基线图训练、目标图评分”，
后续接入多天数据后可切换。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .normalize import minmax_normalize

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


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


def _clustering(adjacency_binary: np.ndarray) -> np.ndarray:
    n = adjacency_binary.shape[0]
    coefficients = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neighbors = np.nonzero(adjacency_binary[i])[0]
        degree = int(neighbors.size)
        if degree < 2:
            continue
        subgraph = adjacency_binary[np.ix_(neighbors, neighbors)]
        links = float(subgraph.sum()) / 2.0
        coefficients[i] = links / (degree * (degree - 1) / 2.0)
    return coefficients


class GraphAutoencoder(nn.Module):
    """两层 GCN 编码器 + 内积解码器。"""

    def __init__(self, num_nodes: int, hidden_dim: int = 32, latent_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.encoder1 = nn.Linear(num_nodes, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, adjacency_norm: "torch.Tensor") -> "torch.Tensor":
        hidden = torch.relu(self.encoder1(adjacency_norm))
        hidden = self.dropout(hidden)
        return self.encoder2(hidden)

    def decode(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(embeddings @ embeddings.t())

    def forward(self, adjacency_norm: "torch.Tensor"):
        embeddings = self.encode(adjacency_norm)
        return self.decode(embeddings), embeddings


class GraphAutoencoderDetector:
    """边掩码训练的 GAE，用被遮住边的重构误差输出图异常指数 G。"""

    def __init__(
        self,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        epochs: int = 150,
        learning_rate: float = 0.02,
        weight_decay: float = 1e-3,
        dropout: float = 0.2,
        mask_fraction: float = 0.2,
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
        self.device = device
        self.random_state = random_state

        self.model: Optional[GraphAutoencoder] = None
        self.masked_edges_: List[Tuple[int, int]] = []
        self.training_history = []

    # -- 训练 -----------------------------------------------------------

    def fit(self, adjacency: np.ndarray) -> "GraphAutoencoderDetector":
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

        corrupted = binary.copy()
        for i, j in self.masked_edges_:
            corrupted[i, j] = corrupted[j, i] = 0.0

        input_norm = torch.tensor(normalize_adjacency(corrupted), dtype=torch.float32, device=self.device)
        target = torch.tensor(binary, dtype=torch.float32, device=self.device)

        loss_mask = torch.zeros_like(target)
        for i, j in self.masked_edges_:
            loss_mask[i, j] = loss_mask[j, i] = 1.0
        if float(loss_mask.sum()) == 0:
            # 没有边可遮时退化为全图重构
            loss_mask = torch.ones_like(target) - torch.eye(n, device=self.device)

        self.model = GraphAutoencoder(
            num_nodes=n,
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
            reconstruction, _ = self.model(input_norm)
            loss_matrix = loss_fn(reconstruction, target) * loss_mask
            loss = loss_matrix.sum() / loss_mask.sum()
            loss.backward()
            optimiser.step()
            self.training_history.append({"epoch": epoch + 1, "loss": round(float(loss.item()), 6)})
        return self

    # -- 推理 -----------------------------------------------------------

    def _reconstruction(self, adjacency: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GAE 尚未训练")
        A = np.asarray(adjacency, dtype=np.float64)
        binary = (A > 0).astype(np.float64)
        np.fill_diagonal(binary, 0.0)
        corrupted = binary.copy()
        for i, j in self.masked_edges_:
            corrupted[i, j] = corrupted[j, i] = 0.0

        input_norm = torch.tensor(normalize_adjacency(corrupted), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            reconstruction, _ = self.model(input_norm)
        return reconstruction.cpu().numpy().astype(np.float64)

    def reconstruction_error(self, adjacency: np.ndarray) -> np.ndarray:
        """节点级重构误差：被遮住边被判为“不存在”的偏差。"""
        A = np.asarray(adjacency, dtype=np.float64)
        binary = (A > 0).astype(np.float64)
        np.fill_diagonal(binary, 0.0)
        n = A.shape[0]
        reconstruction = self._reconstruction(A)

        per_node = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for i, j in self.masked_edges_:
            per_node[i] += abs(reconstruction[i, j] - 1.0)
            per_node[j] += abs(reconstruction[j, i] - 1.0)
            counts[i] += 1
            counts[j] += 1

        valid = counts > 0
        if valid.any():
            per_node[valid] /= counts[valid]

        # 没有被遮住边的节点，用全图重构误差兜底
        if (~valid).any():
            fallback = np.abs(reconstruction - binary).sum(axis=1) / max(n - 1, 1)
            per_node[~valid] = fallback[~valid]
        return per_node

    def score(self, adjacency: np.ndarray) -> Dict[str, np.ndarray]:
        A = np.asarray(adjacency, dtype=np.float64).copy()
        np.fill_diagonal(A, 0.0)
        error = self.reconstruction_error(A)
        binary = (A > 0).astype(np.float64)
        return {
            "graph_anomaly_index": minmax_normalize(error),
            "reconstruction_error": error,
            "degree": A.sum(axis=1),
            "clustering": _clustering(binary),
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
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "dropout": self.dropout,
                    "mask_fraction": self.mask_fraction,
                    "masked_edges": [list(edge) for edge in self.masked_edges_],
                    "training_history": self.training_history,
                },
                f,
                ensure_ascii=False,
            )
        torch.save(self.model.state_dict(), path + ".pt")

    @classmethod
    def load(cls, path: str, num_nodes: int) -> "GraphAutoencoderDetector":
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
        )
        detector.masked_edges_ = [tuple(edge) for edge in payload.get("masked_edges", [])]
        detector.training_history = payload.get("training_history", [])
        detector.model = GraphAutoencoder(
            num_nodes=num_nodes,
            hidden_dim=detector.hidden_dim,
            latent_dim=detector.latent_dim,
            dropout=detector.dropout,
        )
        detector.model.load_state_dict(torch.load(path + ".pt", map_location="cpu", weights_only=True))
        detector.model.eval()
        return detector
