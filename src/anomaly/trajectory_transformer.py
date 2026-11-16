"""Transformer 轨迹预测与预测残差判异（研究要点二·技术路径一）。

思路：
- 用正常轨迹训练一个 Transformer，输入过去 K 个轨迹点，预测未来 M 个点；
- 推理时计算预测点与真实点的位移残差，残差越大越异常；
- 残差归一化后作为“轨迹预测异常指数”，与 Isolation Forest 的行为异常指数 A 并列。

说明：当前仓库真实轨迹样本较少（单摄像头单时段百余条），本模块重点是把
“预测—残差—判异”的链路跑通；要达成“未来 3 秒平均位移误差 ≤ 2 米”的指标，
需要多天、多摄像头的大规模轨迹数据。
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - torch 是项目核心依赖，缺失时给出明确提示
    torch = None
    nn = None


DEFAULT_PAST_LEN = 8
DEFAULT_FUTURE_LEN = 8


def parse_trajectory_points(points) -> np.ndarray:
    """把 ``"x1,y1;x2,y2;..."`` 解析为 (N, 2) 数组。"""
    if points is None:
        return np.empty((0, 2), dtype=np.float64)
    if isinstance(points, np.ndarray):
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)

    coords = []
    for token in str(points).split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 2:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return np.asarray(coords, dtype=np.float64) if coords else np.empty((0, 2), dtype=np.float64)


def make_windows(
    points: np.ndarray,
    past_len: int = DEFAULT_PAST_LEN,
    future_len: int = DEFAULT_FUTURE_LEN,
    stride: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """把一条轨迹切成 (过去, 未来) 窗口。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    total = past_len + future_len
    if points.shape[0] < total:
        return []
    windows = []
    for start in range(0, points.shape[0] - total + 1, max(1, stride)):
        windows.append((points[start:start + past_len], points[start + past_len:start + total]))
    return windows


def normalize_window(past: np.ndarray, future: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """以过去最后一个点为原点、平均步长为尺度做归一化。"""
    past = np.asarray(past, dtype=np.float64)
    future = np.asarray(future, dtype=np.float64)
    last = past[-1]
    steps = np.linalg.norm(np.diff(past, axis=0), axis=1) if past.shape[0] > 1 else np.array([1.0])
    scale = float(np.mean(steps)) if steps.size else 1.0
    if scale <= 1e-6:
        scale = 1.0
    return (past - last) / scale, (future - last) / scale, scale


class TrajectoryTransformer(nn.Module):
    """小型 Transformer 编码器 + 全序列回归头。"""

    def __init__(
        self,
        past_len: int = DEFAULT_PAST_LEN,
        future_len: int = DEFAULT_FUTURE_LEN,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.input_proj = nn.Linear(2, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, past_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, future_len * 2)

    def forward(self, past: "torch.Tensor") -> "torch.Tensor":
        # past: (B, past_len, 2) -> (B, future_len, 2)
        x = self.input_proj(past) + self.position_embedding[:, : past.shape[1], :]
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        output = self.head(pooled)
        return output.view(-1, self.future_len, 2)


class TrajectoryPredictionDetector:
    """训练 Transformer，并用预测残差给出轨迹异常指数。"""

    def __init__(
        self,
        past_len: int = DEFAULT_PAST_LEN,
        future_len: int = DEFAULT_FUTURE_LEN,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        epochs: int = 40,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        contamination: float = 0.1,
        device: str = "cpu",
        random_state: int = 42,
    ):
        if torch is None:
            raise RuntimeError("TrajectoryPredictionDetector 需要 PyTorch")
        self.past_len = past_len
        self.future_len = future_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.contamination = contamination
        self.device = device
        self.random_state = random_state

        self.model: Optional[TrajectoryTransformer] = None
        self.residual_mean: float = 0.0
        self.residual_std: float = 1.0
        # 残差阈值：训练残差 z 分数的 (1 - contamination) 分位数
        self.residual_threshold_: float = 3.0
        self.training_history: List[dict] = []

    # -- 数据准备 -------------------------------------------------------

    def _collect_windows(self, vectors: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
        past_list, future_list = [], []
        for vector in vectors:
            # 只使用行政通道普通车辆，且要求轨迹足够长
            target = str(vector.get("analysis_target", "")).strip().lower()
            if target in {"0", "false", "no"}:
                continue
            points = parse_trajectory_points(vector.get("trajectory_points", ""))
            for past, future in make_windows(points, self.past_len, self.future_len):
                past_n, future_n, _ = normalize_window(past, future)
                past_list.append(past_n)
                future_list.append(future_n)
        if not past_list:
            return np.empty((0, self.past_len, 2)), np.empty((0, self.future_len, 2))
        return np.asarray(past_list, dtype=np.float32), np.asarray(future_list, dtype=np.float32)

    # -- 训练 -----------------------------------------------------------

    def fit(
        self,
        vectors: Sequence[dict],
        validation_split: float = 0.2,
        reuse_existing: bool = False,
    ) -> "TrajectoryPredictionDetector":
        past, future = self._collect_windows(vectors)
        if past.shape[0] < 4:
            raise ValueError("可用的轨迹窗口太少，无法训练 Transformer（至少需要 4 个窗口）")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        # CUDA 上的卷积/规约默认可能是非确定性的，会让同一份数据两次训练的残差末位不同。
        # warn_only=True：个别算子不支持确定性实现时只告警，不中断训练。
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        indices = np.arange(past.shape[0])
        np.random.shuffle(indices)
        split = max(1, int(len(indices) * (1 - validation_split)))
        train_idx, val_idx = indices[:split], indices[split:]

        train_ds = TensorDataset(
            torch.from_numpy(past[train_idx]), torch.from_numpy(future[train_idx])
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if not (reuse_existing and self.model is not None):
            self.model = TrajectoryTransformer(
                past_len=self.past_len,
                future_len=self.future_len,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
            ).to(self.device)
        else:
            # 增量更新：基于已有权重继续微调
            self.model = self.model.to(self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.L1Loss()
        self.training_history = []

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch_past, batch_future in train_loader:
                batch_past = batch_past.to(self.device)
                batch_future = batch_future.to(self.device)
                optimiser.zero_grad()
                prediction = self.model(batch_past)
                loss = loss_fn(prediction, batch_future)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
                total_loss += float(loss.item()) * batch_past.shape[0]

            train_loss = total_loss / max(len(train_idx), 1)
            val_loss = self._evaluate(past[val_idx], future[val_idx]) if len(val_idx) else float("nan")
            self.training_history.append(
                {"epoch": epoch + 1, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6)}
            )

        # 用训练集残差分布做归一化
        residuals = self._raw_residuals(past[val_idx] if len(val_idx) else past, future[val_idx] if len(val_idx) else future)
        self.residual_mean = float(np.mean(residuals)) if residuals.size else 0.0
        self.residual_std = float(np.std(residuals)) if residuals.size else 1.0
        if self.residual_std <= 1e-9:
            self.residual_std = 1.0

        # 推理时取“轨迹内最大残差”，阈值也必须按轨迹级 z 分数标定
        scored = self.score(vectors)
        trajectory_zscores = []
        for record in scored:
            try:
                residual_value = float(record.get("traj_residual", 0) or 0)
            except (TypeError, ValueError):
                residual_value = 0.0
            # 输入向量可能已带有上一次运行的 traj_residual（重训场景），
            # 而这类记录本次没有产生窗口，也就没有 traj_residual_zscore。
            zscore = record.get("traj_residual_zscore")
            if residual_value > 0 and zscore is not None:
                trajectory_zscores.append(zscore)
        if trajectory_zscores:
            self.residual_threshold_ = float(
                np.quantile(trajectory_zscores, 1.0 - self.contamination)
            )
        else:
            self.residual_threshold_ = 3.0
        return self

    def _evaluate(self, past: np.ndarray, future: np.ndarray) -> float:
        if self.model is None or past.shape[0] == 0:
            return float("nan")
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(past).to(self.device)).cpu().numpy()
        return float(np.mean(np.linalg.norm(prediction - future, axis=2)))

    def _raw_residuals(self, past: np.ndarray, future: np.ndarray) -> np.ndarray:
        """归一化尺度下的平均位移残差（每个窗口一个值）。"""
        if self.model is None or past.shape[0] == 0:
            return np.empty(0)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(past).to(self.device)).cpu().numpy()
        return np.mean(np.linalg.norm(prediction - future, axis=2), axis=1)

    # -- 推理 -----------------------------------------------------------

    def score(self, vectors: Sequence[dict]) -> List[dict]:
        """为每条记录补充 ``traj_residual`` 和 ``traj_anomaly``。"""
        if self.model is None:
            raise RuntimeError("TrajectoryPredictionDetector 尚未训练")

        results = [dict(vector) for vector in vectors]
        for record in results:
            record.setdefault("traj_residual", 0.0)
            record.setdefault("traj_anomaly", 0)

        for index, vector in enumerate(vectors):
            target = str(vector.get("analysis_target", "")).strip().lower()
            if target in {"0", "false", "no"}:
                continue
            points = parse_trajectory_points(vector.get("trajectory_points", ""))
            windows = make_windows(points, self.past_len, self.future_len)
            if not windows:
                continue

            past_list, future_list = [], []
            scales = []
            for past, future in windows:
                past_n, future_n, scale = normalize_window(past, future)
                past_list.append(past_n)
                future_list.append(future_n)
                scales.append(scale)

            past_arr = np.asarray(past_list, dtype=np.float32)
            future_arr = np.asarray(future_list, dtype=np.float32)
            residuals = self._raw_residuals(past_arr, future_arr)
            # 取残差最大的窗口作为该轨迹的异常证据
            residual = float(np.max(residuals)) * float(np.mean(scales))  # 还原到像素尺度
            zscore = (float(np.max(residuals)) - self.residual_mean) / self.residual_std

            results[index]["traj_residual"] = round(residual, 2)
            results[index]["traj_residual_zscore"] = round(float(zscore), 3)
            results[index]["traj_anomaly"] = int(zscore > self.residual_threshold_)

        return results

    # -- 持久化 ---------------------------------------------------------

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = {
            "kind": "phase3_trajectory_transformer",
            "past_len": self.past_len,
            "future_len": self.future_len,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "contamination": self.contamination,
            "residual_threshold": self.residual_threshold_,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "training_history": self.training_history,
        }
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        torch.save(self.model.state_dict(), path + ".pt")

    @classmethod
    def load(cls, path: str) -> "TrajectoryPredictionDetector":
        with open(path + ".json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        detector = cls(
            past_len=payload.get("past_len", DEFAULT_PAST_LEN),
            future_len=payload.get("future_len", DEFAULT_FUTURE_LEN),
            d_model=payload.get("d_model", 64),
            nhead=payload.get("nhead", 4),
            num_layers=payload.get("num_layers", 2),
            dim_feedforward=payload.get("dim_feedforward", 128),
            contamination=payload.get("contamination", 0.1),
        )
        detector.residual_mean = float(payload.get("residual_mean", 0.0))
        detector.residual_std = float(payload.get("residual_std", 1.0))
        detector.residual_threshold_ = float(payload.get("residual_threshold", 3.0))
        detector.training_history = payload.get("training_history", [])
        detector.model = TrajectoryTransformer(
            past_len=detector.past_len,
            future_len=detector.future_len,
            d_model=detector.d_model,
            nhead=detector.nhead,
            num_layers=detector.num_layers,
            dim_feedforward=detector.dim_feedforward,
        )
        detector.model.load_state_dict(torch.load(path + ".pt", map_location="cpu", weights_only=True))
        detector.model.eval()
        return detector
