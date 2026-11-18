"""自包含的 Isolation Forest 实现（研究要点二：无监督个体异常检测）。

本项目不引入 scikit-learn 依赖，Isolation Forest 采用纯 NumPy 实现，
与 ByteTrack 的“自包含”风格一致，便于在边缘节点部署。

算法参考：
  Liu, Ting, Zhou. "Isolation Forest." ICDM 2008.

约定：
  - ``score_samples`` 返回 [0, 1] 的异常指数，**越大越异常**，
    直接对应研究要点二中的“行为异常指数 A”。
  - ``predict`` 采用 sklearn 约定：-1 为异常，1 为正常。
  - 阈值由训练样本的 ``1 - contamination`` 分位数确定。
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import List, Optional, Sequence

import numpy as np

from .baseline import DEFAULT_FEATURES, BaselineBuilder, FeatureSpec

_EULER = 0.5772156649015329


def _harmonic(n: int) -> float:
    """调和数 H(n) 的近似（n 较大时用 ln(n)+gamma）。"""
    if n <= 0:
        return 0.0
    return math.log(n) + _EULER


def _c(n: int) -> float:
    """二叉搜索树中一次不成功查找的平均路径长度 c(n)。

    c(n) = 2*H(n-1) - 2*(n-1)/n，用于把树深度归一化到 [0,1] 的异常分数。
    """
    if n <= 1:
        return 0.0
    return 2.0 * _harmonic(n - 1) - 2.0 * (n - 1) / n


# ---------------------------------------------------------------------------
# 单棵隔离树
# ---------------------------------------------------------------------------


def _build_tree(
    X: np.ndarray,
    depth: int,
    height_limit: int,
    rng: np.random.RandomState,
) -> dict:
    """递归构建一棵隔离树，返回可 JSON 序列化的节点字典。"""
    n = int(X.shape[0])
    if depth >= height_limit or n <= 1:
        return {"leaf": True, "size": n}

    ranges = X.max(axis=0) - X.min(axis=0)
    valid = np.flatnonzero(ranges > 0)
    if valid.size == 0:
        return {"leaf": True, "size": n}

    q = int(rng.choice(valid))
    lo = float(X[:, q].min())
    hi = float(X[:, q].max())
    p = float(rng.uniform(lo, hi))

    left_mask = X[:, q] < p
    n_left = int(left_mask.sum())
    if n_left == 0 or n_left == n:
        return {"leaf": True, "size": n}

    return {
        "leaf": False,
        "feature": q,
        "threshold": p,
        "size": n,
        "left": _build_tree(X[left_mask], depth + 1, height_limit, rng),
        "right": _build_tree(X[~left_mask], depth + 1, height_limit, rng),
    }


def _path_length(node: dict, x: np.ndarray) -> float:
    """点 x 落入叶节点后的路径长度（深度 + c(叶节点样本数)）。"""
    depth = 0
    while not node["leaf"]:
        if x[node["feature"]] < node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
        depth += 1
    return depth + _c(int(node["size"]))


# ---------------------------------------------------------------------------
# Isolation Forest 模型
# ---------------------------------------------------------------------------


class IsolationForest:
    """轻量 Isolation Forest 实现。"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        random_state: Optional[int] = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.contamination = float(contamination)
        self.random_state = random_state

        self.trees_: List[dict] = []
        self.n_features_: int = 0
        self.max_samples_: int = 0
        self.height_limit_: int = 0
        self.threshold_: float = 0.5

    # -- training ---------------------------------------------------------

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        n, d = X.shape
        if n < 2:
            raise ValueError("至少需要 2 条样本才能训练 Isolation Forest")

        self.n_features_ = int(d)
        self.max_samples_ = min(self.max_samples, n)
        self.height_limit_ = int(math.ceil(math.log2(max(self.max_samples_, 2))))

        rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=self.max_samples_, replace=False)
            self.trees_.append(_build_tree(X[idx], 0, self.height_limit_, rng))

        scores = self.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    # -- inference --------------------------------------------------------

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的异常指数（0~1，越大越异常）。"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("特征维度不匹配")
        if not self.trees_:
            raise RuntimeError("模型尚未训练")

        norm = _c(self.max_samples_)
        if norm <= 0:
            norm = 1.0

        n_estimators = len(self.trees_)
        scores = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            x = X[i]
            total = 0.0
            for tree in self.trees_:
                total += _path_length(tree, x)
            avg = total / n_estimators
            scores[i] = 2.0 ** (-avg / norm)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """sklearn 约定：-1 异常，1 正常。"""
        scores = self.score_samples(X)
        # 用严格大于，避免全部分数相同时把所有样本都判成异常
        return np.where(scores > self.threshold_, -1, 1)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_features": self.n_features_,
            "max_samples_": self.max_samples_,
            "height_limit": self.height_limit_,
            "threshold": self.threshold_,
            "trees": self.trees_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "IsolationForest":
        model = cls(
            n_estimators=payload.get("n_estimators", 100),
            max_samples=payload.get("max_samples", 256),
            contamination=payload.get("contamination", 0.1),
            random_state=payload.get("random_state", 42),
        )
        model.n_features_ = int(payload.get("n_features", 0))
        model.max_samples_ = int(payload.get("max_samples_", 0))
        model.height_limit_ = int(payload.get("height_limit", 0))
        model.threshold_ = float(payload.get("threshold", 0.5))
        model.trees_ = payload.get("trees", [])
        return model

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "IsolationForest":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# 行为向量适配器
# ---------------------------------------------------------------------------


def _specs_from_names(names: Sequence[str]) -> List[FeatureSpec]:
    """按名称恢复 FeatureSpec（优先复用 DEFAULT_FEATURES 里的可靠性定义）。"""
    by_name = {spec.name: spec for spec in DEFAULT_FEATURES}
    specs = []
    for name in names:
        specs.append(by_name.get(name) or FeatureSpec(name))
    return specs


class IsolationForestDetector:
    """把行为向量字典适配成 Isolation Forest 输入，并输出异常指数 A。

    与 ``BaselineBuilder`` 共享可靠性字段规则：
    ``speed_reliable`` / ``path_reliable`` / ``aggregation_reliable`` 为 0 时，
    对应特征置为缺失（NaN），训练与推理时统一用中位数/训练中位数填补。
    """

    def __init__(
        self,
        features: Optional[List[FeatureSpec]] = None,
        target_groups: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        random_state: Optional[int] = 42,
    ):
        self.features = features or DEFAULT_FEATURES
        self.builder = BaselineBuilder(features=self.features, target_groups=target_groups)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.feature_medians_: Optional[np.ndarray] = None

    # -- data preparation -------------------------------------------------

    def _extract_target_matrix(self, vectors: Sequence[dict]):
        """返回 (参与分析的行索引, 特征矩阵)。非分析对象（货车等）被跳过。"""
        indices: List[int] = []
        rows: List[List[float]] = []
        for index, vector in enumerate(vectors):
            if not self.builder.is_analysis_target(vector):
                continue
            row: List[float] = []
            for spec in self.features:
                if not self.builder._is_reliable(vector, spec):
                    row.append(np.nan)
                    continue
                value = self.builder._to_float(vector.get(spec.name))
                if value is None or not np.isfinite(value):
                    row.append(np.nan)
                else:
                    row.append(float(value))
            indices.append(index)
            rows.append(row)

        if not rows:
            return indices, np.empty((0, len(self.features)), dtype=np.float64)
        return indices, np.asarray(rows, dtype=np.float64)

    # -- training / inference --------------------------------------------

    def fit(self, vectors: Sequence[dict]) -> "IsolationForestDetector":
        _, matrix = self._extract_target_matrix(vectors)
        if matrix.shape[0] < 3:
            raise ValueError("需要至少 3 条可用的分析对象样本才能训练 Isolation Forest")

        # 未标定摄像头会让整列速度/路径特征为 NaN，这里静默处理全 NaN 列
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        filled = np.where(np.isnan(matrix), medians, matrix)

        self.feature_medians_ = np.asarray(medians, dtype=np.float64)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
        ).fit(filled)
        return self

    def score(self, vectors: Sequence[dict]) -> List[dict]:
        """为每条记录补充 ``iforest_score``（异常指数 A）和 ``iforest_anomaly``。"""
        if self.model is None or self.feature_medians_ is None:
            raise RuntimeError("IsolationForestDetector 尚未训练")

        results = [dict(vector) for vector in vectors]
        for record in results:
            record.setdefault("iforest_score", 0.0)
            record.setdefault("iforest_anomaly", 0)

        indices, matrix = self._extract_target_matrix(vectors)
        if not indices:
            return results

        filled = np.where(np.isnan(matrix), self.feature_medians_, matrix)
        scores = self.model.score_samples(filled)
        labels = self.model.predict(filled)

        for local, original in enumerate(indices):
            results[original]["iforest_score"] = round(float(scores[local]), 4)
            results[original]["iforest_available"] = 1
            results[original]["iforest_anomaly"] = int(labels[local] == -1)

        return results

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        if self.model is None or self.feature_medians_ is None:
            raise RuntimeError("IsolationForestDetector 尚未训练")
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = {
            "kind": "phase3_isolation_forest",
            "features": [spec.name for spec in self.features],
            "target_groups": sorted(self.builder.target_groups),
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "feature_medians": self.feature_medians_.tolist(),
            "model": self.model.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        detector = cls(
            features=_specs_from_names(payload.get("features", [])),
            target_groups=payload.get("target_groups"),
            n_estimators=payload.get("n_estimators", 100),
            max_samples=payload.get("max_samples", 256),
            contamination=payload.get("contamination", 0.1),
            random_state=payload.get("random_state", 42),
        )
        detector.feature_medians_ = np.asarray(payload["feature_medians"], dtype=np.float64)
        detector.model = IsolationForest.from_dict(payload["model"])
        return detector
