import numpy as np
from typing import Optional, Dict, List


class FeatureNormalizer:
    """特征标准化器。

    将行为向量各维度标准化到统一尺度，消除量纲差异，
    为后续异常检测（距离/密度类算法）做准备。
    """

    def __init__(self, method: str = "zscore"):
        self.method = method
        self._fitted = False
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def fit(self, vectors: List[np.ndarray]):
        """用一批向量拟合标准化参数。"""
        if not vectors:
            return
        stacked = np.stack(vectors, axis=0)  # (N, D)

        if self.method == "zscore":
            self._mean = np.mean(stacked, axis=0)
            self._std = np.std(stacked, axis=0)
            self._std[self._std == 0] = 1.0  # 避免除零
        elif self.method == "minmax":
            self._min = np.min(stacked, axis=0)
            self._max = np.max(stacked, axis=0)
            range_val = self._max - self._min
            range_val[range_val == 0] = 1.0
            self._max = self._min + range_val

        self._fitted = True

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """标准化单个向量。"""
        if not self._fitted:
            return vector

        if self.method == "zscore":
            return (vector - self._mean) / self._std
        elif self.method == "minmax":
            return (vector - self._min) / (self._max - self._min)
        return vector

    def fit_transform(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """拟合并标准化所有向量。"""
        self.fit(vectors)
        return [self.transform(v) for v in vectors]

    def save(self, filepath: str):
        """保存标准化参数到文件。"""
        data = {}
        if self._mean is not None:
            data["mean"] = self._mean.tolist()
        if self._std is not None:
            data["std"] = self._std.tolist()
        if self._min is not None:
            data["min"] = self._min.tolist()
        if self._max is not None:
            data["max"] = self._max.tolist()
        data["method"] = self.method

        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """从文件加载标准化参数。"""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.method = data.get("method", "zscore")
        if "mean" in data:
            self._mean = np.array(data["mean"])
        if "std" in data:
            self._std = np.array(data["std"])
        if "min" in data:
            self._min = np.array(data["min"])
        if "max" in data:
            self._max = np.array(data["max"])
        self._fitted = True
