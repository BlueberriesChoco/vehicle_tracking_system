"""异常检测的常用评估指标（无需 scikit-learn）。"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _as_arrays(y_true: Sequence, scores: Sequence):
    y = np.asarray([1 if str(v).strip() in {"1", "True", "true", "yes"} else 0 for v in y_true], dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: y={y.shape}, scores={s.shape}")
    return y, s


def roc_auc(y_true: Sequence, scores: Sequence) -> Optional[float]:
    """用 Mann-Whitney U 计算 ROC-AUC；只有一个类别时返回 None。"""
    y, s = _as_arrays(y_true, scores)
    positives = s[y == 1]
    negatives = s[y == 0]
    if positives.size == 0 or negatives.size == 0:
        return None

    greater = float(np.sum(positives[:, None] > negatives[None, :]))
    equal = float(np.sum(positives[:, None] == negatives[None, :]))
    return (greater + 0.5 * equal) / (positives.size * negatives.size)


def precision_recall_f1(y_true: Sequence, y_pred: Sequence) -> dict:
    """二分类的 precision / recall / f1（正类为异常）。"""
    y, p = _as_arrays(y_true, y_pred)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def precision_at_threshold(y_true: Sequence, scores: Sequence, threshold: float) -> dict:
    """把 ``scores >= threshold`` 判为正类时的精度与召回。"""
    y, s = _as_arrays(y_true, scores)
    predicted = (s >= threshold).astype(int)
    return precision_recall_f1(y, predicted)
