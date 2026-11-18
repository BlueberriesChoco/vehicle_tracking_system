"""异常检测评估指标的回归测试。"""

import pytest

from src.anomaly.metrics import precision_at_threshold, precision_recall_f1, roc_auc


def test_roc_auc_perfect_separation():
    assert roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)


def test_roc_auc_reversed():
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)


def test_roc_auc_ties_is_half():
    assert roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_roc_auc_single_class_returns_none():
    assert roc_auc([1, 1], [0.9, 0.8]) is None


def test_precision_recall_f1():
    result = precision_recall_f1([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.3333)
    assert result["specificity"] == pytest.approx(0.6667)
    assert precision_recall_f1([1], [1])["false_positive_rate"] is None


def test_precision_at_threshold():
    result = precision_at_threshold([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], threshold=0.5)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
