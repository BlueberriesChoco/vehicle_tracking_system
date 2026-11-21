"""G 的跨批次可比性标注（`graph_anomaly_comparable`）的回归测试。

## 背景

G 改用分位锚点标定后，锚点从哪来决定了分数能不能横向比较：

- **独立基线**：编码器在另一张图上训练，锚点来自那张图 → 跨批次可比；
- **自参照**：训练图就是评分图 → 误差带被压缩，锚点把窄带放大，
  只能当本批内部相对排序看。

## 这里守的是什么

`graph_anomaly_comparable` 最初写成 ``calibration["source"] == "baseline"``，
但 GAE 的 ``fit`` 总会记录锚点，``score`` 随之走 baseline 分支——
**同图训练时 source 也是 "baseline"**，于是自参照被误标成「可比」。

实测该缺陷让 GAE 同图输出的 `comparable` 撒谎：
`graph_admin_in_gae.json` 报 `comparable=True`，但它的训练图就是评分图。

正确判据是「训练图是否就是评分图」，由 payload 层自己知道，
不能从 ``calibration.source`` 反推。
"""

import numpy as np
import pytest

from src.graph.payload import build_graph_payload
from src.graph.relations import RelationConfig


def _rows(n=30):
    """造一批能建出连边的记录（重叠时间 + 相近轨迹）。"""
    return [
        {
            "track_id": str(i),
            "camera_id": "cam01",
            "analysis_target": "1",
            "vehicle_group": "ordinary",
            "passage_id": f"cam01_20260901080000_{i}",
            "enter_time": f"2026-09-01T08:{i:02d}:00",
            "exit_time": f"2026-09-01T08:{i:02d}:03",
            "trajectory_points": ";".join(
                f"{500 + 3 * s},{100 + 40 * s}" for s in range(12)
            ),
            "avg_speed_ms": 8.0 + 0.1 * i,
            "path_smoothness": 0.1 + 0.005 * i,
            "path_deviation": 0.0,
            "duration_sec": 3.0,
        }
        for i in range(n)
    ]


def test_svd_graph_is_marked_self_referential():
    """SVD 没有独立基线概念，锚点恒由本图自算 → 不可横向比较。"""
    payload = build_graph_payload(_rows(), RelationConfig(top_k=3), method="svd")

    assert payload is not None
    assert payload["graph_anomaly_comparable"] is False
    assert payload["graph_anomaly_self_referential"] is True


def test_gae_trained_on_same_graph_is_not_comparable():
    """同图训练（不传 gae_model）时训练图 == 评分图，必须标为不可比。

    这是本次要修的缺陷：旧实现看 ``calibration.source == "baseline"``，
    而 fit 总会写锚点，导致这里错误地返回 True。
    """
    payload = build_graph_payload(
        _rows(), RelationConfig(top_k=3), method="gae", gae_epochs=10
    )

    assert payload is not None
    assert payload["graph_anomaly_method"] == "gae"
    assert payload["graph_anomaly_comparable"] is False
    assert payload["graph_anomaly_self_referential"] is True
    # 关键：锚点来源确实是 baseline，所以不能靠它判断可比性
    assert payload["graph_anomaly_calibration"].get("source") == "baseline"


def test_gae_loaded_from_baseline_model_is_comparable(tmp_path):
    """加载独立基线模型评分时，锚点来自那张基线图 → 可比。"""
    prefix = str(tmp_path / "gae_baseline")
    # 先在一批图上训练并保存（模拟"基线图训练"）
    build_graph_payload(
        _rows(24),
        RelationConfig(top_k=3),
        method="gae",
        gae_epochs=10,
        save_gae_model=prefix,
    )
    # 再对另一批图评分
    payload = build_graph_payload(
        _rows(30), RelationConfig(top_k=3), method="gae", gae_model=prefix
    )

    assert payload is not None
    assert payload["graph_anomaly_method"] == "gae_baseline"
    assert payload["graph_anomaly_comparable"] is True
    assert payload["graph_anomaly_self_referential"] is False


def test_comparable_flag_is_consistent_with_self_referential():
    """两个字段必须互为反面，避免下游各读一个得到矛盾结论。"""
    for method, kwargs in (("svd", {}), ("gae", {"gae_epochs": 10})):
        payload = build_graph_payload(
            _rows(), RelationConfig(top_k=3), method=method, **kwargs
        )
        assert payload is not None
        assert payload["graph_anomaly_comparable"] != payload["graph_anomaly_self_referential"]


def test_self_referential_flag_survives_json_roundtrip():
    """字段必须写进 JSON 且是原生 bool——下游用 == True 判断，字符串 "False" 会误判。"""
    import json

    payload = build_graph_payload(_rows(), RelationConfig(top_k=3), method="svd")
    restored = json.loads(json.dumps(payload, default=str))

    assert restored["graph_anomaly_comparable"] is False
    assert restored["graph_anomaly_self_referential"] is True
