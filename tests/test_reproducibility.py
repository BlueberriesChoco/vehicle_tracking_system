"""可复现性测试：同一输入、同一随机种子，两次运行必须完全一致。

监管系统如果结果不可复现，就无法作为执法依据。这里覆盖三个随机组件：
- Isolation Forest（树构建随机采样）
- 图谱（Louvain 社区发现 + SVD 重构误差）
- Transformer 轨迹预测（torch 初始化与 DataLoader 打乱）
"""

import hashlib
import json
from datetime import datetime, timedelta

import numpy as np

from src.anomaly.isolation_forest import IsolationForestDetector
from src.anomaly.trajectory_transformer import TrajectoryPredictionDetector
from src.graph.payload import build_graph_payload
from src.graph.relations import RelationConfig


def _synthetic_rows(n=40, points=12):
    rows = []
    start = datetime(2026, 9, 1, 8, 0, 0)
    rng = np.random.RandomState(0)
    for index in range(n):
        enter = start + timedelta(seconds=20 * index)
        exit_time = enter + timedelta(seconds=3)
        trajectory = ";".join(
            f"{500 + 3 * step:.1f},{100 + 40 * step:.1f}" for step in range(points)
        )
        rows.append(
            {
                "track_id": str(index),
                "camera_id": "cam01",
                "analysis_target": "1",
                "vehicle_group": "ordinary",
                "passage_id": f"cam01_20260901080000_{index}",
                "enter_time": enter.isoformat(),
                "exit_time": exit_time.isoformat(),
                "duration_sec": "3.0",
                "trajectory_points": trajectory,
                "avg_speed_ms": str(round(float(rng.uniform(5, 15)), 3)),
                "max_speed_ms": str(round(float(rng.uniform(15, 20)), 3)),
                "path_smoothness": str(round(float(rng.uniform(0.05, 0.3)), 3)),
                "path_deviation": "0.0",
            }
        )
    return rows


def _iforest_signature(rows):
    detector = IsolationForestDetector(
        n_estimators=40, max_samples=64, contamination=0.05, random_state=42
    )
    detector.fit(rows)
    scored = detector.score(rows)
    values = [round(float(row.get("iforest_score") or 0), 6) for row in scored]
    return hashlib.sha1(json.dumps(values).encode()).hexdigest()


def _graph_signature(rows):
    payload = build_graph_payload(rows, RelationConfig(top_k=3), method="svd")
    snapshot = [
        (node["node"], node["community_id"], round(node["graph_anomaly_index"], 6))
        for node in payload["nodes"]
    ]
    return hashlib.sha1(json.dumps(snapshot).encode()).hexdigest()


def _transformer_signature(rows):
    detector = TrajectoryPredictionDetector(
        past_len=5, future_len=5, epochs=1, batch_size=16, random_state=42
    )
    detector.fit(rows)
    scored = detector.score(rows)
    values = [round(float(row.get("traj_residual") or 0), 6) for row in scored]
    return hashlib.sha1(json.dumps(values).encode()).hexdigest()


def test_isolation_forest_is_reproducible():
    rows = _synthetic_rows()

    assert _iforest_signature(rows) == _iforest_signature(rows)


def test_graph_is_reproducible():
    rows = _synthetic_rows()

    assert _graph_signature(rows) == _graph_signature(rows)


def test_transformer_is_reproducible():
    rows = _synthetic_rows(n=30, points=10)

    assert _transformer_signature(rows) == _transformer_signature(rows)
