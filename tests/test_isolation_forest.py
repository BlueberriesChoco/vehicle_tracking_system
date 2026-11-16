"""Isolation Forest 与“最大同屏车辆数”的回归测试。

覆盖：
- Isolation Forest 对明显离群点给出更高异常指数；
- 常数数据不会把所有样本都判为异常；
- 模型字典序列化 / 反序列化一致；
- 行为向量适配器（跳过非分析对象、极端向量被判异常）；
- 接入 AnomalyDetector 后写入 iforest 告警原因；
- 行为向量 CSV 暴露“最大同屏车辆数”。
"""

import os

import numpy as np

from src.anomaly.anomaly_detector import AnomalyDetector
from src.anomaly.isolation_forest import IsolationForest, IsolationForestDetector
from src.behavior.features.aggregation_feature import AggregationFeatureExtractor
from src.output.vector_writer import VectorWriter
from src.trajectory.scene_geometry import SceneGeometry


def _scene():
    return SceneGeometry(((0, 0), (0, 0)), ((0, 0), (0, 0)), 10.0)


def _normal_vectors(n=40, seed=0):
    rng = np.random.RandomState(seed)
    vectors = []
    for index in range(n):
        speed = max(float(rng.normal(2.0, 0.3)), 0.0)
        vectors.append({
            "track_id": index,
            "camera_id": "cam01",
            "vehicle_group": "ordinary",
            "analysis_target": 1,
            "avg_speed_ms": speed,
            "max_speed_ms": speed + 1.0,
            "speed_variance": abs(float(rng.normal(0.1, 0.02))),
            "max_dwell_sec": abs(float(rng.normal(5.0, 1.0))),
            "stop_count": 0,
            "dwell_ratio": abs(float(rng.normal(0.05, 0.01))),
            "path_deviation": abs(float(rng.normal(0.2, 0.05))),
            "path_smoothness": abs(float(rng.normal(0.1, 0.02))),
            "is_night": 0,
            "night_ratio": 0.0,
            "freq_index": abs(float(rng.normal(0.0, 0.2))),
            "freq_count_24h": 1,
            "aggregation_index": abs(float(rng.normal(0.1, 0.02))),
            "nearest_vehicle_m": abs(float(rng.normal(20.0, 3.0))),
            "max_concurrent_vehicles": 2,
            "speed_reliable": 1,
            "path_reliable": 1,
            "aggregation_reliable": 1,
            "exit_time": "2026-06-01T08:00:10",
        })
    return vectors


def test_isolation_forest_flags_obvious_outliers():
    rng = np.random.RandomState(1)
    normal = rng.normal(0, 1, size=(100, 2))
    outliers = np.array([[8.0, 8.0], [9.0, 9.0], [-8.0, -8.0], [10.0, -10.0], [8.5, -8.5]])
    X = np.vstack([normal, outliers])

    model = IsolationForest(
        n_estimators=120, max_samples=64, contamination=0.05, random_state=7
    ).fit(X)

    scores = model.score_samples(X)
    assert scores[-5:].mean() > scores[:-5].mean()
    predictions = model.predict(X)
    assert (predictions[-5:] == -1).sum() >= 3


def test_isolation_forest_constant_data_does_not_flag_everything():
    X = np.ones((10, 3))
    model = IsolationForest(
        n_estimators=20, max_samples=8, contamination=0.1, random_state=1
    ).fit(X)

    predictions = model.predict(X)
    assert (predictions == -1).sum() == 0


def test_isolation_forest_dict_roundtrip_is_stable():
    rng = np.random.RandomState(2)
    X = rng.normal(0, 1, size=(50, 3))
    model = IsolationForest(n_estimators=15, max_samples=16, random_state=3).fit(X)

    clone = IsolationForest.from_dict(model.to_dict())
    np.testing.assert_allclose(clone.score_samples(X), model.score_samples(X), rtol=1e-9)


def test_isolation_forest_save_and_load():
    rng = np.random.RandomState(4)
    X = rng.normal(0, 1, size=(40, 2))
    model = IsolationForest(n_estimators=12, max_samples=16, random_state=5).fit(X)

    # 写入工作区根目录（部分沙箱只允许根目录写入），用唯一文件名并在结束时删除
    path = os.path.join(".", f"dsh_iforest_test_{os.getpid()}.json")
    try:
        model.save(path)
        loaded = IsolationForest.load(path)
        np.testing.assert_allclose(loaded.score_samples(X), model.score_samples(X), rtol=1e-9)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_iforest_detector_scores_extreme_vector_as_anomaly():
    vectors = _normal_vectors(300, seed=2)
    detector = IsolationForestDetector(
        n_estimators=100, max_samples=256, contamination=0.05, random_state=5
    ).fit(vectors)

    extreme = dict(
        vectors[0],
        avg_speed_ms=60.0,
        max_speed_ms=80.0,
        max_dwell_sec=2000.0,
        aggregation_index=1.0,
        nearest_vehicle_m=0.5,
        max_concurrent_vehicles=20,
    )
    scored = detector.score([extreme])[0]

    assert scored["iforest_score"] > 0.5
    assert scored["iforest_anomaly"] == 1


def test_iforest_detector_skips_non_target_rows():
    vectors = _normal_vectors(20, seed=3)
    detector = IsolationForestDetector(
        n_estimators=20, max_samples=16, random_state=1
    ).fit(vectors)

    freight = dict(
        vectors[0],
        vehicle_type="truck",
        vehicle_group="freight",
        analysis_target=0,
        avg_speed_ms=99.0,
    )
    scored = detector.score([freight])[0]

    assert scored["iforest_score"] == 0.0
    assert scored["iforest_anomaly"] == 0


def test_anomaly_detector_records_iforest_reason():
    vectors = _normal_vectors(300, seed=4)
    detector = IsolationForestDetector(
        n_estimators=100, max_samples=256, contamination=0.05, random_state=3
    ).fit(vectors)

    extreme = dict(
        vectors[0],
        avg_speed_ms=60.0,
        max_dwell_sec=3000.0,
        aggregation_index=1.0,
        max_concurrent_vehicles=20,
    )
    engine = AnomalyDetector(
        thresholds={"speed_max_kmh": 9999.0, "dwell_max_sec": 999999.0, "deviation_max_m": 9999.0},
        iforest=detector,
    )
    result = engine.detect([extreme])[0]

    assert result["is_anomaly"] == 1
    assert "iforest_score" in result["alert_reason"]


def test_vector_writer_exposes_max_concurrent_vehicles():
    assert "max_concurrent_vehicles" in VectorWriter.CSV_COLUMNS


def test_aggregation_reports_max_concurrent_vehicles():
    extractor = AggregationFeatureExtractor(_scene(), aggregation_radius_m=5)
    result = extractor.extract(
        [(0.0, 0.0), (10.0, 0.0)],
        {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (20.0, 0.0)},
        current_track_id=1,
    )

    assert result["max_concurrent_vehicles"] == 3
