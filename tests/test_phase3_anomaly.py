import csv

from src.anomaly import AnomalyDetector
from src.anomaly.baseline import BaselineBuilder
from src.anomaly.rule_engine import RuleEngine
from src.output.alert_writer import AlertWriter


def _row(speed=2.0, dwell=5.0, deviation=0.2, aggregation=0.1, reliable=1):
    return {
        "camera_id": "cam01",
        "track_id": "1",
        "avg_speed_ms": speed,
        "max_speed_ms": speed + 1,
        "speed_variance": 0.1,
        "max_dwell_sec": dwell,
        "stop_count": 0,
        "dwell_ratio": 0.0,
        "path_deviation": deviation,
        "path_smoothness": 0.1,
        "is_night": 0,
        "night_ratio": 0.0,
        "freq_index": 1.0,
        "freq_count_24h": 1,
        "aggregation_index": aggregation,
        "nearest_vehicle_m": 20.0,
        "speed_reliable": reliable,
        "path_reliable": reliable,
        "aggregation_reliable": reliable,
        "exit_time": "2026-06-01T08:00:10",
    }


def test_baseline_builder_filters_unreliable_geometry_features():
    rows = [_row(speed=2.0, reliable=1), _row(speed=99.0, reliable=0)]

    baseline = BaselineBuilder().build(rows, camera_id="cam01")

    assert baseline["camera_id"] == "cam01"
    assert baseline["sample_count"] == 2
    assert baseline["stats"]["avg_speed_ms"]["count"] == 1
    assert baseline["stats"]["avg_speed_ms"]["mean"] == 2.0
    assert baseline["stats"]["max_dwell_sec"]["count"] == 2


def test_baseline_score_marks_outlier_when_enough_features_available():
    builder = BaselineBuilder()
    baseline = builder.build([_row(speed=2.0), _row(speed=2.2), _row(speed=1.8)], camera_id="cam01")

    result = builder.score(_row(speed=20.0, dwell=100.0, aggregation=5.0), baseline, threshold=2.0)

    assert result["baseline_anomaly"] == 1
    assert result["baseline_score"] >= 2.0
    assert result["baseline_feature_count"] >= 3


def test_rule_engine_skips_unreliable_speed_and_path_rules():
    engine = RuleEngine({"speed_max_kmh": 10.0, "deviation_max_m": 1.0, "dwell_max_sec": 120.0})

    reasons = engine.check(_row(speed=20.0, deviation=20.0, reliable=0))

    assert reasons == []


def test_anomaly_detector_combines_rules_and_baseline():
    baseline = BaselineBuilder().build([_row(speed=2.0), _row(speed=2.1), _row(speed=1.9)], camera_id="cam01")
    detector = AnomalyDetector(baseline=baseline, baseline_threshold=2.0)

    detected = detector.detect([_row(speed=20.0, dwell=200.0)])[0]

    assert detected["is_anomaly"] == 1
    assert "长时间滞留" in detected["alert_reason"]
    assert detected["baseline_feature_count"] >= 3


def test_alert_writer_outputs_only_anomalies_by_default(tmp_path):
    path = tmp_path / "alerts.csv"
    rows = [
        {"camera_id": "cam01", "track_id": 1, "is_anomaly": 0},
        {"camera_id": "cam01", "track_id": 2, "is_anomaly": 1, "anomaly_score": 4.0, "alert_reason": "test"},
    ]

    AlertWriter().write(rows, str(path))

    with path.open(newline="", encoding="utf-8") as f:
        output = list(csv.DictReader(f))

    assert len(output) == 1
    assert output[0]["track_id"] == "2"
    assert output[0]["severity"] == "medium"
