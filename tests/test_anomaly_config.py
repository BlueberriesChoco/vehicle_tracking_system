"""异常检测配置读取的回归测试。"""

from src.anomaly.config import load_anomaly_config


def test_load_anomaly_config_reads_rules():
    config = load_anomaly_config("config/anomaly_config.yaml")

    assert config["rules"]["dwell_max_sec"] == 120.0
    assert config["rules"]["speed_max_kmh"] == 40.0
    assert config["isolation_forest"]["contamination"] == 0.05


def test_load_anomaly_config_missing_file_returns_empty():
    assert load_anomaly_config("does/not/exist.yaml") == {}


def test_load_anomaly_config_none_returns_empty():
    assert load_anomaly_config(None) == {}
