"""规则引擎的回归测试（重点：夜间通行规则）。"""

from src.anomaly.rule_engine import RuleEngine


def test_night_any_passage_fires_once():
    engine = RuleEngine()

    reasons = engine.check({"is_night": 1, "freq_count_24h": 1, "stop_count": 0})

    assert "夜间通行" in reasons


def test_night_high_freq_takes_precedence():
    engine = RuleEngine()

    reasons = engine.check({"is_night": 1, "freq_count_24h": 6, "stop_count": 0})

    assert any("夜间高频通行" in reason for reason in reasons)
    assert "夜间通行" not in reasons


def test_night_any_passage_can_be_disabled():
    engine = RuleEngine(thresholds={"night_any_passage": 0})

    reasons = engine.check({"is_night": 1, "freq_count_24h": 1, "stop_count": 0})

    assert "夜间通行" not in reasons


def test_daytime_passage_not_flagged():
    engine = RuleEngine()

    reasons = engine.check({"is_night": 0, "freq_count_24h": 1, "stop_count": 0})

    assert "夜间通行" not in reasons
    assert not any("夜间" in reason for reason in reasons)


def test_snake_rule_fires_on_high_curvature():
    """平均转角越大越犹豫；不能把直线判成蛇形。"""
    engine = RuleEngine(thresholds={"snake_smoothness_max": 0.5})

    reasons = engine.check({"path_smoothness": 0.8})

    assert any("蛇形" in reason for reason in reasons)


def test_snake_rule_does_not_fire_on_straight_path():
    engine = RuleEngine(thresholds={"snake_smoothness_max": 0.5})

    reasons = engine.check({"path_smoothness": 0.12})

    assert not any("蛇形" in reason for reason in reasons)


def test_snake_rule_does_not_require_geometry_calibration():
    """角度特征在像素空间有效，path_reliable=0 时也应生效。"""
    engine = RuleEngine(thresholds={"snake_smoothness_max": 0.5})

    reasons = engine.check({"path_smoothness": 0.9, "path_reliable": 0})

    assert any("蛇形" in reason for reason in reasons)


def test_speeding_rule_with_reliable_speed():
    engine = RuleEngine()

    # 13.9 m/s = 50.0 km/h
    reasons = engine.check({"avg_speed_ms": 13.9, "speed_reliable": 1})

    assert any("超速" in reason for reason in reasons)


def test_speeding_rule_skipped_when_speed_unreliable():
    engine = RuleEngine()

    reasons = engine.check({"avg_speed_ms": 13.9, "speed_reliable": 0})

    assert not any("超速" in reason for reason in reasons)


def test_deviation_rule_with_reliable_path():
    engine = RuleEngine()

    reasons = engine.check({"path_deviation": 5.0, "path_reliable": 1})

    assert any("路径偏离" in reason for reason in reasons)


def test_deviation_rule_skipped_when_path_unreliable():
    engine = RuleEngine()

    reasons = engine.check({"path_deviation": 5.0, "path_reliable": 0})

    assert not any("路径偏离" in reason for reason in reasons)


def test_dwell_rule_fires_above_threshold():
    engine = RuleEngine()

    reasons = engine.check({"max_dwell_sec": 200})

    assert any("长时间滞留" in reason for reason in reasons)


def test_night_stop_rule_fires_on_repeated_stops():
    engine = RuleEngine()

    reasons = engine.check({"is_night": 1, "stop_count": 3, "freq_count_24h": 1})

    assert any("夜间频繁停留" in reason for reason in reasons)
