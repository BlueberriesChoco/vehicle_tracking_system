"""告警预算的回归测试。"""

from src.anomaly.alert_budget import apply_alert_budget, choose_contamination


def _row(score, is_anomaly=1, reason=""):
    return {"anomaly_score": score, "is_anomaly": is_anomaly, "alert_reason": reason}


def test_budget_caps_alert_count():
    rows = [_row(score) for score in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]]

    budgeted = apply_alert_budget(rows, target_rate=0.3)

    alerts = sum(1 for row in budgeted if row["is_anomaly"] == 1)
    assert alerts == 3
    # 保留的是分数最高的 3 个
    assert [row["is_anomaly"] for row in budgeted[:3]] == [1, 1, 1]
    assert [row["is_anomaly"] for row in budgeted[3:]] == [0] * 7


def test_budget_keeps_rule_alerts_first():
    rows = [
        _row(0.9),
        _row(0.8),
        _row(0.1, is_anomaly=1, reason="长时间滞留=400s"),
        _row(0.05),
    ]

    budgeted = apply_alert_budget(rows, target_rate=0.25)

    # 预算 1 个：规则命中的那行优先保留
    assert budgeted[2]["is_anomaly"] == 1
    assert budgeted[0]["is_anomaly"] == 0


def test_budget_marks_capped_rows():
    rows = [_row(0.9), _row(0.8), _row(0.1, is_anomaly=0)]

    budgeted = apply_alert_budget(rows, target_rate=0.34)

    # 预算 1 个，被压掉的行标记 alert_capped
    capped = [row for row in budgeted if row["alert_capped"] == 1]
    assert len(capped) == 1
    assert capped[0]["anomaly_score"] == 0.8


def test_budget_zero_clears_alerts():
    rows = [_row(0.9), _row(0.8)]

    budgeted = apply_alert_budget(rows, target_rate=0.0)

    assert sum(row["is_anomaly"] for row in budgeted) == 0


def test_budget_handles_empty_rows():
    assert apply_alert_budget([]) == []


def test_choose_contamination_prefers_largest_within_target():
    trials = [
        {"contamination": 0.02, "alert_rate": 0.03},
        {"contamination": 0.05, "alert_rate": 0.08},
        {"contamination": 0.08, "alert_rate": 0.14},
        {"contamination": 0.10, "alert_rate": 0.20},
    ]
    assert choose_contamination(trials, 0.15)["contamination"] == 0.08


def test_choose_contamination_falls_back_to_closest():
    trials = [
        {"contamination": 0.05, "alert_rate": 0.18},
        {"contamination": 0.10, "alert_rate": 0.30},
    ]
    assert choose_contamination(trials, 0.15)["contamination"] == 0.05


def test_choose_contamination_empty():
    assert choose_contamination([], 0.15) is None
