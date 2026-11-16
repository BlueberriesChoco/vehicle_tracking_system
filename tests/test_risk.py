"""研究要点四：时间频率 T、风险融合 R、证据链与报告的回归测试。"""

import pytest

from src.risk import (
    EvidenceBuilder,
    ReportService,
    RiskFusion,
    TemplateReportGenerator,
    TimeFrequencyScorer,
)


def _row(**kwargs):
    base = {
        "camera_id": "cam01",
        "track_id": "7",
        "_source_file": "vec_cam01_20260901_08.csv",
        "iforest_score": 0.8,
        "graph_anomaly_index": 0.6,
        "freq_count_24h": 12,
        "night_ratio": 0.5,
        "is_night": 1,
        "traj_residual": 25.0,
        "community_id": 3,
        "community_size": 7,
        "alert_reason": "长时间滞留(300s)",
    }
    base.update(kwargs)
    return base


def test_time_frequency_scorer_is_bounded_and_monotonic():
    scorer = TimeFrequencyScorer(freq_threshold=6.0)

    low = scorer.score_one(_row(freq_count_24h=0, night_ratio=0.0, is_night=0))
    high = scorer.score_one(_row(freq_count_24h=12, night_ratio=1.0, is_night=1))

    assert 0.0 <= low["time_freq_index"] <= 1.0
    assert 0.0 <= high["time_freq_index"] <= 1.0
    assert high["time_freq_index"] > low["time_freq_index"]


def test_single_pass_has_zero_time_frequency():
    """单次白天通行是正常基线，T 必须为 0，否则会给每辆车生成“时间频率异常”证据。"""
    scorer = TimeFrequencyScorer(freq_threshold=6.0)

    record = scorer.score_one(_row(freq_count_24h=1, night_ratio=0.0, is_night=0))

    assert record["time_freq_index"] == 0.0


def test_night_passage_raises_time_frequency():
    scorer = TimeFrequencyScorer(freq_threshold=6.0)

    day = scorer.score_one(_row(freq_count_24h=1, night_ratio=0.0, is_night=0))
    night = scorer.score_one(_row(freq_count_24h=1, night_ratio=1.0, is_night=1))

    assert night["time_freq_index"] > day["time_freq_index"]
    assert night["time_freq_index"] >= 0.2


def test_risk_fusion_normalises_weights_and_levels():
    fusion = RiskFusion(alpha=1.0, beta=1.0, gamma=2.0)

    # 权重归一化后 alpha=0.25, beta=0.25, gamma=0.5
    assert fusion.alpha == pytest.approx(0.25)
    assert fusion.beta == pytest.approx(0.25)
    assert fusion.gamma == pytest.approx(0.5)

    record = fusion.fuse_one(
        _row(iforest_score=1.0, graph_anomaly_index=1.0, time_freq_index=1.0)
    )
    assert record["risk_score_100"] == pytest.approx(100.0)
    assert record["risk_level"] == "high"

    low = fusion.fuse_one(_row(iforest_score=0.0, graph_anomaly_index=0.0, time_freq_index=0.0))
    assert low["risk_level"] == "low"


def test_risk_fusion_rejects_all_zero_weights():
    with pytest.raises(ValueError):
        RiskFusion(alpha=0.0, beta=0.0, gamma=0.0)


def test_evidence_builder_traces_every_item():
    record = _row(time_freq_index=0.7)
    evidence = EvidenceBuilder().build(record)

    assert len(evidence) >= 4
    for item in evidence:
        assert item["trace"]["camera_id"] == "cam01"
        assert item["trace"]["track_id"] == "7"
        assert item["trace"]["source_file"] == "vec_cam01_20260901_08.csv"
    categories = {item["category"] for item in evidence}
    assert {"behavior", "graph", "temporal", "rule", "trajectory"} <= categories


def test_evidence_builder_skips_weak_signals():
    record = _row(iforest_score=0.0, graph_anomaly_index=0.0, time_freq_index=0.0,
                  traj_residual=0.0, alert_reason="")

    evidence = EvidenceBuilder().build(record)

    assert evidence == []


def test_template_report_contains_score_and_evidence():
    record = _row(time_freq_index=0.7)
    record.update({"risk_score_100": 72.5, "risk_level": "medium",
                   "risk_components": {"A": 0.8, "G": 0.6, "T": 0.7}})
    evidence = EvidenceBuilder().build(record)

    text = TemplateReportGenerator().generate(record, evidence)

    assert "72.5" in text
    assert "证据链" in text
    assert "排查建议" in text


def test_template_report_marks_missing_graph_as_no_data():
    """没有图谱节点时，不能把“图 0.00”写成“图异常很低”。"""
    record = _row(graph_anomaly_index=0.0)
    record.update({"risk_score_100": 30.0, "risk_level": "low", "graph_available": 0,
                   "risk_components": {"A": 0.4, "G": 0.0, "T": 0.1}})

    text = TemplateReportGenerator().generate(record, [])

    assert "图 无数据" in text
    assert "图 0.00" not in text


def test_template_report_shows_graph_value_when_available():
    record = _row(graph_anomaly_index=0.6)
    record.update({"risk_score_100": 55.0, "risk_level": "medium", "graph_available": 1,
                   "risk_components": {"A": 0.4, "G": 0.6, "T": 0.1}})

    text = TemplateReportGenerator().generate(record, [])

    assert "图 0.60" in text
    assert "图 无数据" not in text
    assert "cam01:7" in text


def test_report_service_falls_back_to_template_without_llm():
    service = ReportService()

    result = service.generate(_row(time_freq_index=0.7), [])

    assert result["backend"] == "template"
    assert result["text"]


class _FakeLLM:
    def available(self):
        return True

    def generate(self, record, evidence):
        return "LLM 生成的排查建议"


def test_report_service_uses_llm_when_available():
    service = ReportService(llm=_FakeLLM())

    result = service.generate(_row(), [])

    assert result["backend"] == "llm"
    assert "LLM 生成的排查建议" in result["text"]
    # 低风险车辆的证据区标题已改为“指标读数”，避免把读数写成疑点
    assert ("证据链" in result["text"]) or ("指标读数" in result["text"])


class _BrokenLLM:
    def available(self):
        return True

    def generate(self, record, evidence):
        raise RuntimeError("endpoint down")


def test_report_service_falls_back_when_llm_fails():
    service = ReportService(llm=_BrokenLLM())

    result = service.generate(_row(time_freq_index=0.7), [])

    assert result["backend"] == "template"
    assert result["text"]


def _grounded_record():
    record = _row(time_freq_index=0.7)
    record.update(
        {
            "risk_score": 0.562,
            "risk_score_100": 56.2,
            "risk_level": "medium",
            "risk_components": {"A": 0.48, "G": 1.0, "T": 0.08},
        }
    )
    evidence = [
        {"category": "graph", "signal": "graph_anomaly", "field": "graph_anomaly_index",
         "value": 1.0, "description": "图异常指数 G=1.00", "trace": {"camera_id": "cam01", "track_id": "7"}},
        {"category": "behavior", "signal": "isolation_forest", "field": "iforest_score",
         "value": 0.48, "description": "行为异常指数 A=0.48", "trace": {"camera_id": "cam01", "track_id": "7"}},
    ]
    return record, evidence


def test_validate_response_accepts_grounded_text():
    from src.risk import validate_response

    record, evidence = _grounded_record()
    result = validate_response("该车综合风险 56.2 分，G=1.0，A=0.48，属于中风险。", record, evidence)

    assert result["valid"] is True
    assert result["issues"] == []


def test_validate_response_rejects_ungrounded_number():
    from src.risk import validate_response

    record, evidence = _grounded_record()
    result = validate_response("该车综合风险 88.8 分，疑似严重违规。", record, evidence)

    assert result["valid"] is False
    assert any(issue.startswith("ungrounded_number") for issue in result["issues"])


def test_validate_response_rejects_foreign_vehicle_id():
    from src.risk import validate_response

    record, evidence = _grounded_record()
    result = validate_response("建议核查 cam02:99 是否与当前车辆同行。", record, evidence)

    assert result["valid"] is False
    assert any(issue.startswith("foreign_vehicle_id") for issue in result["issues"])


class _GroundedLLM:
    def available(self):
        return True

    def generate(self, record, evidence):
        return "该车综合风险 56.2 分，G=1.0，建议核查。"


class _HallucinatingLLM:
    def available(self):
        return True

    def generate(self, record, evidence):
        return "该车综合风险 99.9 分，已确认走私。"


def test_report_service_uses_grounded_llm_output():
    record, evidence = _grounded_record()
    service = ReportService(llm=_GroundedLLM())

    result = service.generate(record, evidence)

    assert result["backend"] == "llm"
    assert result["validation"]["valid"] is True


def test_report_service_falls_back_on_hallucinated_llm_output():
    record, evidence = _grounded_record()
    service = ReportService(llm=_HallucinatingLLM())

    result = service.generate(record, evidence)

    assert result["backend"] == "template_fallback"
    assert result["validation"]["valid"] is False
    assert result["text"]


def test_template_report_has_conclusion_and_business_language():
    record, evidence = _grounded_record()
    record["enter_time"] = "2026-09-01T08:03:12"
    record["exit_time"] = "2026-09-01T08:04:05"
    record["duration_sec"] = 53

    text = ReportService().generate(record, evidence)["text"]

    assert "一句话结论" in text
    assert "08:03:12" in text
    assert "约 53 秒" in text
    assert "证据链" in text
    # A=0.48 未达疑点量级（阈值 0.60），不再写成“行为模式与常态不符”；
    # 但 G=1.00 达到阈值，仍应出现结伴措辞
    assert "疑似结伴" in text


def test_low_risk_conclusion_drops_key_issue():
    """低风险车辆不应声称存在“主要疑点”，否则会误导一线人员。"""
    record, evidence = _grounded_record()
    record["risk_level"] = "low"

    text = ReportService().generate(record, evidence)["text"]

    assert "主要疑点" not in text
    assert "风险较低" in text


def test_low_magnitude_value_is_reported_as_reading_not_anomaly():
    """A=0.48 只是读数，不应被写成“行为模式与常态不符”。"""
    record, evidence = _grounded_record()

    text = TemplateReportGenerator().generate(record, evidence)

    assert "行为模式与常态不符" not in text
    assert "行为异常指数 A=0.48" in text
    assert "疑似结伴" in text


def test_low_risk_uses_neutral_evidence_header():
    """低风险车辆不应出现“证据链（疑点）”的措辞。"""
    record = _row()
    record.update({"risk_level": "low", "risk_score_100": 20.0,
                   "risk_components": {"A": 0.3, "G": 0.3, "T": 0.0}})

    text = TemplateReportGenerator().generate(record, [])

    assert "指标读数" in text
    assert "**证据链：**" not in text


def test_low_risk_evidence_uses_raw_readings_even_for_high_values():
    """低风险车辆即使某项指标很高，也只陈述读数，不下“疑似结伴”之类判断。"""
    record, evidence = _grounded_record()
    record["risk_level"] = "low"

    text = TemplateReportGenerator().generate(record, evidence)

    assert "疑似结伴" not in text
    assert "图异常指数 G=1.00" in text


def test_conclusion_picks_strongest_signal_not_first():
    """G=0.97 明显强于 A=0.62，主要疑点应选图异常（疑似结伴）而非行为异常。"""
    record = _row()
    record.update({"risk_score_100": 59.0, "risk_level": "medium",
                   "risk_components": {"A": 0.62, "G": 0.97, "T": 0.0}})
    evidence = [
        {"category": "behavior", "signal": "isolation_forest", "field": "iforest_score",
         "value": 0.62, "description": "行为异常指数 A=0.62",
         "trace": {"camera_id": "cam01", "track_id": "7"}},
        {"category": "graph", "signal": "graph_anomaly", "field": "graph_anomaly_index",
         "value": 0.97, "description": "图异常指数 G=0.97",
         "trace": {"camera_id": "cam01", "track_id": "7"}},
    ]

    text = TemplateReportGenerator().generate(record, evidence)

    assert "疑似结伴" in text.split("一句话结论")[1].splitlines()[0]


def test_detector_scores_are_not_reported_as_rule_hits():
    """alert_reason 里的 iforest_score=... 不是规则命中，不能写成“命中预设监管规则”。"""
    from src.risk.evidence import EvidenceBuilder

    record = _row()
    record["alert_reason"] = "iforest_score=0.71;traj_residual=177.4px;蛇形行驶(平均转角0.55)"

    signals = [item["signal"] for item in EvidenceBuilder().build(record)]
    descriptions = [item["description"] for item in EvidenceBuilder().build(record)]

    rule_items = [d for d in descriptions if d.startswith("规则命中")]
    assert len(rule_items) == 1
    assert "蛇形行驶" in rule_items[0]
    assert not any("iforest_score" in d or "traj_residual" in d for d in rule_items)


def test_low_risk_suggestions_do_not_claim_deviation():
    """低风险车辆的排查建议不能写“偏离正常分布、建议重点核对”。"""
    record = _row()
    record.update({"risk_level": "low", "risk_score_100": 49.9, "traj_residual": 23.6,
                   "risk_components": {"A": 0.62, "G": 0.72, "T": 0.0}})
    evidence = [
        {"category": "behavior", "signal": "isolation_forest", "field": "iforest_score",
         "value": 0.62, "description": "行为异常指数 A=0.62",
         "trace": {"camera_id": "cam01", "track_id": "7"}},
        {"category": "trajectory", "signal": "transformer_residual", "field": "traj_residual",
         "value": 23.6, "description": "Transformer 轨迹预测残差 23.6px",
         "trace": {"camera_id": "cam01", "track_id": "7"}},
    ]

    text = TemplateReportGenerator().generate(record, evidence)

    assert "偏离正常分布" not in text
    assert "建议重点核对" not in text
    assert "常规抽查" in text


def test_corroboration_rule_raises_multi_signal_vehicles():
    """开启印证规则后：≥2 个独立信号的车至少升到 medium；单信号不受影响。"""
    from src.risk.fusion import RiskFusion

    multi = {
        "iforest_score": 0.82, "graph_anomaly_index": 0.19, "time_freq_index": 0.0,
        "iforest_anomaly": 1, "traj_anomaly": 1, "alert_reason": "蛇形行驶(平均转角0.75)",
    }
    single = {
        "iforest_score": 0.62, "graph_anomaly_index": 0.30, "time_freq_index": 0.0,
        "iforest_anomaly": 1, "traj_anomaly": 0, "alert_reason": "iforest_score=0.62",
    }

    default = RiskFusion()
    assert default.fuse_one(dict(multi))["risk_level"] == "low"

    enabled = RiskFusion(corroboration=True)
    raised = enabled.fuse_one(dict(multi))
    assert raised["risk_level"] == "medium"
    assert raised["risk_score_100"] == 50.0
    assert raised["corroboration_signals"] == 3  # IF + Transformer + 规则

    untouched = enabled.fuse_one(dict(single))
    assert untouched["risk_level"] == "low"
    assert untouched["corroboration_signals"] == 1


def test_detector_score_tokens_do_not_count_as_rule_signal():
    """alert_reason 里的 iforest_score=... / traj_residual=... 不算独立规则信号。"""
    from src.risk.fusion import RiskFusion

    record = {"iforest_score": 0.7, "iforest_anomaly": 1, "traj_anomaly": 0,
              "alert_reason": "iforest_score=0.70;traj_residual=120.0px"}
    assert RiskFusion.count_independent_signals(record) == 1
