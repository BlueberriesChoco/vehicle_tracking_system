"""非监管对象（analysis_target=0）不得进入风险评分体系的测试。

背景：Phase1/2 已明确"货车不参与行政通道异常判断"，``build_vehicle_graph``
也正确排除了它们（所以这些车没有图谱节点）。但风险报告阶段没有同样的过滤，
货车/两轮车仍被写入 ``risk_admin_in.json`` 并生成报告正文：

    综合风险 0.0 分（低风险），风险较低，未发现需要优先处置的异常，可作为常规抽查对象。

这些车实际上**根本没参与评分**（A/G/T 全 0，无 iforest、无规则），
被陈述成"已评估且正常"是把"未参与"错述成"已排查"。
"""

import importlib.util
import os
import sys

import pytest

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "build_risk_report.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_risk_report_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_module()


def _row(**kwargs):
    base = {
        "camera_id": "admin_in",
        "track_id": "1",
        "vehicle_group": "car",
        "analysis_target": 1,
        "risk_score_100": 42.0,
    }
    base.update(kwargs)
    return base


def test_analysis_target_defaults_to_target(module):
    assert module._is_analysis_target({"analysis_target": 1}) is True
    assert module._is_analysis_target({"analysis_target": "1"}) is True


def test_missing_field_is_treated_as_target(module):
    """旧数据没有 analysis_target 字段，按监管对象处理，不能误删。"""
    assert module._is_analysis_target({}) is True


def test_freight_is_not_analysis_target(module):
    assert module._is_analysis_target({"analysis_target": 0}) is False
    assert module._is_analysis_target({"analysis_target": "0"}) is False


def test_split_keeps_targets_and_separates_others(module):
    rows = [
        _row(track_id="1"),
        _row(track_id="2", analysis_target=0, vehicle_group="freight"),
        _row(track_id="3"),
        _row(track_id="4", analysis_target=0, vehicle_group="two_wheeler"),
    ]

    kept, excluded = module._split_non_target(rows)

    assert len(kept) == 2
    assert len(excluded) == 2
    assert {row["track_id"] for row in kept} == {"1", "3"}


def test_excluded_rows_get_not_evaluated_status(module):
    rows = [_row(track_id="2", analysis_target=0, vehicle_group="freight", risk_score_100=0.0)]

    _, excluded = module._split_non_target(rows)

    row = excluded[0]
    assert row["risk_level"] == "not_evaluated"
    assert row["risk_score_100"] == ""
    assert "未参与风险评分" in row["exclusion_reason"]


def test_excluded_report_does_not_claim_normal(module):
    """核心断言：不能再说"风险较低、未发现异常"。"""
    rows = [_row(track_id="2", analysis_target=0, vehicle_group="freight")]

    _, excluded = module._split_non_target(rows)

    text = excluded[0]["report_text"]
    assert "非监管对象" in text
    assert "没有风险分" in text
    assert "风险较低" not in text
    assert "未发现需要优先处置的异常" not in text
    assert "常规抽查对象" not in text


def test_excluded_report_names_the_vehicle_group(module):
    rows = [_row(track_id="2", analysis_target=0, vehicle_group="two_wheeler")]

    _, excluded = module._split_non_target(rows)

    assert "two_wheeler" in excluded[0]["report_text"]


def test_split_is_noop_when_all_are_targets(module):
    rows = [_row(track_id=str(i)) for i in range(3)]

    kept, excluded = module._split_non_target(rows)

    assert len(kept) == 3
    assert excluded == []
