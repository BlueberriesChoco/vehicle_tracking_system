"""跨窗口通行频次统计（T 分量频次那一路的真实输入）的测试。

背景：``freq_count_24h`` 原本由 ``FrequencyFeatureExtractor`` 产生，
而它的计数器只在**单段视频内**累加、每段 ``reset()``。单窗口运行时每辆车
``passage_id`` 唯一，``get_count`` 恒返回 1，于是 T 恒为 0，γ 成为死权重。

修复思路：用跨窗口稳定的身份键重新统计次数；身份键覆盖率不足时如实标记
"不可用"，让融合层剔除该分量，而不是把"没有数据"当成"时间上很干净"。
"""

import pytest

from src.risk.repeat_frequency import (
    apply_passage_counts,
    count_passages,
    identity_key,
)


def _rec(**kwargs):
    base = {"camera_id": "admin_in", "track_id": "1", "freq_count_24h": 1}
    base.update(kwargs)
    return base


# -- 身份键层级 ---------------------------------------------------------------


def test_identity_key_prefers_plate_hash():
    key = identity_key(_rec(plate_hash="h1", global_vehicle_id="g1", plate_number="京A12345"))

    assert key == ("plate_hash", "h1")


def test_identity_key_falls_back_to_global_id():
    key = identity_key(_rec(plate_hash="", global_vehicle_id="g1", plate_number="京A12345"))

    assert key == ("global_vehicle_id", "g1")


def test_identity_key_uses_camera_scoped_plate_number():
    """原始车牌要带 camera，避免不同机位的同号车被并成一辆。"""
    key = identity_key(_rec(plate_hash="", plate_number="京A12345"))

    assert key == ("plate_number", "admin_in:京A12345")


def test_identity_key_returns_none_without_any_key():
    assert identity_key(_rec()) is None


# -- 计数 ---------------------------------------------------------------------


def test_counts_repeated_vehicle_across_windows():
    rows = [
        _rec(track_id="1", plate_hash="h1"),
        _rec(track_id="2", plate_hash="h1"),  # 同一辆车第二个窗口
        _rec(track_id="3", plate_hash="h2"),
    ]

    stats = count_passages(rows)

    assert stats["counts"] == [2, 2, 1]
    assert stats["distinct"] == 2
    assert stats["coverage"] == 1.0


def test_unknown_identity_counts_as_zero_not_one():
    """无法确定身份时填 0（未知），不能填 1（确认只来一次）。"""
    rows = [_rec(track_id="1"), _rec(track_id="2", plate_hash="h1")]

    stats = count_passages(rows)

    assert stats["counts"][0] == 0
    assert stats["coverage"] == pytest.approx(0.5)


def test_coverage_gate_marks_unavailable_when_plates_missing():
    """本项目实测情形：444 辆车里只有 2 辆有 plate_hash。"""
    rows = [_rec(track_id=str(i)) for i in range(10)]
    rows[0]["plate_hash"] = "h1"
    rows[5]["plate_hash"] = "h1"

    stats = count_passages(rows)

    assert stats["coverage"] == pytest.approx(0.2)
    assert stats["available"] is False


def test_dominant_key_level_reported():
    rows = [_rec(track_id=str(i), plate_hash=f"h{i}") for i in range(3)]
    rows.append(_rec(track_id="9", global_vehicle_id="g9"))

    stats = count_passages(rows)

    assert stats["key_level"] == "plate_hash"


# -- 写回 ---------------------------------------------------------------------


def test_apply_passage_counts_writes_when_coverage_sufficient():
    rows = [
        _rec(track_id="1", plate_hash="h1"),
        _rec(track_id="2", plate_hash="h1"),
        _rec(track_id="3", plate_hash="h2"),
    ]

    enriched, summary = apply_passage_counts(rows, min_coverage=0.5)

    assert summary["applied"] is True
    assert enriched[0]["freq_count_24h"] == 2
    assert enriched[0]["freq_count_source"] == "identity_key"
    assert enriched[2]["freq_count_24h"] == 1


def test_apply_passage_counts_keeps_original_when_coverage_low():
    """覆盖率不足时不能改写，否则会造出一堆假的"来了很多次"。"""
    rows = [
        _rec(track_id="1", plate_hash="h1", freq_count_24h=1),
        _rec(track_id="2", freq_count_24h=1),
        _rec(track_id="3", freq_count_24h=1),
        _rec(track_id="4", freq_count_24h=1),
    ]

    enriched, summary = apply_passage_counts(rows, min_coverage=0.5)

    assert summary["applied"] is False
    assert all(row["freq_count_24h"] == 1 for row in enriched)
    assert all(row["freq_count_source"] == "segment_local" for row in enriched)


def test_apply_passage_counts_annotates_coverage_on_every_row():
    rows = [_rec(track_id="1", plate_hash="h1"), _rec(track_id="2")]

    enriched, _ = apply_passage_counts(rows)

    assert all("freq_identity_coverage" in row for row in enriched)
    assert enriched[0]["freq_identity_coverage"] == pytest.approx(0.5)


def test_empty_input_is_safe():
    stats = count_passages([])
    assert stats["counts"] == []
    assert stats["distinct"] == 0

    enriched, summary = apply_passage_counts([])
    assert enriched == []
    assert summary["applied"] is False
