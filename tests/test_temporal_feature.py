"""时间特征提取的边界测试（夜间时长计算）。"""

from datetime import datetime, timedelta

from src.behavior.features.temporal_feature import TemporalFeatureExtractor


def _dt(day, hour, minute=0):
    return datetime(2026, 9, day, hour, minute, 0)


def test_night_overlap_short_passage():
    extractor = TemporalFeatureExtractor()

    overlap = extractor._compute_night_overlap(_dt(1, 23, 0), _dt(1, 23, 0) + timedelta(seconds=30))

    assert overlap == 30.0


def test_night_overlap_crossing_midnight():
    extractor = TemporalFeatureExtractor()

    # 23:00 -> 次日 06:30：06:00 之后已是白天，应计 7 小时
    overlap = extractor._compute_night_overlap(_dt(1, 23, 0), _dt(2, 6, 30))

    assert overlap == 7.0 * 3600


def test_night_overlap_early_morning():
    extractor = TemporalFeatureExtractor()

    # 05:00 -> 07:00 只有 05:00-06:00 属于夜间
    overlap = extractor._compute_night_overlap(_dt(1, 5, 0), _dt(1, 7, 0))

    assert overlap == 3600.0


def test_night_overlap_stops_at_six_am():
    """23:00 -> 08:00 只应计算到 06:00，即 7 小时（不是 8 小时）。"""
    extractor = TemporalFeatureExtractor()

    overlap = extractor._compute_night_overlap(_dt(1, 23, 0), _dt(2, 8, 0))

    assert overlap == 7.0 * 3600


def test_night_overlap_daytime_is_zero():
    extractor = TemporalFeatureExtractor()

    overlap = extractor._compute_night_overlap(_dt(1, 12, 0), _dt(1, 12, 10))

    assert overlap == 0.0


def test_night_overlap_partial_evening():
    extractor = TemporalFeatureExtractor()

    # 21:00 -> 23:00 只有 22:00-23:00 属于夜间
    overlap = extractor._compute_night_overlap(_dt(1, 21, 0), _dt(1, 23, 0))

    assert overlap == 3600.0


def test_is_night_flag_uses_majority_ratio():
    extractor = TemporalFeatureExtractor()

    late_evening = extractor.extract(_dt(1, 21, 30), _dt(1, 21, 31))
    night = extractor.extract(_dt(1, 23, 0), _dt(1, 23, 0) + timedelta(seconds=5))

    assert late_evening["is_night"] == 0
    assert night["is_night"] == 1
