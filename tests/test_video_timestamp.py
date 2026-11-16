"""录像起始时间解析的回归测试。

历史问题：文件名没有时间戳时，`_parse_video_start_time` 返回 None，
调用方回退到 `datetime.now()` —— 同一段录像在白天或晚上跑会得到不同的夜间判定。
现在改为固定占位时间，并用 `time_known=0` 禁用夜间判断。
"""

from datetime import datetime

from src.pipeline.batch_processor import BatchProcessor


def test_parse_start_time_from_standard_filename():
    name = "Download_南京南通海关综保A区行政车道进_20260901080110_20260901090712.h264"

    parsed = BatchProcessor._parse_video_start_time(name)

    assert parsed == datetime(2026, 9, 1, 8, 1, 10)


def test_parse_start_time_returns_none_for_untimed_clip():
    assert BatchProcessor._parse_video_start_time("clip_08_short.h264") is None


def test_parse_start_time_returns_none_for_garbage_name():
    assert BatchProcessor._parse_video_start_time("video.mp4") is None
