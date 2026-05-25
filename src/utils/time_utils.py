from datetime import datetime, time, timedelta
from typing import Tuple


def is_night_time(dt: datetime, night_start: int = 22, night_end: int = 6) -> bool:
    """判断给定时间是否处于夜间时段。

    Args:
        dt: 日期时间
        night_start: 夜间开始小时 (22)
        night_end: 夜间结束小时 (6)

    Returns:
        True if night time
    """
    hour = dt.hour
    if night_start < night_end:
        return night_start <= hour < night_end
    else:
        return hour >= night_start or hour < night_end


def compute_overlap_seconds(
    start1: datetime, end1: datetime,
    start2: datetime, end2: datetime,
) -> float:
    """计算两个时间段的交叠秒数。"""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0.0, (overlap_end - overlap_start).total_seconds())


def parse_timestamp(frame_idx: int, base_time: datetime, fps: int = 25) -> datetime:
    """根据帧号和基准时间计算实际时间戳。"""
    offset_sec = frame_idx / fps
    return base_time + timedelta(seconds=offset_sec)


def format_duration(seconds: float) -> str:
    """格式化时长字符串。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


def get_hour_bucket(dt: datetime) -> str:
    """获取小时分桶标签。"""
    return dt.strftime("%Y%m%d_%H")
