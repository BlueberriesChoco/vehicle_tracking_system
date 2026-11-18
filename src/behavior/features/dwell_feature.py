from datetime import datetime
from typing import List, Optional, Sequence


class DwellFeatureExtractor:
    """停留特征提取器。

    根据瞬时速度序列识别车辆在通道内的停留行为：
    - 停留判定：连续观测区间速度低于阈值，按观测间隔累计
    - 提取：停留次数、最大停留时长、停留占比
    """

    def __init__(
        self,
        dwell_threshold_ms: float = 0.5,
        min_dwell_duration_sec: float = 3.0,
        frame_rate: int = 25,
    ):
        self.dwell_threshold_ms = dwell_threshold_ms
        self.min_dwell_duration_sec = min_dwell_duration_sec
        self.frame_rate = frame_rate

    def extract(
        self,
        instant_speeds: List[float],
        timestamps: Optional[Sequence[datetime]] = None,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> dict:
        """从瞬时速度序列中提取停留特征。

        Args:
            instant_speeds: 瞬时速度列表 (m/s)

        Returns:
            dict with max_dwell_sec, stop_count, dwell_ratio
        """
        count = len(instant_speeds)
        if not count or (timestamps is not None and len(timestamps) < 2):
            return {
                "max_dwell_sec": 0.0,
                "stop_count": 0,
                "dwell_ratio": 0.0,
                "dwell_segments": [],
            }

        if timestamps is not None and len(timestamps) != count + 1:
            raise ValueError("Dwell timestamps must bound every speed interval")
        if frame_indices is not None and len(frame_indices) != count + 1:
            raise ValueError("Dwell frame indices must bound every speed interval")

        dwell_segments = []
        start = 0
        dwell_sec = 0.0
        observed_sec = 0.0
        for i in range(count + 1):
            if i < count:
                delta = (timestamps[i + 1] - timestamps[i]).total_seconds() if timestamps is not None else 1 / self.frame_rate
                contiguous = frame_indices is None or frame_indices[i + 1] == frame_indices[i] + 1
                if delta > 0 and contiguous:
                    observed_sec += delta
                    if instant_speeds[i] < self.dwell_threshold_ms:
                        dwell_sec += delta
                        continue
            if dwell_sec > 0 and dwell_sec + 1e-9 >= self.min_dwell_duration_sec:
                dwell_segments.append({
                    "start_frame": start,
                    "end_frame": i,
                    "duration_frames": i - start,
                    "duration_sec": dwell_sec,
                })
            start = i + 1
            dwell_sec = 0.0

        total_dwell_sec = sum(seg["duration_sec"] for seg in dwell_segments)
        dwell_ratio = total_dwell_sec / observed_sec if observed_sec else 0.0

        max_dwell_sec = max(
            (seg["duration_sec"] for seg in dwell_segments), default=0.0
        )

        return {
            "max_dwell_sec": round(max_dwell_sec, 2),
            "stop_count": len(dwell_segments),
            "dwell_ratio": round(dwell_ratio, 4),
            "dwell_segments": dwell_segments,
        }
