from typing import List
import numpy as np


class DwellFeatureExtractor:
    """停留特征提取器。

    根据瞬时速度序列识别车辆在通道内的停留行为：
    - 停留判定：连续N帧速度 < threshold → 一个停留段
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
        self.min_dwell_frames = int(min_dwell_duration_sec * frame_rate)

    def extract(self, instant_speeds: List[float]) -> dict:
        """从瞬时速度序列中提取停留特征。

        Args:
            instant_speeds: 瞬时速度列表 (m/s)

        Returns:
            dict with max_dwell_sec, stop_count, dwell_ratio
        """
        if not instant_speeds:
            return {
                "max_dwell_sec": 0.0,
                "stop_count": 0,
                "dwell_ratio": 0.0,
                "dwell_segments": [],
            }

        # 识别停留段
        dwell_segments = []
        in_dwell = False
        dwell_start = 0

        for i, speed in enumerate(instant_speeds):
            if speed < self.dwell_threshold_ms and not in_dwell:
                in_dwell = True
                dwell_start = i
            elif speed >= self.dwell_threshold_ms and in_dwell:
                in_dwell = False
                duration_frames = i - dwell_start
                if duration_frames >= self.min_dwell_frames:
                    dwell_segments.append({
                        "start_frame": dwell_start,
                        "end_frame": i,
                        "duration_frames": duration_frames,
                        "duration_sec": duration_frames / self.frame_rate,
                    })

        # 处理末尾仍在停留的情况
        if in_dwell:
            duration_frames = len(instant_speeds) - dwell_start
            if duration_frames >= self.min_dwell_frames:
                dwell_segments.append({
                    "start_frame": dwell_start,
                    "end_frame": len(instant_speeds),
                    "duration_frames": duration_frames,
                    "duration_sec": duration_frames / self.frame_rate,
                })

        total_dwell_frames = sum(seg["duration_frames"] for seg in dwell_segments)
        dwell_ratio = total_dwell_frames / len(instant_speeds) if instant_speeds else 0.0

        max_dwell_sec = max(
            (seg["duration_sec"] for seg in dwell_segments), default=0.0
        )

        return {
            "max_dwell_sec": round(max_dwell_sec, 2),
            "stop_count": len(dwell_segments),
            "dwell_ratio": round(dwell_ratio, 4),
            "dwell_segments": dwell_segments,
        }
