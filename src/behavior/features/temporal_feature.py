from datetime import datetime, time


class TemporalFeatureExtractor:
    """时间特征提取器。

    判断车辆的通行时段，计算夜间通行比例。
    """

    def __init__(self, night_start_hour: int = 22, night_end_hour: int = 6):
        self.night_start = night_start_hour
        self.night_end = night_end_hour

    def extract(
        self,
        enter_time: datetime,
        exit_time: datetime,
    ) -> dict:
        """提取时间特征。

        Args:
            enter_time: 进入通道时间
            exit_time: 离开通道时间

        Returns:
            dict with is_night, night_ratio, hour_of_day
        """
        duration_sec = (exit_time - enter_time).total_seconds()
        if duration_sec <= 0:
            return {
                "is_night": 0,
                "night_ratio": 0.0,
                "hour_of_day": enter_time.hour,
            }

        # 计算夜间重叠时长
        night_overlap_sec = self._compute_night_overlap(enter_time, exit_time)

        night_ratio = night_overlap_sec / duration_sec
        is_night = 1 if night_ratio > 0.5 else 0

        return {
            "is_night": is_night,
            "night_ratio": round(night_ratio, 4),
            "hour_of_day": enter_time.hour,
        }

    def _compute_night_overlap(self, start: datetime, end: datetime) -> float:
        """计算时间段与夜间时段的交叠秒数。"""
        total = 0.0
        current = start

        while current < end:
            if self._is_night_hour(current.hour):
                # 找到夜间结束时间
                next_boundary = current.replace(
                    minute=0, second=0, microsecond=0
                )
                if current.hour >= self.night_start:
                    next_boundary = next_boundary + self._hours_to_timedelta(
                        24 - self.night_start + self.night_end
                    )
                else:
                    next_boundary = next_boundary.replace(hour=self.night_end)

                segment_end = min(next_boundary, end)
                total += (segment_end - current).total_seconds()
                current = segment_end
            else:
                # 跳到下一个夜间开始或结束
                next_start = current.replace(minute=0, second=0, microsecond=0)
                if current.hour < self.night_start:
                    next_start = next_start.replace(hour=self.night_start)
                else:
                    next_start = next_start + self._hours_to_timedelta(
                        24 - self.night_end + self.night_start
                    )
                current = min(next_start, end)

        return total

    def _is_night_hour(self, hour: int) -> bool:
        if self.night_start < self.night_end:
            return self.night_start <= hour < self.night_end
        else:
            # 跨日夜间 (22:00 ~ 06:00)
            return hour >= self.night_start or hour < self.night_end

    @staticmethod
    def _hours_to_timedelta(hours: int):
        from datetime import timedelta
        return timedelta(hours=hours)
