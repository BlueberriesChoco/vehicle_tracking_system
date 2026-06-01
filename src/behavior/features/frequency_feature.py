from typing import Optional, Dict
from collections import defaultdict


class FrequencyFeatureExtractor:
    """高频通行特征提取器。

    统计车辆（按车牌/ReID标识）在时间窗口内出现频次，
    输出归一化后的高频通行指数。
    """

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        # vehicle_key -> 出现次数
        self._frequency_map: Dict[str, int] = defaultdict(int)
        self._total_vehicles: int = 0

    def register_vehicle(self, vehicle_key: str):
        """注册一次车辆通行事件。

        Args:
            vehicle_key: 车辆唯一标识（车牌哈希 或 临时 track_id）
        """
        self._frequency_map[vehicle_key] += 1
        self._total_vehicles += 1

    def compute_freq_index(self, vehicle_key: str) -> float:
        """计算高频通行指数（Z-score归一化）。

        指数 > 0 表示高于平均通行频次，指数越高越可疑。
        """
        if self._total_vehicles == 0 or len(self._frequency_map) == 0:
            return 0.0

        count = self._frequency_map.get(vehicle_key, 1)

        # 计算均值和标准差
        all_counts = list(self._frequency_map.values())
        mean = sum(all_counts) / len(all_counts)

        if len(all_counts) <= 1:
            return 0.0

        variance = sum((c - mean) ** 2 for c in all_counts) / len(all_counts)

        if variance == 0:
            return 0.0

        freq_index = (count - mean) / (variance ** 0.5)
        return round(freq_index, 3)

    def get_count(self, vehicle_key: str) -> int:
        return self._frequency_map.get(vehicle_key, 0)

    def reset(self):
        """Clear counters when starting an independent video segment."""
        self._frequency_map.clear()
        self._total_vehicles = 0
