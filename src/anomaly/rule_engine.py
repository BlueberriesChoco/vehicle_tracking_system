from typing import List, Dict


class RuleEngine:
    """规则引擎（Phase 3 启用）。

    基于硬阈值对行为向量做规则判定：
    - 超速：avg_speed_ms > threshold
    - 长时间滞留：max_dwell_sec > threshold
    - 逆行：path_deviation 超过正常范围
    - 夜间异常通行：is_night + 停留过长
    """

    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "speed_max_kmh": 40.0,
            "dwell_max_sec": 120.0,
            "deviation_max_m": 3.0,
        }

    def check(self, vector: dict) -> List[str]:
        """检查行为向量，返回触发的告警原因列表。"""
        reasons = []

        speed_kmh = vector.get("avg_speed_ms", 0) * 3.6
        if speed_kmh > self.thresholds.get("speed_max_kmh", 40):
            reasons.append(f"超速({speed_kmh:.1f}km/h)")

        if vector.get("max_dwell_sec", 0) > self.thresholds.get("dwell_max_sec", 120):
            reasons.append(f"长时间滞留({vector['max_dwell_sec']:.0f}s)")

        if vector.get("path_deviation", 0) > self.thresholds.get("deviation_max_m", 3):
            reasons.append(f"路径偏离({vector['path_deviation']:.1f}m)")

        if vector.get("is_night", 0) and vector.get("stop_count", 0) > 2:
            reasons.append("夜间频繁停留")

        return reasons
