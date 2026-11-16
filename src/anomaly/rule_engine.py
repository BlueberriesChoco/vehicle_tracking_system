from typing import Dict, List, Optional


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


class RuleEngine:
    """Rule-based anomaly checks for Phase 3 behavior review.

    规则覆盖“已知异常类型”，与无监督模型互补：
    对于训练集中恒定的特征（例如从未出现过的夜间通行），Isolation Forest
    无法在训练时分裂，因此必须由规则兜底。
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        defaults = {
            "speed_max_kmh": 40.0,
            "dwell_max_sec": 120.0,
            "deviation_max_m": 3.0,
            "night_stop_count": 2,
            "night_high_freq_count": 6.0,
            "night_any_passage": 1,
            "snake_smoothness_max": 0.5,
        }
        if thresholds:
            defaults.update(thresholds)
        self.thresholds = defaults

    def check(self, vector: dict) -> List[str]:
        reasons = []

        if _is_truthy(vector.get("speed_reliable", 1)):
            speed_kmh = _to_float(vector.get("avg_speed_ms")) * 3.6
            if speed_kmh > self.thresholds.get("speed_max_kmh", 40.0):
                reasons.append(f"超速({speed_kmh:.1f}km/h)")

        dwell_sec = _to_float(vector.get("max_dwell_sec"))
        if dwell_sec > self.thresholds.get("dwell_max_sec", 120.0):
            reasons.append(f"长时间滞留({dwell_sec:.0f}s)")

        if _is_truthy(vector.get("path_reliable", 1)):
            deviation = _to_float(vector.get("path_deviation"))
            if deviation > self.thresholds.get("deviation_max_m", 3.0):
                reasons.append(f"路径偏离({deviation:.1f}m)")

        # 蛇形判定看“平均转角”：值越大越犹豫（见 PathReference.compute_smoothness）。
        # 角度在像素空间也有意义（相似变换下角度不变），因此不依赖几何标定。
        smoothness = _to_float(vector.get("path_smoothness"), 0.0)
        if smoothness >= self.thresholds.get("snake_smoothness_max", 0.5):
            reasons.append(f"蛇形行驶(平均转角{smoothness:.2f})")

        if _is_truthy(vector.get("is_night", 0)):
            stop_count = _to_float(vector.get("stop_count"))
            if stop_count > self.thresholds.get("night_stop_count", 2):
                reasons.append("夜间频繁停留")

            freq_count = _to_float(vector.get("freq_count_24h"))
            if freq_count >= self.thresholds.get("night_high_freq_count", 6.0):
                reasons.append(f"夜间高频通行({freq_count:.0f}次)")
            elif _is_truthy(self.thresholds.get("night_any_passage", 1)):
                # 该行政车道夜间实测基本无车（00:00 与 20:57 两个时段 ROI 内检出均为 0），
                # 因此“夜间出现”本身就是值得核查的信号，不必等到高频。
                reasons.append("夜间通行")

        return reasons
