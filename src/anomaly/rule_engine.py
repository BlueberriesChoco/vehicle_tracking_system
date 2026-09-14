from typing import Dict, List, Optional


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


class RuleEngine:
    """Rule-based anomaly checks for Phase 3 behavior review."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "speed_max_kmh": 40.0,
            "dwell_max_sec": 120.0,
            "deviation_max_m": 3.0,
            "night_stop_count": 2,
        }

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

        stop_count = _to_float(vector.get("stop_count"))
        if _is_truthy(vector.get("is_night", 0)) and stop_count > self.thresholds.get("night_stop_count", 2):
            reasons.append("夜间频繁停留")

        return reasons
