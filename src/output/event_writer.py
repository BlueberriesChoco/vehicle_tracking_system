import csv
import os
import re
from typing import Iterable, List


class EventWriter:
    """Write one passage event for each completed vehicle trajectory."""

    CSV_COLUMNS = [
        "event_id",
        "camera_id",
        "track_id",
        "global_vehicle_id",
        "event_type",
        "event_time",
        "enter_time",
        "exit_time",
        "duration_sec",
        "vehicle_type",
        "vehicle_group",
        "analysis_target",
        "vehicle_color",
        "plate_number",
        "plate_hash",
        "plate_confidence",
        "geometry_level",
        "speed_reliable",
        "path_reliable",
        "aggregation_reliable",
        "passage_reliable",
        "avg_speed_ms",
        "max_speed_ms",
        "max_dwell_sec",
        "stop_count",
        "dwell_ratio",
        "trajectory_length_m",
        "aggregation_index",
        "nearest_vehicle_m",
        "reid_available",
        "anomaly_score",
        "is_anomaly",
        "alert_reason",
    ]

    EVENT_TYPE = "vehicle_passage"

    def write(self, vectors: List[dict], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        events = self.build_events(vectors)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(events)

    def build_events(self, vectors: Iterable[dict]) -> List[dict]:
        events = []
        for index, vector in enumerate(vectors):
            event_time = vector.get("exit_time") or vector.get("enter_time") or ""
            event = {
                "event_id": self._event_id(vector, event_time, index),
                "camera_id": vector.get("camera_id", ""),
                "track_id": vector.get("track_id", ""),
                "global_vehicle_id": vector.get("global_vehicle_id", ""),
                "event_type": self.EVENT_TYPE,
                "event_time": event_time,
                "enter_time": vector.get("enter_time", ""),
                "exit_time": vector.get("exit_time", ""),
                "duration_sec": vector.get("duration_sec", ""),
                "vehicle_type": vector.get("vehicle_type", ""),
                "vehicle_group": vector.get("vehicle_group", ""),
                "analysis_target": vector.get("analysis_target", ""),
                "vehicle_color": vector.get("vehicle_color", ""),
                "plate_number": vector.get("plate_number", ""),
                "plate_hash": vector.get("plate_hash", ""),
                "plate_confidence": vector.get("plate_confidence", ""),
                "geometry_level": vector.get("geometry_level", ""),
                "speed_reliable": vector.get("speed_reliable", ""),
                "path_reliable": vector.get("path_reliable", ""),
                "aggregation_reliable": vector.get("aggregation_reliable", ""),
                "passage_reliable": vector.get("passage_reliable", ""),
                "avg_speed_ms": vector.get("avg_speed_ms", ""),
                "max_speed_ms": vector.get("max_speed_ms", ""),
                "max_dwell_sec": vector.get("max_dwell_sec", ""),
                "stop_count": vector.get("stop_count", ""),
                "dwell_ratio": vector.get("dwell_ratio", ""),
                "trajectory_length_m": vector.get("trajectory_length_m", ""),
                "aggregation_index": vector.get("aggregation_index", ""),
                "nearest_vehicle_m": vector.get("nearest_vehicle_m", ""),
                "reid_available": int(self._has_reid(vector.get("reid_embedding"))),
                "anomaly_score": vector.get("anomaly_score", ""),
                "is_anomaly": vector.get("is_anomaly", ""),
                "alert_reason": vector.get("alert_reason", ""),
            }
            events.append(event)
        return events

    @staticmethod
    def _event_id(vector: dict, event_time: str, index: int) -> str:
        raw = f"{vector.get('camera_id', 'cam')}_{vector.get('track_id', index)}_{event_time or index}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")

    @staticmethod
    def _has_reid(value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip()
            return bool(stripped and stripped != "[]")
        try:
            return len(value) > 0
        except TypeError:
            return bool(value)
