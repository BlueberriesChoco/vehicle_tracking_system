import csv
import os
from typing import Iterable


class AlertWriter:
    """Write Phase 3 anomaly alerts as a reviewable CSV."""

    CSV_COLUMNS = [
        "alert_id",
        "camera_id",
        "track_id",
        "global_vehicle_id",
        "event_time",
        "severity",
        "anomaly_score",
        "is_anomaly",
        "alert_reason",
        "vehicle_type",
        "vehicle_group",
        "analysis_target",
        "vehicle_color",
        "plate_number",
        "plate_confidence",
        "baseline_feature_count",
        "iforest_score",
        "iforest_anomaly",
        "traj_residual",
        "traj_anomaly",
    ]

    def write(self, records: Iterable[dict], filepath: str, anomaly_only: bool = True) -> None:
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        rows = [self._row(record, index) for index, record in enumerate(records)]
        if anomaly_only:
            rows = [row for row in rows if str(row.get("is_anomaly", "0")) in {"1", "true", "True"}]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _row(self, record: dict, index: int) -> dict:
        event_time = record.get("event_time") or record.get("exit_time") or record.get("enter_time") or ""
        alert_id = f"alert_{record.get('camera_id', 'cam')}_{record.get('track_id', index)}_{index}"
        return {
            "alert_id": alert_id,
            "camera_id": record.get("camera_id", ""),
            "track_id": record.get("track_id", ""),
            "global_vehicle_id": record.get("global_vehicle_id", ""),
            "event_time": event_time,
            "severity": self._severity(record),
            "anomaly_score": record.get("anomaly_score", 0.0),
            "is_anomaly": record.get("is_anomaly", 0),
            "alert_reason": record.get("alert_reason", ""),
            "vehicle_type": record.get("vehicle_type", ""),
            "vehicle_group": record.get("vehicle_group", ""),
            "analysis_target": record.get("analysis_target", ""),
            "vehicle_color": record.get("vehicle_color", ""),
            "plate_number": record.get("plate_number", ""),
            "plate_confidence": record.get("plate_confidence", ""),
            "baseline_feature_count": record.get("baseline_feature_count", ""),
            "iforest_score": record.get("iforest_score", ""),
            "iforest_anomaly": record.get("iforest_anomaly", ""),
            "traj_residual": record.get("traj_residual", ""),
            "traj_anomaly": record.get("traj_anomaly", ""),
        }

    @staticmethod
    def _severity(record: dict) -> str:
        try:
            score = float(record.get("anomaly_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score >= 5.0:
            return "high"
        if score >= 3.0:
            return "medium"
        return "low"
