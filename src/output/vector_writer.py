import csv
import json
import os
from typing import List


class VectorWriter:
    """行为向量输出器。"""

    CSV_COLUMNS = [
        "track_id", "passage_id", "global_vehicle_id", "segment_count",
        "camera_id", "vehicle_type", "vehicle_group", "analysis_target", "vehicle_color",
        "plate_number", "plate_hash", "plate_confidence",
        "geometry_level", "speed_reliable", "path_reliable",
        "aggregation_reliable", "passage_reliable",
        "enter_time", "exit_time", "duration_sec", "trajectory_length_m",
        "avg_speed_ms", "max_speed_ms", "speed_variance", "speed_cv",
        "max_dwell_sec", "stop_count", "dwell_ratio",
        "path_deviation", "path_smoothness",
        "is_night", "night_ratio", "time_known",
        "freq_index", "freq_count_24h",
        "aggregation_index", "nearest_vehicle_m", "max_concurrent_vehicles",
        "trajectory_points",
        "reid_embedding",
        "anomaly_score", "is_anomaly", "alert_reason",
        "iforest_score", "iforest_anomaly", "traj_residual", "traj_anomaly",
    ]

    def __init__(self, output_format: str = "csv"):
        self.output_format = output_format

    def write(self, vectors: List[dict], filepath: str):
        if not vectors:
            self._write_empty(filepath)
            return

        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.output_format == "csv":
            self._write_csv(vectors, filepath)
        elif self.output_format == "json":
            self._write_json(vectors, filepath)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def _write_csv(self, vectors: List[dict], filepath: str):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(vectors)

    def _write_json(self, vectors: List[dict], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vectors, f, ensure_ascii=False, indent=2, default=str)

    def _write_empty(self, filepath: str):
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if self.output_format == "csv":
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)
        elif self.output_format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([], f)

    @staticmethod
    def generate_filename(
        camera_id: str,
        date_str: str,
        hour_str: str,
        prefix: str = "vec",
    ) -> str:
        return f"{prefix}_{camera_id}_{date_str}_{hour_str}.csv"
