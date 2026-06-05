import csv
import json
import os
from typing import List


class VectorWriter:
    """行为向量输出器。

    将行为向量字典列表写入文件，支持 CSV 和 JSON 格式。
    按命名规范自动生成输出文件路径。
    """

    CSV_COLUMNS = [
        "track_id", "global_vehicle_id", "segment_count",
        "camera_id", "vehicle_type", "vehicle_color",
        "plate_number", "plate_hash",
        "geometry_level", "speed_reliable", "path_reliable",
        "aggregation_reliable", "passage_reliable",
        "enter_time", "exit_time", "duration_sec", "trajectory_length_m",
        "avg_speed_ms", "max_speed_ms", "speed_variance",
        "max_dwell_sec", "stop_count", "dwell_ratio",
        "path_deviation", "path_smoothness",
        "is_night", "night_ratio",
        "freq_index", "freq_count_24h",
        "aggregation_index", "nearest_vehicle_m",
        "trajectory_points",
        "reid_embedding",
        "anomaly_score", "is_anomaly", "alert_reason",
    ]

    def __init__(self, output_format: str = "csv"):
        self.output_format = output_format

    def write(self, vectors: List[dict], filepath: str):
        """写入行为向量到文件。

        Args:
            vectors: 行为向量字典列表
            filepath: 输出文件完整路径
        """
        if not vectors:
            self._write_empty(filepath)
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if self.output_format == "csv":
            self._write_csv(vectors, filepath)
        elif self.output_format == "json":
            self._write_json(vectors, filepath)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def _write_csv(self, vectors: List[dict], filepath: str):
        """写入 CSV 文件。"""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(vectors)

    def _write_json(self, vectors: List[dict], filepath: str):
        """写入 JSON 文件。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vectors, f, ensure_ascii=False, indent=2, default=str)

    def _write_empty(self, filepath: str):
        """输出空文件（无有效轨迹时）。"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
        """生成符合命名规范的输出文件名。

        Args:
            camera_id: 摄像头编号
            date_str: 日期字符串 YYYYMMDD
            hour_str: 小时字符串 HH
            prefix: 文件名前缀 ("vec" / "alert")

        Returns:
            文件名: vec_cam01_20260520_08.csv
        """
        return f"{prefix}_{camera_id}_{date_str}_{hour_str}.csv"
