from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    reliability_field: Optional[str] = None


# Phase3 只使用可解释的行为字段建基线；带 reliability_field 的字段会自动跳过未标定样本。
DEFAULT_FEATURES = [
    FeatureSpec("avg_speed_ms", "speed_reliable"),
    FeatureSpec("max_speed_ms", "speed_reliable"),
    FeatureSpec("speed_variance", "speed_reliable"),
    FeatureSpec("max_dwell_sec"),
    FeatureSpec("stop_count"),
    FeatureSpec("dwell_ratio"),
    FeatureSpec("path_deviation", "path_reliable"),
    FeatureSpec("path_smoothness", "path_reliable"),
    FeatureSpec("is_night"),
    FeatureSpec("night_ratio"),
    FeatureSpec("freq_index"),
    FeatureSpec("freq_count_24h"),
    FeatureSpec("aggregation_index", "aggregation_reliable"),
    FeatureSpec("nearest_vehicle_m", "aggregation_reliable"),
]


class BaselineStore:
    """Load and save Phase 3 normal behavior baselines."""

    @staticmethod
    def save(baseline: dict, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(baseline, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(filepath: str) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


class BaselineBuilder:
    """Build and apply camera-level normal behavior baselines from vector CSV rows."""

    def __init__(self, features: Optional[List[FeatureSpec]] = None, target_groups: Optional[List[str]] = None):
        self.features = features or DEFAULT_FEATURES
        self.target_groups = set(target_groups or ["ordinary"])

    def load_vectors(self, vector_dir: str, recursive: bool = True) -> List[dict]:
        pattern = os.path.join(vector_dir, "**", "vec_*.csv") if recursive else os.path.join(vector_dir, "vec_*.csv")
        rows = []
        for path in sorted(glob.glob(pattern, recursive=recursive)):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["_source_file"] = path
                    rows.append(row)
        return rows

    def build(self, vectors: Iterable[dict], camera_id: str = "unknown") -> dict:
        # 行政通道基线默认只使用普通车辆，避免货车通行模式污染正常范围。
        vectors = self.filter_analysis_targets(vectors)
        stats = {}
        for spec in self.features:
            values = self._valid_values(vectors, spec)
            stats[spec.name] = self._stats(values)
            if spec.reliability_field:
                stats[spec.name]["reliability_field"] = spec.reliability_field

        return {
            "version": 1,
            "kind": "phase3_behavior_baseline",
            "camera_id": camera_id,
            "target_groups": sorted(self.target_groups),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "sample_count": len(vectors),
            "feature_columns": [spec.name for spec in self.features],
            "stats": stats,
            "scoring": {
                "method": "max_abs_zscore",
                "default_threshold": 3.0,
                "min_feature_count": 3,
            },
        }

    def score(self, vector: dict, baseline: dict, threshold: Optional[float] = None) -> dict:
        threshold = threshold or baseline.get("scoring", {}).get("default_threshold", 3.0)
        feature_scores = {}
        reasons = []

        for spec in self.features:
            stat = baseline.get("stats", {}).get(spec.name, {})
            if not stat or stat.get("count", 0) <= 0:
                continue
            # 未标定摄像头的速度/路径/聚集字段不参与统计评分。
            if not self._is_reliable(vector, spec):
                continue
            value = self._to_float(vector.get(spec.name))
            if value is None:
                continue
            std = float(stat.get("std") or 1.0)
            if std == 0:
                std = 1.0
            zscore = round(float((value - float(stat.get("mean", 0.0))) / std), 3)
            feature_scores[spec.name] = zscore
            if abs(zscore) >= threshold:
                reasons.append(f"{spec.name}_zscore={zscore}")

        max_abs = max((abs(score) for score in feature_scores.values()), default=0.0)
        min_count = baseline.get("scoring", {}).get("min_feature_count", 3)
        enough_features = len(feature_scores) >= min_count
        return {
            "baseline_score": round(max_abs, 3),
            "baseline_feature_count": len(feature_scores),
            "baseline_anomaly": int(enough_features and max_abs >= threshold),
            "baseline_reasons": reasons if enough_features else [],
            "feature_zscores": feature_scores,
        }

    def filter_analysis_targets(self, vectors: Iterable[dict]) -> List[dict]:
        return [vector for vector in vectors if self.is_analysis_target(vector)]

    def is_analysis_target(self, vector: dict) -> bool:
        # 新 CSV 优先使用 analysis_target；兼容旧 CSV 时再退回 vehicle_group。
        explicit = vector.get("analysis_target")
        if explicit is not None and str(explicit).strip() != "":
            return str(explicit).strip().lower() in {"1", "true", "yes"}
        group = str(vector.get("vehicle_group", "ordinary")).strip().lower()
        return group in self.target_groups

    def _valid_values(self, vectors: List[dict], spec: FeatureSpec) -> np.ndarray:
        values = []
        for vector in vectors:
            # 未标定摄像头的速度/路径/聚集字段不参与统计评分。
            if not self._is_reliable(vector, spec):
                continue
            value = self._to_float(vector.get(spec.name))
            if value is not None and np.isfinite(value):
                values.append(value)
        return np.array(values, dtype=np.float64)

    @staticmethod
    def _stats(values: np.ndarray) -> dict:
        if values.size == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 1.0,
                "median": 0.0,
                "q1": 0.0,
                "q3": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        std = float(np.std(values))
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": std if std > 0 else 1.0,
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    @staticmethod
    def _is_reliable(vector: dict, spec: FeatureSpec) -> bool:
        if not spec.reliability_field:
            return True
        value = vector.get(spec.reliability_field, 1)
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
