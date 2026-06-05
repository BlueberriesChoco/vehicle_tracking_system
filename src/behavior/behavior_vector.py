from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from ..tracking.tracklet import Tracklet
from ..trajectory.scene_geometry import SceneGeometry
from ..trajectory.trajectory_extractor import TrajectoryExtractor
from ..trajectory.trajectory_smoother import TrajectorySmoother
from ..trajectory.path_reference import PathReference

from .features.speed_feature import SpeedFeatureExtractor
from .features.dwell_feature import DwellFeatureExtractor
from .features.path_feature import PathFeatureExtractor
from .features.temporal_feature import TemporalFeatureExtractor
from .features.frequency_feature import FrequencyFeatureExtractor
from .features.aggregation_feature import AggregationFeatureExtractor
from .feature_normalizer import FeatureNormalizer


class BehaviorVectorizer:
    """行为向量化主入口。

    对每辆完成通道穿越的车辆，调用各特征提取器，
    构建标准化的行为向量 V = [v1, v2, ..., vn]。

    Phase 2 新增：车身颜色识别、车型细分、车牌检测+OCR、ReID 特征提取。
    """

    VECTOR_LABELS = [
        "avg_speed_ms",
        "max_speed_ms",
        "speed_variance",
        "max_dwell_sec",
        "stop_count",
        "dwell_ratio",
        "path_deviation",
        "path_smoothness",
        "is_night",
        "night_ratio",
        "freq_index",
        "aggregation_index",
    ]

    def __init__(
        self,
        scene_geometry: SceneGeometry,
        trajectory_extractor: TrajectoryExtractor,
        smoother: TrajectorySmoother,
        path_reference: PathReference,
        frame_rate: int = 25,
        night_start_hour: int = 22,
        night_end_hour: int = 6,
        dwell_threshold_ms: float = 0.5,
        dwell_min_duration_sec: float = 3.0,
        aggregation_radius_m: float = 5.0,
        freq_window_hours: int = 24,
        # Phase 2 提取器（可选注入）
        color_recognizer=None,
        vehicle_classifier=None,
        plate_detector=None,
        plate_ocr=None,
        reid_extractor=None,
    ):
        self.scene = scene_geometry
        self.trajectory_extractor = trajectory_extractor
        self.smoother = smoother
        self.frame_rate = frame_rate

        # 各特征提取器
        self.speed_extractor = SpeedFeatureExtractor(
            scene_geometry, frame_rate, dwell_threshold_ms
        )
        self.dwell_extractor = DwellFeatureExtractor(
            dwell_threshold_ms, dwell_min_duration_sec, frame_rate
        )
        self.path_extractor = PathFeatureExtractor(path_reference, scene_geometry)
        self.temporal_extractor = TemporalFeatureExtractor(night_start_hour, night_end_hour)
        self.frequency_extractor = FrequencyFeatureExtractor(freq_window_hours)
        self.aggregation_extractor = AggregationFeatureExtractor(
            scene_geometry, aggregation_radius_m
        )

        # Phase 2 提取器
        self.color_recognizer = color_recognizer
        self.vehicle_classifier = vehicle_classifier
        self.plate_detector = plate_detector
        self.plate_ocr = plate_ocr
        self.reid_extractor = reid_extractor

        self.normalizer = FeatureNormalizer(method="zscore")

    def vectorize(
        self,
        tracklet: Tracklet,
        all_active_positions: Optional[Dict[int, Tuple[float, float]]] = None,
        position_histories: Optional[Dict[int, Dict[int, Tuple[float, float]]]] = None,
    ) -> dict:
        """对一条轨迹构建完整行为向量。

        Args:
            tracklet: 轨迹对象（含 best_crop）
            all_active_positions: 同时段其他车辆的位置

        Returns:
            dict: 包含所有特征的字典 + 元数据
        """
        if all_active_positions is None:
            all_active_positions = {}

        # 1. 获取平滑轨迹
        xs_smooth, ys_smooth = self.trajectory_extractor.get_smoothed_trajectory(tracklet)
        centers_smooth = [(float(xs_smooth[i]), float(ys_smooth[i])) for i in range(len(xs_smooth))]

        # 2. 速度特征
        speed_features = self.speed_extractor.extract(
            xs_smooth, ys_smooth, tracklet.timestamps
        )

        # 3. 停留特征
        dwell_features = self.dwell_extractor.extract(speed_features["instant_speeds"])

        # 4. 路径特征
        path_features = self.path_extractor.extract(centers_smooth)

        # 5. 时间特征
        enter_time = tracklet.first_seen_time or datetime.now()
        exit_time = tracklet.last_seen_time or datetime.now()
        temporal_features = self.temporal_extractor.extract(enter_time, exit_time)

        # 6. 高频通行指数
        # 7. 聚集行为指数
        aggregation_features = self.aggregation_extractor.extract(
            centers_smooth,
            all_active_positions,
            tracklet.track_id,
            tracklet.frame_indices,
            position_histories,
        )

        # 8. 轨迹摘要
        traj_summary = self.trajectory_extractor.build_trajectory_summary(tracklet)

        # ▸▸▸ Phase 2: 车辆属性特征提取 ◂◂◂
        vehicle_color = "unknown"
        vehicle_type = tracklet.class_name  # Phase 1 默认用 YOLO 类别
        plate_number = ""
        plate_hash = ""

        # ▸▸▸ Phase 2: 车辆属性特征提取 ◂◂◂
        ocr_crop = None  # 用于车牌 OCR 的裁剪图

        if tracklet.best_crop is not None:
            # 颜色识别（用最高置信度裁剪）
            if self.color_recognizer is not None:
                vehicle_color = self.color_recognizer.recognize(tracklet.best_crop)

            # 车型细分（用最高置信度裁剪）
            if self.vehicle_classifier is not None:
                bbox = tracklet.bboxes[-1] if tracklet.bboxes else tracklet.best_crop_bbox
                vehicle_type = self.vehicle_classifier.classify(tracklet.best_crop, bbox)
                if tracklet.class_name in ("truck", "bus", "motorcycle"):
                    vehicle_type = self.vehicle_classifier.classify_from_yolo(
                        tracklet.class_name, bbox
                    )

            # ReID 特征提取
            if self.reid_extractor is not None and self.reid_extractor.enabled:
                reid_emb = self.reid_extractor.extract(tracklet.best_crop)
                tracklet.reid_embedding = reid_emb

        # 确定 OCR 输入：优先用最大裁剪（车最近时），回退到最佳裁剪
        if tracklet.largest_crop is not None:
            lw = tracklet.largest_crop.shape[1]
            if lw >= 250:  # 车辆宽度 >= 250px 时车牌约 80+ px，足够 OCR
                ocr_crop = tracklet.largest_crop
        if ocr_crop is None and tracklet.best_crop is not None:
            ocr_crop = tracklet.best_crop

        # 车牌检测 + OCR
        if ocr_crop is not None and self.plate_detector is not None:
            plate_result = self.plate_detector.detect(ocr_crop)
            if plate_result is not None:
                plate_img, _ = plate_result
                if self.plate_ocr is not None and plate_img.size > 0:
                    plate_number, _ = self.plate_ocr.recognize(plate_img)
                    plate_hash = self.plate_ocr.hash_plate(plate_number)
                    tracklet.plate_number = plate_number
                    tracklet.plate_hash = plate_hash

        # Register frequency after OCR so newly extracted plates are used.
        vehicle_key = tracklet.plate_hash or str(tracklet.track_id)
        self.frequency_extractor.register_vehicle(vehicle_key)
        freq_index = self.frequency_extractor.compute_freq_index(vehicle_key)
        freq_count = self.frequency_extractor.get_count(vehicle_key)

        # 9. 组合完整向量
        vector = {
            # 元数据
            "track_id": tracklet.track_id,
            "camera_id": tracklet.camera_id,
            "vehicle_type": vehicle_type,
            "vehicle_color": vehicle_color,
            "plate_number": plate_number,
            "plate_hash": plate_hash,
            "geometry_level": self.scene.geometry_level,
            "speed_reliable": self.scene.speed_reliable,
            "path_reliable": self.scene.path_reliable,
            "aggregation_reliable": self.scene.aggregation_reliable,
            "passage_reliable": self.scene.passage_reliable,
            "enter_time": traj_summary.get("enter_time", ""),
            "exit_time": traj_summary.get("exit_time", ""),
            "duration_sec": round(tracklet.duration_seconds or 0, 2),
            "trajectory_length_m": traj_summary["trajectory_length_m"],

            # 速度特征
            "avg_speed_ms": speed_features["avg_speed_ms"],
            "max_speed_ms": speed_features["max_speed_ms"],
            "speed_variance": speed_features["speed_variance"],

            # 停留特征
            "max_dwell_sec": dwell_features["max_dwell_sec"],
            "stop_count": dwell_features["stop_count"],
            "dwell_ratio": dwell_features["dwell_ratio"],

            # 路径特征
            "path_deviation": path_features["path_deviation"],
            "path_smoothness": path_features["path_smoothness"],

            # 时间特征
            "is_night": temporal_features["is_night"],
            "night_ratio": temporal_features["night_ratio"],

            # 频次特征
            "freq_index": freq_index,
            "freq_count_24h": freq_count,

            # 聚集特征
            "aggregation_index": aggregation_features["aggregation_index"],
            "nearest_vehicle_m": aggregation_features["nearest_vehicle_m"],

            # 轨迹点串
            "trajectory_points": self._serialize_trajectory(centers_smooth),

            # ReID embedding（序列化时单独处理）
            "reid_embedding": tracklet.reid_embedding.tolist() if tracklet.reid_embedding is not None else [],

            # 异常检测占位
            "anomaly_score": 0.0,
            "is_anomaly": 0,
            "alert_reason": "",
        }

        return vector

    def get_numeric_vector(self, vector_dict: dict) -> np.ndarray:
        values = []
        for label in self.VECTOR_LABELS:
            values.append(float(vector_dict.get(label, 0.0)))
        return np.array(values, dtype=np.float64)

    def normalize(self, vectors: List[dict]) -> List[dict]:
        numeric_arrays = [self.get_numeric_vector(v) for v in vectors]
        normalized = self.normalizer.fit_transform(numeric_arrays)
        for i, vec in enumerate(vectors):
            vec["normalized_vector"] = normalized[i].tolist()
        return vectors

    def save_normalizer(self, filepath: str):
        self.normalizer.save(filepath)

    def load_normalizer(self, filepath: str):
        self.normalizer.load(filepath)

    def reset(self):
        """Clear per-video feature state while keeping extractor models loaded."""
        self.frequency_extractor.reset()

    @staticmethod
    def _serialize_trajectory(centers: List[Tuple[float, float]], max_points: int = 200) -> str:
        if len(centers) <= max_points:
            sampled = centers
        else:
            indices = np.linspace(0, len(centers) - 1, max_points, dtype=int)
            sampled = [centers[i] for i in indices]
        return ";".join(f"{x:.1f},{y:.1f}" for x, y in sampled)
