"""行为向量流水线的可靠性回归测试。

本文件重点验证轨迹连续性、时空特征计算、跨视频聚合、场景标定状态，
以及车牌、ReID 和车辆业务分类在模型缺失或低置信度情况下的降级行为。
"""

from datetime import datetime, timedelta

import numpy as np

from src.behavior.features.aggregation_feature import AggregationFeatureExtractor
from src.behavior.features.speed_feature import SpeedFeatureExtractor
from src.pipeline.frame_pipeline import FramePipeline
from src.pipeline.daily_aggregator import DailyAggregator
from src.pipeline.batch_processor import BatchProcessor
from src.output.vector_writer import VectorWriter
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.tracking.tracklet import Tracklet
from src.trajectory.scene_geometry import SceneGeometry
from src.trajectory.trajectory_extractor import TrajectoryExtractor
from src.trajectory.trajectory_smoother import TrajectorySmoother


def _scene(px_per_meter=10.0):
    """创建不含有效进出口线的最小场景，供特征单元测试使用。"""
    return SceneGeometry(((0, 0), (0, 0)), ((0, 0), (0, 0)), px_per_meter)


def _detection(bbox=(0, 0, 100, 100), confidence=0.9):
    """构造一条格式与检测器输出一致的车辆检测结果。"""
    return {
        "bbox": list(bbox),
        "confidence": confidence,
        "class_id": 2,
        "class_name": "car",
    }


def test_tracker_reactivates_recently_lost_track():
    """车辆短暂漏检后再次出现时，应恢复原轨迹而不是创建新轨迹。"""
    tracker = ByteTrackTracker(track_buffer=3, match_thresh=0.5)
    start = datetime(2026, 6, 1, 8, 0, 0)

    tracker.update([_detection()], 0, start)
    tracker.update([], 1, start + timedelta(seconds=1))
    assert tracker.get_tracklet(0).is_lost

    tracker.update([_detection()], 2, start + timedelta(seconds=2))

    assert tracker.get_tracklet(0).is_active
    assert tracker.get_tracklet(0).trajectory_length == 2
    assert len(tracker.get_all_tracklets()) == 1


def test_tracker_default_iou_threshold_tolerates_bbox_jitter():
    """检测框发生常见抖动时，默认 IoU 阈值仍应保持轨迹连续。"""
    tracker = ByteTrackTracker()
    start = datetime(2026, 6, 1, 8, 0, 0)

    tracker.update([_detection()], 0, start)
    tracker.update([_detection((20, 0, 120, 100))], 1, start + timedelta(seconds=1))

    assert tracker.get_tracklet(0).trajectory_length == 2
    assert len(tracker.get_all_tracklets()) == 1


def test_aggregation_excludes_current_vehicle():
    """计算车辆聚集程度时，不能把当前车辆计为自己的邻车。"""
    extractor = AggregationFeatureExtractor(_scene(), aggregation_radius_m=5)

    result = extractor.extract([(10, 10)], {7: (10, 10)}, current_track_id=7)

    assert result["aggregation_index"] == 0.0
    assert result["nearest_vehicle_m"] == -1.0


def test_aggregation_uses_positions_from_matching_frames():
    """邻车距离必须比较同一帧的位置，不能使用其他时刻的位置。"""
    extractor = AggregationFeatureExtractor(_scene(), aggregation_radius_m=5)
    histories = {
        7: {1: (0, 0), 2: (10, 0)},
        8: {1: (100, 0), 2: (100, 0)},
    }

    result = extractor.extract(
        [(0, 0), (10, 0)],
        {7: (10, 0), 8: (10, 0)},
        current_track_id=7,
        frame_indices=[1, 2],
        position_histories=histories,
    )

    assert result["aggregation_index"] == 0.0
    assert result["nearest_vehicle_m"] == 9.0


def test_speed_uses_measured_timestamp_interval():
    """速度应使用实际时间戳间隔，而不是仅按配置帧率推算。"""
    extractor = SpeedFeatureExtractor(_scene(px_per_meter=10), frame_rate=25)
    timestamps = [
        datetime(2026, 6, 1, 8, 0, 0),
        datetime(2026, 6, 1, 8, 0, 2),
    ]

    result = extractor.extract(np.array([0.0, 20.0]), np.array([0.0, 0.0]), timestamps)

    assert result["avg_speed_ms"] == 1.0


def test_trajectory_summary_uses_track_local_timestamp_index():
    """进场时间索引应基于当前轨迹自身的数据，而不是全局帧编号。"""
    extractor = TrajectoryExtractor(_scene(), TrajectorySmoother())
    start = datetime(2026, 6, 1, 8, 0, 0)
    tracklet = Tracklet(1, "cam01", 2, "car")
    tracklet.add_detection(100, start, [0, 0, 10, 10], (5, 10), 0.9)
    tracklet.add_detection(101, start + timedelta(seconds=1), [1, 0, 11, 10], (6, 10), 0.9)
    tracklet.mark_entered(101)

    summary = extractor.build_trajectory_summary(tracklet)

    assert summary["enter_time"] == (start + timedelta(seconds=1)).isoformat()


class _Resettable:
    """记录 reset 调用次数的测试替身。"""

    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def test_frame_pipeline_reset_clears_video_state():
    """处理新视频前必须清空上一段视频的帧号、结果及各模块状态。"""
    tracker = _Resettable()
    trajectory = _Resettable()
    vectorizer = _Resettable()
    pipeline = FramePipeline(None, None, tracker, trajectory, vectorizer)
    pipeline._frame_idx = 12
    pipeline._start_time = datetime(2026, 6, 1)
    pipeline._completed_vectors.append({"track_id": 1})

    pipeline.reset()

    assert tracker.reset_count == 1
    assert trajectory.reset_count == 1
    assert vectorizer.reset_count == 1
    assert pipeline.frame_idx == 0
    assert pipeline.completed_vectors == []


def test_daily_aggregator_parses_time_and_csv_numeric_values():
    """日聚合器应正确解析 CSV 字符串，并按字段语义合并数值。"""
    aggregator = DailyAggregator()
    group = [
        {
            "camera_id": "cam01",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "enter_time": "2026-06-01T08:00:00",
            "exit_time": "2026-06-01T08:00:02",
            "duration_sec": "2.0",
            "trajectory_length_m": "4.0",
            "avg_speed_ms": "2.0",
            "max_speed_ms": "3.0",
            "speed_variance": "0.2",
            "max_dwell_sec": "0",
            "stop_count": "1",
            "dwell_ratio": "0.25",
            "path_deviation": "0.1",
            "path_smoothness": "0.2",
            "aggregation_index": "0.4",
            "nearest_vehicle_m": "2.5",
        },
        {
            "camera_id": "cam01",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "enter_time": "2026-06-01T09:00:00",
            "exit_time": "2026-06-01T09:00:03",
            "duration_sec": "3.0",
            "trajectory_length_m": "6.0",
            "avg_speed_ms": "2.0",
            "max_speed_ms": "4.0",
            "speed_variance": "0.4",
            "max_dwell_sec": "1",
            "stop_count": "2",
            "dwell_ratio": "0.5",
            "path_deviation": "0.2",
            "path_smoothness": "0.4",
            "aggregation_index": "0.6",
            "nearest_vehicle_m": "1.5",
        },
    ]

    merged = aggregator._merge_group(3, group)

    assert aggregator._parse_time("2026-06-01T08:00:00") == datetime(2026, 6, 1, 8)
    assert merged["duration_sec"] == 5.0
    assert merged["trajectory_length_m"] == 10.0
    assert merged["stop_count"] == 3
    assert merged["aggregation_index"] == 0.6


def test_daily_aggregator_keys_include_video_segment():
    """不同视频片段中相同的局部轨迹号必须生成不同内部键。"""
    aggregator = DailyAggregator()

    first = aggregator._get_track_key({"_segment_id": "08.csv", "track_id": "1"})
    second = aggregator._get_track_key({"_segment_id": "09.csv", "track_id": "1"})

    assert first != second
    assert aggregator._parse_embedding("[0.1, 0.2]").tolist() == [0.1, 0.2]


def test_daily_aggregator_requires_identity_evidence_for_cross_segment_merge():
    """跨视频合并轨迹必须有车牌等身份依据，不能只依赖相同轨迹号。"""
    aggregator = DailyAggregator()
    previous = {
        "_segment_id": "08.csv",
        "track_id": "1",
        "exit_time": "2026-06-01T08:59:59",
        "trajectory_points": "0,0;10,10",
    }
    current = {
        "_segment_id": "09.csv",
        "track_id": "1",
        "enter_time": "2026-06-01T09:00:01",
        "trajectory_points": "11,11;20,20",
    }
    previous_key = aggregator._get_track_key(previous)
    current_key = aggregator._get_track_key(current)
    id_map = {previous_key: 3}

    aggregator._match_across_segments([previous], [current], id_map)
    assert current_key not in id_map

    previous["plate_hash"] = "same"
    current["plate_hash"] = "same"
    aggregator._match_across_segments([previous], [current], id_map)
    assert id_map[current_key] == 3


def test_parse_video_start_time_from_hourly_filename():
    """按小时分段的视频文件名应能解析出录像开始时间。"""
    parsed = BatchProcessor._parse_video_start_time("ch01_20260520_080000_090000")

    assert parsed == datetime(2026, 5, 20, 8, 0, 0)


def test_parse_download_video_timestamp():
    """下载视频的长文件名应能解析业务日期、小时和精确开始时间。"""
    name = "Download_CH05综保区A区中央路北_20260422080046_20260422093204_202606010928376"

    assert BatchProcessor._parse_video_datetime(name) == ("20260422", "08")
    assert BatchProcessor._parse_video_start_time(name) == datetime(2026, 4, 22, 8, 0, 46)


def test_geometry_reliability_flags_for_unconfigured_camera():
    """未标定摄像头产生的时空指标必须标记为不可靠。"""
    scene = SceneGeometry(((0, 0), (0, 0)), ((0, 0), (0, 0)), 20)

    assert scene.geometry_level == "none"
    assert scene.speed_reliable == 0
    assert scene.path_reliable == 0
    assert scene.aggregation_reliable == 0
    assert scene.passage_reliable == 0


def test_geometry_reliability_flags_for_calibrated_camera():
    """完成进出口线和 ROI 标定后，相关时空指标应标记为可靠。"""
    scene = SceneGeometry(
        ((0, 0), (10, 0)),
        ((0, 10), (10, 10)),
        20,
        geometry_level="calibrated",
        has_roi=True,
    )

    assert scene.speed_reliable == 1
    assert scene.path_reliable == 1
    assert scene.aggregation_reliable == 1
    assert scene.passage_reliable == 1


def test_infer_geometry_level_modes():
    """根据 ROI、进出口线及比例尺配置推断场景标定等级。"""
    calibration = {
        "ref_length_m": 5,
        "ref_point_1": [0, 0],
        "ref_point_2": [100, 0],
    }

    assert BatchProcessor._infer_geometry_level(False, {}, {}, 20, calibration) == "none"
    assert BatchProcessor._infer_geometry_level(True, {}, {}, 20, calibration) == "roi"
    assert BatchProcessor._infer_geometry_level(
        True,
        {"p1": [0, 0], "p2": [10, 0]},
        {"p1": [0, 10], "p2": [10, 10]},
        20,
        calibration,
    ) == "calibrated"


def test_vector_writer_includes_geometry_reliability_columns():
    """行为向量 CSV 必须输出场景等级及各指标可靠性字段。"""
    for column in [
        "geometry_level",
        "speed_reliable",
        "path_reliable",
        "aggregation_reliable",
        "passage_reliable",
    ]:
        assert column in VectorWriter.CSV_COLUMNS


def test_plate_detector_falls_back_when_yolo_weight_missing():
    """车牌 YOLO 权重缺失时应关闭该后端，而不是导致流水线崩溃。"""
    from src.feature_extraction.plate_detector import PlateDetector

    detector = PlateDetector(model_path="models/missing_license_plate.pt")

    assert detector.use_yolo is False
    assert detector.model is None


def test_reid_histogram_fallback_returns_normalized_embedding():
    """ReID 权重缺失时应退化为颜色直方图，并输出归一化特征。"""
    from src.feature_extraction.reid_extractor import ReIDExtractor

    crop = np.zeros((80, 120, 3), dtype=np.uint8)
    crop[:, :60] = (255, 255, 255)
    extractor = ReIDExtractor(weights_path="models/missing_reid_model.pth")

    embedding = extractor.extract(crop)

    assert extractor.enabled is True
    assert extractor.active_backend == "histogram"
    assert embedding.shape == (74,)
    assert np.linalg.norm(embedding) > 0
    assert np.isclose(np.linalg.norm(embedding), 1.0)



def test_vector_writer_includes_plate_confidence_column():
    """行为向量 CSV 必须保留车牌识别置信度。"""
    assert "plate_confidence" in VectorWriter.CSV_COLUMNS


def test_plate_ocr_filters_low_confidence_result():
    """低于阈值的车牌文本应被丢弃，同时保留置信度供诊断。"""
    from src.feature_extraction.plate_ocr import PlateOCR

    ocr = PlateOCR(engine="none", lang="en", min_confidence=0.55)
    ocr._recognize_internal = lambda image: ("FDK7S", 0.305)

    plate, confidence = ocr.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

    assert plate == ""
    assert confidence == 0.305


def test_vehicle_classifier_groups_freight_and_ordinary_types():
    """车辆细分类应正确归并为货运车辆和行政通道普通车辆。"""
    from src.feature_extraction.vehicle_classifier import VehicleClassifier

    assert VehicleClassifier.group_vehicle_type("truck") == "freight"
    assert VehicleClassifier.group_vehicle_type("suv") == "ordinary"
    assert VehicleClassifier.is_ordinary_vehicle("sedan") is True
    assert VehicleClassifier.is_ordinary_vehicle("truck") is False


def test_vector_writer_includes_analysis_scope_columns():
    """行为向量 CSV 必须标明车辆业务分组和是否属于分析对象。"""
    assert "vehicle_group" in VectorWriter.CSV_COLUMNS
    assert "analysis_target" in VectorWriter.CSV_COLUMNS
